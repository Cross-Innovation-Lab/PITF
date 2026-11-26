import torch
from torch.nn import functional as F
from .difference_geometry import DifferenceGeometryJoint, difference_weights


@torch.no_grad()
def gather_teacher(samples, bank, device):
    rows = []
    for sample in samples:
        indices = torch.as_tensor(sample['index'], device=bank.device)
        if indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError('Teacher lookup requires a vector of TRAIN integer indices')
        if len(indices) != len(sample['regression_labels']):
            raise ValueError('Teacher/sample alignment mismatch')
        if bool(((indices < 0) | (indices >= len(bank))).any()):
            raise ValueError('Teacher index outside TRAIN bank')
        rows.append(bank.index_select(0, indices))
    return torch.cat(rows, 0).to(device=device, dtype=torch.float32).detach()


class PredictedLabelGeometryJoint(DifferenceGeometryJoint):
    def __init__(self, original, teacher_TAV, temperature):
        super().__init__(original, 'polarity_intensity', (1., 1., 1.), temperature)
        bank = torch.as_tensor(teacher_TAV, dtype=torch.float32).detach().clone()
        if bank.ndim != 2 or bank.shape[1] != 3 or not len(bank):
            raise ValueError('Expected TRAIN teacher bank [N,3] in T/A/V order')
        if not torch.isfinite(bank).all() or bool((bank.abs() > 1).any()):
            raise ValueError('Teacher predictions must be finite and clipped to [-1,1]')
        self.register_buffer('_teacher_TAV', bank, persistent=False)

    def _forward_geometry(self,sample1,sample2=None,return_loss=True,device=None):
        device=next(self.parameters()).device if device is None else device
        if not return_loss:return self.infer(sample1,device)
        encoded,first,weights1,fused1=self.features(sample1,device)
        samples=[sample1];blocks=[first];fused=[fused1];weight_rows=[weights1['log_weights']]
        contrastive=fused1.new_zeros(())
        if sample2 is not None:
            _,second,weights2,fused2=self.features(sample2,device)
            samples.append(sample2);blocks.append(second);fused.append(fused2);weight_rows.append(weights2['log_weights'])
            indices=tuple(torch.tensor(getattr(self.config,key),device=device) for key in ('t1','p','t2','n'))
            pair_labels=torch.tensor([0,0,0,1,2,3,4,0,0,0,1,2,3,4,0,0,0,1,2,3,4,
                                      5,5,5,6,7,8,9,5,5,5,6,7,8,9,5,5,5,6,7,8,9])
            for index in range(len(fused1)):
                paired=[torch.cat([first[index,j].unsqueeze(0),second[6*index:6*(index+1),j]],0) for j in range(6)]
                contrastive=contrastive+self.ntxent_loss(torch.cat(paired,0),pair_labels,indices_tuple=indices)
            contrastive=contrastive/len(fused1)
        truth=torch.cat([s['regression_labels'].to(device).float().reshape(-1) for s in samples])
        mono_truth=torch.cat([torch.cat([s['regression_labels'].to(device).float().reshape(-1)]*3+
                                       [s['regression_labels_'+m].to(device).float().reshape(-1) for m in 'TVA']) for s in samples])
        mono_features=torch.cat([torch.cat(list(b.unbind(1)),0) for b in blocks],0)
        prediction=self.TVA_decoder(torch.cat(fused,0)).reshape(-1)
        mse=F.mse_loss(prediction,truth)
        mono_loss=F.mse_loss(self.mono_decoder(mono_features).reshape(-1),mono_truth)
        kl=mse.new_zeros(())
        if self.coefficient:
            unimodal=gather_teacher(samples,self._teacher_TAV,device)
            target=difference_weights(unimodal,truth,self.difference_mode,self.scales,self.temperature).detach()
            kl=F.kl_div(torch.cat(weight_rows,0),target,reduction='batchmean')
        loss=mse+.1*contrastive+.01*mono_loss+self.coefficient*kl
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint ConFEDE/LDW loss')
        return prediction,encoded,loss,mse,contrastive
