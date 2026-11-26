import torch
from torch import nn
from torch.nn import functional as F
from reproductions.confede.modules import encode
from reproductions.ldw.weighting import label_weights
from .gate import ModalityGate, apply_modality_gains


class JointLDWFusion(nn.Module):
    def __init__(self, backbone, enabled=True, hidden=64, coefficient=.35, input_dim=768):
        super().__init__()
        self.backbone=backbone
        self.gate=ModalityGate(input_dim,hidden)
        self.enabled=enabled
        self.coefficient=coefficient if enabled else 0.

    def __getattr__(self,name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name=='backbone':raise
            return getattr(super().__getattr__('backbone'),name)

    def set_train(self,train_module=None):
        self.backbone.set_train(train_module)
        self.gate.requires_grad_(self.enabled)

    def features(self,sample,device):
        t,v,a=encode(self.backbone,sample,device)
        encoded=torch.stack([t,a,v],1)
        blocks=torch.stack([self.T_simi_proj(t),self.V_simi_proj(v),self.A_simi_proj(a),
                            self.T_dissimi_proj(t),self.V_dissimi_proj(v),self.A_dissimi_proj(a)],1)
        weights=self.gate(encoded)
        fused=apply_modality_gains(blocks,weights['weights']) if self.enabled else blocks.flatten(1)
        return (t,v,a),blocks,weights,fused

    def infer(self,sample,device=None):
        device=next(self.parameters()).device if device is None else device
        _,blocks,weights,fused=self.features(sample,device)
        mono=self.mono_decoder(torch.cat(list(blocks.unbind(1)),0)).reshape(6,len(fused))
        return {'M':self.TVA_decoder(fused).reshape(-1),'T':mono[3],'V':mono[4],'A':mono[5],
                'weights':weights['weights']}

    def forward(self,sample1,sample2=None,return_loss=True,device=None):
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
            unimodal=torch.cat([torch.stack([s['regression_labels_'+m].to(device).float().reshape(-1) for m in 'TAV'],1) for s in samples])
            target=label_weights(unimodal,truth).detach()
            kl=F.kl_div(torch.cat(weight_rows,0),target,reduction='batchmean')
        loss=mse+.1*contrastive+.01*mono_loss+self.coefficient*kl
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint ConFEDE/LDW loss')
        return prediction,encoded,loss,mse,contrastive
