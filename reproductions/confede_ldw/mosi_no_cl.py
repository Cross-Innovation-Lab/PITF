import torch
from torch import nn
from torch.nn import functional as F
from reproductions.confede.modules import encode
from reproductions.ldw.weighting import label_weights
from .joint import JointLDWFusion
from .gate import ModalityGate, apply_modality_gains
from .difference_geometry import difference_weights
from .predicted_label_teacher import gather_teacher

ARMS=('no_ldw','absolute','polarity_intensity')


@torch.no_grad()
def teacher_weights(unimodal,truth,mode,temperature=1.):
    if mode=='absolute':
        return label_weights(unimodal.detach(),truth.detach())
    if mode!='polarity_intensity':raise ValueError(mode)
    return difference_weights(unimodal,truth,'polarity_intensity',temperature=temperature)


class MOSINoCLFusion(JointLDWFusion):
    def __init__(self,backbone,arm,teacher_TAV=None,temperature=1.,input_dim=768,hidden=64):
        if arm not in ARMS:raise ValueError(arm)
        nn.Module.__init__(self)
        self.backbone=backbone
        self.arm=arm
        self.enabled=arm!='no_ldw'
        self.coefficient=.35 if self.enabled else 0.
        self.temperature=float(temperature)
        saved=torch.get_rng_state()
        try:gate=ModalityGate(input_dim,hidden)
        finally:torch.set_rng_state(saved)
        self.gate=gate if self.enabled else None
        bank=torch.empty(0,3) if teacher_TAV is None else torch.as_tensor(teacher_TAV,dtype=torch.float32).detach().clone()
        if bank.ndim!=2 or bank.shape[1]!=3 or not len(bank):
            raise ValueError('Expected TRAIN predictions [N,3] T/A/V')
        if not torch.isfinite(bank).all() or bool((bank.abs()>1).any()):
            raise ValueError('Teacher predictions must be finite and normalized to [-1,1]')
        self.register_buffer('_teacher_TAV',bank,persistent=False)
        self.last_losses={}

    def set_train(self,train_module=None):
        self.backbone.set_train(train_module)
        if self.gate is not None:self.gate.requires_grad_(True)

    def features(self,sample,device):
        t,v,a=encode(self.backbone,sample,device)
        encoded=torch.stack([t,a,v],1)
        blocks=torch.stack([self.T_simi_proj(t),self.V_simi_proj(v),self.A_simi_proj(a),
                            self.T_dissimi_proj(t),self.V_dissimi_proj(v),self.A_dissimi_proj(a)],1)
        if self.enabled:
            weights=self.gate(encoded)
            fused=apply_modality_gains(blocks,weights['weights'])
        else:
            w=encoded.new_full((len(t),3),1/3)
            weights=dict(weights=w,log_weights=w.log())
            fused=blocks.flatten(1)
        return (t,v,a),blocks,weights,fused

    def forward(self,sample1,sample2=None,return_loss=True,device=None):
        device=next(self.parameters()).device if device is None else device
        if not return_loss:return self.infer(sample1,device)
        encoded,first,w,first_fused=self.features(sample1,device)
        samples=[sample1];blocks=[first];fused=[first_fused];logw=[w['log_weights']]
        if sample2 is not None:
            _,second,w2,second_fused=self.features(sample2,device)
            samples.append(sample2);blocks.append(second);fused.append(second_fused);logw.append(w2['log_weights'])
        truth=torch.cat([s['regression_labels'].to(device).float().reshape(-1) for s in samples])
        prediction=self.TVA_decoder(torch.cat(fused,0)).reshape(-1)
        mono_features=torch.cat([torch.cat(list(b.unbind(1)),0) for b in blocks],0)
        teachers=[gather_teacher([s],self._teacher_TAV,device) for s in samples]
        mono_truth=torch.cat([torch.cat([s['regression_labels'].to(device).float().reshape(-1)]*3+
                              [teacher[:,i] for i in (0,2,1)]) for s,teacher in zip(samples,teachers)])
        mse=F.mse_loss(prediction,truth)
        mono=F.mse_loss(self.mono_decoder(mono_features).reshape(-1),mono_truth)
        kl=mse.new_zeros(())
        if self.enabled:
            teacher=torch.cat(teachers,0)
            target=teacher_weights(teacher,truth,self.arm,self.temperature)
            kl=F.kl_div(torch.cat(logw,0),target,reduction='batchmean')
        loss=mse+.01*mono+self.coefficient*kl
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MOSI CL-off loss')
        self.last_losses=dict(main=mse.detach(),mono=mono.detach(),kl=kl.detach())
        return prediction,encoded,loss,mse,mse.new_zeros(())
