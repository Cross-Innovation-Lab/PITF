import torch
from torch import nn
from torch.nn import functional as F
from reproductions.ldw.weighting import label_weights
from .intensity_contrast import IntensityContrastJoint

ARMS=('relative_gap','polarity_intensity','scale_calibrated')
EPSILON=.1
POLARITY_SCALE=.1
POLARITY_WEIGHT=.5


@torch.no_grad()
def difference_values(unimodal,multimodal,mode,scales=(1.,1.,1.)):
    if mode not in ('absolute',*ARMS):raise ValueError(mode)
    if unimodal.ndim!=2 or unimodal.shape[1]!=3 or multimodal.numel()!=len(unimodal):
        raise ValueError('Expected unimodal[N,3] T/A/V and multimodal[N]')
    if not torch.isfinite(unimodal).all() or not torch.isfinite(multimodal).all():raise ValueError('Nonfinite labels')
    if (unimodal.abs()>1+1e-6).any() or (multimodal.abs()>1+1e-6).any():raise ValueError('Labels outside[-1,1]')
    if mode=='absolute':return (unimodal.detach()-multimodal.detach().reshape(-1,1)).abs()
    x=unimodal.detach().double();y=multimodal.detach().double().reshape(-1,1)
    if mode=='relative_gap':d=2*(x-y).abs()/(x.abs()+y.abs()+EPSILON)
    elif mode=='polarity_intensity':
        polarity=.5*((x/POLARITY_SCALE).tanh()-(y/POLARITY_SCALE).tanh()).abs()
        intensity=(x.abs()-y.abs()).abs()
        d=2*(POLARITY_WEIGHT*polarity+(1-POLARITY_WEIGHT)*intensity)
    else:
        a=x.new_tensor(scales).reshape(1,3)
        if not torch.isfinite(a).all() or bool(((a<.5)|(a>2)).any()):raise ValueError('Scales must be in[.5,2]')
        d=((x*a).clamp(-1,1)-y).abs()
    return d.to(unimodal.dtype)


@torch.no_grad()
def difference_weights(unimodal,multimodal,mode,scales=(1.,1.,1.),temperature=1.):
    if temperature<=0:raise ValueError('Temperature must be positive')
    if mode=='absolute':return label_weights(unimodal,multimodal)
    return (-difference_values(unimodal,multimodal,mode,scales)/temperature).softmax(-1)


@torch.no_grad()
def fit_label_scales(unimodal,multimodal):
    x=unimodal.detach().double();y=multimodal.detach().double().reshape(-1,1)
    denominator=x.square().sum(0)
    ratio=(x*y).sum(0)/denominator.clamp_min(1e-12)
    return torch.where(denominator>1e-12,ratio,torch.ones_like(ratio)).clamp(.5,2.)


@torch.no_grad()
def mean_entropy(distances,temperature):
    logq=(-distances.double()/temperature).log_softmax(-1)
    return float(-(logq.exp()*logq).sum(-1).mean())


@torch.no_grad()
def fit_entropy_temperature(distances,target_entropy):
    low,high=1e-6,1e6
    if not mean_entropy(distances,low)<=target_entropy<=mean_entropy(distances,high):
        raise ValueError('Requested entropy is unattainable for this geometry')
    for _ in range(80):
        middle=(low*high)**.5
        if mean_entropy(distances,middle)<target_entropy:low=middle
        else:high=middle
    result=(low*high)**.5
    if abs(mean_entropy(distances,result)-target_entropy)>1e-10:raise AssertionError('Entropy fit failed')
    return result


class DifferenceGeometryJoint(IntensityContrastJoint):
    def __init__(self,original,mode,scales=(1.,1.,1.),temperature=1.):
        if mode not in ('absolute',*ARMS):raise ValueError(mode)
        if original.ntxent_loss.mode!='off':raise ValueError('Contrastive must stay OFF')
        nn.Module.__init__(self)
        self.backbone,self.gate=original.backbone,original.gate
        self.enabled,self.coefficient=original.enabled,original.coefficient
        self.difference_mode,self.scales,self.temperature=mode,tuple(float(v) for v in scales),float(temperature)
        self.alpha=1.

    def forward(self,sample1,sample2=None,return_loss=True,device=None):
        if self.difference_mode=='absolute' or self.alpha==0:
            return super().forward(sample1,sample2,return_loss,device)
        if not return_loss or sample2 is None:return self._forward_geometry(sample1,sample2,return_loss,device)
        adapter=self.backbone.ntxent_loss;n=len(sample1['regression_labels'])
        adapter.begin(torch.cat([sample1['regression_labels'].detach().reshape(n,1),sample2['regression_labels'].detach().reshape(n,6)],1))
        try:
            result=self._forward_geometry(sample1,sample2,return_loss,device)
            assert adapter._position==n
            return result
        finally:adapter.end()

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
            unimodal=torch.cat([torch.stack([s['regression_labels_'+m].to(device).float().reshape(-1) for m in 'TAV'],1) for s in samples])
            target=difference_weights(unimodal,truth,self.difference_mode,self.scales,self.temperature).detach()
            kl=F.kl_div(torch.cat(weight_rows,0),target,reduction='batchmean')
        loss=mse+.1*contrastive+.01*mono_loss+self.coefficient*kl
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint ConFEDE/LDW loss')
        return prediction,encoded,loss,mse,contrastive
