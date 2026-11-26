from __future__ import annotations
import torch
from torch import nn
from pytorch_metric_learning.utils import common_functions as c_f
from .joint import JointLDWFusion


@torch.no_grad()
def sentiment_negative_weights(seven_labels,negative_anchors,negative_partners,positions=7,floor=.1):
    y=seven_labels.detach().reshape(-1)
    if len(y)!=positions or not torch.isfinite(y).all():
        raise ValueError('Expected seven finite TRAIN sentiment labels')
    if bool((y.abs()>1+1e-6).any()):
        raise ValueError('SIMS contrastive labels must lie in [-1,1]')
    a,n=negative_anchors,negative_partners
    cross=(a//positions<3)&(n//positions<3)&(a%positions!=n%positions)
    difference=(y[a%positions]-y[n%positions]).abs().clamp(max=2)
    raw=floor+(1-floor)*difference/2
    opposite=(y[a%positions]>0)!=(y[n%positions]>0)
    raw=torch.where(opposite,torch.ones_like(raw),raw)
    peers=(a[:,None]==a[None,:])&cross[None,:]
    means=(raw[None,:]*peers).sum(1)/peers.sum(1).clamp_min(1)
    return torch.where(cross,raw/means.clamp_min(1e-12),torch.ones_like(raw)),cross


def weighted_pair_loss(original,embeddings,labels,indices,negative_weights):
    a1,p,a2,n=indices
    if len(a1)==0 or len(a2)==0:
        return original(embeddings,labels,indices_tuple=indices)
    mat=original.distance(embeddings)
    pos_pairs,neg_pairs=mat[a1,p],mat[a2,n]
    if not original.distance.is_inverted:
        pos_pairs=-pos_pairs;neg_pairs=-neg_pairs
    positive=pos_pairs.unsqueeze(1)/original.temperature
    negative=neg_pairs/original.temperature+negative_weights.to(neg_pairs).log()
    per_positive=c_f.to_dtype(a2.unsqueeze(0)==a1.unsqueeze(1),dtype=negative.dtype)
    negative=negative*per_positive
    negative[per_positive==0]=c_f.neg_inf(negative.dtype)
    maximum=torch.max(positive,torch.max(negative,dim=1,keepdim=True)[0]).detach()
    numerator=torch.exp(positive-maximum).squeeze(1)
    denominator=torch.sum(torch.exp(negative-maximum),dim=1)+numerator
    losses=-torch.log(numerator/denominator+c_f.small_val(negative.dtype))
    return original.reducer({'loss':{'losses':losses,'indices':(a1,p),'reduction_type':'pos_pair'}},embeddings,labels)


class IntensityContrastiveLoss(nn.Module):
    def __init__(self,original,mode='weighted'):
        super().__init__()
        if mode not in ('original','off','weighted'):raise ValueError(mode)
        self.original=original
        self.mode=mode
        self.weighting_active=True
        self._contexts=None
        self._position=0
        self._summaries=[]
        self.last_diagnostics={}

    def begin(self,labels):
        if self._contexts is not None:raise RuntimeError('Nested contrastive context')
        self._contexts=labels.detach()
        self._position=0;self._summaries=[]

    def end(self):
        calls=self._position
        self._contexts=None;self._position=0
        if self._summaries:
            x=torch.stack(self._summaries).mean(0).cpu().tolist()
            self.last_diagnostics={'cross_weight_mean':x[0],'cross_weight_min_mean':x[1],
                                   'cross_weight_max_mean':x[2],'same_binary_cross_negative_fraction':x[3],
                                   'mean_cross_sentiment_gap':x[4],'calls':calls}
        else:self.last_diagnostics={'calls':calls}
        self._summaries=[]

    def forward(self,embeddings,labels,indices_tuple=None):
        position=self._position;self._position+=1
        if self.mode=='off':return embeddings.new_zeros(())
        if self.mode=='original' or not self.weighting_active:
            return self.original(embeddings,labels,indices_tuple=indices_tuple)
        if self._contexts is None:raise RuntimeError('Missing TRAIN sentiment context')
        y=self._contexts[position].to(embeddings)
        a1,p,a2,n=indices_tuple
        weights,cross=sentiment_negative_weights(y,a2,n)
        with torch.no_grad():
            w=weights[cross]
            same=(y[a2[cross]%7]>0)==(y[n[cross]%7]>0)
            gap=(y[a2[cross]%7]-y[n[cross]%7]).abs()
            if w.numel():self._summaries.append(torch.stack([w.mean(),w.min(),w.max(),same.float().mean(),gap.mean()]).detach())
        return weighted_pair_loss(self.original,embeddings,labels,indices_tuple,weights)


class IntensityContrastJoint(JointLDWFusion):
    def __init__(self,backbone,contrastive_mode='weighted',**kwargs):
        super().__init__(backbone,**kwargs)
        self.backbone.ntxent_loss=IntensityContrastiveLoss(self.backbone.ntxent_loss,contrastive_mode)

    def forward(self,sample1,sample2=None,return_loss=True,device=None):
        adapter=self.backbone.ntxent_loss
        if not return_loss or sample2 is None:
            return super().forward(sample1,sample2,return_loss,device)
        n=len(sample1['regression_labels'])
        anchor=sample1['regression_labels'].detach().reshape(n,1)
        partners=sample2['regression_labels'].detach().reshape(n,6)
        adapter.begin(torch.cat([anchor,partners],1))
        try:
            result=super().forward(sample1,sample2,return_loss,device)
            if adapter._position!=n:raise AssertionError('Contrastive anchor/partner ordering mismatch')
            return result
        finally:adapter.end()
