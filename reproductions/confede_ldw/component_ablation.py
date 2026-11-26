from types import FunctionType
import torch
from .difference_geometry import DifferenceGeometryJoint
from .mosi_no_cl import MOSINoCLFusion

ARMS=('polarity_only','intensity_only')

@torch.no_grad()
def component_values(unimodal,multimodal,mode):
    if mode not in ARMS:raise ValueError(mode)
    if unimodal.ndim!=2 or unimodal.shape[1]!=3 or multimodal.numel()!=len(unimodal):raise ValueError('shape')
    if not torch.isfinite(unimodal).all() or not torch.isfinite(multimodal).all():raise ValueError('nonfinite')
    if (unimodal.abs()>1+1e-6).any() or (multimodal.abs()>1+1e-6).any():raise ValueError('range')
    x=unimodal.detach().double();y=multimodal.detach().double().reshape(-1,1)
    d=.5*((x/.1).tanh()-(y/.1).tanh()).abs() if mode=='polarity_only' else (x.abs()-y.abs()).abs()
    return d.to(unimodal.dtype)

@torch.no_grad()
def component_weights(unimodal,multimodal,mode,scales=(1.,1.,1.),temperature=1.):
    if temperature<=0:raise ValueError('temperature')
    return (-component_values(unimodal,multimodal,mode)/temperature).softmax(-1)

_method=DifferenceGeometryJoint._forward_geometry
_forward=FunctionType(_method.__code__,{**_method.__globals__,'difference_weights':component_weights},
                      _method.__name__,_method.__defaults__,_method.__closure__)

class ComponentAblationJoint(DifferenceGeometryJoint):
    _forward_geometry=_forward
    def __init__(self,original,mode,temperature):
        if mode not in ARMS:raise ValueError(mode)
        super().__init__(original,'polarity_intensity',temperature=temperature)
        self.difference_mode=mode

def _bank_teacher(unimodal,truth,mode,temperature=1.):
    return component_weights(unimodal,truth,mode,temperature=temperature)

_bank_method=MOSINoCLFusion.forward
_bank_forward=FunctionType(_bank_method.__code__,{**_bank_method.__globals__,'teacher_weights':_bank_teacher},
    _bank_method.__name__,_bank_method.__defaults__,_bank_method.__closure__)

class ComponentAblationV2(MOSINoCLFusion):
    forward=_bank_forward
    def __init__(self,original,mode,temperature):
        if mode not in ARMS:raise ValueError(mode)
        torch.nn.Module.__init__(self)
        self.backbone,self.gate=original.backbone,original.gate
        self.arm,self.temperature=mode,float(temperature)
        self.enabled,self.coefficient=original.enabled,original.coefficient
        self.register_buffer('_teacher_TAV',original._teacher_TAV,persistent=False)
        self.last_losses={}
