from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from reproductions.confede_ldw.component_ablation import component_values,component_weights
from reproductions.confede_ldw.difference_geometry import difference_values,mean_entropy,fit_entropy_temperature
from reproductions.confede_ldw.gate import ModalityGate,apply_modality_gains
from reproductions.confede.metrics import sentiment_metrics
from triagedmsa.mmap_pickle import load_mmap_pickle

def test_local_package_import():
    import reproductions.confede_ldw.component_ablation as m
    assert Path(m.__file__).is_relative_to(Path(__file__).resolve().parents[1])
def test_component_identity_and_absolute_relation():
    x=torch.tensor([[.5,-.5,0.],[.2,-.2,-.9]],dtype=torch.float64);y=torch.tensor([.5,-.4],dtype=torch.float64)
    p=component_values(x,y,'polarity_only');i=component_values(x,y,'intensity_only')
    torch.testing.assert_close(p+i,difference_values(x,y,'polarity_intensity'),rtol=0,atol=0)
    correction=2*torch.minimum(x.abs(),y[:,None].abs())*(x*y[:,None]<0)
    torch.testing.assert_close((x-y[:,None]).abs(),i+correction)
def test_entropy_calibration_and_no_teacher_grad():
    torch.manual_seed(3);x=(torch.rand(100,3,dtype=torch.float64)*2-1).requires_grad_();y=torch.rand(100,dtype=torch.float64)*2-1
    for m in ('polarity_only','intensity_only'):
        d=component_values(x,y,m);tau=fit_entropy_temperature(d,1.08)
        assert abs(mean_entropy(d,tau)-1.08)<1e-10
        q=component_weights(x,y,m,temperature=tau)
        assert not q.requires_grad
        torch.testing.assert_close(q.sum(-1),torch.ones(100,dtype=q.dtype))
def test_uniform_gate_preserves_original_six_blocks():
    gate=ModalityGate(input_dim=8,hidden=4);x=torch.randn(2,3,8);blocks=torch.randn(2,6,4)
    w=gate(x)['weights'];torch.testing.assert_close(w,torch.full_like(w,1/3))
    torch.testing.assert_close(apply_modality_gains(blocks,w),blocks.flatten(1))
def test_memmap_pickle_roundtrip(tmp_path):
    import pickle
    for protocol in (4,5):
        p=tmp_path/f'{protocol}.pkl';v={'train':np.arange(30,dtype=np.float32).reshape(5,6)}
        with p.open('wb') as f:pickle.dump(v,f,protocol=protocol)
        np.testing.assert_array_equal(load_mmap_pickle(p)['train'],v['train'])
def test_strict_conflict_zero_rule_and_metrics():
    ym=np.array([0.,.2,-.2]);uni=np.array([[1.,-1.,0.],[0.,-.1,.5],[0.,-.1,-.3]])
    np.testing.assert_array_equal(np.any(uni*ym[:,None]<0,axis=1),[False,True,False])
    t=np.array([-1.,0.,1.],dtype=np.float32);m=sentiment_metrics(t,t,'sims')
    assert m['acc2']==m['acc3']==m['acc5']==1. and m['mae']==0.
