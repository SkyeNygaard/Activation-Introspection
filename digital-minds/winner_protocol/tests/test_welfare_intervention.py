import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from welfare_intervention import PublishedActAdd, targeted_published_actadd, _apply

def test_raw_targeted_semantics():
    h = torch.zeros(1, 6, 4)
    v = torch.tensor([1.,2.,3.,4.])
    e = PublishedActAdd(0,v,2.0,(1,4),(+1,-1))
    y = _apply(h,e)
    assert torch.equal(y[0,1],2*v)
    assert torch.equal(y[0,4],-2*v)
    assert torch.equal(h,torch.zeros_like(h))

def test_prehook_cleanup():
    class B(torch.nn.Module):
        def forward(self,x): return x+10
    b=B(); blocks=torch.nn.ModuleList([b])
    v=torch.ones(4)
    e=PublishedActAdd(0,v,2.0,(1,),(+1,))
    x=torch.zeros(1,3,4)
    with targeted_published_actadd(blocks,e):
        assert torch.equal(b(x)[0,1],torch.full((4,),12.))
    assert torch.equal(b(x),torch.full_like(x,10.))
