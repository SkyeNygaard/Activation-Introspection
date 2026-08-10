import sys
from pathlib import Path
import types
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scoring import appended_tokens, score_options

class Tok:
    # whitespace-ish tokenizer with optional split for "LONG"
    def __init__(self):
        self.vocab = {"P":0," A":1," B":2," L":3,"ONG":4}
    def _enc(self, text):
        if text == "P": return [0]
        if text == "P A": return [0,1]
        if text == "P B": return [0,2]
        if text == "P LONG": return [0,3,4]
        raise KeyError(text)
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = self._enc(text)
        if return_tensors:
            return types.SimpleNamespace(input_ids=torch.tensor([ids]))
        return types.SimpleNamespace(input_ids=ids)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
    def forward(self, ids):
        b,s = ids.shape
        logits = torch.zeros(b,s,5)
        # after prompt, A better than B
        logits[:,-1,1]=2.0
        logits[:,-1,2]=1.0
        # sequence LONG: first L moderate, then ONG high
        if s >= 2:
            logits[:,0,3]=1.5
            logits[:,1,4]=2.5
        return types.SimpleNamespace(logits=logits)

def test_single_token_fast_path():
    t=Tok(); m=M()
    r=score_options(m,t,torch.nn.ModuleList([]),"P",(" A"," B"))
    assert r["single_token_fast_path"]
    assert r["predicted"]==" A"

def test_multitoken_fallback():
    t=Tok(); m=M()
    r=score_options(m,t,torch.nn.ModuleList([]),"P",(" A"," LONG"))
    assert not r["single_token_fast_path"]
    assert len(r["logprobs"])==2

def test_appended_tokens():
    t=Tok()
    assert appended_tokens(t,"P"," A")==[1]
    assert appended_tokens(t,"P"," LONG")==[3,4]
