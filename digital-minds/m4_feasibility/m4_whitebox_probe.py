#!/usr/bin/env python3
"""M4 white-box feasibility probe.

Purpose: answer "can this exact class of activation experiment run locally?"
rather than merely "can the model generate text?"

Tested workload:
1. direct MPS model load;
2. representative sequence forward;
3. forward-pre-hook at requested transformer block;
4. target-position residual edit;
5. logits materialization;
6. optional KV-cache retention + one continuation token;
7. MPS allocator telemetry.

This is NOT a scientific experiment.
"""
from __future__ import annotations
import argparse, gc, json, os, platform, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
# Fail fast instead of committing more than physical RAM and swapping. Both must
# be set: the low watermark defaults to 1.4, and a high watermark below it makes
# large allocations raise "invalid low watermark ratio".
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO","0.6")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parent))
from memory_guard import guard, params_for

def gib(x): return float(x)/(1024**3)

def mps_stats():
    if not torch.backends.mps.is_available():
        return {}
    out={}
    for name in ("current_allocated_memory","driver_allocated_memory","recommended_max_memory"):
        fn=getattr(torch.mps,name,None)
        if fn is not None:
            try: out[name+"_gib"]=gib(fn())
            except Exception as e: out[name+"_error"]=repr(e)
    if "driver_allocated_memory_gib" in out and "recommended_max_memory_gib" in out:
        r=out["recommended_max_memory_gib"]
        out["driver_over_recommended"]=out["driver_allocated_memory_gib"]/r if r else None
    return out

def resolve_blocks(model):
    candidates=[
        ("model.layers", lambda m:m.model.layers),
        ("model.model.layers", lambda m:m.model.model.layers),
        ("transformer.h", lambda m:m.transformer.h),
        ("gpt_neox.layers", lambda m:m.gpt_neox.layers),
        ("model.decoder.layers", lambda m:m.model.decoder.layers),
    ]
    for name,getter in candidates:
        try:
            blocks=getter(model)
            if len(blocks)>0:
                return name,blocks
        except Exception:
            pass
    raise RuntimeError("could not resolve transformer blocks")

def make_input(tok,seq_len):
    base=(
        "This is a neutral feasibility passage used only to exercise a transformer "
        "forward pass and activation hook. No scientific conclusion should be "
        "drawn from its content. "
    )
    text=base
    while len(tok(text,add_special_tokens=False)["input_ids"]) < seq_len:
        text += base
    ids=tok(text,add_special_tokens=False,return_tensors="pt")["input_ids"][:,:seq_len]
    return ids

def timed_sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--layer",type=int,required=True)
    ap.add_argument("--seq-len",type=int,default=512)
    ap.add_argument("--with-cache",action="store_true")
    ap.add_argument("--trust-remote-code",action="store_true")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--params-b",type=float,
                    help="billions of parameters, for models not in memory_guard.PARAMS_B")
    ap.add_argument("--wait-s",type=float,default=1800,
                    help="how long to wait for memory to free up before giving up")
    a=ap.parse_args()

    report={
        "model":a.model,"layer":a.layer,"seq_len":a.seq_len,
        "with_cache":a.with_cache,
        "platform":platform.platform(),
        "python":platform.python_version(),
        "torch":torch.__version__,
        "mps_built":torch.backends.mps.is_built(),
        "mps_available":torch.backends.mps.is_available(),
        "stages":[],
    }
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this Python/PyTorch environment")

    def stage(name,t0=None,**extra):
        timed_sync()
        row={"stage":name,**mps_stats(),**extra}
        if t0 is not None: row["elapsed_s"]=time.perf_counter()-t0
        report["stages"].append(row)
        print(json.dumps(row,sort_keys=True),flush=True)

    params_b=a.params_b if a.params_b else params_for(a.model)
    with guard(params_b=params_b,dtype_bytes=2,timeout_s=a.wait_s) as budget:
        report["memory_budget"]=budget.as_dict()
        _run(a,report,stage)

    text=json.dumps(report,indent=2,sort_keys=True)
    print(text)
    if a.out:
        a.out.write_text(text)

def _run(a,report,stage):
    stage("start")
    t=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    stage("tokenizer_loaded",t)

    t=time.perf_counter()
    # Current HF Apple-Silicon loading can map weights directly into Metal buffers.
    model=AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
        trust_remote_code=a.trust_remote_code,
    )
    model.eval()
    model.requires_grad_(False)
    stage("model_loaded",t,dtype=str(next(model.parameters()).dtype))

    block_path,blocks=resolve_blocks(model)
    if not (0<=a.layer<len(blocks)):
        raise ValueError(f"layer {a.layer} outside 0..{len(blocks)-1}")
    report["block_path"]=block_path
    report["n_blocks"]=len(blocks)

    ids=make_input(tok,a.seq_len).to("mps")
    attn=torch.ones_like(ids)
    stage("inputs_ready",tokens=int(ids.numel()))

    captured={}
    def capture_hook(module,args):
        x=args[0]
        captured["shape"]=list(x.shape)
        captured["dtype"]=str(x.dtype)
        # Do NOT retain the full tensor: this is a memory feasibility test.
        captured["last_token_norm"]=float(x[:,-1,:].float().norm().item())

    h=blocks[a.layer].register_forward_pre_hook(capture_hook)
    t=time.perf_counter()
    with torch.inference_mode():
        out=model(input_ids=ids,attention_mask=attn,use_cache=a.with_cache)
        # Force logits to be realized before timing/memory.
        last_logits=out.logits[:,-1,:]
        _=float(last_logits.float().norm().item())
    h.remove()
    stage("capture_forward",t,captured=captured,logits_shape=list(last_logits.shape))

    hidden=int(captured["shape"][-1])
    direction=torch.randn(hidden,device="mps",dtype=torch.bfloat16)
    direction=direction/direction.float().norm().to(direction.dtype)

    hook_fired={"n":0}
    def edit_hook(module,args):
        x=args[0]
        y=x.clone()
        y[:,-1,:]=y[:,-1,:] + direction.to(y.dtype)*1e-3
        hook_fired["n"]+=1
        return (y,*args[1:])

    h=blocks[a.layer].register_forward_pre_hook(edit_hook)
    t=time.perf_counter()
    with torch.inference_mode():
        edited=model(input_ids=ids,attention_mask=attn,use_cache=False)
        edited_logits=edited.logits[:,-1,:]
        _=float(edited_logits.float().norm().item())
    h.remove()
    if hook_fired["n"]!=1:
        raise RuntimeError(f"edit hook fired {hook_fired['n']} times")
    stage("edit_forward",t,hook_fired=hook_fired["n"])

    if a.with_cache:
        past=out.past_key_values
        next_id=torch.argmax(last_logits,dim=-1,keepdim=True)
        t=time.perf_counter()
        with torch.inference_mode():
            step=model(input_ids=next_id,past_key_values=past,use_cache=True)
            _=float(step.logits[:,-1,:].float().norm().item())
        stage("cached_one_token_step",t)

    # Clean big outputs, then report allocator floor after empty_cache.
    del out, edited, last_logits, edited_logits, direction, ids, attn
    if "step" in locals(): del step
    if "past" in locals(): del past
    gc.collect()
    torch.mps.empty_cache()
    stage("after_cleanup")

    rec=max(
        (x.get("driver_over_recommended",0) or 0)
        for x in report["stages"]
    )
    report["peak_driver_over_recommended"]=rec
    report["feasibility_interpretation"]=(
        "PROMISING" if rec < .80 else
        "TIGHT" if rec < .95 else
        "UNSAFE_OR_SWAPPING_RISK"
    )

    # Release the weights before returning. An MPS process that exits without
    # unwinding can leave GPU buffers wired with no owning process, which no
    # amount of waiting reclaims.
    del model, blocks
    gc.collect()
    torch.mps.empty_cache()
    stage("after_model_release")

if __name__=="__main__":
    main()
