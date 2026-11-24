"""Experimental RoPE arithmetic; frequency angles are still generated in FP64."""
import torch
from aura import model
ORIGINAL_APPLY=model.rope_apply
ORIGINAL_FREQS=model.rope_freqs

@torch.autocast("cuda", enabled=False)
def fp32_inline(x, freqs):
    b,s,n,d=x.shape
    pairs=x.float().reshape(b,s,n,d//2,2)
    xr,xi=pairs[...,0],pairs[...,1]
    f=torch.view_as_real(freqs).float()
    fr,fi=f[...,0],f[...,1]
    return torch.stack([xr*fr-xi*fi,xr*fi+xi*fr],dim=-1).reshape(b,s,n,d).to(x.dtype)

@torch.autocast("cuda", enabled=False)
def fp32_prepacked(x, freqs):
    b,s,n,d=x.shape
    pairs=x.float().reshape(b,s,n,d//2,2)
    xr,xi=pairs[...,0],pairs[...,1]
    fr,fi=freqs[...,0],freqs[...,1]
    return torch.stack([xr*fr-xi*fi,xr*fi+xi*fr],dim=-1).reshape(b,s,n,d).to(x.dtype)

def real_fp32_freqs(*args,**kwargs):
    return torch.view_as_real(ORIGINAL_FREQS(*args,**kwargs)).float().contiguous()

def install(name):
    model.rope_apply={'fp64_control':ORIGINAL_APPLY,'fp32_inline':fp32_inline,'fp32_prepacked':fp32_prepacked}[name]
    model.rope_freqs=real_fp32_freqs if name=='fp32_prepacked' else ORIGINAL_FREQS
