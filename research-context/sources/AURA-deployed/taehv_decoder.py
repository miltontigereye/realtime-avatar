"""Streaming TAEHV (taew2_1) drop-in for Aura.decode: same normalized Wan2.1 latents in,
[3, F, H, W] video in [-1, 1] out, decoder memory carried across chunks."""
import sys
import torch
sys.path.insert(0, '/workspace/avatar/research/aura/research_kernels/taehv-011dfc2')
from taehv import TAEHV, StreamingTAEHV
WEIGHTS = '/workspace/avatar/research/aura/research_kernels/taehv-011dfc2/taew2_1.safetensors'

class TaehvDecode:
    def __init__(self, device='cuda', dtype=torch.float16, weights=WEIGHTS, raw_scale=None):
        """raw_scale=(mean, inv_std): convert normalized latents to raw Wan latents first (LightTAE expects raw;
        taew2_1 expects normalized). Verified on real frames: scaling-probe.txt in runs/rtx1-taehv-20261124-03."""
        self.dtype = dtype
        self.raw_scale = raw_scale
        self.taehv = TAEHV(weights).to(device, dtype).eval()
        self.stream = StreamingTAEHV(self.taehv)
    @torch.no_grad()
    def __call__(self, latents, first):
        if first:
            self.stream.reset()
        with torch.autocast('cuda', enabled=False):
            if self.raw_scale is not None:
                mean, inv_std = self.raw_scale
                latents = latents.float() / inv_std.view(-1, 1, 1, 1).float() + mean.view(-1, 1, 1, 1).float()
            x = latents.permute(1, 0, 2, 3)[None].to(self.dtype)  # [C,F,h,w] -> NTCHW
            frames = []
            frame = self.stream.decode(x)
            while frame is not None:
                frames.append(frame)
                frame = self.stream.decode()
            video = torch.cat(frames, dim=1)[0]  # T,C,H,W in [0,1]
            return video.permute(1, 0, 2, 3).float().mul_(2).sub_(1)

def install(aura, weights=WEIGHTS, raw=False):
    """Replace aura.decode; the Wan VAE stays loaded (session cleanup still calls it)."""
    dec = TaehvDecode(aura.device, weights=weights, raw_scale=aura.vae.scale if raw else None)
    aura.wan_decode = aura.decode
    aura.decode = dec
    return dec
