"""GPU compatibility check before loading the full AURA checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "deps"))
import torch
from torch import nn
from fp8_policy import install, BF16InputLinear


@torch.no_grad()
def main():
    model = nn.Module()
    block = nn.Module()
    block.ffn = nn.Sequential(nn.Linear(128, 256), nn.GELU(approximate="tanh"), nn.Linear(256, 128))
    model.blocks = nn.ModuleList([block])
    model.to("cuda", torch.bfloat16).eval().requires_grad_(False)
    x = torch.randn(1, 128, 128, device="cuda", dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = model.blocks[0].ffn(x)
    names = install(model, "ffn", "row")
    assert len(names) == 2
    assert isinstance(model.blocks[0].ffn[0], BF16InputLinear)
    compiled = torch.compile(model.blocks[0].ffn, dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager = model.blocks[0].ffn(x)
        actual = compiled(x)
    assert actual.shape == baseline.shape and actual.dtype == baseline.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    relative = float((actual.float() - baseline.float()).norm() / baseline.float().norm())
    assert relative < 0.1, relative
    print({"dtype": str(actual.dtype), "shape": list(actual.shape), "relative_l2": relative,
           "eager_compiled_relative_l2": float((actual.float() - eager.float()).norm() / eager.float().norm()),
           "quantized_modules": names}, flush=True)


if __name__ == "__main__":
    main()
