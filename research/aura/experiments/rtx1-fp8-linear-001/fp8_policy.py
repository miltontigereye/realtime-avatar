"""Selective TorchAO FP8 policy; apply before torch.compile, never to conditioning."""
import re

import torch
from torch import nn
from torch.nn import functional as F


class BF16InputLinear(nn.Linear):
    """Match AURA's BF16-autocast boundary for FP32 modulation activations.

    TorchAO 0.13 otherwise chooses its output dtype from the pre-autocast input,
    requesting FP32 row-scaled GEMM and bypassing the fast BF16-output kernel.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input.to(torch.bfloat16), self.weight, self.bias)


def selected(name: str, module: nn.Module, policy: str) -> bool:
    if policy not in {"ffn", "self_ffn"}:
        raise ValueError(f"Unknown policy: {policy}")
    suffix = r"ffn\.(0|2)" if policy == "ffn" else r"(ffn\.(0|2)|self_attn\.(q|k|v|o))"
    return isinstance(module, nn.Linear) and re.fullmatch(r"blocks\.\d+\." + suffix, name) is not None


def install(model: nn.Module, policy: str, granularity: str) -> list[str]:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
    from torchao.quantization.granularity import PerRow, PerTensor

    names = [name for name, module in model.named_modules() if selected(name, module, policy)]
    if not names:
        raise ValueError("No eligible uncompiled DiT linears found")
    if granularity not in {"row", "tensor"}:
        raise ValueError(granularity)
    config = Float8DynamicActivationFloat8WeightConfig(
        granularity=PerRow() if granularity == "row" else PerTensor()
    )
    for name in names:
        original = model.get_submodule(name)
        if original.weight.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 baseline weight at {name}")
        replacement = BF16InputLinear(original.in_features, original.out_features,
            bias=original.bias is not None, device="meta", dtype=torch.bfloat16)
        replacement.weight = original.weight
        replacement.bias = original.bias
        replacement.train(original.training)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, replacement)
    quantize_(model, config, filter_fn=lambda module, name: selected(name, module, policy))
    for name in names:
        weight = model.get_submodule(name).weight
        if type(weight) is torch.nn.Parameter:
            raise RuntimeError(f"Quantization did not transform {name}")
    return names
