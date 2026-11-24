"""The experiment must never quantize speech/text conditioning by accident."""
import unittest

import torch
from torch import nn
from fp8_policy import BF16InputLinear, selected


class PolicyTests(unittest.TestCase):
    def test_exact_target_names(self):
        linear = nn.Linear(16, 16)
        for name in ["blocks.0.ffn.0", "blocks.29.ffn.2"]:
            self.assertTrue(selected(name, linear, "ffn"))
            self.assertTrue(selected(name, linear, "self_ffn"))
        for projection in ["q", "k", "v", "o"]:
            name = f"blocks.12.self_attn.{projection}"
            self.assertFalse(selected(name, linear, "ffn"))
            self.assertTrue(selected(name, linear, "self_ffn"))

    def test_conditioning_and_compiled_names_excluded(self):
        linear = nn.Linear(16, 16)
        for name in ["blocks.0.cross_attn.q", "audio_injectors.0.attn.q",
                     "time_projection.1", "text_embedding.0", "head.head",
                     "casual_audio_encoder.final_linear", "blocks.0._orig_mod.ffn.0",
                     "blocks.0.ffn.0.extra", "other.blocks.0.ffn.0"]:
            self.assertFalse(selected(name, linear, "self_ffn"), name)
        self.assertFalse(selected("blocks.0.ffn.0", nn.LayerNorm(16), "self_ffn"))

    def test_invalid_policy_fails(self):
        with self.assertRaises(ValueError):
            selected("blocks.0.ffn.0", nn.Linear(16, 16), "all")

    def test_explicit_bf16_matches_original_autocast(self):
        # CPU autocast exercises the same dtype boundary without consuming GPU time.
        original = nn.Linear(32, 64, dtype=torch.bfloat16)
        boundary = BF16InputLinear(32, 64, dtype=torch.bfloat16)
        boundary.load_state_dict(original.state_dict())
        x = torch.randn(2, 17, 32, dtype=torch.float32)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = original(x)
            actual = boundary(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
