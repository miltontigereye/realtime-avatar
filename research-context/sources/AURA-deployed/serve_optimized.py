"""Native AURA profiles on rtx1; four steps, SDPA, BF16 DiT, streaming LightTAE."""
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch._dynamo.utils import counters

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/rtx1-taehv-001"))
from aura.pipeline import Aura, DEFAULT_PROMPT
from aura.streaming_audio import LiveAudioFeatures, SAMPLE_RATE
from rope_variants import install as install_rope
import taehv_decoder
from web import server

PROFILES = {
    "aura-1.3b": (768, 512, ROOT / "research_kernels/lightx2v-autoencoders/lighttaew2_1_v2.safetensors"),
    "aura-5b": (1280, 704, ROOT / "research_kernels/lightx2v-autoencoders-wan22/lighttaew2_2_v2.safetensors"),
}


class OptimizedAura(Aura):
    def __init__(self, name="aura-5b", **kwargs):
        width, height, weights = PROFILES[name]
        if not weights.is_file():
            raise FileNotFoundError(weights)
        kwargs.update(compile=False, bf16_transformer=True,
                      offload_text_encoder=True, offload_transformer_for_decode=False)
        super().__init__(name, **kwargs)
        self.cfg = replace(self.cfg, buckets={height / width: (height, width)})
        self.prepare_static = self.gpu_pixels = True
        install_rope("fp32_inline")
        torch.backends.cudnn.benchmark = False
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
        for index, block in enumerate(self.model.blocks):
            self.model.blocks[index] = torch.compile(
                block, dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
        taehv_decoder.install(self, weights=str(weights), raw=True)
        self._warmed = False
        self.runtime_info = {
            "model": name, "resolution": [width, height], "playback_fps": self.fps,
            "steps": list(self.cfg.denoising_steps), "attention": "SDPA",
            "profile": "max-autotune-no-cudagraphs", "compile_dynamic": False,
            "prepared_conditioning": True, "gpu_pixel_packing": True,
            "transformer": "BF16 except time/audio FP32", "text_encoder_device": "CPU",
            "decoder": weights.name, "decoder_dtype": "FP16", "raw_latents": True,
            "cold_start_requires_compilation": True,
            "quality_note": "Lightweight decoder changes pixels; 5B quality validation is recorded separately.",
        }

    @torch.no_grad()
    def start_stream(self, image, prompt=DEFAULT_PROMPT, seed=42):
        if not self._warmed:
            started = time.perf_counter()
            dummy = audio = None
            stable = 0
            try:
                dummy = super().start_stream(image, prompt, seed)
                audio = LiveAudioFeatures(self)
                frames = self.cfg.frames_per_block * self.cfg.vae_stride[0]
                pcm = np.zeros(round(SAMPLE_RATE * frames / self.fps), dtype=np.float32)
                for index in range(30):
                    before = int(counters["stats"]["unique_graphs"])
                    video = dummy.render(audio.extract(pcm, frames))
                    if not torch.isfinite(video).all():
                        raise RuntimeError("Nonfinite warm-up output")
                    stable = stable + 1 if int(counters["stats"]["unique_graphs"]) == before else 0
                    logging.info("Disposable warm-up chunk %d; stable=%d", index + 1, stable)
                    if index >= 5 and stable >= 3:
                        break
                else:
                    raise RuntimeError("Renderer warm-up did not stabilize")
                self._warmed = True
                self.runtime_info["warmup_seconds"] = time.perf_counter() - started
                self.runtime_info["warmup_chunks"] = index + 1
            finally:
                if audio is not None:
                    audio.close()
                if dummy is not None:
                    dummy.close()
        return super().start_stream(image, prompt, seed)


if __name__ == "__main__":
    server.Aura = OptimizedAura
    server.main()
