"""5B geometry: conditioning reuse through eviction and Wan22 decoder streaming."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from aura.configs import CONFIGS
from aura.model import AuraDiT
from aura.kv_cache import KVCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_kernels/taehv-011dfc2"))
from taehv import TAEHV, StreamingTAEHV


@torch.no_grad()
def main():
    torch.manual_seed(41)
    cfg = replace(CONFIGS["aura-5b"], dim=24, ffn_dim=48, num_heads=3,
                  num_layers=2, audio_inject_layers=[])
    model = AuraDiT(cfg).cuda().eval()
    def cache():
        return KVCache(2, 3, 8, 4, 4, 2, cfg.segment_frames, cfg.ref_rope_offset,
                       rope_max_block=None)
    ref = torch.randn(1, 48, 1, 4, 4, device="cuda")
    context = torch.randn(7, 4096, device="cuda")
    prepared = model.prepare_conditioning(context, ref)
    a, b = cache(), cache()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk in range(10):
            for i, step in enumerate(cfg.denoising_steps):
                x = torch.randn(48, 4, 4, 4, device="cuda")
                t = torch.tensor([step], device="cuda")
                expected = model(x, t, context, ref, None, a)
                actual = model(x, t, context, ref, None, b, prepared=prepared[i])
                torch.testing.assert_close(actual, expected, atol=.003, rtol=.003)
            for ka, kb in zip(a.k, b.k):
                torch.testing.assert_close(ka, kb, atol=.003, rtol=.003)
            a.advance()
            b.advance()
    del model, a, b, prepared
    weights = ROOT / "research_kernels/lightx2v-autoencoders-wan22/lighttaew2_2_v2.safetensors"
    decoder = TAEHV(str(weights)).cuda().half().eval()
    assert decoder.latent_channels == 48 and decoder.patch_size == 2
    latent = torch.randn(1, 8, 48, 4, 6, device="cuda", dtype=torch.float16) * .1
    full = decoder.decode_video(latent, parallel=False, show_progress_bar=False)
    stream = StreamingTAEHV(decoder)
    chunks = []
    for block in latent.split(4, dim=1):
        frames = []
        frame = stream.decode(block)
        while frame is not None:
            frames.append(frame)
            frame = stream.decode()
        chunks.append(torch.cat(frames, dim=1))
    assert [v.shape[1] for v in chunks] == [13, 16]
    assert tuple(full.shape) == (1, 29, 3, 64, 96)
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=.01, rtol=.01)
    # Reset must remove all history before another portrait/session.
    stream.reset()
    frame = stream.decode(latent[:, :4])
    torch.testing.assert_close(frame, chunks[0][:, :1], atol=0, rtol=0)
    print(json.dumps({"conditioning_through_eviction_and_segment_boundary": "pass",
                      "wan22_streaming_vs_whole": "pass", "decoder_reset": "pass",
                      "first_and_subsequent_frames": [13, 16],
                      "max_decoder_difference": float((torch.cat(chunks, dim=1) - full).abs().max())}))


if __name__ == "__main__":
    main()
