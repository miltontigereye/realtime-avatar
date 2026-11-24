"""Isolated native AURA/LightTAE FP8 experiment, configured by command-line arguments.

Run each policy in a fresh process. Profiling and quality renders are outside timing.
Dependencies are imported from existing, frozen rtx1 engineering experiments.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = HERE.parent
sys.path[:0] = [str(HERE / "deps"), str(EXP / "rtx1-conditioning-001"),
                str(EXP / "rtx1-taehv-001"), str(ROOT)]

import numpy as np
import torch
from dataclasses import replace
from aura.pipeline import Aura, DEFAULT_PROMPT
from aura.streaming_audio import LiveAudioFeatures
from rope_variants import install as install_rope
import taehv_decoder
import render_quality_lighttae as quality
from fp8_policy import install
from torch._dynamo.utils import counters


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=["bf16", "ffn", "self_ffn"], default="bf16")
    parser.add_argument("--granularity", choices=["row", "tensor"], default="row")
    parser.add_argument("--chunks", type=int, default=22)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.chunks < 4 or args.warmup < 6:
        parser.error("Require at least four timed chunks and six disposable warm-up chunks")
    args.out.mkdir(parents=True, exist_ok=False)
    quality.health()
    dump(args.out / "config.json", {**vars(args), "out": str(args.out)})
    metadata = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "resolution": [768, 512],
                "steps": [1000, 745, 498, 255], "playback_fps": 16,
                "decoder": "LightTAE v2 FP16 raw latents", "attention": "SDPA",
                "dtype": "BF16 DiT, FP32 time/audio, selective FP8 W8A8 if requested",
                "source_sha256": {}, "timing": "audio extraction, render, GPU pixel transfer and encoder submission; excludes setup/warmup/profile/drain"}
    for path in [Path(__file__), HERE / "fp8_policy.py", Path(quality.__file__),
                 Path(sys.modules[Aura.__module__].__file__), Path(sys.modules["aura.model"].__file__),
                 Path(sys.modules["rope_variants"].__file__), Path(taehv_decoder.__file__)]:
        metadata["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(args.out / "metadata.json", metadata)
    print("LOADING", args.policy, flush=True)
    aura = Aura("aura-1.3b", compile=False, offload_text_encoder=True, bf16_transformer=True)
    from huggingface_hub import hf_hub_download
    checkpoint = Path(hf_hub_download(aura.cfg.repo, aura.cfg.checkpoint, local_files_only=True))
    metadata["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    aura.cfg = replace(aura.cfg, buckets={512 / 768: (512, 768)})
    aura.prepare_static = True
    aura.gpu_pixels = True
    context = aura.text_encoder([DEFAULT_PROMPT], aura.text_device)
    aura.text_encoder = lambda prompts, device: context
    weights = ROOT / "research_kernels/lightx2v-autoencoders/lighttaew2_1_v2.safetensors"
    taehv_decoder.install(aura, weights=str(weights), raw=True)
    metadata["decoder_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata["quantized_modules"] = [] if args.policy == "bf16" else install(aura.model, args.policy, args.granularity)
    dump(args.out / "metadata.json", metadata)
    install_rope("fp32_inline")
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
    for i, block in enumerate(aura.model.blocks):
        aura.model.blocks[i] = torch.compile(block, dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
    speech = quality.read_audio(EXP / "rtx1-quality-001/inputs/speech-24k.wav")
    image = EXP / "rtx1-quality-001/inputs/portrait.jpg"
    metadata["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [image, EXP / "rtx1-quality-001/inputs/speech-24k.wav"]}
    dump(args.out / "metadata.json", metadata)

    def run_chunks(count: int, measured: bool = False, profile: bool = False) -> list[dict]:
        from web.media import LiveMuxer
        session = aura.start_stream(str(image), DEFAULT_PROMPT, 42)
        audio = LiveAudioFeatures(aura)
        mux = LiveMuxer(args.out / ("speed.mp4" if measured else "warmup.mp4"), 768, 512, 16, lambda data: None)
        rows = []
        try:
            for i in range(count):
                quality.health()
                pcm = speech[np.arange(i * 24000, (i + 1) * 24000) % len(speech)].copy()
                before = int(counters["stats"]["unique_graphs"])
                # Deliberate benchmark synchronization, outside recorded wall boundary.
                torch.cuda.synchronize()
                started = time.perf_counter()
                video = session.render(audio.extract(pcm, 16))
                mux.push(video, pcm, 16)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if not torch.isfinite(video).all():
                    raise RuntimeError("Nonfinite output")
                row = {"index": i, "seconds": elapsed, "new_graphs": int(counters["stats"]["unique_graphs"]) - before}
                rows.append(row)
                print("MEASURE" if measured else "WARMUP", args.policy, row, flush=True)
                if measured:
                    dump(args.out / "speed-rows.json", rows)
        finally:
            mux.close()
            audio.close()
            session.close()
        return rows

    run_chunks(args.warmup)
    rows = run_chunks(args.chunks, measured=True)
    # First two chunks have initial-cache shapes; report them separately.
    interior = rows[2:]
    latencies = np.array([r["seconds"] for r in interior])
    result = {"policy": args.policy, "granularity": args.granularity,
        "measured_chunks": len(interior), "excluded_initial_chunks": rows[:2],
        "generation_fps": float(16 * len(interior) / latencies.sum()),
        "latency_seconds": {"mean": float(latencies.mean()), "max": float(latencies.max()),
            **{f"p{p}": float(np.percentile(latencies, p)) for p in [50, 95, 99]}},
        "deadline_misses": int((latencies > 1).sum()), "new_graphs": sum(r["new_graphs"] for r in interior),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "rows": rows}
    dump(args.out / "results.json", result)
    print("RESULT", json.dumps(result), flush=True)
    if args.profile:
        session = aura.start_stream(str(image), DEFAULT_PROMPT, 42)
        audio = LiveAudioFeatures(aura)
        try:
            for _ in range(3):
                session.render(audio.extract(speech[:24000].copy(), 16))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                session.render(audio.extract(speech[:24000].copy(), 16))
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.out / "profile.json"))
            (args.out / "profile.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=45))
        finally:
            audio.close()
            session.close()
    if args.quality:
        quality.OUT = args.out
        quality.POLICIES["native"] = ("native",) * 4
        for case in quality.CASES:
            quality.render(aura, case, "native", seconds=16)
    (args.out / "DONE").write_text("Completed; no automatic deployment.\n")


if __name__ == "__main__":
    main()
