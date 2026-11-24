# Tiger Eye Avatar research benchmarks

## RTX PRO 6000 Blackwell baseline

[`builds/rtx-pro-6000-blackwell-001/`](builds/rtx-pro-6000-blackwell-001/README.md)
preserves the remote RunPod measurements from
`runs/rtx1-fps-20261123-01`: **8.004 generation FPS for 1.3B at 768 × 512**
and **4.047 generation FPS for 5B at 1280 × 704**, both with four steps,
BF16 transformer storage and compilation disabled. Each result uses 20 measured
chunks after two warm-up chunks. This is a short throughput benchmark.

The build contains the remote source snapshot, raw results, generated videos,
package versions, model revision/blob identities and integrity checksums. Its
`optimization-plan.json` records proposed experiments toward 16 generation FPS,
with correctness and sustained-streaming gates. The target requires about 2×
the 1.3B baseline throughput or 3.95× the 5B baseline throughput; unchanged 5B
playback requires 25 generation FPS to keep up with audio.

## RTX 5090 historical baseline

`builds/baseline-001/` preserves the working demo before performance research:
source snapshot (including local changes and sample inputs), dependency lock,
installed package versions, hardware/software metadata, cached model revisions,
existing measurements, reference videos, and SHA-256 checksums. Both 1.3B and
5B models are supported. The web UI includes microphone/file input, synchronized
playback, microphone testing, and Tiger Eye Avatar branding.

The existing short-clip measurements are historical evidence, not a new benchmark
of sustained real time. `results.json` documents timing boundaries. CLI results
in `evidence/demo/outputs/validation.json` use different frame accounting and must
not be merged with browser streaming results. Use the file-input runs as the
initial comparison pair. Neither model was compiled in those runs.

## Baseline settings

| Setting | 1.3B web demo | 5B web demo on this RTX 5090 |
| --- | --- | --- |
| Sample output, width × height | 768 × 512 | 1280 × 704 |
| Playback rate | 16 fps | 25 fps |
| Audio per chunk | 1 second | 0.64 seconds |
| Diffusion steps | 4 | 4 |
| Transformer storage | BF16, selected modules FP32 | BF16, selected modules FP32 |
| Text encoder | CPU during portrait setup | CPU during portrait setup |
| Transformer offload during decoding | Disabled | Enabled |
| VAE decode group | 4 | 1 |
| Compilation in reference runs | Disabled | Disabled |

`run_web_demo.sh` supplies `OMP_NUM_THREADS=8`, `HF_HUB_DISABLE_XET=1` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless overridden. The captured
environment records shell values; launcher defaults are in the source snapshot.
Model weights remain in the shared Hugging Face cache and are not bundled.

## Save another build

Make changes in the normal checkout, record measurements, then capture a new ID:

```bash
.venv/bin/python benchmarks/capture.py compile-001 \
  --note '1.3B transformer compilation; original resolution and four steps' \
  --result demo/outputs/my-run-qa.json \
  --artifact demo/outputs/my-run.mp4
```

Repeat `--result` and `--artifact` for additional evidence. Each ID gets a separate
folder. The capture command refuses to overwrite one. It copies existing results;
it does not run inference. Do not use baseline results to describe a modified build.
Compare the same portrait, exact audio bytes, resolution, precision, chunk size,
and diffusion steps. Record changes to those settings explicitly. Separate cold
load, compilation/warm-up and steady-state timings. For a real-time claim, measure
several minutes without a growing queue, dropped audio or capture pauses, along
with microphone-to-display latency and visual/lip-sync quality. RTF ≤ 1 means
rendering keeps up within the recorded timing boundaries; it alone does not prove
end-to-end real time. Do not average per-chunk FPS: divide total nominal frames
by total chunk time instead.

## Verify or restore the baseline

```bash
cd benchmarks/builds/baseline-001
sha256sum -c SHA256SUMS
```

Restore into a separate empty directory to keep research changes safe:

```bash
mkdir -p /home/knoledge_base/tiger-eye-baseline-001
tar -xzf benchmarks/builds/baseline-001/source.tar.gz -C /home/knoledge_base/tiger-eye-baseline-001
cd /home/knoledge_base/tiger-eye-baseline-001
uv sync --frozen
HF_HOME=/home/cache/huggingface ./run_web_demo.sh --port 7861
```

The source archive is the authoritative snapshot of uncommitted and untracked
source. `working-tree.patch` supplements it for tracked changes. `packages.txt`
records installed versions; `uv.lock` controls the project dependencies. Match
`environment.json` when comparing hardware-specific results. Browser automation
helpers and their requirements are saved as evidence; they require Playwright,
local Chrome, the noted fake-microphone asset, and a running demo on port 7860.
