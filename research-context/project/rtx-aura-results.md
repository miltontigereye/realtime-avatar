# RTX Aura experiment results

Updated: 2026-11-24 (Asia/Dhaka). Historical sections consolidate archived local baseline evidence and rtx1 reports. The FP8 feed-forward and 5B engineering-transfer sections add fresh measurements collected on this date.

This is the single maintained results ledger for RTX Aura. See [plan and milestones](rtx-aura-plan.md) for future work and [literature review](literature-aura-acceleration.md) for untested ideas. Earlier per-experiment reports are retained as historical evidence; statements in them that the demo was unchanged describe their writing time.

Canonical locations, maintained with identical contents:
- Local: `/home/knoledge_base/RealTime-Avatar/docs/rtx-aura-results.md`
- rtx1: `/workspace/avatar/docs/rtx-aura-results.md`

Unless a local path is explicit, `runs/`, `experiments/` and `research_kernels/` below are relative to `/workspace/avatar/research/aura/` on rtx1. Historical raw remote evidence was not copied by the consolidation. New 5B JSON summaries and contact sheets are retained locally under the matching run IDs; videos and raw sampled arrays remain on rtx1. FP8 summaries, logs and source snapshots are also retained locally under `runs/rtx1-fp8-linear-20261124-01/`; videos, sampled faces and profiler traces remain on rtx1.

Current [public demo](https://2ipsk1ylcb0iqi-8000.proxy.runpod.net): the dual-model service defaults to 5B, with 1.3B still selectable. Its [5B configuration and verification](#native-5b-engineering-transfer-short-web-screen) are recorded below; the earlier 1.3B deployment section is historical.

## Publication evidence and claim ledger

All existing measurements below are **development/engineering evidence**. They have not been collected under the held-out paper protocol in [the plan](rtx-aura-plan.md). No result for the proposed mouth/audio-aware reuse method is recorded. Do not relabel historical four-case screens as the final benchmark or count chunks as independent quality samples.

| Claim ID | Candidate statement | Current supporting evidence | Publication status / missing evidence |
| --- | --- | --- | --- |
| C0 | Native 1.3B lightweight-decoder demo exceeds 16 FPS in a short web test | `rtx1-lighttae-web-20261124-01`: 22.5035 FPS, rounded P99 0.72 s | Supported only within stated short-test boundaries; needs repeated sustained testing for broader claims |
| C1 | LightTAE v2 preserves lip sync and identity | Four development cases in `rtx1-taehv-20261124-04` | Limited automated screen; producer mouth fidelity fails 35 dB; needs held-out data, uncertainty and human review |
| C2 / H1 | Proposed policy improves lip-sync–latency frontier over general reuse | None | Untested; P0–P3 required |
| C3 / H2 | Mouth/audio gating explains the improvement | None | Untested; matched-budget mechanism ablations required |
| C4 / H3 | Proposed system sustains native playback with bounded state | None for proposed method; historical paced tests use other profiles | Untested for proposed method and current ten-minute demo target; P4 required |
| C5 | Proposed method transfers to 5B or achieves native 25 FPS | General engineering profile: public 5B web screen 18.824 FPS, P99 0.85 s (`rtx1-5b-public-web-20261124-01`) | 25 FPS target fails; no proposed-method transfer evidence |
| C6 | Benefit generalizes beyond AURA / this GPU | None | Unsupported; additional models/hardware needed before making that claim |

No acceptance, novelty or main-conference-readiness claim follows from this ledger. Final conclusions must follow the observed results, including null or negative findings.

### Paper experiment register

Planned IDs below are reservations, not executed runs. Enter immutable run IDs and evidence when each is actually performed.

| ID | Experiment | Status | Intended output |
| --- | --- | --- | --- |
| E-P01 | Verified general-reuse versus candidate-policy pilot | Not run | Go/no-go and frozen budgets for P1 |
| E-P02 | Held-out main B0/B1/B2/B3/M comparison | Not run | Main table and Pareto curve |
| E-P03 | Uniform/random, region/audio and refresh ablations | Not run | Mechanism table |
| E-P04 | Speech transitions, pose/mask errors and drift | Not run | Failure/robustness analysis |
| E-P05 | Repeated ten-minute native tests and independent display latency | Not run under paper protocol | Sustained systems table/figure |
| E-P06 | 5B transfer and strong compatible acceleration comparisons | Not run | Model transfer table |
| E-P07 | Blinded paired human A/V evaluation | Not run | Perceptual evidence and uncertainty |

### Required record for each publication experiment

Use this schema here when results arrive; raw metrics/configuration stay in the referenced run directory:

```text
Experiment ID / hypothesis / development-or-final split:
Status: planned | running | valid | failed | invalid | superseded
Run IDs / date / source revision and hash / environment and kernel versions:
Model and checkpoint hashes / GPU / dimensions / playback FPS:
Steps / dtype / attention / decoder / cache window / audio lookahead:
Policy and gating overhead / requested and achieved compute budget:
Dataset manifest hash / rights / identity and clip counts / durations / seeds:
Predeclared metrics, regression margins and protocol revision:
Timing boundary / warm-up and measured sample counts / process repetitions:
Per-case quality and detection coverage / paired effects / 95% CI method:
Mean, P50, P95, P99, max latency / FPS / deadline misses:
Per-stage timing / queues / memory trajectory / dropped input / recompilation:
Causality, cache equivalence, eviction and conditioning checks:
Failures and exclusions with reasons / representative-case selection rule:
Raw evidence and commands to regenerate tables/figures:
Claim IDs supported, rejected or still unresolved / deployment state:
```

Freeze a protocol revision before final evaluation, record deviations, and retain every attempt. A final table row needs traceable metrics and settings, valid sample counts and uncertainty; otherwise mark it missing. A null result belongs in this ledger with the same provenance as a successful one. Citation verification belongs in literature; future work belongs in the plan.

## Prior 1.3B demo: verified on rtx1, 2026-11-24

Before the dual-model deployment below, process `python -u serve_lighttae.py --host 0.0.0.0 --port 8000` ran from `experiments/rtx1-lighttae-demo-001`. Launcher: `run_lighttae_server.sh` there. It subclasses FastAura and installs raw-latent LightTAE v2 (`research_kernels/lightx2v-autoencoders/lighttaew2_1_v2.safetensors`).

| Setting | Prior deployed value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Model / output / playback | AURA-1.3B / 768×512 / 16 FPS |
| Diffusion timesteps | [1000,745,498,255] |
| Precision | Transformer BF16, time/audio FP32; lightweight decoder FP16 |
| Attention | Original SDPA; no Sage quantization |
| Compilation | Static full-graph DiT blocks, `max-autotune-no-cudagraphs` |
| Runtime optimizations | Prepared conditioning, GPU pixel packing, disposable warm-up |
| Decoder | Streaming LightTAE v2, raw Wan latents |

**Actual short web integration result:** `runs/rtx1-lighttae-web-20261124-01`; also recorded in `experiments/rtx1-lighttae-demo-001/web-check.log`. Twenty measured chunks / 320 frames after two warm-up chunks: 14.22 s render time, **22.5035 generation FPS**, mean/P50 0.71 s and P95/P99 0.72 s. Server timing is rounded to 0.01 s. Request-to-ready 163.55 s, including 64.32 s disposable warm-up; first real warm-up chunks 0.96/0.60 s. Sampled peak whole-GPU memory 8,893 MiB. FFprobe confirms 768×512 H.264 at 16 FPS plus AAC, 22.043 s media duration. Input is the demo portrait and repeated demo speech, no dropped samples in this check.

Timing includes audio features, diffusion, decode, pixel transfer/conversion and encoder queue submission; excludes model/portrait setup, audio waits, encoder drain and browser/network playback. The metadata's `vae_dtype=BF16` describes the retained original VAE; the installed LightTAE decoder is FP16. A short web integration pass is not a ten-minute sustained or microphone-to-display result.

The browser live-edge fix in `WEB_LATENCY_FIX.md` adds a 300 ms start cushion, catch-up playback and a microphone latency readout. Its roughly two-second voice-to-screen estimate is explanatory, not an independently measured latency distribution. JavaScript syntax passed; a transient broken upload was restored and is recorded in that report.

## Baselines and fast profile

All rows use four released steps. 1.3B uses BF16 transformer storage except FP32 time/audio; original 5B VAE is FP32. Frame rates are generation throughput, not playback rate.

| Experiment / source run | Model, dimensions | FPS | P99 chunk s | Scope / decision |
| --- | --- | ---: | ---: | --- |
| Eager baseline: `rtx1-fps-20261123-01` | 1.3B, 768×512 | 8.004 | 2.01 | 20 chunks after 2 warm-up; frozen archive |
| Same baseline | 5B, 1280×704 | 4.047 | 3.9881 | Native playback 25 FPS; optimized engineering screen below |
| Conditioning: `rtx1-conditioning-20261123-01/conditioning_gpu_pixels` | 1.3B, 768×512 | 15.187 | 1.065 | Best native result with the original Wan VAE (no decoder fidelity loss); below the 16 FPS target. Superseded for speed by the LightTAE v2 rows below |
| Autotune: `rtx1-autotune-20261123-01/max_autotune_672x448` | 1.3B, 672×448 | 20.998 | 0.769 | Short reduced-resolution screen |
| Paced: `rtx1-realtime-20261123-01` | 1.3B, 672×448 | 20.857 | 0.778 | 180 measured chunks, max 0.781 s, no dropped harness input |
| Web: `rtx1-fast-web-20261123-01` | 1.3B, 672×448 | 20.766 | 0.78 | Historical fast demo, superseded by active native LightTAE demo |

Baseline archive locally: `research/aura/benchmarks/builds/rtx-pro-6000-blackwell-001/`. Fast-profile source: `experiments/rtx1-conditioning-001/REALTIME_RESULTS.md`. Three-minute reduced-resolution pace test had maximum input-start lag 5.53 ms, video queue ≤1, sampled audio queue 0, allocated GPU memory approximately 9,271.49 MiB and no measured recompilations. This establishes only the renderer/encoder pace gate for that configuration.

Compile/RoPE/VAE/stage-profile experiments also exist (`rtx1-compile-001`, `rtx1-rope-001`, `rtx1-vae-001`, `rtx1-vae-002`, `rtx1-stage-profile-001`; corresponding run directories include compiled/static profiles). Their individual raw outputs were not audited for a separate gain in this consolidation; the table reports the integrated outcome, not an invented attribution to each optimization.

## Attention experiments

Source reports: `experiments/rtx1-sage-001/RESULTS.md`, `rtx1-quality-001/RESULTS.md`, `rtx1-selective-sage-001/RESULTS.md`, `rtx1-fa4-001/RESULTS.md`.

| Run / configuration | Dimensions | FPS | P99 s | Quality / outcome |
| --- | --- | ---: | ---: | --- |
| `rtx1-sage-20261123-01`, SDPA control | 672×448 | 20.976 | 0.769 | Matched short screen |
| Same run, Sage INT8 QK / FP16 values | 672×448 | 22.216 | 0.726 | +5.91% throughput; subsequent quality failure |
| Same run, Sage INT8 QK / FP8 values | 672×448 | 22.962 | 0.703 | +9.47%; subsequent quality failure |
| `rtx1-quality-20261123-02`, FP8 paced | 672×448 | 22.762 | 0.710097 | Stopped at 312 measured seconds; 2 deadline misses; not completed ten minutes |
| `rtx1-quality-fp16-20261123-01`, FP16 paced | 672×448 | 22.139 | 0.729473 | Completed 300 measured seconds; pace pass, quality fail |
| `rtx1-sage-native-paced-20261123-01`, FP8 | 768×512 | 16.774 | 0.968472 | Three-minute renderer pace pass; not current demo |
| `rtx1-selective-sage-20261123-02`, SDPA | 768×512 | 14.598 | 1.104 | Matched selective-policy control |
| Same run, FP16 all steps | 768×512 | 15.775 | 1.021 | Four-case automated quality pass, deadline fail |
| Same run, FP8 all steps | 768×512 | 16.553 | 0.973 | Four-case automated quality pass, thin mouth-fidelity margin |

At 672×448, FP8 fails mouth PSNR for fashion (33.42 dB) and producer (32.56); FP16 fails producer (33.48), against the predeclared 35 dB gate. Lip-sync, identity and sparse temporal regression checks pass all four cases. A pace pass does not override the fidelity failures.

At native 768×512, the later selective-policy screen passes all automated checks for all seven candidates. FP8-all mouth PSNR is 41.3/35.5/37.0/38.5 dB across existing/fashion/grandmother/producer; fashion has only about 0.47 dB margin. Mixed per-step policies lose speed without a clear quality gain (FP8 first three steps: 15.988 FPS, P99 1.007 s; first two: 15.447 FPS, P99 1.044 s). Sage remains experimental; native quality results cannot erase reduced-resolution failures or be transferred to a different decoder.

FA4 microbenchmark (`runs/rtx1-fa4-20261123-01`, BF16, SM120): native long attention 2.787 ms SDPA versus 2.724 ms FA4, relative L2 8.8e-5. Estimated pipeline saving under 1%, below 2% screen; no end-to-end follow-up. Initial tuple-return handling bug was fixed and failed artifacts retained.

## Decoder overlap and VAE layout

Source: `experiments/rtx1-overlap-001/RESULTS.md`. Corrected matched-interior timings supersede preliminary overlap FPS; terminal drain had biased the earlier estimates.

| Configuration | Serial FPS | Overlap FPS | Change |
| --- | ---: | ---: | ---: |
| 1.3B, 768×512, four steps, original BF16 SDPA/VAE | 14.682 | 14.789 | +0.73% |
| Same, 672×448 | 20.953 | 20.891 | −0.29% |

Neither reaches the 2% improvement screen. Native four-case outputs match unsplit rendering bit for bit (64 chunk hashes). At most two chunks in flight; not deployed. Smaller-resolution quality follow-up was interrupted, not completed. VAE-only canonical-layout screens at 672×448 save at most 0.32% (296.225 to 295.270 ms), with synthetic-latent equality; these are decoder diagnostics, not avatar FPS.

Evidence: `runs/rtx1-overlap-20261123-01`, `rtx1-overlap-fast-20261123-01`, `rtx1-vae-layout-20261123-01`, including corrected-window JSON and preserved executed sources.

## Lightweight decoders

Source: `experiments/rtx1-taehv-001/RESULTS.md`. AURA-1.3B, native 768×512, four steps; lightweight decoders FP16. Short matched speed run: `runs/rtx1-taehv-20261124-02`, 20 measured chunks after warm-up.

| Configuration | FPS | P99 s |
| --- | ---: | ---: |
| Original Wan VAE + SDPA | 14.684 | 1.108 |
| TAEHV taew2_1 + SDPA | 22.111 | 0.730 |
| Wan VAE + Sage FP8 | 16.593 | 0.971 |
| TAEHV + Sage FP8 | 26.879 | 0.601 |

Decoder-only times: TAEHV 22.9 ms/chunk; LightTAE v1/v2 22.5–23.2 ms; Wan VAE approximately 390 ms in this screen. The TAEHV throughput rows are not direct LightTAE measurements; the active LightTAE web result above supplies its own evidence.

Quality: four cases ×16 seconds, paired with Wan VAE; mouth PSNR order is existing/fashion/grandmother/producer.

| Decoder / attention | Mouth PSNR dB | Producer identity delta | Failed checks |
| --- | --- | ---: | --- |
| TAEHV / SDPA | 35.10 / 37.82 / 36.12 / 34.30 | −0.027 | Producer identity and mouth |
| TAEHV / Sage FP8 | 34.27 / 34.48 / 35.04 / 33.07 | −0.028 | Three mouth cases; producer identity and raw fidelity |
| LightTAE v1 / SDPA | 34.03 / 36.65 / 34.32 / 33.21 | −0.032 | Three mouth cases; producer identity |
| LightTAE v2 / SDPA | 35.35 / 38.57 / 36.00 / 34.76 | −0.014 | Producer mouth, 0.24 dB below 35 dB gate |

LightTAE v2 SyncNet confidence deltas: −0.016/−0.051/−0.093/+0.078 (limit −0.25); identity deltas −0.011/−0.004/−0.008/−0.014 (limit −0.02). Lip-sync and identity screens pass; this is not a lossless decoder or a documented human A/V review pass. It is now deployed with the fidelity tradeoff disclosed.

Correctness and provenance: chunkwise versus whole-video decoder maximum absolute difference 0.008 in FP16; 13/16/16 initial/subsequent frame accounting retained. TAEHV expects normalized latents; LightTAE expects raw latents. `runs/rtx1-taehv-20261124-03` used wrong normalized LightTAE inputs and is INVALID, superseded by `-04`. Microbench evidence: `-01`; corrected paired quality: `-04`. TAEHV pinned `011dfc2` (MIT); LightX2V Autoencoders pinned `043793f` (Apache-2.0), per source report.

## Native 5B engineering transfer: short web screen

Experiment: `experiments/rtx1-5b-optimized-001/`. Web evidence: `runs/rtx1-5b-web-20261124-02/`; this is a development screen. The existing 1.3B optimizations were adapted to AURA-5B: static full-graph `max-autotune-no-cudagraphs` DiT compilation, prepared invariant conditioning, FP32-inline RoPE, BF16 transformer storage except time/audio modules FP32, CPU text encoder, GPU pixel packing and disposable warm-up. Attention remains SDPA and timesteps remain [1000,745,498,255]. There is no FP8/INT8 quantization or step reduction.

5B uses **Wan2.2, 48 latent channels and spatial stride 16**, so the 1.3B Wan2.1 decoder checkpoint is incompatible. This profile uses streaming **`lighttaew2_2_v2.safetensors`**, FP16, with raw Wan2.2 latent scaling. Checkpoint source: [LightX2V Autoencoders](https://huggingface.co/lightx2v/Autoencoders), pinned `043793f3b3a2ea8db7d0ed8b5a0e8134436d529b` (Apache-2.0). Original Wan2.2 FP32 VAE remains loaded for reference-image encoding and paired diagnostic decoding. The audio encoder, DiT and decoders are co-resident on one RTX PRO 6000 Blackwell; the text encoder is on CPU.

| Short web result | Value |
| --- | ---: |
| Output / playback | 1280×704 / 25 FPS |
| Timed chunks / decoded display frames | 20 / 320 |
| Timed rendering total | 17.08 s |
| Generation throughput | 18.7354 FPS |
| Chunk mean / P50 / P95 / P99 / max | 0.854 / 0.85 / 0.86 / 0.86 / 0.86 s |
| 0.64 s deadline misses in timed window | 20 / 20 |
| Request to ready, including model/portrait setup | 363.25 s |
| Disposable warm-up | 227.51 s / 6 chunks |
| First two real-session chunks, excluded from timing window | 0.69 / 0.76 s |
| Sampled peak whole-GPU memory | 25,532 MiB |

The archived eager 5B baseline is 4.047 FPS, so the observed throughput ratio is approximately **4.63×**. This is a historical configuration comparison, not an isolated attribution or a fresh paired control. **Native 25 FPS fails**: generation takes about 1.334× playback duration. This short test does not establish sustained stability, microphone-to-display latency, concurrency or held-out quality. Idle 1.3B/Sage services remained resident; the memory figure includes them.

Web validation received fragmented H.264/AAC output through the HTTP/WebSocket API. FFprobe confirms 1280×704, 25 FPS and audio; output duration is 14.1227 s for 22 chunks including two warm-up chunks. The timing boundary includes audio features, diffusion, decoding, GPU pixel packing/CPU transfer and encoder queue submission; setup, audio waits, encoder drain and browser/network playback are excluded. Server durations are rounded to 0.01 s. The video encoder profile now permits H.264 level 4.0.

Correctness: `check_correctness.py` passes conditioning reuse versus uncached execution through eviction and a 5B segment boundary on a small 48-channel model; streaming versus whole-video Wan2.2 LightTAE decoding has maximum difference **0.0** on the tested FP16 tensors. Reset and 13-first/16-subsequent frame counts pass. Eight existing web streaming tests and two RoPE precision tests pass. These checks do not replace full-size quality evaluation.

**Paired decoder quality:** `runs/rtx1-5b-quality-20261124-01/`, four development portraits ×16 seconds, 25 FPS, paired on identical generated latents and PCM. Raw pixel diagnostics sample 64 frames per case at approximately four frames/second. This isolates the decoder; it is not a comparison of the entire optimized DiT against upstream eager generation.

| Case | Raw PSNR dB | Mouth PSNR dB | Sync confidence delta | Sync distance delta | Identity cosine delta | Automated checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Existing | 40.36 | 36.75 | -0.0196 | -0.0878 | -0.0035 | Pass |
| Fashion | 43.48 | 39.63 | -0.0422 | +0.0025 | -0.0012 | Pass |
| Grandmother | 41.93 | 38.21 | +0.0115 | -0.0094 | -0.0010 | Pass |
| Producer | 41.69 | 38.07 | -0.0584 | +0.0555 | +0.0007 | Pass |

All four cases pass the unchanged 35 dB raw/mouth regression gates, sync confidence/distance/offset, identity, temporal, detection, shifted-audio and wrong-audio controls. Producer baseline face detection is 62/64 samples (96.875%), candidate 64/64; other cases detect 64/64 in both. Temporal ratios range from 0.969 to 0.983. The decoder still changes pixels; no lossless equivalence, human paired A/V review, held-out generalization or sustained quality claim follows. The original 1.3B producer-mouth failure remains unchanged.

**Deployment:** the dual-model service now runs on public port 8000 from `experiments/rtx1-5b-optimized-001/run_server.sh`; `deployment.json` records the process and idle-service preflight. Default is 5B (1280×704, 25 FPS playback); 1.3B remains selectable at 768×512 and 16 FPS playback. Browser origin and H.264 codec declaration are configured for the public HTTPS endpoint. The prior 1.3B launcher is retained for rollback. The temporary port-8004 listener was stopped after promotion. Final public checks passed:

- `runs/rtx1-5b-browser-20261124-01/`: actual Chrome 149 through public HTTPS/WebSocket and MSE, default 5B selected, 1280×704 video playing, 99 video frames decoded and 48,533 audio bytes decoded, no JavaScript or media errors. The sample completed and exposed a working download. Total browser test time was 217.67 s, including fresh-process model loading and warm-up; it is not a microphone-to-display measurement or human A/V review.
- `runs/rtx1-5b-public-web-20261124-01/`: final warmed public process, 20 measured chunks / 320 frames after two warm-up chunks, 17.00 s summed server render time, **18.8235 FPS**, mean/P50/P95/P99/max all **0.85 s** at the server's 0.01 s precision, 20/20 misses of the 0.64 s deadline. Initial real-session chunks were 0.70/0.76 s. Request-to-ready was 5.41 s with the model already loaded; the final process's disposable warm-up was 82.92 s using the populated compile cache. Sampled whole-GPU peak was 20,485 MiB; the temporary listener and old 1.3B process were gone, while the idle Sage service remained resident. Client wall throughput for all 22 chunks after ready was 18.563 FPS, including drain/network effects. FFprobe again confirms H.264 1280×704 at 25 FPS plus AAC. This repeat used the public URL and matching browser Origin header. The historical eager-baseline ratio is approximately **4.65×**; the 25 FPS target still fails.

Reproducibility: `runs/rtx1-5b-deployment-20261124-01/` contains executed source hashes and `source.tar.gz`, package versions, AURA/Wan/LightTAE checkpoint hashes and all eight portrait/speech input hashes. Supplementary `dependency-provenance.json` records cached T5/tokenizer and wav2vec2/processor files and hashes. Local JSON summaries, source archive and browser screenshot mirror remote evidence; quality videos and sampled arrays remain on rtx1. No held-out paper-protocol, ten-minute sustained or human A/V review gate is closed.

Attempts preserved: `rtx1-5b-web-20261124-01` failed before inference because port 8003 was occupied by an existing review server. That review server was left running; the successful `-02` test uses port 8004. Startup failure is retained in `server.port-conflict.log`; no result from `-01` is used.

## FP8 feed-forward linear screen: rejected

Source: `experiments/rtx1-fp8-linear-001/`; immutable evidence: `runs/rtx1-fp8-linear-20261124-01/`. This is an engineering screen, not a publication-protocol experiment. The then-active port-8000 demo was not changed during that FP8 experiment.

Matched settings: AURA-1.3B, RTX PRO 6000 Blackwell Server Edition, native 768×512, four steps [1000,745,498,255], SDPA, streaming raw-latent LightTAE v2 FP16, BF16 DiT with the existing FP32 time/audio modules, prepared conditioning and GPU pixel packing. Each variant used a fresh process, six disposable warm-up chunks and twenty timed chunks. The first two timed chunks have initial-cache workloads and are reported separately; steady-state statistics below use the remaining **18 chunks / 288 decoded frames**. There was one process per variant, so these short-run percentiles do not establish sustained tail latency or uncertainty.

The candidate quantizes **60 FFN linears** (`blocks.*.ffn.0` and `.ffn.2`) with TorchAO **0.13.0**, dynamic E4M3 FP8 activations and weights, and per-row scales. Selected linear inputs/outputs retain the BF16 boundary of the original autocast execution. Audio/text cross-attention, modulation, norms, self-attention projections and the decoder retain their baseline precision. TorchAO lives in the experiment's `deps/`; the demo environment was not upgraded. Compilation is static full-graph `max-autotune-no-cudagraphs`.

| Variant / evidence subdirectory | Generation FPS | Mean s | P50 s | P95 s | P99 s | Max s | Peak PyTorch allocation MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 control / `bf16` | 22.4138 | 0.7138 | 0.7122 | 0.7200 | 0.7212 | 0.7215 | 8,006.78 |
| FP8 FFNs / `ffn_bf16` | 23.7426 | 0.6739 | 0.6779 | 0.6807 | 0.6813 | 0.6814 | 7,154.89 |

Throughput improves **5.93%**. Both steady-state windows have zero one-second deadline misses and zero new graphs. Allocation peaks cover process setup/warm-up as well as measurement and exclude the other demo process; they are not whole-GPU residency measurements. Timings include audio extraction, rendering, GPU-to-CPU pixel transfer and encoder queue submission, with deliberate synchronization at timing boundaries. Setup, disposable warm-up, health checks, finite-output checks, profiler capture and encoder drain are excluded; network/browser display is unmeasured.

**Startup caveat:** the first two timed chunks are BF16 0.9176/0.5952 s versus FP8 1.1935/0.5604 s. FP8 therefore has a first-chunk one-second deadline miss even after disposable warm-up; the zero-miss statement applies only to the 18-chunk interior. The first two disposable FP8 warm-up chunks took 158.57/90.59 s including compilation, versus 31.47/15.87 s for BF16. BF16 already had a populated kernel cache; this is not a repeated cold-start comparison.

**Profile:** `bf16/profile.txt` records 342.199 ms in `aten::_flash_attention_forward` in one profiled chunk (49.08% of reported self CUDA time). FP8 records 342.237 ms there and 240 native `aten::_scaled_mm` calls totaling 72.425 ms. Attention already consumes roughly half the control chunk; the earlier literature assumption that linears dominate the DiT is not supported by this profile. Profiling is diagnostic and outside the throughput window.

**Paired quality:** four development cases ×16 seconds, with the same portrait, speech and seed in BF16 and FP8. Evidence: `paired-ffn_bf16/quality-ffn_bf16.json`, `quality-summary.json` and `judge-provenance.json`. The baseline is **BF16 + LightTAE v2**, so this measures additional quantization regression and does not erase the decoder's earlier producer-mouth failure versus Wan VAE.

| Case | Raw PSNR dB | Mouth PSNR dB | Sync confidence delta | Sync distance delta | Identity cosine delta | Failed checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Existing | 30.80 | 27.34 | +0.1582 | +0.0231 | −0.0096 | Raw and mouth fidelity |
| Fashion | 33.67 | 28.01 | −0.0587 | +0.1119 | −0.0068 | Raw and mouth fidelity |
| Grandmother | 33.37 | 26.54 | −0.2820 | +0.3045 | −0.0088 | Sync confidence/distance, raw and mouth fidelity |
| Producer | 34.30 | 31.80 | +0.0473 | −0.1135 | −0.0117 | Raw and mouth fidelity |

The unchanged gates require raw/mouth PSNR ≥35 dB, sync-confidence delta ≥−0.25, sync-distance delta ≤+0.25 and identity delta ≥−0.02. Identity, face detection, temporal checks, offset checks and shifted/wrong-audio controls pass all four cases; temporal ratios are 1.0001–1.0262. Fidelity fails all four and sync fails grandmother. Pixel fidelity is a regression diagnostic, not a calibrated perceptual judgment; no human A/V pass or ten-minute FP8 paced run is claimed.

**Decision: reject this FFN-only post-training FP8 configuration for deployment.** The 5.93% throughput improvement does not satisfy the existing quality criteria. The live demo retains BF16 linears. Other selective policies, QAT, INT8 and NVFP4 remain unmeasured; this result does not establish that every FP8 policy fails.

Attempts retained:
- `ffn/`: stopped during warm-up after discovering that TorchAO chose FP32 GEMM outputs from pre-autocast FP32 modulation inputs. Sources and `STOPPED` explain the retry. The corrected BF16 boundary has an exact pre-quantization autocast-equivalence test and a passing eager/compiled GPU compatibility smoke check.
- `self_ffn_bf16/`: expanded self-attention/FFN experiment stopped at the busy-demo preflight before model loading or timing. `PREFLIGHT_BUSY` and `preflight.log` are evidence of an unexecuted candidate, not a performance result. It was not retried after the FFN-only quality rejection.

Reproduction entry points: `run.sh` accepts `--policy bf16` for the control or `--policy ffn` for the candidate, with `--chunks 20 --profile --quality` and a fresh `--out` directory. The paired layout uses `bf16/` and `ffn_bf16/` below a common run root. `judge.py` accepts that root with `--run` and the label `--candidate ffn_bf16`, using `environments/quality-20261123-01`. `summarize.py` regenerates `comparison.json`; `archive.py` creates source snapshots and `provenance.json`, including checkpoint/input hashes and package versions. The supplementary provenance records the actual imported `rtx1-conditioning-001/aura/model.py`; the initial BF16 manifest also hashed the root model file. Four focused policy/autocast tests pass locally and on rtx1; the GPU compatibility check is diagnostic, not a quality gate.

## Failures, limitations and pending work

- Selective Sage attempt `rtx1-selective-sage-20261123-01` failed on compiler specialization limits; use `-02` results. Quality preflight `rtx1-quality-20261123-01` was abandoned; use `-02`. LightTAE web attempt `.failed-venv` is retained; use the successful unsuffixed run.
- No completed ten-minute native LightTAE v2 paced run or independent microphone-to-display distribution was found in the reviewed reports. The FP8 FFN result above is rejected on quality. The short optimized 5B screen above remains below 25 FPS; no reduced-step result or cross-chunk research comparison is established here.
- Four-input automated screens are limited evidence. Pixel fidelity thresholds are regression limits, not a calibrated perceptual verdict. Results at different resolution, decoder or attention settings must not be merged into a single validated profile.
- Historical runs have different measurement counts and boundaries. Compare variants within their matched experiment; differences between control rows are not isolated gains. Short-run P99 values have limited tail coverage.

## Adding an experiment

Append the experiment/run IDs, date, hypothesis, exact GPU/model/resolution/steps/dtypes/decoder/attention, input and source/config/checkpoint hashes, timing boundaries, sample and warm-up counts, mean/P50/P95/P99 and memory/queue diagnostics. Link raw evidence, record quality and causal/cache checks, failures, decision and deployment state. Use a fresh run directory; preserve previous evidence. Update the companion plan only after its stated exit criteria are met, then synchronize all three maintained documents (literature, plan, results) and verify SHA-256.
