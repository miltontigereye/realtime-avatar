# Literature review: accelerating AURA without losing lip sync

## Publication positioning and evidence policy

Publication objective added 2026-11-24. This is the single literature file for the paper; [rtx-aura-plan.md](rtx-aura-plan.md) owns experiment design and milestones, and [rtx-aura-results.md](rtx-aura-results.md) owns measured evidence. The historical survey below is retained as a discovery record. Its novelty verdicts, paper identities, dates, venues and speedup statements are provisional until checked against primary sources. No new primary-source verification was performed during this publication-oriented reorganization. Do not copy unverified survey statements into a manuscript as established facts.

**Working research question:** Can a causal, mouth/audio-aware allocation of computation improve the lip-sync–latency tradeoff over general acceleration policies in few-step streaming talking avatars, at matched compute and bounded state?

**Candidate contribution, not yet demonstrated:** a cross-chunk reuse policy that protects speech-sensitive regions using only information available at the current chunk boundary. Compilation, tiny decoders and standard quantization establish a competitive engineering baseline. Their use alone does not establish a new algorithm. If experiments reject the proposed advantage, change the paper scope to the supported systems findings rather than retaining the novelty claim.

### Related-work comparison to complete before method selection

| Family / candidate papers from historical survey | Comparison question | Evidence required before writing related work |
| --- | --- | --- |
| AURA / upstream optimized pipeline | Which acceleration and rolling-cache choices already belong to AURA? | Verified paper/version, official implementation, optimized settings and timing boundaries |
| ScoutCache, X-Cache, PreSTP | Does general few-step streaming reuse/pruning already account for regions or conditioning? | Full method and supplement, causal/cache behavior, code availability and exact baseline configuration |
| EchoCache, LightningCP | Is audio- or foreground-aware reuse already equivalent to the proposed policy? | Full spatial/temporal gating definition, model step regime, lip-sync evaluation and streaming assumptions |
| UnStep and step-reduction methods | Can a simpler step policy match the proposed tradeoff? | Actual schedules, timestep-history compatibility, retraining requirements and quality protocol |
| Sparse/quantized attention and lightweight decoders | How much gain is explained by ordinary engineering? | Precision/decoder tradeoffs, matched hardware measurements and component ablations |

Every accepted citation needs title, authors, primary URL/DOI/arXiv ID, version/date, verification depth (metadata/abstract/full text/code reproduced), exact supporting section/table, model/hardware/timing boundary, limitations and relevance. Keep these records in this file. A repository README or abstract is not a full-paper reading; a search snippet is only a lead. Record inaccessible or nonexistent sources explicitly, and exclude unsupported claims. Compare only primary-source numbers with their original qualifiers; our GPU results must come from our results ledger.

### Novelty decision record

Status: **open**. No full-text novelty audit, reproduced closest-method baseline or demonstrated proposed-method result is recorded yet. The historical statement that no paper combines certain features is a search observation, not proof of novelty. Novelty requires a specific mechanism difference and an experimentally useful advantage over the strongest verified relevant baseline, beyond changing the application domain.

Manuscript related-work structure: (1) causal talking-avatar generation; (2) few-step streaming reuse and pruning; (3) audio/region-aware acceleration; (4) practical single-GPU inference. State our verified difference only after the comparison above is complete.

## Historical survey (2026-11-23, sections 5–6 added 2026-11-24)

Decoder note: sections 4–6 were written while TAEHV was the candidate decoder. TAEHV later failed the producer identity check and LightTAE v2 was selected and deployed; see [results](rtx-aura-results.md#lightweight-decoders). Ordering note: the plan's "Next actions in order" is authoritative; the orderings in §5.4 and §6.8 are survey inputs to it.

Scope: is speeding up AURA (Kyutai's streaming talking-head model, BMVC 2026, [arXiv 2611.26540](https://arxiv.org/abs/2611.26540)) on rtx1 publishable research?

**Status of evidence:** this review was built from search-result summaries and abstracts, not full readings. No reported speed-ups have been checked. ScoutCache, EchoCache and LightningCP must be read in full before committing to a direction.

## 1. Starting point

Build `rtx-pro-6000-blackwell-001` (run `rtx1-fps-20261123-01`, RTX PRO 6000 Blackwell, compilation off, 20 measured chunks after 2 warm-up):

| Model | Resolution | Measured | Target | Speed-up needed |
|---|---|---|---|---|
| AURA-1.3B | 768×512 | 8.0 fps (2.0 s per 16-frame chunk) | 16 fps | ~2× |
| AURA-5B | 1280×704 | 4.0 fps (3.95 s per chunk) | 25 fps (native playback) | ~6.2× (3.95× for 16 fps) |

The AURA authors already report 20.7 fps for 1.3B and 9.7 fps for 5B on an H100 with "an optimized pipeline." Making 1.3B real time with standard techniques therefore reproduces their claim and is not a research contribution by itself.

## 2. Prior work by idea

### 2.1 Full attention only on the face, cached background
- **LightningCP**, "Lightning Fast Caching-based Parallel Denoising Prediction for Accelerating Talking Head Generation" (AAAI 2026, [arXiv 2509.00052](https://arxiv.org/abs/2509.00052)). Decoupled Foreground Attention runs attention only on foreground (face) tokens and reuses cached background outputs. It also caches static features to skip most layers and predicts denoising steps in parallel. It targets multi-step, non-streaming talking-head diffusion.
- **Region-Adaptive Sampling (RAS)** ([arXiv 2502.10389](https://arxiv.org/abs/2502.10389)). A general diffusion-transformer method that updates only the regions in focus and reuses cached noise elsewhere. Up to about 2.4–2.5× faster on SD3 and Lumina-Next.

**Verdict:** face-region compute for talking heads is taken in the multi-step setting.

### 2.2 Caching guided by audio
- **EchoCache** (ACM MM 2026, [arXiv 2608.02474](https://arxiv.org/abs/2608.02474), [code](https://github.com/IF-LAB-PKU/EchoCache)). A training-free plug-in. A "Cross-modal Saliency Anchor" built from audio time-frequency energy controls how much each step reuses cached results. Tested on Wan2.2-S2V and LongCat-Avatar. The authors say the audio-energy prior is limited and the scheduling partly heuristic.
- **CAVC**, convergence-adaptive audio-visual cache ([arXiv 2610.30854](https://arxiv.org/pdf/2610.30854)).
- **Synchrony-aware cross-modal sparse attention** ([arXiv 2608.15522](https://arxiv.org/pdf/2608.15522)).

**Verdict:** audio-guided caching is taken for multi-step speech-to-video models.

### 2.3 Reuse across chunks in few-step streaming models
- **X-Cache** ([arXiv 2604.20289](https://arxiv.org/abs/2604.20289)). Reuses block residuals across consecutive chunks and recomputes a block only when a fingerprint of its input changes. Its key argument: few-step distillation removes redundancy between steps, but redundancy between chunks remains. Tested on a driving world model.
- **ScoutCache** ([arXiv 2611.21038](https://arxiv.org/abs/2611.21038), November 2026). Training-free; combines denoising and KV-cache optimization for few-step streaming video diffusion. **The closest paper; read first.**
- **PreSTP** ([arXiv 2611.07551](https://arxiv.org/abs/2611.07551)). Training-free spatio-temporal token pruning for few-step streaming video diffusion; up to 3.2× faster.
- **C³ache** ([arXiv 2606.08962](https://arxiv.org/abs/2606.08962)), **Frequency-Step Cache** ([arXiv 2610.15922](https://arxiv.org/abs/2610.15922)), **TGuide** ([arXiv 2610.14657](https://arxiv.org/abs/2610.14657)), **In-Flight KV Cache** ([arXiv 2609.32540](https://arxiv.org/abs/2609.32540)).

**Verdict:** active and crowded, but general-purpose (video, world models). None evaluates talking heads or lip sync.

### 2.4 Sparse attention in streaming video
- **Light Forcing** ([arXiv 2602.04789](https://arxiv.org/abs/2602.04789)). Sparse attention + FP8 + LightVAE: 2.3× faster, 19.7 fps on an RTX 5090 (general video model, not an avatar).
- **Sparse Forcing** ([arXiv 2604.21221](https://arxiv.org/abs/2604.21221)). Trainable native sparse attention for streaming video.
- **IASA**, "Input-Aware Sparse Attention for Real-Time Co-Speech Video Generation" (SIGGRAPH Asia 2025, [arXiv 2510.02617](https://arxiv.org/abs/2510.02617)). Pose-guided sparse attention restricted to matched face and hand regions, plus a loss that protects lips and hands during distillation. Pose-driven, not a direct audio-to-video model like AURA.
- **TalkingMachines** ([arXiv 2506.03099](https://arxiv.org/abs/2506.03099)). Each chunk attends only to the current, previous and first chunk. AURA's rolling cache (first chunk as identity anchor + most recent chunk) already does this.

### 2.5 Overlapping VAE decode with denoising
- **LiveTalk** ([arXiv 2512.23576](https://arxiv.org/abs/2512.23576)): denoising and decoding run in parallel, so per-block latency is the larger of the two rather than their sum.
- **TalkingMachines**: diffusion on one GPU, VAE on another; VAE plus transfer was ~30% of chunk time.
- **StreamAvatar** ([arXiv 2512.22065](https://arxiv.org/abs/2512.22065)): two-GPU pipeline; VAE still over half the processing time.
- **Live Avatar** ([arXiv 2512.04677](https://arxiv.org/abs/2512.04677)): denoising steps spread over GPUs plus a dedicated VAE GPU.

**Verdict:** taken. Most papers use a second GPU; a single-GPU version is engineering.

### 2.6 Lighter decoders for Wan
- **PortraitDirector** ([arXiv 2604.19129](https://arxiv.org/abs/2604.19129)): in its 4-step streaming model the Wan VAE decoder was 50% of latency; a quarter-width retrained decoder was about 4× faster at comparable quality.
- **LightVAE** (secondary source, [ComfyUI wiki](https://comfyui-wiki.com/ru/news/2026-10-13-lightvae)): drop-in Wan2.1/2.2 decoder, up to ~2.9× faster decoding.
- **Reference-guided deep-compression portrait VAE** ([arXiv 2606.01620](https://arxiv.org/abs/2606.01620)): new causal VAE, 42 fps; not a drop-in Wan replacement.
- Our own measurement: see the results ledger only (decoder-only TAEHV 22.9 ms and LightTAE v1/v2 22.5–23.2 ms per chunk vs ~390 ms Wan VAE, `runs/rtx1-taehv-20261124-01`). An earlier unsourced "~4 ms at −3.2 dB PSNR" note was removed because no source run was found.

**Verdict:** usable as an engineering step only.

### 2.7 Fewer steps / distillation for streaming avatars
AvatarForcing (1 step, [arXiv 2603.14331](https://arxiv.org/abs/2603.14331)), TurboTalk ([arXiv 2604.14580](https://arxiv.org/abs/2604.14580)), REST (ICML 2026, [arXiv 2512.11229](https://arxiv.org/abs/2512.11229)), DynaForcing ([arXiv 2608.17707](https://arxiv.org/abs/2608.17707)), Decoupled Self-Forcing / Motar ([arXiv 2609.10317](https://arxiv.org/abs/2609.10317)), Hallo-Live ([arXiv 2604.23632](https://arxiv.org/abs/2604.23632)), JoyAvatar ([arXiv 2512.11423](https://arxiv.org/abs/2512.11423)), Omni-LiveAvatar ([arXiv 2608.13602](https://arxiv.org/abs/2608.13602)), SoulX-FlashHead ([arXiv 2602.07449](https://arxiv.org/abs/2602.07449)), Proxy Avatar with low-rank caching ([arXiv 2608.01978](https://arxiv.org/abs/2608.01978)).

**Verdict:** crowded. Not a promising angle for us.

## 3. Possible gap

No paper found combines all three:

1. **A few-step streaming student** (AURA: 4 steps, rolling KV cache). Step-to-step caching (LightningCP, EchoCache) has little to reuse here; reuse has to happen between chunks.
2. **Per-token reuse across chunks, decided by face/mouth position and by audio.** Example policy: background and torso tokens reuse the previous chunk unless head motion is detected; mouth tokens are always recomputed during speech; silence allows more reuse.
3. **Evaluation on lip sync and identity** (Sync-C/D, LipScore, CSIM) and on worst-case latency (P99) during sustained, paced streaming. The general caching papers report overall video quality and average speed.

**Risk:** this is close to ScoutCache or PreSTP with talking-head-specific gating, so reviewers may call it incremental. It only becomes a paper if:
- the general methods measurably **hurt lip sync** on AURA while ours does not; and
- it brings **AURA-5B to real time** on one GPU, which needs about 4–6×.

## 4. Proposed plan

**Go/no-go experiment (do first).** On AURA-1.3B, apply a general cross-chunk reuse method (ScoutCache or a PreSTP-style token pruning) at the same speed-up as a candidate mouth/audio-gated policy. Compare Sync-C/D, LipScore, CSIM and FVD against the uncached baseline.
- If the general method keeps lip sync, drop the research direction and treat the work as engineering.
- If it loses lip sync and the gated policy does not, continue to 5B.

**In parallel (needed either way), the engineering baseline:** torch.compile, text K/V caching, RoPE precision fix, FP8 where safe, a lighter VAE (TAEHV or LightVAE; LightTAE v2 was later selected), and decode overlap on a single GPU. Each change goes in its own build ID with the existing quality gates.

**Output options (historical; the venue is undecided per the plan — check scope and deadlines when selecting):**

| Route | Contribution | Venue level | Risk |
|---|---|---|---|
| Engineering only | AURA-1.3B and 5B real time on one Blackwell GPU, with honest P99 and sustained-streaming benchmarks | Workshop, demo or technical report | Low; not novel |
| Mouth/audio-gated cross-chunk reuse | Lip-sync-preserving acceleration of few-step streaming avatars; 5B real time | Main conference (e.g. BMVC, WACV, ACM MM) | Medium–high; depends on the go/no-go result and fast-moving competition |

## 5. Ideas from the local research projects

Added 2026-11-24. Taken from the README, docs and code layout of the projects in `RealTime-Avatar/research/` (local) — UnStep, TurboDiffusion, SparkDiffusion / `sparkdiffusion`, FastVideo and Self-Forcing. A later TorchAO FP8 FFN screen is now recorded and rejected in the [results ledger](rtx-aura-results.md#fp8-feed-forward-linear-screen-rejected); other candidates below remain untested unless that ledger states otherwise. Literature gains are estimates for AURA.

**Context when written (rtx1, AURA-1.3B, native 768×512):** with the TAEHV decoder (now replaced by LightTAE v2, same ~23 ms decode cost) the chunk takes about 0.73 s and the DiT about 0.70 s of it (TAEHV decode 23 ms vs ~390 ms for the Wan VAE). The DiT is now the bottleneck.

### 5.1 No retraining

1. **UnStep** (Meta, [arXiv 2609.32518](https://arxiv.org/abs/2609.32518), `research/UnStep`) — training-free acceleration of few-step causal Wan2.1 models (Self-Forcing). Closest fit: AURA is a 4-step causal Wan2.1-1.3B student with a rolling KV cache.
   - Later chunks use 2 denoising steps plus one cheap clean pass (clean sigma 0.02); the first chunk keeps 4. Expected ~40–50% less DiT time; lip quality is the risk.
   - Truncated SVD of V and O projections, layers 8–29, rank fraction 0.92 (small gain).
   - Per-layer attention windows with a 3-frame sink (little gain: AURA already attends to the anchor plus the 2 most recent blocks).
   - Fast VAE cache reset (irrelevant with a lightweight decoder).
2. **8-bit linear layers.** A subsequent native LightTAE profile finds about 342 ms in attention, roughly half the profiled chunk, so the earlier assumption that linears consume most DiT time is not supported. Dynamic per-row FP8 on 60 FFN linears gives +5.93% throughput, but fails all four mouth/raw fidelity checks and one lip-sync case; it is not deployed. See the results ledger for the matched configuration and limits. Other linear policies remain hypotheses.
   - TurboDiffusion (`research/TurboDiffusion`): INT8 W8A8 linear kernels, used for RTX 5090-class GPUs and also for video-VAE decoder blocks.
   - SparkDiffusion (`sparkdiffusion/ops/quantization/fp8_linear.py`, `nvfp4_quant.py`): FP8 linear for Blackwell tensor cores, plus an NVFP4 path.
3. **Fused kernels** (`sparkdiffusion/ops/fused_kernel/`: fused QKV+RoPE, fused gate, fused low-rank). AURA blocks do modulation and gating in FP32 over all tokens, which is memory-bound. Expected modest gain (~5–10%), low quality risk.
4. **SageAttention3 FP4** (FastVideo `fastvideo/attention/backends/sage_attn3.py`): Blackwell FP4 attention. Only worth trying together with 5.2 item 5, since Sage FP8 is already borderline on quality.

### 5.2 Needs fine-tuning

5. **Attention QAT** (FastVideo `docs/training/attn_qat.md`): fine-tune with simulated low-bit attention, then re-distil with DMD2. Would make FP8/FP4 attention safe for AURA instead of borderline; also a possible research contribution.
6. **Trainable sparse attention:** RoLa (SparkDiffusion, [arXiv 2609.06712](https://arxiv.org/abs/2609.06712)), SLA / SageSLA (TurboDiffusion), VSA (FastVideo). 90–97% sparsity after fine-tuning. A face-preserving sparsity pattern connects to the gap in section 3.
7. **Fewer-step distillation:** rCM (TurboDiffusion), CrossDistill (SparkDiffusion, [arXiv 2609.14725](https://arxiv.org/abs/2609.14725)). The trained version of UnStep's step reduction; try UnStep first.

### 5.3 For AURA-5B
TAEHV `taew2_2` / `taew2_2_super` and lightx2v `lighttaew2_2` (`_v2`) are lightweight decoders for the Wan2.2 VAE used by AURA-5B.

Implementation update (2026-11-24): the compatible Wan2.2 LightTAE v2 checkpoint from the [official model card](https://huggingface.co/lightx2v/Autoencoders) has now been screened with the compiled native 5B profile and deployed. Measurements, four-case decoder regression results, provenance and the unmet 25 FPS target belong in the [results ledger](rtx-aura-results.md#native-5b-engineering-transfer-short-web-screen). Other decoder variants above remain candidates.

### 5.4 Suggested order (historical; superseded by the plan's order, which tests FP8 linears before reduced steps)
1. UnStep-style 2-step later chunks plus clean pass, on top of the selected lightweight decoder (LightTAE v2), judged for lip sync.
2. FP8 or INT8 linear layers.
3. Fused modulation and gate kernels.
4. Attention QAT or sparse-attention fine-tuning, if quantized attention is needed without borderline quality or as a paper contribution.

## 6. Broad survey: research that could speed up any part of AURA (2025-11 to 2026-11)

Added 2026-11-24 from a delegated web survey. **[V]** = abstract, README or model card was opened; **[S]** = search snippet only, numbers unconfirmed. Speed-ups are the authors' figures on their own hardware (mostly H100/H200/GB200; no item reports RTX PRO 6000 numbers). Items already covered above are excluded.

**State when written (rtx1, AURA-1.3B, native 768×512):** 14.7 fps baseline (compiled) → 22.1 fps with TAEHV → 26.9 fps with TAEHV + Sage FP8 attention. The DiT is now ~0.70 s of a ~0.73 s chunk. AURA-5B is still at 4.0 fps.

### 6.1 Step reduction for causal video
| Method | Link | Idea | Reported | Training | Fit |
|---|---|---|---|---|---|
| LongLive-2.0 [V] | [2605.18739](https://arxiv.org/abs/2605.18739), [code](https://github.com/NVlabs/LongLive) (Apache-2.0) | NVFP4 training/inference, 4→2-step LoRA, NVFP4 KV cache, async streaming VAE | 5B: BF16 24.8 → NVFP4 2-step 45.7 fps (DiT only, GB200) | fine-tune | **High** (closest public recipe to AURA-5B) |
| Diagonal Distillation [V] (ICLR 2026) | [2603.09488](https://arxiv.org/abs/2603.09488) | More steps on early chunks, fewer later | up to 31 fps | fine-tune | **High** (matches the identity-anchor chunk) |
| TGuide [V] | [2610.14657](https://arxiv.org/abs/2610.14657) | Reuse previous chunk's same-timestep features to skip steps | up to 1.89× on Self Forcing | small predictor | **High** |
| ForcingBooster [V] | [2610.12625](https://arxiv.org/abs/2610.12625) | Learned per-patch early exit | ~50% fewer patch-steps | light training | **High** (static background) |
| One-Forcing [V], Any Forcing [V], AAD-1 [S], Adversarial Self-Distillation [S] | [2605.23458](https://arxiv.org/abs/2605.23458), [2610.42031](https://arxiv.org/abs/2610.42031), [2606.03972](https://arxiv.org/abs/2606.03972), [2511.01419](https://arxiv.org/abs/2511.01419) | One-step causal generation via adversarial losses | Any Forcing 20.9–24.7 fps (H100) | fine-tune | Medium |
| rCM [V] (ICLR 2026) | [2510.08431](https://arxiv.org/abs/2510.08431) | Consistency + score regularisation, 1–4 steps on Wan2.1 | — | fine-tune | Low–medium |

### 6.2 Attention and KV cache
| Method | Link | Idea | Reported | Training | Fit |
|---|---|---|---|---|---|
| HeadCast [V] | [2607.20125](https://arxiv.org/abs/2607.20125), [code](https://github.com/sjlgaga/HeadCast) | Per-head class (sink/dummy/spatial/global) with its own KV path | 1.62× at 720p, 1.95× at 1080p | training-free | **High for 5B** |
| Forcing-KV [V] | [2605.09681](https://arxiv.org/abs/2605.09681) | Prune static/dynamic heads in the KV cache | 1.5× (480p) to 2.82× (1080p) on Self Forcing | n/s | High |
| FPSAttention [V] (NeurIPS 2025) | [2506.04648](https://arxiv.org/abs/2506.04648) | Joint FP8 + sparsity training | 4.96× end-to-end (Wan2.1, 720p) | fine-tune | Medium (fixes borderline FP8 quality) |
| TRT-LLM SAGE + skip-softmax [V] | NVIDIA TensorRT-LLM blog 28 | INT8/FP8 QK, FP8 PV, skip-softmax | 1.43–1.54× on Wan2.2-A14B (B200) | training-free | Medium |
| LoSA [S], SVOO [S], FAST-AR [V], LVSA [V], ESTRA [V] | [2608.12032](https://arxiv.org/abs/2608.12032), [2603.18636](https://arxiv.org/abs/2603.18636), [2602.01801](https://arxiv.org/abs/2602.01801), [2605.31057](https://arxiv.org/abs/2605.31057), [2611.10448](https://arxiv.org/abs/2611.10448) | Training-free sparse attention / KV compression | 1.36×–4.7× (mostly long sequences) | training-free | Low–medium (AURA's KV is short) |
| ARL2 [V], Hybrid Forcing [V], ReHyAt [S] | [2605.16579](https://arxiv.org/abs/2605.16579), [2604.10103](https://arxiv.org/abs/2604.10103), [2601.04342](https://arxiv.org/abs/2601.04342) | Hybrid linear/softmax attention | ARL2 2.26×; Hybrid Forcing 29.5 fps (H100) | fine-tune | Medium |
| Quant VideoGen [V] (ICML 2026) | [2602.02958](https://arxiv.org/abs/2602.02958) | 2-bit KV cache | 7× less KV memory, no speed-up | training-free | Low (memory for 5B on a 5090) |

### 6.3 Quantisation and pruning of linears
| Method | Link | Idea | Reported | Training | Fit |
|---|---|---|---|---|---|
| LightX2V Self-Forcing-NVFP4 [V] | [model card](https://huggingface.co/lightx2v/Self-Forcing-NVFP4) (Apache-2.0) | NVFP4 QAT of Self-Forcing on Wan2.1-1.3B with an NVFP4 kernel | "approaching real-time" on RTX 5090 | QAT | **High** (same backbone and GPU class) |
| 6Bit-Diffusion [V] | [2603.18742](https://arxiv.org/abs/2603.18742) | Per-layer NVFP4/INT8 + temporal-delta block cache | 1.92× end-to-end | PTQ | High |
| Timestep-aware SVDQuant-GPTQ W4A4 [S] | [2605.27003](https://arxiv.org/abs/2605.27003) | Low-rank branch + GPTQ | −0.9% VBench on Wan2.2-I2V | PTQ | Medium–high |
| DVD-Quant [S], QVGen [S] | [2505.18663](https://arxiv.org/abs/2505.18663), [2505.11497](https://arxiv.org/abs/2505.11497) | W4A4 PTQ / QAT for video DiTs | near-lossless (claimed) | PTQ / QAT | Medium |
| PARE [S], MobileWan [S] | [2605.27336](https://arxiv.org/abs/2605.27336), [2607.06173](https://arxiv.org/abs/2607.06173) | Width/depth/head pruning | n/s | fine-tune | Medium (MobileWan targets Wan2.2-5B) |

### 6.4 Caching for streaming video
Frequency-Step Cache ([2610.15922](https://arxiv.org/abs/2610.15922)) [V], TRAC ([2610.02779](https://arxiv.org/abs/2610.02779)) [V] and FlowCache ([2602.10825](https://arxiv.org/abs/2602.10825)) [V] report 2–6.7×, but on 30–50-step models; with AURA's 4 steps there is little to skip. TGuide (6.1) and ScoutCache (section 2.3) are the relevant ones.

### 6.5 Decoder
- **taew2_2 / taew2_2_super** [V] ([taehv](https://github.com/madebyollin/taehv), MIT): tiny streaming decoder for the Wan2.2 latent of AURA-5B; ~1.8 ms/frame vs 50.5 ms for the Wan2.2 VAE. **High for 5B**, no training.
- **FlashDecoder** [V] (CVPR 2026, [2607.14898](https://arxiv.org/abs/2607.14898)): transformer decoder with rolling KV on Wan latents, PSNR matching the conv decoder, 3.6–12×. Higher-quality alternative if TAEHV loses lip/teeth detail; code not found.
- **Flash-VAED** [S] ([2602.19161](https://arxiv.org/abs/2602.19161)), **Turbo-VAED** [S] ([2508.09136](https://arxiv.org/abs/2508.09136)), **DC-VideoGen** [V] ([2509.25182](https://arxiv.org/abs/2509.25182); 4× fewer DiT tokens but needs re-distilling AURA).

### 6.6 Audio, streaming and runtime
- **Side-stream overlap** [S] ([2606.05981](https://arxiv.org/abs/2606.05981)): audio encoder and decode on a second CUDA stream, overlapped with the DiT of the next chunk. Mainly a first-frame latency win.
- **DyStream** [V] ([2512.24408](https://arxiv.org/abs/2512.24408)): causal audio encoders (Mimi etc.) with ~60 ms lookahead keep lip sync; relevant to replacing 5B's non-causal wav2vec2.
- **Block Cascading** [V] ([2511.20426](https://arxiv.org/abs/2511.20426)), **StreamDiffusionV2** [V] ([2511.07399](https://arxiv.org/abs/2511.07399)): pipeline chunks/steps across GPUs; low fit on one GPU.
- **CUDA graphs:** TRT-LLM VisualGen captures the transformer forward per shape; FastUSP [S] ([2602.10940](https://arxiv.org/abs/2602.10940)) finds launch overhead significant on the RTX 5090. AURA currently compiles with `max-autotune-no-cudagraphs`, so this is untried.
- **cuDNN SM120 AOT attention** [V]: experimental attention benchmarked on the RTX 5090 and RTX PRO 6000.
- **SANA-Streaming** [V] ([2605.30409](https://arxiv.org/abs/2605.30409)): 1280×704 at 24 fps end-to-end on one RTX 5090 with a co-designed hybrid DiT. Shows 5B-class resolution in real time on this GPU class is possible; no code.

### 6.7 Efficient talking-head systems
- **SoulX-FlashHead** [V] ([2602.07449](https://arxiv.org/abs/2602.07449), Apache-2.0): Wan2.1-1.3B base; Lite reaches 96 fps on an RTX 4090 mainly by using the high-compression LTX-Video VAE.
- **Lip Forcing** [V] ([2606.11180](https://arxiv.org/abs/2606.11180)): 1.3B 2-step causal lip sync at 31 fps with a SyncNet reward.
- **JoyStreamer-Flash** [V] ([2512.11423](https://arxiv.org/abs/2512.11423)): more steps on initial frames (same idea as Diagonal Distillation).
- Face-crop generation with paste-back: no 2025–26 paper found; still an open idea.

### 6.8 Top 10 for AURA (survey ranking: expected DiT gain × feasibility on one Blackwell GPU without retraining first)
1. **FP8, then NVFP4 linear layers** (historical survey priority; keep audio/text cross-attention and modulation in higher precision). The first TorchAO dynamic FP8 FFN screen has since produced a 5.93% speed gain but failed fidelity in all four cases and sync in one. That configuration is rejected; any further selective policy, QAT or NVFP4 work needs new matched quality evidence. See the results ledger rather than treating this ranking as an observed gain.
2. **taew2_2_super decoder for AURA-5B** via the same streaming adapter.
3. **Asymmetric steps:** 4 steps on the anchor chunk, 2–3 on later chunks (training-free probe; LongLive-2.0-style 2-step LoRA if it degrades). Overlaps with UnStep (section 5.1).
4. **Head-wise KV/attention routing** (HeadCast, Forcing-KV); mainly a 5B win.
5. **Region-adaptive steps:** freeze background tokens after step 2 using a face/mouth mask (probe), then a TGuide-style predictor.
6. **CUDA-graph capture** of the per-chunk 4-step loop with static KV buffers.
7. **Side-stream overlap** of audio encoding and decode with the next chunk's DiT.
8. **Skip-softmax / block-sparse self-attention** with a conservative threshold.
9. **QAT for FP8 attention** (FPSAttention, Attn-QAT) to remove the borderline quality of Sage FP8.
10. **Retraining options:** hybrid linear attention, head pruning for 5B, or a high-compression latent (LTX-VAE / DC-AE-V style); largest possible win but needs re-distillation.

**Not verified:** all [S] items; the hardware behind most arXiv numbers; whether LongLive-2.0's 5B is Wan2.2-TI2V-5B; Nunchaku support for Wan; FA4/FlashInfer on SM120; FlashDecoder code; Lip Forcing's first-frame latency claim.

## 7. Reading list before committing

1. ScoutCache ([2611.21038](https://arxiv.org/abs/2611.21038)): closest; check whether it already handles conditioning signals or regions.
2. EchoCache ([2608.02474](https://arxiv.org/abs/2608.02474)): check whether its audio anchor works per token and whether it was tried on few-step models.
3. LightningCP ([2509.00052](https://arxiv.org/abs/2509.00052)): check speed-ups, lip-sync results and whether it handles streaming.
4. PreSTP ([2611.07551](https://arxiv.org/abs/2611.07551)) and X-Cache ([2604.20289](https://arxiv.org/abs/2604.20289)): baselines for the go/no-go test.
5. AURA paper ([2611.26540](https://arxiv.org/abs/2611.26540)): what "optimized pipeline" includes, to avoid claiming their work.
