# RTX Aura research plan and milestones

Updated: 2026-11-24 (Asia/Dhaka).

This is the single maintained plan and milestone tracker for RTX Aura acceleration. Measured outcomes belong in [rtx-aura-results.md](rtx-aura-results.md); the [literature review](literature-aura-acceleration.md) supplies research candidates, not completed results. This tracker covers the AURA acceleration branch, not completion of the separate LiveAvatar-Tiny milestones in PROJECT_PLAN.md.

Canonical locations, maintained with identical contents:
- Local: `/home/knoledge_base/RealTime-Avatar/docs/rtx-aura-plan.md`
- rtx1: `/workspace/avatar/docs/rtx-aura-plan.md`

## Initial Guidelines
Facial Identity Stabilization for Wan 2.1 1.3B and Wan 2.2 S2V 5BThe current facial distortion issue may be related to Identity Drift, Temporal Inconsistency, or inference acceleration.

Before increasing model size, we should determine whether the degradation originates from the base model itself or from the modifications introduced to achieve real-time video generation.
1. Potential Root CausesCauseSymptomsPriorityIdentity DriftThe face gradually changes into a different personCriticalLatent DriftEyes, nose, mouth, and facial contours become distortedCriticalChunk Boundary InconsistencyFacial distortion occurs periodicallyHighInsufficient Reference ConditioningGenerated faces deviate from the original imageHighReduced Denoising StepsLoss of facial details and structural integrityHighAudio Conditioning OverloadExcessive deformation of the mouth and jawHighCache AccumulationQuality deteriorates during continuous generationHighQuantizationFine facial features become unstableMediumVAE ReconstructionFacial details are lost during encoding and decodingMedium2. Recommended ArchitectureInstead of relying entirely on the video generation model, we should introduce an independent facial identity stabilization mechanism.
3. Technical SolutionsTechnologyPurposePriorityReference Latent Re-injectionPeriodically restore original facial identity informationCriticalAudio Timestep ModulationPrevent excessive facial deformation caused by audio conditioningCriticalTemporal MemoryMaintain consistency across generated framesHighAdaptive Cache ResetPrevent accumulated errors in inference cachesHighIdentity LoRAImprove facial identity preservationHighFacial Landmark ControlMaintain stable facial geometryHighFace ROI RefinementImprove facial quality without increasing full-frame computationMediumIdentity LossOptimize identity preservation during trainingHighOverlapping Chunk GenerationReduce discontinuities between video chunksHigh4. Research Direction: StableAvatarStableAvatar is particularly relevant because it addresses long-duration audio-driven video generation using Wan 2.1 1.3B.

We should investigate its approach to:


Long-duration identity preservation
Audio conditioning optimization
Temporal consistency
Error accumulation prevention
Reference image conditioning
Continuous video generation


We should also evaluate whether these techniques can be integrated into the existing accelerated inference pipeline without significantly increasing GPU consumption.

Note that the official Wan 2.2 S2V model is 14B, while the official 5B model is TI2V. If the current S2V 5B implementation is a customized or distilled model, its audio-conditioning architecture should also be investigated.
5. Root Cause InvestigationWhen distortion occursPotential causeImmediately after generation startsBase model, VAE, denoising stepsGradually over timeLatent drift and accumulated errorsAt regular intervalsChunk boundaries and cache managementDuring speechAudio conditioningWhen opening the mouth widelyFacial geometry instabilityDuring head rotationInsufficient reference or pose conditioningOnly with accelerated inferenceQuantization, distillation, or reduced denoising steps6. Proposed Development KPIsKPITargetEffective generation speed≥25 FPSInitial video response latency≤500 msFacial identity cosine similarity≥0.85, subject to calibrationFacial distortion rateBelow 0.1%Continuous generation≥60 minutesAudio-visual synchronization error≤80 msThese are proposed engineering targets, not measured performance results.
7. Identity Guardian ModuleWe should develop an independent Identity Guardian module to continuously monitor facial identity and structural integrity during video generation.

The module should evaluate facial embeddings, facial landmarks, structural integrity, and temporal consistency. When degradation is detected, it should trigger reference reconditioning, adaptive cache resets, or localized facial refinement, depending on the identified cause.

This approach could potentially improve both Wan 2.1 1.3B and Wan 2.2 5B without requiring a significantly larger video generation model.
8. Recommended Development StrategyWan 2.1 1.3B + StableAvatar Techniques + Identity Guardian + Accelerated Inference

The objective should not simply be to maximize FPS, but to achieve:


Stable facial identity
Natural facial expressions
Accurate lip synchronization
Long-duration video generation
Low inference latency
Minimal GPU hosting costs


The first step is to determine whether facial distortion occurs in both the 1.3B and 5B models, or only in the 1.3B model.

We should then compare standard inference against accelerated inference under identical conditions to isolate the root cause.

The ultimate goal is to achieve identity-preserving, real-time video generation without compromising visual quality or computational efficiency.


## Objective and current stage

Accelerate AURA on one RTX PRO 6000 Blackwell while preserving speech timing, identity, causal execution and bounded streaming state. Native targets are 16 playback FPS at 768×512 for 1.3B and 25 playback FPS at 1280×704 for 5B. Retain four released diffusion steps unless explicitly labeling a separate experiment.

Current stage: native 1.3B demo deployed with LightTAE v2; short web throughput and four-input quality screens recorded. Sustained validation of this exact deployed configuration remains open. 5B has an archived eager baseline only in the reviewed evidence.

Active demo: `experiments/rtx1-lighttae-demo-001/run_lighttae_server.sh`, port 8000 on rtx1. AURA-1.3B, 768×512, steps [1000,745,498,255], compiled DiT, ordinary SDPA, FP16 streaming LightTAE v2 decoder; transformer BF16 except time/audio modules FP32. Sage attention and reduced-step inference are not enabled. Deployment verification and measured web result are recorded in the results file.

## Milestones

| ID | Deliverable and exit criterion | State / evidence |
| --- | --- | --- |
| A0 | Archive reproducible eager baselines for both sizes, with settings and timing boundaries | Complete: `rtx1-fps-20261123-01` and local frozen build |
| A1 | Profile and optimize compilation, invariant conditioning and pixel transfer; establish renderer pace gate | Complete for reduced 672×448 1.3B: three-minute paced and short web runs. Native compiled Wan-VAE path still below 16 FPS |
| A2 | Screen attention backends, decoder overlap and lightweight decoders with paired quality checks | Initial screen complete: FA4 and overlap below 2% speed gate; Sage quality depends on resolution; LightTAE v2 chosen with disclosed fidelity loss |
| A3 | Deploy native LightTAE v2 demo and verify HTTP/WebSocket output, audio, dimensions and short throughput | Complete for short web integration: 22.504 FPS, P99 0.72 s. Does not close A4 |
| A4 | Validate the exact native deployed profile over at least ten measured minutes, broader inputs and human A/V review | Pending; immediate next milestone |
| A5 | Reduce DiT time using isolated, quality-gated FP8/INT8 linear or asymmetric-step experiments | Planned; no completed AURA results found for these candidates |
| A6 | Accelerate 5B to native 25 FPS (16 frames per 0.64 s), with sustained and quality evidence | Planned; baseline 4.047 FPS |
| R0 | Establish whether mouth/audio-aware cross-chunk reuse has a defensible research gap | Pending full primary-source reading and matched-compute go/no-go experiment |

Completion refers only to each row's stated scope. A running demo or short throughput pass does not prove sustained end-to-end real time.

## Next actions in order

Publication priority: P0/P1 run first, alongside A4, before substantial work on A5/A6. A faster demo alone does not establish the paper's research claim. This list is the authoritative order; literature §5.4 and §6.8 rankings are inputs to it.

1. **P0 novelty audit.** Read ScoutCache, EchoCache, LightningCP, PreSTP, X-Cache and the AURA paper (plus its official optimized pipeline) in full. Record each in the literature citation format and complete its related-work comparison table and novelty decision record.
2. **P1 data and pilot.** Identify licensed/consented portrait and speech sources for ≥30 held-out identities and build the split manifest (research-only licences such as CC-BY-NC are acceptable for the paper; new identifiable people follow `docs/ethics-and-consent.md`). Check the AURA weights licence for publishing derived results and code. Run the disposable matched-compute pilot (E-P01) on development data only.
3. Freeze the exact A3 source/configuration and input/checkpoint hashes in a fresh run. Complete human paired A/V review of LightTAE v2 versus Wan VAE, including the side-view producer case. Expand portraits and speech; retain the original four cases.
4. Run the exact deployed native SDPA + LightTAE v2 profile for at least ten measured minutes after warm-up. Record mean/P50/P95/P99, maximum chunk time, deadline misses, queue growth, dropped samples, recompilation and memory over time. Measure microphone-to-display latency separately; browser latency display alone is not an independent validation.
5. Profile DiT with the selected decoder. Test FP8 dynamic linears first, then INT8/NVFP4 only if compatible and useful. Keep audio/text cross-attention and modulation in higher precision initially. Save each variant separately and compare at identical resolution, audio, portrait, steps and seed. Literature speedups remain hypotheses.
6. Probe four anchor steps and two or three later steps in a separate experiment. Explicitly invalidate incompatible timestep caches; verify cached/uncached equivalence and lips. Do not silently reuse missing timestep histories.
7. For 5B, test a compatible Wan2.2 lightweight decoder and compilation independently, then combine only validated changes. Native playback needs P99 within 0.64 s per 16-frame chunk; a 16 FPS result is a separately labeled reduced-playback target.

## Research go/no-go

Read ScoutCache, EchoCache, LightningCP, PreSTP and the AURA paper in full before claiming novelty. Compare a general cross-chunk reuse baseline against mouth/audio-gated reuse at equal average compute on 1.3B. Measure SyncNet confidence/distance/offset, identity, lip detail, temporal stability and suitable distributional video quality on enough clips. If general reuse preserves lips equally well, pursue an engineering report/demo. Continue the research route only if the proposed policy demonstrates a reproducible lip-sync advantage, then evaluate 5B. No publication claim is established by the current screens.

## Measurement and update rules

- Keep all stages co-resident on one GPU for integrated claims; distinguish diagnostic kernels from pipeline timing. Report GPU, dimensions, dtype, steps, playback rate and timing boundary with every result.
- FPS is total measured decoded frames / total measured generation time. Separate setup, warm-up, pacing waits and terminal drain. Preserve failures and superseded results; never overwrite a run.
- Retain existing quality thresholds: a failed lossless threshold stays failed. LightTAE v2's producer mouth PSNR miss is an explicit fidelity tradeoff, even though lip-sync and identity screens pass.
- For accelerated candidates, validate bounded caches/queues, causal audio accounting and conditioning, alongside quality and speed. Compare latency distributions and equal-compute baselines.
- Record new experiments in the single results file with source run IDs, settings, quality decision, deployment state and limitations. Update milestone state here only when its exit evidence exists.
- Maintain the three literature/plan/results documents in both locations and compare SHA-256 after copying. Existing experiment reports and raw runs remain historical evidence, not competing current trackers. Synchronize only intended edits to these three files; preserve unrelated project plans.

## Paper objective and hypotheses

Working title: **Lip-Sync-Preserving Acceleration of Few-Step Streaming Talking Avatars**. This is a provisional scope, not a claim of novelty or publication readiness. Venue and submission date remain undecided; check actual venue scope, deadlines and artifact policies when selecting one.

- H1 (research): mouth/audio-aware reuse gives a better lip-sync–latency Pareto frontier than the strongest verified general reuse baseline at matched compute on the same model.
- H2 (mechanism): the benefit comes from protecting speech-sensitive computation, beyond generic face masks, audio energy alone or uniformly allocating the same compute.
- H3 (systems): the combined method can meet native playback deadlines during sustained single-GPU operation with bounded state and acceptable identity/temporal quality. Report separately for 1.3B and 5B; a 5B real-time result is an ambitious target, not a prerequisite for every possible paper scope.

Current engineering results support baseline feasibility only. H1–H3 are unproven for the proposed method. Decide scope from evidence; do not promise a venue or acceptance.

## Publication milestones and gates

| ID | Deliverable | Exit criterion | Current state |
| --- | --- | --- | --- |
| P0 | Verified related work and novelty audit | Full readings of closest methods and upstream AURA; exact differences and reproducible baseline choice documented in literature | Pending |
| P1 | Frozen evaluation protocol and research pilot | Disjoint development/final splits, fixed metrics/budgets and a matched-compute pilot comparing general reuse with candidate gating | Pending |
| P2 | Method specification and implementation | Causal inputs, gating equations/pseudocode, per-timestep reuse, invalidation, failure behavior, overhead and bounded-state analysis; focused correctness tests | Pending |
| P3 | Main comparisons and mechanism ablations | H1/H2 tested with paired clips/seeds, uncertainty, budget sweep and full run provenance; negative outcomes retained | Pending |
| P4 | Generalization and sustained deployment evaluation | Held-out identities/audio and stress cases; A4 plus proposed-method pace tests; 5B transfer where supported | Pending |
| P5 | Paper and artifact freeze | Every table/figure/claim linked to valid runs, reproducible configs and scripts, limitations, data rights and citations checked | Pending |

P0/P1 are the immediate research priorities. Do not start large training or claim the method is new before P0. A small disposable pilot can test feasibility while reading, but cannot serve as the held-out final result.

## Evaluation protocol to freeze at P1

**Data and separation.** Build a manifest of licensed/consented portrait and speech sources, identities, durations, conditions, hashes and split assignments. Keep appearance references disjoint from target evaluation footage. Select data under the repository's provenance and consent rules; check official data terms before acquisition. Use speaker/identity-disjoint development and final test sets. The existing four demo cases are development screens, not an unseen benchmark. Reserve the final set before policy tuning; do not select thresholds or best checkpoints on it.

Planning floor: at least 30 held-out identities with two speech clips each and three paired generation seeds per configuration, subject to a pilot-based uncertainty/power assessment and data availability. Freeze the final count and rationale before final evaluation; increase it if uncertainty is too wide. Cover frontal/profile pose, speech rate, silence, head motion and difficult illumination; report subgroup counts. Generated seeds are repeated measurements, not new independent identities. Long stress streams are evaluated separately from short-clip quality.

**Methods and controls.** Compare (B0) upstream eager AURA, (B1) strong compiled exact-compute baseline, (B2) compiled baseline plus selected lightweight decoder, (B3) strongest verified compatible general reuse/pruning method, and (M) proposed mouth/audio-aware policy on the same B2 foundation. Add the strongest compatible quantization/step baseline after screening. If an official closest method cannot run on AURA, disclose the adaptation and fidelity checks; label a simplified reimplementation honestly. Compare M against B3 on a shared Wan decoder as well, so decoder fidelity cannot explain a method advantage. Keep portraits, audio bytes, seeds, dimensions, cache windows and schedules paired; explicitly label any changed schedule/precision.

**Budgets and ablations.** On development data choose a sweep from low to high compute, then freeze configurations. Compare matched average measured DiT time (with a declared matching tolerance) and matched quality, reporting total pipeline time including policy overhead and requested versus achieved budgets. Include no reuse, uniform/random reuse, face-only, audio-only, mouth-only, combined mouth+audio, and periodic refresh controls at equal budget; ablate cache age/history and each engineering component. Test silence-to-speech transitions, mask/tracker error, sudden pose changes, accumulated drift and recovery. Keep masks/gating causal and include their runtime. Do not infer a causal mechanism solely from a favorable aggregate score.

**Metrics and statistics.** Primary outcome: lip-sync–latency Pareto comparison using SyncNet confidence/distance/offset with validated audio-shift controls. Report face identity, lip detail, temporal stability, face detection coverage and failed samples. Use LipScore only after its implementation and validity are established. Pixel PSNR to the unchanged renderer measures regression, not true perceptual quality. Add distributional video quality only with a justified real reference set, preprocessing and enough clips; the four demo cases do not support a reliable FVD claim. Use paired, randomized and blinded A/V preference comparisons under an appropriate consent/review protocol; freeze participant/task/sample counts and analysis before collecting judgments.

Predeclare primary metrics, acceptable regression margins and the comparison rule at P1 using development data and measurement variability. Preserve the existing 35 dB engineering threshold and its historical failures; any perceptual acceptance criterion is a separately justified criterion, not a retroactive replacement. Report per-case results, paired effects and 95% confidence intervals using identity-clustered resampling or an appropriate repeated-measures model. Never treat correlated chunks as independent test subjects. Report seed variation and all failed/missing evaluations; no method selection on final-test scores.

**Systems evaluation.** Run all stages on one GPU without competing inference. Report stage timings, mean/P50/P95/P99/max latency, total-frame throughput, deadline misses, audio lookahead, startup, policy overhead, memory/queue trajectories and input drops. Run at least ten measured minutes per selected final profile after warm-up and repeat with three fresh process starts to expose runtime variability. Report cold and warm behavior separately. Measure microphone-to-display with an independent timestamp/alignment method; current UI estimates are diagnostic. P99 deadline targets: ≤1 s for native 1.3B, ≤0.64 s for native 5B, alongside no growing queues or unreported audio loss. State streams/GPU at the P99 deadline; if only one stream was tested, report that limit.

## Planned paper tables and figures

| Output | Content / supporting gate |
| --- | --- |
| Table 1 | Verified closest-work comparison: causal/few-step setting, conditioning, region granularity, quality and timing protocol (P0) |
| Table 2 | Held-out quality and integrated throughput/latency for B0–B3 and M, including uncertainty (P3) |
| Table 3 | Matched-budget component and policy ablations (P3) |
| Table 4 | 1.3B/5B transfer, sustained deadline misses, memory and concurrency (P4; label missing cells) |
| Figure 1 | Causal method diagram and bounded cache lifecycle (P2) |
| Figure 2 | Lip-sync–latency Pareto curves with uncertainty (P3) |
| Figure 3 | Representative paired successes/failures and speech-transition behavior, selected by a declared rule (P3/P4) |
| Figure 4 | Long-stream latency, queue, memory and quality drift (P4) |

Manuscript outline: problem and measured gap; verified related work; causal method; frozen evaluation protocol; main results and ablations; failure analysis and limitations; reproducibility and data/consent statement. Keep drafting material within these three maintained files until a manuscript is explicitly created. Do not add another status/plan/results document.

Artifact gate: archive source revision, configs, environment/kernel versions, checkpoint/input hashes, seeds, manifests, raw metrics and table/figure generation commands; rerun selected results from the frozen artifact. Release only data/weights permitted by their licenses and consent. Distinguish measured on-device throughput from end-to-end experience, tested hardware from generalization claims, and engineering integration from algorithmic contribution.
