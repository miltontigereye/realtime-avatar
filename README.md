# RealTime Avatar research sources

The [`research/`](research/) directory contains the full locally available readable source for FastVideo, LightX2V, Self-Forcing, SparkDiffusion, TurboDiffusion, UnStep, and AURA. The `sparkdiffusion` implementation is physically stored next to the `SparkDiffusion` framework in the local project and is preserved that way here.

Included: model implementations, training and inference pipelines, kernels, tests, demos, configuration, dependency declarations, build scripts, documentation, and licenses.

Excluded: model weights, datasets and prompt corpora, audio/video/images, Git and agent metadata, environments, caches, compiled binaries, generated outputs, and dependency lockfiles. Data-loading and checkpoint-loading source code is included. Some documentation references omitted assets.

This is a source snapshot, not a complete installed environment. Uninitialized upstream submodules and third-party dependencies are not vendored. [`research/SOURCE_MANIFEST.json`](research/SOURCE_MANIFEST.json) records upstream revisions, submodule declarations and availability, per-project counts, and file hashes. Working copies may differ from the recorded upstream commits.

For a smaller algorithm reading selection, start with [`research-context/INDEX.txt`](research-context/INDEX.txt). Current AURA plan, results, and literature notes are in [`research-context/project/`](research-context/project/). The deployed optimized AURA snapshot is separately represented in [`research-context/sources/AURA-deployed/`](research-context/sources/AURA-deployed/); `research/aura/` preserves the local source tree.

GitHub is the shared handoff point for Codex, Claude, local development, rtx1, and rtx2. Commit proposals and prototype changes on separate branches for review and refresh imported project files after GitHub changes. GPU deployment is a separate step.

To regenerate the source snapshot from a local checkout into a new directory:

```bash
python tools/export_research_sources.py /path/to/RealTime-Avatar /path/to/new-export/research
```
