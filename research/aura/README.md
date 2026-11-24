# AURA: AUdio-dRiven streaming Avatar

[**Live demo**](https://aura.kyutai.org/) · [**Project page**](http://kyutai.org/aura) · [**Paper**](https://arxiv.org/abs/2611.26540) ·
[**AURA-1.3B**](https://huggingface.co/kyutai/AURA-1.3B) · [**AURA-5B**](https://huggingface.co/kyutai/AURA-5B) ·
[**AURA-1.3B teacher**](https://huggingface.co/kyutai/AURA-1.3B-Teacher-TalkVerse) · [**AURA dataset**](https://huggingface.co/datasets/kyutai/AURA-Dataset)

Nathaniel Cohen<sup>1,2</sup>, Nicolas Dufour<sup>1</sup>, Amélie Royer<sup>1</sup>, Alasdair Newson<sup>2</sup>, Patrick Pérez<sup>3*</sup><br>
<sup>1</sup>Kyutai · <sup>2</sup>Sorbonne Université, ISIR · <sup>3</sup>Google DeepMind · <sup>*</sup>work done while at Kyutai<br>
*BMVC 2026*

AURA turns a single portrait image and a speech audio stream into a talking-head video, **chunk by chunk,
in streaming**. Each chunk needs only its own audio, with no multi-second look-ahead. The models are 4-step
**causal** students distilled from 50-step bidirectional speech-to-video teachers. A rolling KV cache keeps
the first chunk as an identity anchor plus the most recent chunk, so memory stays constant on arbitrarily
long streams. Samples are on the [project page](http://kyutai.org/aura), and you can talk to AURA-1.3B combined with our cascaded voice model Unmute in
the [live demo] too (https://aura.kyutai.org/).



https://github.com/user-attachments/assets/de813f1c-0836-4c17-9851-096e8dfcbdd9



## Models

| Model | Teacher | Audio encoder | Output | Speed (paper, 1× H100) | Weights license |
|---|---|---|---|---|---|
| [AURA-1.3B](https://huggingface.co/kyutai/AURA-1.3B) | [AURA-1.3B-Teacher-TalkVerse](https://huggingface.co/kyutai/AURA-1.3B-Teacher-TalkVerse) (on [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)) | [Mimi](https://huggingface.co/kyutai/mimi) | 576p, 16 fps | **20.7 fps (real time)**, 23× faster than its teacher | CC BY-NC 4.0 |
| [AURA-5B](https://huggingface.co/kyutai/AURA-5B) | [TalkVerse-S2V-5B](https://huggingface.co/snap-research/talkverse-s2v-5b) (on [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)) | [wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) | 720p, 25 fps | 9.7 fps, 17× faster than its teacher | Snap Inc. Non-Commercial |

The reported  speeds are end-to-end, measured with an optimized pipeline on a H100 GPU. With `--compile`, the reference
code in this repository runs AURA-1.3B at ~16 fps (real time for 16 fps playback) and AURA-5B at ~8 fps
on one H100.

<img width="1800" height="651" alt="V3_method" src="https://github.com/user-attachments/assets/122ed3a4-8122-4513-bee0-af0c2a625f7b" />


## Installation

To use AURA, you will need a CUDA GPU and [uv](https://docs.astral.sh/uv/) for the python setup:

```bash
git clone https://github.com/kyutai-labs/aura.git && cd aura
uv sync
```

## Usage

```bash
uv run generate.py --image portrait.jpg --audio speech.wav --output out.mp4                  # AURA-1.3B
uv run generate.py --model aura-5b --image portrait.jpg --audio speech.wav --output out.mp4  # AURA-5B
```

AURA is trained on LoRA starting from a Text-to-Video (T2V) non-causal model. 
On first use, the required weights are downloaded from the Hugging Face Hub: The final AURA checkpoint
(`stage2_dmd/*_merged.safetensors`), the VAE and text encoder of the Wan base model, and the audio
encoder (Mimi or Wav2vec depending on the side). The video covers the whole audio. 

Options: `--prompt` (a short description of the video),
`--num_chunks` (length, in chunks of 16 frames), `--seed`, `--compile` (`torch.compile` the transformer:
faster, after a few slow chunks).

The portrait is resized and center-cropped to the closest training resolution. Square or landscape
portraits where the face fills a good part of the frame work best.

You can also use `generate` via the Python API

```python
from aura import Aura

aura = Aura("aura-1.3b")
for chunk in aura.generate("portrait.jpg", "speech.wav"):  # [3, frames, H, W], in [-1, 1]
    ...
```

## Results

| Model | Benchmark | FID ↓ | FVD ↓ | JEDi ↓ | CSIM ↑ | Sync-C ↑ | Sync-D ↓ | WE ↓ |
|---|---|---|---|---|---|---|---|---|
| AURA-1.3B | HALLO3 test (100 clips, 10 s) | 56.8 | 177.1 | 0.403 | 0.84 | 4.07 | 9.93 | 0.0036 |
| AURA-1.3B | CelebV-HQ test (150 clips) | 51.5 | 248.8 | 0.574 | 0.77 | 4.76 | 10.1 | 0.0046 |
| AURA-5B | HALLO3 test (100 clips, 10 s) | 56.9 | 195.5 | 0.313 | 0.79 | 4.30 | 9.87 | 0.0032 |
| AURA-5B | CelebV-HQ test (150 clips) | 54.0 | 208.1 | 0.320 | 0.72 | 4.94 | 10.0 | 0.0043 |

AURA-1.3B has the best identity preservation (CSIM) of all compared methods on both test sets. In a user
study on HALLO3 test, AURA-5B is preferred over Hallo3 on all four axes (visual quality, lip-sync,
reference fidelity, naturalness) and is on par with or above its 50-step teacher. See the
[paper](https://arxiv.org/abs/2611.26540) for the full tables and ablations.

## Code

| File | Content |
|---|---|
| `aura/pipeline.py` | Model loading, streaming generation loop (few-step sampling, causal VAE decoding) |
| `aura/model.py` | Diffusion transformer with audio cross-attention |
| `aura/kv_cache.py` | Rolling KV cache: anchor chunk + sliding window, RoPE positions |
| `aura/audio.py` | Mimi / wav2vec2 audio features |
| `aura/configs.py` | Configurations of the released models |
| `aura/wan/` | VAEs and text encoder vendored from [Wan2.2](https://github.com/Wan-Video/Wan2.2) (Apache 2.0) |

## License

The code builds on [TalkVerse](https://github.com/snap-research/TalkVerse) and
[Wan2.2](https://github.com/Wan-Video/Wan2.2); see [`LICENSE.txt`](LICENSE.txt).

The weights have their own licenses, both non-commercial:
- **AURA-1.3B**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), following the terms of
  its training data (reference images and prompts from the
  [Hallo3 training data](https://huggingface.co/datasets/fudan-generative-ai/hallo3_training_data)).
- **AURA-5B**: [Snap Inc. Non-Commercial License](https://huggingface.co/kyutai/AURA-5B/blob/main/LICENSE),
  as it is derived from [TalkVerse-S2V-5B](https://huggingface.co/snap-research/talkverse-s2v-5b).

The Wan VAEs and T5 text encoder (Apache 2.0), Mimi and wav2vec2 keep their own licenses.

## Responsible use

AURA generates realistic videos of real-looking people speaking arbitrary audio. **Do not** use it to
impersonate people, create non-consensual content or produce disinformation. Only animate portraits for
which you have the subject's consent, and disclose that videos are AI-generated.

## Contact

For questions about the code or the models, contact Nathaniel Cohen (nathaniel.cohen@kyutai.org) or
Amélie Royer (amelie@kyutai.org), or open an issue.

## Citation

```bibtex
@inproceedings{cohen2026aura,
  title     = {{AURA}: {AU}dio-d{R}iven streaming {A}vatar},
  author    = {Cohen, Nathaniel and Dufour, Nicolas and Royer, Am{\'e}lie
               and Newson, Alasdair and P{\'e}rez, Patrick},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026},
}
```

AURA-5B builds on TalkVerse; please also cite:

```bibtex
@article{wang2025talkverse,
  title   = {TalkVerse: Democratizing Minute-Long Audio-Driven Video Generation},
  author  = {Wang, Zhenzhi and Wang, Jian and Ma, Ke and Lin, Dahua and Zhou, Bing},
  journal = {arXiv preprint arXiv:2512.14938},
  year    = {2025}
}
```
