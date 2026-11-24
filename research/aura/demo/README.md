# Local AURA demos

For microphone input and live browser playback with either model, run
`./run_web_demo.sh` and open http://localhost:7860. See
[the web demo guide](../web/README.md) for streaming behavior and controls.

Run from `/home/knoledge_base/aura`:

```bash
./run_local_demo.sh              # both models, sequentially
./run_local_demo.sh aura-1.3b    # only 1.3B
./run_local_demo.sh aura-5b      # only 5B
./run_local_demo.sh both /absolute/path/portrait.jpg /absolute/path/speech.wav
```

Videos are written to `demo/outputs/aura-1.3b.mp4` and
`demo/outputs/aura-5b.mp4`. Logs are in `demo/logs/`.
These are AI-generated talking-head samples. The default inputs are copied
from `../LiveAvatar/examples/boy.jpg` and the first four seconds of `boy.wav`.

The launcher runs the T5 text encoder on CPU. For 5B, it also stores the
transformer weights in BF16 while retaining float32 time and audio modules,
decodes one latent frame at a time, and moves the transformer to CPU during VAE
decoding to reduce peak VRAM use. The upstream model
resolutions and four denoising steps are preserved.
The 5B launcher also enables expandable CUDA allocations to reduce fragmentation.

The repository's demo is a command-line MP4 generator. For full control:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python generate.py --model aura-5b \
  --image demo/portrait.jpg --audio demo/speech.wav \
  --offload-text-encoder --bf16-transformer --vae-decode-group 1 \
  --offload-transformer-for-decode \
  --output demo/outputs/custom.mp4
```

Add `--compile` for transformer compilation, with extra startup time.
Model files are cached under the existing `HF_HOME` location,
`/home/cache/huggingface`. Download missing weights with:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python demo/download_models.py
```

## Verified results on the RTX 5090

Both four-second MP4s contain H.264 video and AAC audio, and sampled frames
were visually checked. Runs used the launcher defaults without `--compile`.

| Model | Output | Playback rate | Generation time, excluding loading |
| --- | --- | --- | --- |
| AURA-1.3B | 768 × 512 | 16 fps | 11.59 seconds |
| AURA-5B | 1280 × 704 | 25 fps | 52.00 seconds |

Generation times sum the chunk timings before audio/video muxing. They are
not playback speeds. The 5B run peaked at 32,044 MiB of total GPU memory use,
including the desktop, so it uses nearly all available VRAM on this GPU.
Machine-readable results are in `demo/outputs/validation.json`.
