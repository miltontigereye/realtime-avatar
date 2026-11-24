# Tiger Eye Avatar

Run from the repository checkout:

```bash
./run_web_demo.sh
```

Open **http://localhost:7860** in Chrome or Edge. Select a target portrait and
either **Aura 1.3B** or **Aura 5B**, then choose **Start microphone** and grant
microphone access. Or select **Upload audio**, choose a file and click **Stream audio file**.
The generated avatar plays with synchronized sound. Capture starts after the model and portrait are ready.
**Try sample audio** uses the included four-second speech sample without
requesting microphone permission.

**Finish stream** stops capture, renders the queued audio, and enables the video
download. **Cancel** discards pending audio and stops after the current GPU
operation. **Release GPU memory** unloads the cached model when no stream is
active. Switching models automatically unloads the previous model; both sets of
weights remain available on disk.

Microphone capture requires localhost or HTTPS. For a remote GPU machine, keep
the default loopback binding and forward the port:

```bash
ssh -L 7860:127.0.0.1:7860 user@gpu-machine
```

Then open `http://localhost:7860` on the computer with your microphone. Choose a
different port with `./run_web_demo.sh --port 7861` if needed.

For RunPod's HTTPS proxy, expose HTTP port 8000 and explicitly allow the public
origin so the browser can open its live WebSocket through the TLS proxy:

```bash
export AURA_PUBLIC_ORIGIN=https://YOUR_POD_ID-8000.proxy.runpod.net
./run_web_demo.sh --host 0.0.0.0 --port 8000
```

## Streaming behavior

This is live, incremental generation, but the tested RTX 5090 does not generate
at playback speed. Video will wait for new chunks. The page displays actual
generation speed separately from the video's playback frame rate.

Microphone capture pauses automatically when about four seconds of audio are
pending, and resumes when the queue falls to about one second. **Speech during
the displayed capture pause is not recorded.** Speak when the page says
**Listening**. This prevents an indefinitely growing audio delay. The server
also enforces a six-second input-buffer limit. Sample-audio playback pauses its
input position instead, so it retains the entire sample.

- **1.3B:** stateful Mimi encoding in 80 ms audio frames, with persistent encoder
  state and absolute feature timing. Video blocks represent one second.
- **5B:** wav2vec2 processes each new 0.64-second block with up to 1.28 seconds
  of left context. This is a bounded-window adaptation of a bidirectional audio
  encoder; its conditioning can differ from whole-file inference. It requires
  no future block, but matching whole-file quality is not guaranteed.
- Both models preserve the reference conditioning, rolling transformer KV
  cache, and causal VAE cache for the full session. Do not repeatedly call the
  file-based `generate()` method for individual audio fragments.
- The causal VAE produces 13 frames in the first block. The player stream
  repeats its first frame three extra times to keep the audio/video clocks
  aligned. The final partial audio block is zero-padded to a full block.
- A continuous FFmpeg encoder emits H.264/AAC fragmented MP4 over a WebSocket.
  Media Source Extensions plays incoming fragments before generation finishes.
  The completed fragmented MP4 can also be downloaded and played normally.

One GPU worker serves one active stream at a time. On GPUs below 64 GB, 5B
uses BF16 transformer weights, CPU text encoding, one-latent-frame VAE decoding,
and transformer offloading during decoding. The launcher enables expandable CUDA
allocations. It uses nearly all the 5090's memory during decoding. The 1.3B
compilation checkbox is optional and adds a potentially long first-use warm-up;
the default browser tests do not enable it.

Generated sessions are stored in `demo/web_sessions/<session-id>/`, including
the target portrait, MP4, and FFmpeg error log. This directory is ignored by
Git. Download links are available while the server retains the session; the
files remain on disk after restarting. No portraits or microphone audio are
sent to an external service. Missing model weights download from Hugging Face.

## Code and validation

- `web/server.py`: local HTTP/WebSocket server, GPU ownership, and bounded PCM queue.
- `web/media.py`: continuous video/audio encoding and stream delivery.
- `web/static/`: browser interface, microphone AudioWorklet, and streamed playback.
- `aura/streaming_audio.py`: live Mimi and windowed wav2vec2 conditioning.
- `Aura.start_stream()` / `AuraVideoSession.render()`: persistent video state,
  also used by the original file-generation API.

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
```

Tests cover queue limits, EOF padding, cancellation, session ownership, image
validation, MP4 emission before EOF, monotonic audio/video timestamps, and
extension of absolute RoPE positions beyond the initial 1024-frame table.
Browser checks use actual GPU inference, uploaded portraits, microphone capture
with a simulated audio device, sample audio, automatic capture pause/resume, and
downloads. A physical microphone has not been exercised automatically.
Long-duration visual quality has not been evaluated.

Detailed browser results and screenshots are saved locally under
`demo/outputs/web-*-qa.json` and `demo/outputs/web-*.png`.

### Audio file input

Choose **Upload audio**, select a WAV, MP3 or another browser-supported audio file (up to 50 MB), then click **Stream audio file**. Both Aura 1.3B and Aura 5B support this input. The browser resamples to 24 kHz and mixes stereo to mono. Input pauses under backpressure without skipping samples and finishes automatically at the end of the file. The generated player and downloadable MP4 contain both the avatar and synchronized audio; original input audio is not played separately. Generation still depends on GPU speed and may buffer between chunks.

### Microphone test

Click **Test microphone**, grant browser access and speak for up to five seconds. Watch the input level, then use the recording player to listen. **Stop microphone test** ends it early. Testing stays in your browser and does not use the GPU. The microphone is released when the test ends.
