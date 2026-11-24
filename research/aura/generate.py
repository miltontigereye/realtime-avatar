"""Generate a talking-head video from a portrait image and a speech audio file.

uv run generate.py --image portrait.jpg --audio speech.wav --output out.mp4
uv run generate.py --model aura-5b --image portrait.jpg --audio speech.wav --output out.mp4
"""

import argparse
import logging
import subprocess
import tempfile

import imageio
import imageio_ffmpeg
import torch

from aura.configs import CONFIGS
from aura.pipeline import DEFAULT_PROMPT, Aura


def save_video(path: str, chunks, fps: int, audio: str):
    """Write the chunks [3, T, H, W] in [-1, 1] as H.264 and mux the audio (cut to the video)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        with imageio.get_writer(tmp.name, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for chunk in chunks:
                frames = ((chunk.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()
                for frame in frames:
                    writer.append_data(frame)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp.name, "-i", audio, "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", path]
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", required=True, help="Reference portrait.")
    parser.add_argument("--audio", required=True, help="Speech audio (wav, mp3, ...).")
    parser.add_argument("--output", default="out.mp4")
    parser.add_argument("--model", default="aura-1.3b", choices=list(CONFIGS))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text description of the video.")
    parser.add_argument(
        "--num_chunks", type=int, default=None, help="Video length in chunks of 16 frames (default: whole audio)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the transformer blocks (slow first chunks)."
    )
    parser.add_argument(
        "--offload-text-encoder", action="store_true", help="Run the T5 text encoder on CPU to save GPU memory."
    )
    parser.add_argument(
        "--bf16-transformer", action="store_true", help="Store transformer weights in bf16 to save GPU memory."
    )
    parser.add_argument(
        "--vae-decode-group", type=int, choices=range(1, 5), default=None,
        help="Latent frames per VAE decode call; use 1 to reduce peak memory."
    )
    parser.add_argument(
        "--offload-transformer-for-decode", action="store_true",
        help="Move the transformer to CPU during VAE decoding to reduce peak GPU memory."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    aura = Aura(
        args.model, compile=args.compile, offload_text_encoder=args.offload_text_encoder,
        bf16_transformer=args.bf16_transformer,
        vae_decode_group=args.vae_decode_group,
        offload_transformer_for_decode=args.offload_transformer_for_decode,
    )
    chunks = aura.generate(args.image, args.audio, args.prompt, num_chunks=args.num_chunks, seed=args.seed)
    save_video(args.output, chunks, aura.fps, args.audio)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
