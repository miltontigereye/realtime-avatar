"""AURA streaming inference: reference image + speech -> talking-head video, chunk by chunk."""

import itertools
import logging
import time
from dataclasses import replace

import torch
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors.torch import load_file

from .audio import MimiFeatures, Wav2Vec2Features, to_video_rate
from .configs import CONFIGS, ModelConfig
from .kv_cache import KVCache
from .model import AuraDiT
from .wan.t5 import T5EncoderModel
from .wan.vae2_1 import Wan2_1_VAE
from .wan.vae2_2 import Wan2_2_VAE, unpatchify

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "A person is talking to the camera, with natural facial expressions and head movements."


class FlowSchedule:
    """Noise levels of the flow-matching schedule the students were distilled on (50 steps,
    shift 5); a timestep maps to the sigma of the nearest scheduler step."""

    def __init__(self, steps=50, shift=5.0, sigma_min=0.003 / 1.002):
        sigmas = torch.linspace(1.0, sigma_min, steps)
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * 1000

    def sigma(self, t: float) -> float:
        return float(self.sigmas[(self.timesteps - t).abs().argmin()].double())


class Aura:
    def __init__(
        self, name: str = "aura-1.3b", device: str = "cuda", compile: bool = False,
        offload_text_encoder: bool = False, bf16_transformer: bool = False,
        vae_decode_group: int | None = None,
        offload_transformer_for_decode: bool = False,
    ):
        self.cfg: ModelConfig = CONFIGS[name]
        if vae_decode_group is not None:
            if not 1 <= vae_decode_group <= self.cfg.frames_per_block:
                raise ValueError("vae_decode_group must be between 1 and frames_per_block")
            self.cfg = replace(self.cfg, vae_decode_group=vae_decode_group)
        self.device = torch.device(device)
        self.text_device = torch.device("cpu") if offload_text_encoder else self.device
        self.offload_transformer_for_decode = offload_transformer_for_decode
        cfg = self.cfg
        wan_dir = snapshot_download(
            cfg.wan_repo, allow_patterns=[cfg.vae, "models_t5_umt5-xxl-enc-bf16.pth", "google/*"]
        )

        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.text_device,
            checkpoint_path=f"{wan_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path=f"{wan_dir}/google/umt5-xxl",
        )
        vae_cls = Wan2_1_VAE if cfg.vae == "Wan2.1_VAE.pth" else Wan2_2_VAE
        self.vae = vae_cls(vae_pth=f"{wan_dir}/{cfg.vae}", dtype=cfg.vae_dtype, device=self.device)

        with torch.device("meta"):
            self.model = AuraDiT(cfg)
        state = load_file(hf_hub_download(cfg.repo, cfg.checkpoint))
        state = {k: v for k, v in state.items() if not k.startswith("frame_packer.")}  # unused when streaming
        state["trainable_cond_mask.weight"] = state["trainable_cond_mask.weight"][:2]
        if bf16_transformer:
            # Preserve original float32 weights for modules outside bf16 autocast.
            fp32_modules = ("time_embedding.", "time_projection.", "casual_audio_encoder.")
            state = {k: v if k.startswith(fp32_modules) else v.to(torch.bfloat16) for k, v in state.items()}
        self.model.load_state_dict(state, assign=True)
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)
        del state
        if compile:
            for i, block in enumerate(self.model.blocks):
                self.model.blocks[i] = torch.compile(block, dynamic=True)

        audio_cls = MimiFeatures if cfg.audio_backend == "mimi" else Wav2Vec2Features
        self.audio_features = audio_cls(self.device)
        self.schedule = FlowSchedule()

    @property
    def fps(self) -> int:
        return self.cfg.fps

    def load_image(self, path: str) -> torch.Tensor:
        """Resize and center-crop to the closest training resolution -> [1, 3, 1, H, W] in [-1, 1]."""
        image = Image.open(path).convert("RGB")
        ratio = image.height / image.width
        h, w = self.cfg.buckets[min(self.cfg.buckets, key=lambda k: abs(k - ratio))]
        image = TF.center_crop(TF.resize(image, min(h, w)), [h, w])
        return (TF.to_tensor(image) * 2 - 1)[None, :, None].to(self.device)

    def decode(self, latents: torch.Tensor, first: bool) -> torch.Tensor:
        """Decode one chunk [C, F, h, w] with the causal VAE, continuing from the previous chunk."""
        vae = self.vae.model
        mean, inv_std = self.vae.scale
        is_v22 = isinstance(self.vae, Wan2_2_VAE)
        with torch.autocast("cuda", dtype=self.cfg.vae_dtype):
            z = vae.conv2(latents[None] / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1))
            t, group = z.shape[2], self.cfg.vae_decode_group
            # The first latent frame of the video decodes alone, into a single frame.
            edges = [0, *range(1, t, group), t] if first else [*range(0, t, group), t]
            out = []
            for start, end in itertools.pairwise(edges):
                vae._conv_idx = [0]
                kwargs = {"first_chunk": True} if is_v22 and first and start == 0 else {}
                out.append(vae.decoder(z[:, :, start:end], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kwargs))
            video = torch.cat(out, dim=2).float().clamp_(-1, 1)
        return unpatchify(video, patch_size=2)[0] if is_v22 else video[0]

    @torch.no_grad()
    def generate(
        self, image: str, audio: str, prompt: str = DEFAULT_PROMPT, num_chunks: int | None = None, seed: int = 42
    ):
        """Yield the video chunk by chunk, as [3, frames, H, W] tensors in [-1, 1].

        Each chunk is `frames_per_block` latent frames (16 video frames, 13 for the first chunk).
        The video covers the whole audio, rounded up to a whole chunk, or `num_chunks` chunks.
        """
        cfg = self.cfg
        frames_per_chunk = cfg.vae_stride[0] * cfg.frames_per_block
        features, n = to_video_rate(self.audio_features(audio), cfg.fps, frames_per_chunk)
        features = features.to(self.device, torch.bfloat16)
        num_chunks = n if num_chunks is None else min(num_chunks, n)

        session = self.start_stream(image, prompt, seed)
        try:
            for r in range(num_chunks):
                t0 = time.perf_counter()
                video = session.render(features[..., r * frames_per_chunk : (r + 1) * frames_per_chunk])
                logger.info(f"chunk {r + 1}/{num_chunks}: {video.shape[1]} frames in {time.perf_counter() - t0:.2f}s")
                yield video
        finally:
            session.close()

    def start_stream(self, image: str, prompt: str = DEFAULT_PROMPT, seed: int = 42):
        """Start one persistent video session; pass one block of audio features to render()."""
        return AuraVideoSession(self, image, prompt, seed)


class AuraVideoSession:
    """Reference conditioning, rolling attention cache, and causal VAE state for one stream.

    An Aura instance supports one active session at a time. Audio feature extraction is
    deliberately separate, so live PCM and prerecorded input share the same renderer.
    """

    @torch.no_grad()
    def __init__(self, aura: Aura, image: str, prompt: str, seed: int):
        self.aura = aura
        self.seed = seed
        self.index = 0
        self.closed = False
        cfg = aura.cfg
        ref = aura.load_image(image)
        self.ref_latent = torch.stack(aura.vae.encode(list(ref.to(cfg.vae_dtype))))
        self.context = aura.text_encoder([prompt], aura.text_device)[0].to(aura.device)

        _, _, _, height, width = ref.shape
        self.lat_h, self.lat_w = height // cfg.vae_stride[1], width // cfg.vae_stride[2]
        tokens_per_frame = (self.lat_h // cfg.patch_size[1]) * (self.lat_w // cfg.patch_size[2])
        self.kv_cache = KVCache(
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            head_dim=cfg.dim // cfg.num_heads,
            tokens_per_frame=tokens_per_frame,
            frames_per_block=cfg.frames_per_block,
            window=cfg.kv_window,
            segment_frames=cfg.segment_frames,
            ref_rope_offset=cfg.ref_rope_offset,
            # Last block that fits in a training clip.
            rope_max_block=cfg.segment_frames // cfg.frames_per_block - 1 if cfg.rope_reindex else None,
            device=aura.device,
        )
        self.prepared = aura.model.prepare_conditioning(self.context, self.ref_latent) if getattr(aura, "prepare_static", False) else None
        aura.vae.model.clear_cache()


    @torch.no_grad()
    def render(self, features: torch.Tensor) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("Video session is closed")
        aura, cfg = self.aura, self.aura.cfg
        expected = (1, cfg.num_audio_layers, cfg.audio_dim, cfg.frames_per_block * cfg.vae_stride[0])
        if tuple(features.shape) != expected:
            raise ValueError(f"Expected audio feature shape {expected}, received {tuple(features.shape)}")
        features = features.to(aura.device, torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if aura.offload_transformer_for_decode:
                aura.model.to(aura.device)
            generator = torch.Generator(aura.device).manual_seed(self.seed + self.index)
            shape = (cfg.in_dim, cfg.frames_per_block, self.lat_h, self.lat_w)
            x = torch.randn(shape, dtype=torch.bfloat16, device=aura.device, generator=generator)
            audio_tokens = aura.model.encode_audio(features)
            steps = cfg.denoising_steps
            for i, t in enumerate(steps):
                timestep = torch.full((1,), t, dtype=torch.long, device=aura.device)
                velocity = aura.model(x, timestep, self.context, self.ref_latent, audio_tokens, self.kv_cache, prepared=None if self.prepared is None else self.prepared[i])
                x0 = (x.double() - aura.schedule.sigma(t) * velocity.double()).float()
                if i + 1 < len(steps):
                    sigma = aura.schedule.sigma(steps[i + 1])
                    noise = torch.randn(x0.shape, dtype=x0.dtype, device=aura.device, generator=generator)
                    x = (1 - sigma) * x0 + sigma * noise
            self.kv_cache.advance()
            if aura.offload_transformer_for_decode:
                aura.model.cpu()
                torch.cuda.empty_cache()
            video = aura.decode(x0, first=self.index == 0)
            torch.cuda.synchronize()
            self.index += 1
            return video if getattr(aura, "gpu_pixels", False) else video.cpu()

    def close(self):
        if not self.closed:
            self.aura.vae.model.clear_cache()
            self.kv_cache = None
            self.ref_latent = self.context = self.prepared = None
            self.closed = True
