"""Configurations of the released AURA models."""

from dataclasses import dataclass, field

import torch

# Training resolutions, as {height / width: (height, width)}. The reference image is resized and
# center-cropped to the bucket closest to its aspect ratio.
BUCKETS_480 = {0.67: (512, 768), 1.00: (576, 576)}
BUCKETS_720 = {
    0.22: (448, 2048), 0.29: (512, 1792), 0.36: (576, 1600), 0.45: (640, 1408),
    0.55: (704, 1280), 0.63: (768, 1216), 0.76: (832, 1088), 0.88: (896, 1024),
    1.00: (960, 960), 1.14: (1024, 896), 1.31: (1088, 832), 1.50: (1152, 768),
    1.58: (1216, 768), 1.82: (1280, 704), 1.91: (1344, 704), 2.20: (1408, 640),
    2.30: (1472, 640), 2.67: (1536, 576), 2.89: (1664, 576), 3.62: (1856, 512),
    3.75: (1920, 512),
}  # fmt: skip


@dataclass
class ModelConfig:
    repo: str  # Hugging Face repo of the AURA checkpoint
    checkpoint: str  # merged DiT checkpoint, in `repo`
    wan_repo: str  # Wan base model: VAE + umT5 text encoder
    vae: str
    vae_stride: tuple[int, int, int]
    vae_dtype: torch.dtype
    vae_decode_group: int  # latent frames decoded per VAE call
    audio_backend: str  # "mimi" or "wav2vec2"
    audio_dim: int
    num_audio_layers: int
    fps: int
    buckets: dict
    dim: int
    ffn_dim: int
    num_heads: int
    in_dim: int  # latent channels
    segment_frames: int  # latent frames per training clip (RoPE layout)
    ref_rope_offset: int  # RoPE temporal position of the reference frame in a segment
    rope_reindex: bool  # bounded RoPE re-indexing for long videos
    num_layers: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    audio_inject_layers: list[int] = field(default_factory=lambda: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29])
    frames_per_block: int = 4  # latent frames per streamed chunk
    kv_window: int = 2  # cached blocks besides the anchor block, the current one included
    denoising_steps: tuple[int, ...] = (1000, 745, 498, 255)


CONFIGS = {
    "aura-1.3b": ModelConfig(
        repo="kyutai/AURA-1.3B",
        checkpoint="stage2_dmd/step-6500_ema_merged.safetensors",
        wan_repo="Wan-AI/Wan2.1-T2V-1.3B",
        vae="Wan2.1_VAE.pth",
        vae_stride=(4, 8, 8),
        vae_dtype=torch.bfloat16,
        vae_decode_group=4,
        audio_backend="mimi",
        audio_dim=512,
        num_audio_layers=1,
        fps=16,
        buckets=BUCKETS_480,
        dim=1536,
        ffn_dim=8960,
        num_heads=12,
        in_dim=16,
        segment_frames=20,
        ref_rope_offset=30,
        rope_reindex=True,
    ),
    "aura-5b": ModelConfig(
        repo="kyutai/AURA-5B",
        checkpoint="stage2_dmd/step-5000_merged.safetensors",
        wan_repo="Wan-AI/Wan2.2-TI2V-5B",
        vae="Wan2.2_VAE.pth",
        vae_stride=(4, 16, 16),
        vae_dtype=torch.float32,
        vae_decode_group=2,
        audio_backend="wav2vec2",
        audio_dim=1024,
        num_audio_layers=25,
        fps=25,
        buckets=BUCKETS_720,
        dim=3072,
        ffn_dim=14336,
        num_heads=24,
        in_dim=48,
        segment_frames=30,
        ref_rope_offset=40,
        rope_reindex=False,
    ),
}
