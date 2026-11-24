"""AURA diffusion transformer: a Wan DiT with audio cross-attention, run causally chunk by chunk
over a rolling KV cache.

Adapted from Wan2.2 (https://github.com/Wan-Video/Wan2.2, Apache-2.0) and TalkVerse.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .configs import ModelConfig
from .kv_cache import KVCache


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full (non-causal) attention on [B, L, heads, head_dim] tensors, computed in bf16."""
    q, k, v = (u.to(torch.bfloat16).transpose(1, 2) for u in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_freqs(freqs: torch.Tensor, frames: list[int], h: int, w: int) -> torch.Tensor:
    """3D RoPE frequencies for `len(frames)` frames of h x w tokens, frame i sitting at
    temporal position `frames[i]`. Returns a complex tensor [1, len(frames) * h * w, 1, d]."""
    d = freqs.shape[1]
    freqs_f, freqs_h, freqs_w = freqs.split([d - 2 * (d // 3), d // 3, d // 3], dim=1)
    n = len(frames)
    f_idx = torch.tensor(frames, device=freqs.device)
    out = torch.cat(
        [
            freqs_f[f_idx].view(n, 1, 1, -1).expand(n, h, w, -1),
            freqs_h[:h].view(1, h, 1, -1).expand(n, h, w, -1),
            freqs_w[:w].view(1, 1, w, -1).expand(n, h, w, -1),
        ],
        dim=-1,
    )
    return out.reshape(1, n * h * w, 1, -1)


@torch.autocast("cuda", enabled=False)
def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate x [B, L, heads, head_dim] by complex freqs [1, L, 1, head_dim / 2] (in float64)."""
    b, s, n, d = x.shape
    x2 = x.to(torch.float64).reshape(b, s, n, d // 2, 2)
    xr, xi = x2[..., 0], x2[..., 1]
    f = torch.view_as_real(freqs)
    fr, fi = f[..., 0], f[..., 1]
    return torch.stack([xr * fr - xi * fi, xr * fi + xi * fr], dim=-1).reshape(b, s, n, d).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def project(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return q, k, v

    def forward(self, x, context):
        """Cross-attention of x [B, L1, C] to context [B, L2, C]."""
        q, k, v = self.project(x, context)
        return self.o(attention(q, k, v).type_as(q).flatten(2))


def modulate(x, shift, scale, split: int):
    """x * (1 + scale) + shift, with modulation row 0 for the noisy tokens x[:, :split] and
    row 1 for the reference-frame tokens x[:, split:]. shift, scale: [B, 2, C]."""
    return torch.cat(
        [
            x[:, :split] * (1 + scale[:, 0:1]) + shift[:, 0:1],
            x[:, split:] * (1 + scale[:, 1:2]) + shift[:, 1:2],
        ],
        dim=1,
    )


def gate(x, g, split: int):
    return torch.cat([x[:, :split] * g[:, 0:1], x[:, split:] * g[:, 1:2]], dim=1)


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = Attention(dim, num_heads, eps)
        self.norm3 = LayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = Attention(dim, num_heads, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(self, x, e, split, context, cache, freqs_q, freqs_k):
        """x: [1, L, C] = noisy tokens x[:, :split] then the reference frame tokens.
        e: [1, 6, 2, C] time modulation (row 0: noisy tokens, row 1: reference frame).
        cache: (k_cache, v_cache, start, end): this layer's KV cache and the current block's slot."""
        with torch.autocast("cuda", dtype=torch.float32):
            e = [u.squeeze(1) for u in (self.modulation.unsqueeze(2) + e).chunk(6, dim=1)]

        # Self-attention over the cached blocks. Only the noisy tokens are cached; the reference
        # frame is appended at every step.
        y = modulate(self.norm1(x).float(), e[0], e[1], split)
        q, k, v = self.self_attn.project(y, y)
        k_cache, v_cache, start, end = cache
        k_cache[:, start:end] = k[:, :split]
        v_cache[:, start:end] = v[:, :split]
        k = torch.cat([k_cache[:, :end], k[:, split:]], dim=1)
        v = torch.cat([v_cache[:, :end], v[:, split:]], dim=1)
        y = attention(rope_apply(q, freqs_q), rope_apply(k, freqs_k), v).type_as(q)
        y = self.self_attn.o(y.flatten(2))
        with torch.autocast("cuda", dtype=torch.float32):
            x = x + gate(y, e[2], split)

        # Cross-attention to the text, then feed-forward.
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(modulate(self.norm2(x).float(), e[3], e[4], split))
        with torch.autocast("cuda", dtype=torch.float32):
            x = x + gate(y, e[5], split)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size, eps: float = 1e-6):
        super().__init__()
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

    def forward(self, x, e):
        with torch.autocast("cuda", dtype=torch.float32):
            shift, scale = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            return self.head(self.norm(x) * (1 + scale) + shift)


class CausalConv1d(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0), mode="replicate"))


class MotionEncoder(nn.Module):
    """Temporal audio encoder from Wan2.2-S2V (`MotionEncoder_tc`): downsamples the audio features
    by 4 in time (to the latent frame rate) and returns, per latent frame, `num_heads + 1` local
    audio tokens and one global token."""

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        self.final_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def _tail(self, x):
        """[b, t, c] after conv1 -> [b, t / 4, hidden_dim]."""
        x = F.silu(self.norm1(x))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(self.norm2(x))
        x = self.conv3(x.transpose(1, 2)).transpose(1, 2)
        return F.silu(self.norm3(x))

    def forward(self, x):
        b = x.shape[0]
        x = x.transpose(1, 2)  # b c t
        local = rearrange(self.conv1_local(x), "b (n c) t -> (b n) t c", n=self.num_heads)
        local = rearrange(self._tail(local), "(b n) t c -> b t n c", b=b)
        local = torch.cat([local, self.padding_tokens.repeat(b, local.shape[1], 1, 1)], dim=-2)
        glob = self.final_linear(self._tail(self.conv1_global(x).transpose(1, 2)))
        return glob[:, :, None], local


class AudioEncoder(nn.Module):
    """Weighted sum over the audio feature layers, then the temporal MotionEncoder."""

    def __init__(self, in_dim: int, num_layers: int, out_dim: int, num_tokens: int = 4):
        super().__init__()
        self.encoder = MotionEncoder(in_dim, out_dim, num_tokens)
        self.weights = nn.Parameter(torch.full((1, num_layers, 1, 1), 0.01))

    def forward(self, features):
        """features: [B, num_layers, dim, T] -> global [B, T/4, 1, C], local [B, T/4, n, C]."""
        with torch.autocast("cuda", dtype=torch.float32):
            weights = F.silu(self.weights)
            x = ((features * weights) / weights.sum(dim=1, keepdim=True)).sum(dim=1)
            return self.encoder(x.permute(0, 2, 1))


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x, temb):
        shift, scale = self.linear(F.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class AudioInjector(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.injector = nn.ModuleList([Attention(dim, num_heads) for _ in range(num_layers)])
        self.injector_adain_layers = nn.ModuleList([AdaLayerNorm(dim) for _ in range(num_layers)])


class AuraDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.patch_embedding = nn.Conv3d(cfg.in_dim, dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(4096, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([Block(dim, cfg.ffn_dim, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.head = Head(dim, cfg.in_dim, cfg.patch_size)
        # Token-type embedding: 0 = noisy latents, 1 = reference frame.
        self.trainable_cond_mask = nn.Embedding(2, dim)
        self.casual_audio_encoder = AudioEncoder(cfg.audio_dim, cfg.num_audio_layers, dim)
        self.audio_injector = AudioInjector(dim, cfg.num_heads, len(cfg.audio_inject_layers))
        self.inject_id = {layer: i for i, layer in enumerate(cfg.audio_inject_layers)}

        d = dim // cfg.num_heads
        with torch.device("cpu"):  # also when the model is built on the meta device
            self.freqs = torch.cat(
                [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
                dim=1,
            )

    def encode_audio(self, audio):
        """audio: [1, num_layers, dim, frames] features of one chunk (video frame rate).
        Returns the per-latent-frame (global, local) audio tokens."""
        # Wan-S2V prepends `motion_frames` copies of the first feature (the motion-frame context of
        # the bidirectional model) and drops the matching latent frames after the encoder.
        motion_frames = 73
        audio = torch.cat([audio[..., :1].repeat(1, 1, 1, motion_frames), audio], dim=-1)
        glob, local = self.casual_audio_encoder(audio)
        lat_motion = (motion_frames + 3) // 4
        return glob[:, lat_motion:], local[:, lat_motion:]

    def inject_audio(self, idx, x, audio_global, audio_local, split):
        i = self.inject_id[idx]
        t = audio_local.shape[1]
        y = rearrange(x[:, :split], "b (t n) c -> (b t) n c", t=t)
        y = self.audio_injector.injector_adain_layers[i](y, rearrange(audio_global, "b t n c -> (b t) n c")[:, 0])
        y = self.audio_injector.injector[i](y, rearrange(audio_local, "b t n c -> (b t) n c"))
        y = rearrange(y, "(b t) n c -> b (t n) c", t=t)
        return torch.cat([x[:, :split] + y, x[:, split:]], dim=1)

    def forward(self, x, t, context, ref_latent, audio, kv_cache: KVCache):
        """Velocity prediction for one chunk.

        x: [C, F, H, W] noisy latents of the chunk; t: [1] timestep; context: [L, 4096] text
        embedding; ref_latent: [1, C, 1, H, W]; audio: (global, local) tokens from `encode_audio`.
        """
        cfg = self.cfg
        required_positions = max(kv_cache.query_frame_indices() + kv_cache.key_frame_indices()) + 1
        if required_positions > self.freqs.shape[0]:
            # 5B uses absolute temporal positions. Extend its table rather than
            # failing once a live session passes the original 1024 latent frames.
            length = max(required_positions, self.freqs.shape[0] * 2)
            d = cfg.dim // cfg.num_heads
            self.freqs = torch.cat(
                [rope_params(length, d - 4 * (d // 6)), rope_params(length, 2 * (d // 6)),
                 rope_params(length, 2 * (d // 6))], dim=1,
            )
        self.freqs = self.freqs.to(x.device)
        x = self.patch_embedding(x[None])
        f, h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        ref = self.patch_embedding(ref_latent).flatten(2).transpose(1, 2)
        split = x.shape[1]
        x = torch.cat([x, ref], dim=1)
        token_type = torch.zeros(1, x.shape[1], dtype=torch.long, device=x.device)
        token_type[:, split:] = 1
        x = x + self.trainable_cond_mask(token_type).to(x.dtype)

        # Time modulation: timestep t for the noisy tokens, 0 for the reference frame.
        with torch.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(256, torch.cat([t, t.new_zeros(1)])).float())
            e0 = self.time_projection(e).unflatten(1, (6, cfg.dim))
        e0 = torch.stack([e0[:1], e0[1:]], dim=2)  # [1, 6, 2, C]
        e = e[:1]

        context = self.text_embedding(
            torch.cat([context, context.new_zeros(512 - context.size(0), context.size(1))])[None]
        )

        # RoPE: the chunk's frames + the reference frame for queries, the cached frames + the
        # reference frame for keys.
        freqs_q = rope_freqs(self.freqs, kv_cache.query_frame_indices(), h, w)
        freqs_k = rope_freqs(self.freqs, kv_cache.key_frame_indices(), h, w)

        start, end = kv_cache.slot()
        for idx, block in enumerate(self.blocks):
            x = block(x, e0, split, context, (kv_cache.k[idx], kv_cache.v[idx], start, end), freqs_q, freqs_k)
            if idx in self.inject_id:
                x = self.inject_audio(idx, x, *audio, split)

        x = self.head(x[:, :split], e)
        return self.unpatchify(x[0], (f, h, w)).float()

    def unpatchify(self, x, grid):
        c = self.cfg.in_dim
        p = self.cfg.patch_size
        x = x.view(*grid, *p, c)
        x = torch.einsum("fhwpqrc->cfphqwr", x)
        return x.reshape(c, *[g * s for g, s in zip(grid, p)])
