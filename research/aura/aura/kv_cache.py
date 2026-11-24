"""Rolling KV cache for chunk-by-chunk causal generation."""

import torch


class KVCache:
    """Per-layer keys / values of the noisy tokens of past chunks.

    A chunk ("block") is `frames_per_block` latent frames. The cache keeps the first block
    (identity anchor, never evicted) plus the latest `window` blocks, the current one included:
    [block 0 | block r-window+1 | ... | block r]. The reference frame is not cached: its keys and
    values are appended at every step.

    RoPE positions follow the real block index, except that past `rope_max_block` the positions
    are frozen as if the current block were `rope_max_block` (bounded RoPE re-indexing), so the
    geometry stays the one seen in training on arbitrarily long videos.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        tokens_per_frame: int,
        frames_per_block: int,
        window: int,
        segment_frames: int,
        ref_rope_offset: int,
        rope_max_block: int | None = None,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        self.frames_per_block = frames_per_block
        self.block_tokens = frames_per_block * tokens_per_frame
        self.window = window
        self.num_slots = window + 1
        self.segment_frames = segment_frames
        self.ref_rope_offset = ref_rope_offset
        self.rope_max_block = rope_max_block
        shape = (1, self.num_slots * self.block_tokens, num_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.block_idx = 0

    def slot(self) -> tuple[int, int]:
        """Token range of the current block in the cache; the valid cache is [:end]. Every
        denoising step of a block overwrites the same slot."""
        slot = min(self.block_idx, self.num_slots - 1)
        return slot * self.block_tokens, (slot + 1) * self.block_tokens

    @torch.no_grad()
    def advance(self):
        """Move to the next block; once the window is full, evict its oldest block."""
        self.block_idx += 1
        if self.block_idx >= self.num_slots:
            b = self.block_tokens
            for t in self.k + self.v:
                t[:, b:-b] = t[:, 2 * b :].clone()

    def _rope_block(self):
        if self.rope_max_block is None:
            return self.block_idx
        return min(self.block_idx, self.rope_max_block)

    def _block_frames(self, block):
        return list(range(block * self.frames_per_block, (block + 1) * self.frames_per_block))

    def _with_ref(self, frames):
        # The reference frame sits at `ref_rope_offset` within the current segment.
        base = (frames[-1] // self.segment_frames) * self.segment_frames
        return frames + [base + self.ref_rope_offset]

    def query_frame_indices(self):
        """RoPE temporal positions of the current block's frames, then of the reference frame."""
        return self._with_ref(self._block_frames(self._rope_block()))

    def key_frame_indices(self):
        """RoPE temporal positions of the cached frames, then of the reference frame."""
        r = self._rope_block()
        blocks = [0] + list(range(max(1, r - self.window + 1), r + 1))
        return self._with_ref([f for b in blocks for f in self._block_frames(b)])
