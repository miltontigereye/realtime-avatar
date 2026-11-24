"""Audio features: Mimi semantic latent (AURA 1.3B) or wav2vec2 hidden states (AURA 5B)."""

import librosa
import numpy as np
import torch
import torch.nn.functional as F

FEATURE_RATE = 25  # Hz, rate of the features before resampling to the video frame rate


class MimiFeatures:
    """Dequantized first (semantic) RVQ level of Kyutai's Mimi codec: 512-d, 12.5 Hz."""

    def __init__(self, device):
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        path = hf_hub_download("kyutai/moshiko-pytorch-bf16", loaders.MIMI_NAME)
        self.model = loaders.get_mimi(path, device=device, num_codebooks=1).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, audio_path: str) -> torch.Tensor:
        """-> [num_layers=1, T @ 25 Hz, 512]"""
        wav, _ = librosa.load(audio_path, sr=self.model.sample_rate)
        wav = torch.as_tensor(wav, device=self.device)[None, None]
        feat = self.model.decode_latent(self.model.encode(wav)).transpose(1, 2)  # [1, T, 512]
        n = int(feat.shape[1] / self.model.frame_rate * FEATURE_RATE)
        feat = F.interpolate(feat.transpose(1, 2), size=n, mode="linear", align_corners=True)
        return feat.transpose(1, 2).float()


class Wav2Vec2Features:
    """All 25 hidden states of wav2vec2-large-xlsr-53-english: 1024-d, 50 Hz -> 25 Hz."""

    def __init__(self, device, model_id="jonatasgrosman/wav2vec2-large-xlsr-53-english"):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, audio_path: str) -> torch.Tensor:
        """-> [num_layers=25, T @ 25 Hz, 1024]"""
        wav, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(wav, sampling_rate=sr, return_tensors="pt").input_values
        hidden = self.model(inputs.to(self.device), output_hidden_states=True).hidden_states
        return torch.cat(hidden)[:, ::2].float()


def to_video_rate(feat: torch.Tensor, fps: int, frames_per_chunk: int) -> tuple[torch.Tensor, int]:
    """Sample the 25 Hz features at the video frame rate, zero-padded to a whole number of chunks.

    feat: [num_layers, T, dim] -> ([1, num_layers, dim, num_chunks * frames_per_chunk], num_chunks)
    """
    num_layers, t, dim = feat.shape
    num_chunks = int(t / (frames_per_chunk * FEATURE_RATE / fps)) + 1
    n = num_chunks * frames_per_chunk
    idx = np.round(np.linspace(0, n / fps, n, endpoint=False) * FEATURE_RATE).astype(int)
    out = feat.new_zeros(num_layers, dim, n)
    valid = idx < t
    out[:, :, valid] = feat[:, idx[valid]].transpose(1, 2)
    return out[None], num_chunks
