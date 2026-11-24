"""Bounded live-PCM conditioning for AURA; all input is mono float32 at 24 kHz."""

import librosa
import numpy as np
import torch

SAMPLE_RATE = 24000


class LiveAudioFeatures:
    def __init__(self, aura):
        self.backend = aura.audio_features
        self.kind = aura.cfg.audio_backend
        self.fps = aura.fps
        self.device = aura.device
        self.pending = np.empty(0, dtype=np.float32)
        self.history = np.empty(0, dtype=np.float32)
        self.semantic = None
        self.semantic_count = 0
        self.frame_offset = 0
        self.context = None
        if self.kind == "mimi":
            self.context = self.backend.model.streaming(batch_size=1)
            self.context.__enter__()

    @torch.no_grad()
    def extract(self, samples: np.ndarray, frames: int) -> torch.Tensor:
        if self.kind == "mimi":
            features = self._mimi(samples, frames)
        else:
            features = self._wav2vec(samples, frames)
        self.frame_offset += frames
        return features.to(self.device, torch.bfloat16)

    def _mimi(self, samples, frames):
        model = self.backend.model
        self.pending = np.concatenate((self.pending, samples))
        hop = model.frame_size
        outputs = []
        while len(self.pending) >= hop:
            wav = torch.from_numpy(self.pending[:hop].copy()).to(self.device)[None, None]
            self.pending = self.pending[hop:]
            outputs.append(model.decode_latent(model.encode(wav)).transpose(1, 2).float())
        if outputs:
            new = torch.cat(outputs, dim=1)
            self.semantic_count += new.shape[1]
            self.semantic = new if self.semantic is None else torch.cat((self.semantic, new), dim=1)
            self.semantic = self.semantic[:, -64:]
        if self.semantic is None:
            raise ValueError("A full Mimi frame is required before rendering")
        # Absolute time positions avoid drift between alternating 12/13 codec frames
        # in one-second video blocks. Hold the newest feature where future audio is absent.
        times = (torch.arange(frames, device=self.device) + self.frame_offset) / self.fps
        positions = torch.round(times * 25) * (model.frame_rate / 25)
        offset = self.semantic_count - self.semantic.shape[1]
        positions = (positions - offset).clamp(0, self.semantic.shape[1] - 1)
        low = positions.floor().long()
        high = (low + 1).clamp(max=self.semantic.shape[1] - 1)
        weight = (positions - low)[None, :, None]
        features = self.semantic[:, low] * (1 - weight) + self.semantic[:, high] * weight
        return features.permute(0, 2, 1)[None]

    def _wav2vec(self, samples, frames):
        # Wav2vec2 is windowed, not a stateful causal encoder. Retain 1.28 seconds
        # of left context; never require audio beyond the current video block.
        left_samples = len(self.history)
        window = np.concatenate((self.history, samples))
        self.history = window[-int(1.28 * SAMPLE_RATE):].copy()
        wav = librosa.resample(window, orig_sr=SAMPLE_RATE, target_sr=16000)
        values = self.backend.processor(wav, sampling_rate=16000, return_tensors="pt").input_values
        hidden = self.backend.model(values.to(self.device), output_hidden_states=True).hidden_states
        layers = torch.cat(hidden)
        indices = torch.round(
            (left_samples / SAMPLE_RATE + torch.arange(frames, device=self.device) / self.fps) * 50
        ).long().clamp(max=layers.shape[1] - 1)
        return layers[:, indices].permute(0, 2, 1)[None].float()

    def close(self):
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None
        self.semantic = None
