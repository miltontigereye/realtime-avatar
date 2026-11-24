"""Checks for bounded audio, session ownership, and genuinely incremental MP4 output."""

import asyncio
import json
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

import aiohttp
import numpy as np
import torch
from aiohttp.test_utils import TestClient, TestServer

from web.media import LiveMuxer
from web.server import AudioBuffer, create_app
from aura.configs import CONFIGS
from aura.kv_cache import KVCache
from aura.model import AuraDiT


class AudioBufferTests(unittest.TestCase):
    def test_packet_boundaries_eof_and_final_padding(self):
        audio = AudioBuffer()
        audio.feed(np.arange(7, dtype=np.float32))
        audio.feed(np.arange(7, 13, dtype=np.float32))
        np.testing.assert_array_equal(audio.read(9), np.arange(9, dtype=np.float32))
        audio.finish()
        np.testing.assert_array_equal(audio.read(9), [9, 10, 11, 12, 0, 0, 0, 0, 0])
        self.assertIsNone(audio.read(9))
        self.assertEqual(audio.received, 13)

    def test_cancel_unblocks_waiting_reader(self):
        audio = AudioBuffer()
        result = []
        thread = threading.Thread(target=lambda: result.append(audio.read(24000)))
        thread.start()
        audio.finish(cancel=True)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])

    def test_capacity_is_bounded_by_samples(self):
        audio = AudioBuffer(max_seconds=1)
        audio.feed(np.zeros(24000, dtype=np.float32))
        with self.assertRaises(ValueError):
            audio.feed(np.zeros(1, dtype=np.float32))


class LongStreamTests(unittest.TestCase):
    @torch.no_grad()
    def test_absolute_rope_extends_without_changing_existing_positions(self):
        cfg = replace(CONFIGS["aura-1.3b"], dim=128, ffn_dim=256, num_heads=2,
                      num_layers=1, audio_dim=32, audio_inject_layers=[0], rope_reindex=False)
        model = AuraDiT(cfg).eval()
        original = model.freqs.clone()
        cache = KVCache(num_layers=1, num_heads=2, head_dim=64, tokens_per_frame=4,
                        frames_per_block=4, window=2, segment_frames=20, ref_rope_offset=30, device="cpu")
        cache.block_idx = 256
        audio = model.encode_audio(torch.randn(1, 1, 32, 16))
        result = model(torch.randn(16, 4, 4, 4), torch.tensor([1000]), torch.randn(8, 4096),
                       torch.randn(1, 16, 1, 4, 4), audio, cache)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(model.freqs.shape[0], 1024)
        torch.testing.assert_close(model.freqs[:1024], original, rtol=0, atol=0)


class MuxerTests(unittest.TestCase):
    def test_sends_media_before_input_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.mp4"
            chunks = []
            encoder = LiveMuxer(path, 64, 64, 16, chunks.append)
            try:
                encoder.push(torch.zeros(3, 13, 64, 64), np.zeros(24000, dtype=np.float32), 16)
                deadline = time.monotonic() + 5
                while b"moof" not in b"".join(chunks) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertIn(b"moof", b"".join(chunks), "No media was emitted before EOF")
                encoder.push(torch.zeros(3, 16, 64, 64), np.zeros(24000, dtype=np.float32), 16)
                encoder.close()
                encoder = None
                self.assertGreater(path.stat().st_size, 2000)
                packets = json.loads(subprocess.check_output([
                    "ffprobe", "-v", "error", "-show_packets", "-of", "json", str(path)
                ]))["packets"]
                for stream in (0, 1):
                    times = [float(p["pts_time"]) for p in packets if p["stream_index"] == stream]
                    self.assertTrue(times)
                    self.assertTrue(all(a < b for a, b in zip(times, times[1:])),
                                    "Fragmented MP4 has overlapping or reordered timestamps")
            finally:
                if encoder is not None:
                    encoder.close(cancel=True)


class SessionRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.dict("os.environ", {"AURA_PUBLIC_ORIGIN": "https://avatar.example.test"}):
            self.client = TestClient(TestServer(create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def post_session(self, model="aura-1.3b", image=None):
        form = aiohttp.FormData()
        form.add_field("model", model)
        # Force multipart, matching the browser's FormData even without an image.
        form.add_field("compile", "false", content_type="text/plain")
        if image is not None:
            form.add_field("image", image, filename="portrait.png", content_type="image/png")
        return await self.client.post("/api/sessions", data=form)

    async def test_gpu_reservation_and_cancellation_before_websocket(self):
        response = await self.post_session()
        self.assertEqual(response.status, 200)
        session = await response.json()
        response = await self.post_session("aura-5b")
        self.assertEqual(response.status, 409)
        self.assertEqual((await self.client.post("/api/unload")).status, 409)
        self.assertEqual((await self.client.delete(f"/api/sessions/{session['id']}")).status, 200)
        self.assertEqual((await self.post_session("aura-5b")).status, 200)

    async def test_invalid_portrait_and_model_do_not_reserve_gpu(self):
        self.assertEqual((await self.post_session(image=b"not an image")).status, 400)
        self.assertEqual((await self.post_session(model="unknown")).status, 400)
        response = await self.client.get("/api/config")
        self.assertFalse((await response.json())["busy"])

    async def test_public_https_origin_and_rejection_of_other_origins(self):
        response = await self.post_session()
        session = await response.json()
        for origin in ("https://other.example.test", "https://avatar.example.test.attacker.test"):
            with self.assertRaises(aiohttp.WSServerHandshakeError) as raised:
                await self.client.ws_connect(session["websocket"], headers={"Origin": origin})
            self.assertEqual(raised.exception.status, 403)
        with patch("web.server.ModelManager.run", lambda manager, stream: None):
            async with self.client.ws_connect(
                session["websocket"], headers={"Origin": "https://avatar.example.test"}
            ):
                pass


if __name__ == "__main__":
    unittest.main()
