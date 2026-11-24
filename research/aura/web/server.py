"""Local AURA microphone demo. Run with ./run_web_demo.sh."""

import argparse
import asyncio
import concurrent.futures
import gc
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch
from aiohttp import WSMsgType, web
from PIL import Image, UnidentifiedImageError

from aura.pipeline import Aura, DEFAULT_PROMPT
from aura.streaming_audio import LiveAudioFeatures, SAMPLE_RATE
from web.media import LiveMuxer

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web/static"
RUNS = ROOT / "demo/web_sessions"
LOG = logging.getLogger("aura.web")


class AudioBuffer:
    """Single producer/single consumer PCM queue, bounded in samples, not packet count."""

    def __init__(self, max_seconds=6):
        self.condition = threading.Condition()
        self.parts = deque()
        self.count = self.received = 0
        self.maximum = int(max_seconds * SAMPLE_RATE)
        self.finished = self.cancelled = False

    def feed(self, samples):
        with self.condition:
            if self.finished or self.cancelled:
                raise ValueError("Audio input is already closed")
            if self.count + len(samples) > self.maximum:
                raise ValueError("Audio queue is full; pause capture while the model catches up")
            self.parts.append(samples)
            self.count += len(samples)
            self.received += len(samples)
            self.condition.notify()

    def read(self, size):
        with self.condition:
            self.condition.wait_for(lambda: self.count >= size or self.finished or self.cancelled)
            if self.cancelled or (self.finished and self.count == 0):
                return None
            result = np.zeros(size, dtype=np.float32)
            position = 0
            while self.parts and position < size:
                item = self.parts.popleft()
                take = min(size - position, len(item))
                result[position:position + take] = item[:take]
                position += take
                self.count -= take
                if take < len(item):
                    self.parts.appendleft(item[take:])
            return result

    def finish(self, cancel=False):
        with self.condition:
            self.finished = True
            self.cancelled |= cancel
            if cancel:
                self.parts.clear()
                self.count = 0
            self.condition.notify_all()


class Stream:
    def __init__(self, model, compile_model, image, directory):
        self.id = directory.name
        self.model = model
        self.compile = compile_model
        self.image = image
        self.directory = directory
        self.created = time.monotonic()
        self.connected = self.complete = False
        self.audio = AudioBuffer()
        self.generated_seconds = 0.0
        self.loop = None
        self.out = None

    def emit(self, item):
        if self.audio.cancelled:
            return
        future = asyncio.run_coroutine_threadsafe(self.out.put(item), self.loop)
        while True:
            try:
                future.result(timeout=0.5)
                return
            except concurrent.futures.TimeoutError:
                if self.audio.cancelled:
                    future.cancel()
                    return


class ModelManager:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-gpu")
        self.model = None
        self.key = None
        self.active = None
        self.sessions = {}

    def unload(self):
        self.model = None
        self.key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, session):
        key = (session.model, session.compile)
        if self.key != key:
            self.unload()
            session.emit({"type": "status", "state": "loading", "message": f"Loading {session.model}…"})
            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required for this demo")
            small_gpu = torch.cuda.get_device_properties(0).total_memory < 64 * 1024**3
            large = session.model == "aura-5b"
            self.model = Aura(
                session.model, compile=session.compile, offload_text_encoder=True,
                bf16_transformer=True, vae_decode_group=1 if large and small_gpu else None,
                offload_transformer_for_decode=large and small_gpu,
            )
            self.key = key
        return self.model

    @torch.no_grad()
    def run(self, session):
        video_session = audio_features = muxer = None
        try:
            if session.audio.cancelled:
                return
            aura = self.load(session)
            if session.audio.cancelled:
                return
            session.emit({"type": "status", "state": "preparing", "message": "Preparing your portrait…"})
            video_session = aura.start_stream(str(session.image), DEFAULT_PROMPT)
            audio_features = LiveAudioFeatures(aura)
            frames = aura.cfg.frames_per_block * aura.cfg.vae_stride[0]
            audio_samples = round(frames / aura.fps * SAMPLE_RATE)
            height = video_session.lat_h * aura.cfg.vae_stride[1]
            width = video_session.lat_w * aura.cfg.vae_stride[2]
            session.emit({
                "type": "ready", "fps": aura.fps, "width": width, "height": height,
                "chunk_seconds": frames / aura.fps, "sample_rate": SAMPLE_RATE,
            })
            while (samples := session.audio.read(audio_samples)) is not None:
                started = time.perf_counter()
                features = audio_features.extract(samples, frames)
                video = video_session.render(features)
                if session.audio.cancelled:
                    break
                if muxer is None:
                    muxer = LiveMuxer(session.directory / "stream.mp4", width, height, aura.fps, session.emit)
                muxer.push(video, samples, frames)
                elapsed = time.perf_counter() - started
                session.generated_seconds = video_session.index * frames / aura.fps
                session.emit({
                    "type": "chunk", "chunk": video_session.index,
                    "generation_fps": round(frames / elapsed, 2), "render_seconds": round(elapsed, 2),
                    "generated_seconds": session.generated_seconds,
                    "received_seconds": session.audio.received / SAMPLE_RATE,
                })
                LOG.info("%s %s chunk %d in %.2fs", session.id, session.model, video_session.index, elapsed)
            if muxer is not None:
                muxer.close(cancel=session.audio.cancelled)
                muxer = None
            if not session.audio.cancelled:
                session.complete = video_session.index > 0
                session.emit({
                    "type": "done", "download": f"/api/sessions/{session.id}/video" if session.complete else None,
                    "generated_seconds": session.generated_seconds,
                })
        except Exception as error:
            LOG.exception("Stream %s failed", session.id)
            session.emit({"type": "error", "message": str(error).split("\n")[0]})
        finally:
            session.audio.finish()
            if muxer is not None:
                muxer.close(cancel=True)
            if audio_features is not None:
                audio_features.close()
            if video_session is not None:
                video_session.close()
            gc.collect()
            torch.cuda.empty_cache()


def create_app():
    app = web.Application(client_max_size=12 * 1024**2)
    manager = ModelManager()
    public_origin = os.environ.get("AURA_PUBLIC_ORIGIN", "").rstrip("/")

    async def index(request):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def config(request):
        return web.json_response({
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA GPU unavailable",
            "busy": manager.active is not None, "loaded_model": manager.key[0] if manager.key else None,
            "sample_rate": SAMPLE_RATE,
        })

    async def create_session(request):
        if manager.active is not None:
            raise web.HTTPConflict(text="The GPU is busy. Finish or cancel the current session first.")
        reader = await request.multipart()
        model, compile_model, image_data = "aura-1.3b", False, None
        async for field in reader:
            if field.name == "model":
                model = (await field.text()).strip()
            elif field.name == "compile":
                compile_model = (await field.text()) == "true"
            elif field.name == "image":
                data = bytearray()
                while chunk := await field.read_chunk():
                    data.extend(chunk)
                    if len(data) > 10 * 1024**2:
                        raise web.HTTPBadRequest(text="Choose a portrait smaller than 10 MB.")
                image_data = bytes(data)
        if model not in ("aura-1.3b", "aura-5b"):
            raise web.HTTPBadRequest(text="Unknown model")
        if model == "aura-5b" and compile_model:
            raise web.HTTPBadRequest(text="Compilation is available for 1.3B in this demo.")
        try:
            if image_data is None:
                image_data = (ROOT / "demo/portrait.jpg").read_bytes()
            with Image.open(io.BytesIO(image_data)) as image:
                if image.width * image.height > 24_000_000:
                    raise web.HTTPBadRequest(text="Choose a portrait below 24 megapixels.")
                image = image.convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise web.HTTPBadRequest(text="Please choose a valid PNG, JPEG, or WebP portrait.")
        if manager.active is not None:
            raise web.HTTPConflict(text="The GPU is busy.")
        directory = RUNS / uuid.uuid4().hex
        directory.mkdir(parents=True)
        path = directory / "portrait.jpg"
        image.save(path, quality=95)
        session = Stream(model, compile_model, path, directory)
        manager.sessions[session.id] = session
        manager.active = session
        return web.json_response({"id": session.id, "websocket": f"/api/sessions/{session.id}/stream"})

    async def socket(request):
        session = manager.sessions.get(request.match_info["id"])
        if session is None or manager.active is not session or session.connected:
            raise web.HTTPNotFound(text="Session unavailable")
        origin = request.headers.get("Origin")
        allowed_origins = {f"{request.scheme}://{request.host}"}
        if public_origin:
            allowed_origins.add(public_origin)
        if origin and origin not in allowed_origins:
            raise web.HTTPForbidden(text="Open the demo from the same origin as the server.")
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=65536)
        await ws.prepare(request)
        session.connected = True
        session.loop = asyncio.get_running_loop()
        session.out = asyncio.Queue(maxsize=128)

        async def sender():
            while True:
                item = await session.out.get()
                if item is None:
                    return
                if isinstance(item, bytes):
                    await ws.send_bytes(item)
                else:
                    await ws.send_json(item)

        sending = asyncio.create_task(sender())
        worker = session.loop.run_in_executor(manager.executor, manager.run, session)

        async def receive():
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    if len(message.data) % 4 or not message.data:
                        raise ValueError("Invalid PCM packet")
                    samples = np.frombuffer(message.data, dtype="<f4").copy()
                    if not np.isfinite(samples).all():
                        raise ValueError("PCM samples must be finite")
                    session.audio.feed(np.clip(samples, -1, 1))
                    await session.out.put({
                        "type": "buffer", "received_seconds": session.audio.received / SAMPLE_RATE,
                        "generated_seconds": session.generated_seconds,
                    })
                elif message.type == WSMsgType.TEXT:
                    command = json.loads(message.data)
                    if command.get("type") == "finish":
                        session.audio.finish()
                    elif command.get("type") == "cancel":
                        session.audio.finish(cancel=True)
                        return
                elif message.type == WSMsgType.ERROR:
                    break

        receiving = asyncio.create_task(receive())
        try:
            done, _ = await asyncio.wait((worker, receiving, sending), return_when=asyncio.FIRST_COMPLETED)
            if receiving in done or sending in done:
                session.audio.finish(cancel=True)
                for task in (receiving, sending):
                    if task.done() and not task.cancelled() and task.exception():
                        LOG.warning("Client stream ended: %s", task.exception())
                # CUDA work cannot be interrupted midway. Keep the GPU reserved until
                # the worker has closed its caches, pipes, and encoder process.
                await asyncio.shield(worker)
            else:
                await worker
                await session.out.put(None)
                await sending
        finally:
            session.audio.finish(cancel=not session.complete)
            receiving.cancel()
            sending.cancel()
            await asyncio.gather(receiving, sending, return_exceptions=True)
            if manager.active is session:
                manager.active = None
            await ws.close()
        return ws

    async def download(request):
        session = manager.sessions.get(request.match_info["id"])
        if not session or not session.complete:
            raise web.HTTPNotFound(text="This session has no completed video.")
        return web.FileResponse(session.directory / "stream.mp4", headers={
            "Content-Disposition": f'attachment; filename="{session.model}-{session.id[:8]}.mp4"'
        })

    async def unload(request):
        if manager.active is not None:
            raise web.HTTPConflict(text="Stop the current stream before releasing the GPU.")
        await asyncio.get_running_loop().run_in_executor(manager.executor, manager.unload)
        return web.json_response({"ok": True})

    async def cancel_pending(request):
        session = manager.sessions.get(request.match_info["id"])
        if session is None:
            raise web.HTTPNotFound()
        session.audio.finish(cancel=True)
        if not session.connected and manager.active is session:
            manager.active = None
        return web.json_response({"ok": True})

    async def sample_portrait(request):
        return web.FileResponse(ROOT / "demo/portrait.jpg")

    async def sample_speech(request):
        return web.FileResponse(ROOT / "demo/speech.wav")

    async def maintenance(app):
        async def expire():
            while True:
                await asyncio.sleep(10)
                session = manager.active
                if session and not session.connected and time.monotonic() - session.created > 60:
                    session.audio.finish(cancel=True)
                    manager.active = None
        task = asyncio.create_task(expire())
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if manager.active:
            manager.active.audio.finish(cancel=True)
        await asyncio.get_running_loop().run_in_executor(manager.executor, manager.unload)
        manager.executor.shutdown(wait=True)

    app.cleanup_ctx.append(maintenance)
    app.router.add_get("/", index)
    app.router.add_get("/api/config", config)
    app.router.add_post("/api/sessions", create_session)
    app.router.add_delete("/api/sessions/{id}", cancel_pending)
    app.router.add_get("/api/sessions/{id}/stream", socket)
    app.router.add_get("/api/sessions/{id}/video", download)
    app.router.add_post("/api/unload", unload)
    app.router.add_get("/sample/portrait.jpg", sample_portrait)
    app.router.add_get("/sample/speech.wav", sample_speech)
    app.router.add_static("/static", STATIC)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    RUNS.mkdir(parents=True, exist_ok=True)
    web.run_app(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
