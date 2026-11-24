"""Continuous H.264/AAC fragmented MP4 output with bounded writer queues."""

import os
import queue
import shutil
import subprocess
import threading

import imageio_ffmpeg
import numpy as np
import torch


class LiveMuxer:
    def __init__(self, path, width, height, fps, emit):
        self.path = path
        self.emit = emit
        self.errors = []
        self.video_queue = queue.Queue(maxsize=2)
        self.audio_queue = queue.Queue(maxsize=2)
        audio_read, audio_write = os.pipe()
        command = [
            shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-probesize", "32", "-analyzeduration", "1",
            "-thread_queue_size", "64", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
            "-probesize", "32", "-analyzeduration", "1",
            "-thread_queue_size", "64", "-f", "f32le", "-ar", "24000", "-ac", "1",
            "-i", f"pipe:{audio_read}", "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast", "-tune", "zerolatency",
            "-profile:v", "baseline", "-level:v", "3.1", "-pix_fmt", "yuv420p",
            "-crf", "22", "-g", "16", "-keyint_min", "16", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "empty_moov+default_base_moof+frag_keyframe",
            "-frag_duration", "100000",
            "-flush_packets", "1", "-f", "mp4", "pipe:1",
        ]
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, pass_fds=(audio_read,)
        )
        os.close(audio_read)
        self.audio_pipe = os.fdopen(audio_write, "wb", buffering=0)
        self.threads = [
            threading.Thread(target=self._write, args=(self.video_queue, self.process.stdin), daemon=True),
            threading.Thread(target=self._write, args=(self.audio_queue, self.audio_pipe), daemon=True),
            threading.Thread(target=self._read, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _write(self, items, pipe):
        try:
            while True:
                data = items.get()
                if data is None:
                    break
                view = memoryview(data)
                while view:
                    written = pipe.write(view)
                    view = view[written:]
                pipe.flush()
        except (BrokenPipeError, OSError) as error:
            self.errors.append(error)
        finally:
            try:
                pipe.close()
            except (BrokenPipeError, OSError):
                pass

    def _read(self):
        try:
            with self.path.open("wb") as output:
                while data := os.read(self.process.stdout.fileno(), 65536):
                    output.write(data)
                    self.emit(data)
        except Exception as error:
            self.errors.append(error)

    def _put(self, items, data):
        while True:
            if self.errors or self.process.poll() is not None:
                raise RuntimeError("Video encoder stopped; inspect the session FFmpeg log")
            try:
                items.put(data, timeout=0.2)
                return
            except queue.Full:
                continue

    def push(self, video, samples, frames):
        if video.shape[1] < frames:
            # The causal VAE starts with 13 rather than 16 frames. Hold its first
            # image for the warm-up interval so audio/video clocks stay aligned.
            pad = video[:, :1].expand(-1, frames - video.shape[1], -1, -1)
            video = torch.cat((pad, video), dim=1)
        pixels = ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0)
        self._put(self.audio_queue, np.asarray(samples, dtype="<f4").tobytes())
        self._put(self.video_queue, pixels.contiguous().numpy().tobytes())

    def close(self, cancel=False):
        if cancel:
            self.process.terminate()
            for items in (self.video_queue, self.audio_queue):
                while not items.empty():
                    try:
                        items.get_nowait()
                    except queue.Empty:
                        break
        for items in (self.video_queue, self.audio_queue):
            try:
                items.put(None, timeout=2)
            except queue.Full:
                self.process.terminate()
        for thread in self.threads[:2]:
            thread.join(timeout=10)
        try:
            status = self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            status = self.process.wait()
        self.threads[2].join(timeout=10)
        self.process.stdout.close()
        self.log.close()
        if not cancel and (status != 0 or self.errors):
            raise RuntimeError("Could not finalize the video; inspect the session FFmpeg log")
