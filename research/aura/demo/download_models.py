"""Prefetch only the weights needed by both local AURA demos."""

from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download

JOBS = [
    ("Wan-AI/Wan2.1-T2V-1.3B", ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/*"]),
    ("Wan-AI/Wan2.2-TI2V-5B", ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/*"]),
    ("kyutai/AURA-1.3B", ["stage2_dmd/step-6500_ema_merged.safetensors"]),
    ("kyutai/AURA-5B", ["stage2_dmd/step-5000_merged.safetensors"]),
    ("kyutai/moshiko-pytorch-bf16", ["tokenizer*.safetensors"]),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-english", ["*.json", "model.safetensors"]),
]


def download(job):
    repo, patterns = job
    print(f"Downloading {repo}", flush=True)
    path = snapshot_download(repo, allow_patterns=patterns, max_workers=3)
    print(f"Ready: {repo}: {path}", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, JOBS))
    print("All downloads complete", flush=True)
