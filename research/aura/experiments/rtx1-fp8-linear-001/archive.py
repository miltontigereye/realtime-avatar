"""Archive matching executed sources and supplementary provenance, without rerunning inference."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    notes = {"packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "experiment_target_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[str(here / "deps")])},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
        "quality_inputs_sha256": {}, "variants": {},
        "preflight_only_attempts": [p.name for p in args.run.iterdir() if (p / "PREFLIGHT_BUSY").exists()],
        "baseline_provenance_note": "Initial BF16 manifest also hashed root aura/model.py; actual imported model is conditioning-001/aura/model.py, included below. BF16 used no quantization."}
    for path in (here.parent / "rtx1-quality-001/inputs").iterdir():
        if path.suffix.lower() in {".wav", ".jpg"}:
            notes["quality_inputs_sha256"][str(path)] = sha(path)
    experiment_snapshot = args.run / "experiment-source"
    experiment_snapshot.mkdir(exist_ok=True)
    notes["experiment_sources_sha256"] = {}
    for path in here.iterdir():
        if path.is_file() and path.suffix in {".py", ".sh", ".txt"}:
            shutil.copyfile(path, experiment_snapshot / path.name)
            notes["experiment_sources_sha256"][path.name] = sha(path)
    actual_model = here.parent / "rtx1-conditioning-001/aura/model.py"
    for variant in [p for p in args.run.iterdir() if (p / "metadata.json").exists()]:
        metadata = json.loads((variant / "metadata.json").read_text())
        snapshot = variant / "source"
        snapshot.mkdir(exist_ok=True)
        files = {}
        for filename, expected in metadata["source_sha256"].items():
            path = Path(filename)
            candidates = [path, variant / "executed.py", variant / "executed-fp8-policy.py",
                          args.run / "ffn/executed-fp8-policy.py"]
            match = next((p for p in candidates if p.is_file() and sha(p) == expected), None)
            if match is None:
                raise RuntimeError(f"No matching source for {variant.name}: {filename} {expected}")
            target = snapshot / (expected[:12] + "-" + path.name)
            shutil.copyfile(match, target)
            files[filename] = {"sha256": expected, "snapshot": str(target)}
        target = snapshot / (sha(actual_model)[:12] + "-actual-aura-model.py")
        shutil.copyfile(actual_model, target)
        files[str(actual_model)] = {"sha256": sha(actual_model), "snapshot": str(target)}
        notes["variants"][variant.name] = {"source_snapshot": files,
            "complete": (variant / "DONE").exists(), "stopped": (variant / "STOPPED").exists()}
    for candidate in ["ffn_bf16", "self_ffn_bf16"]:
        meta = args.run / candidate / "metadata.json"
        if meta.exists():
            checkpoint_sha = json.loads(meta.read_text()).get("checkpoint_sha256")
            if checkpoint_sha:
                notes["checkpoint_sha256_all_variants_same_released_checkpoint"] = checkpoint_sha
                break
    (args.run / "provenance.json").write_text(json.dumps(notes, indent=2))
    shutil.copyfile(Path(__file__), args.run / "archive-executed.py")
    print("Archived", list(notes["variants"]), flush=True)


if __name__ == "__main__":
    main()
