"""Reuse the established CPU-only four-case regression screen on fresh paired renders."""
import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    # Interpreter and weights are the same as the prior quality screen.
    spec = importlib.util.spec_from_file_location("existing_judge", root / "rtx1-taehv-001/judge_lighttae.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POLICY["note"] = (
        "Paired against fresh BF16 SDPA + LightTAE v2, at native 768x512 and four steps. "
        "Measures additional FP8 regression; does not erase LightTAE's prior fidelity loss versus Wan VAE."
    )
    view = args.run / ("paired-" + args.candidate)
    view.mkdir(exist_ok=False)
    for case in ["existing", "fashion", "grandmother", "producer"]:
        folder = view / case
        folder.mkdir()
        for name, variant in [("native", "bf16"), (args.candidate, args.candidate)]:
            source = (args.run / variant / case / "native").resolve()
            if not (source / "summary.json").exists():
                raise RuntimeError(f"Incomplete render: {source}")
            (folder / name).symlink_to(source, target_is_directory=True)
    module.OUT = view
    module.CANDIDATES = [args.candidate]
    module.main()


if __name__ == "__main__":
    main()
