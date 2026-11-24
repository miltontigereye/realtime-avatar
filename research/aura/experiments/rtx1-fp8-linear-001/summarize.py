"""Regenerate the engineering comparison directly from saved per-variant results."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    base = json.loads((args.run / "bf16/results.json").read_text())
    rows = []
    for variant in ["bf16", "ffn_bf16", "self_ffn_bf16"]:
        path = args.run / variant / "results.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        quality = args.run / ("paired-" + variant) / ("quality-" + variant + ".json")
        verdict = json.loads(quality.read_text()) if quality.exists() else None
        rows.append({"variant": variant, "generation_fps": result["generation_fps"],
            "throughput_gain_percent": 100 * (result["generation_fps"] / base["generation_fps"] - 1),
            "latency_seconds": result["latency_seconds"], "measured_chunks": result["measured_chunks"],
            "deadline_misses": result["deadline_misses"], "new_graphs": result["new_graphs"],
            "automated_quality_passed": verdict["automated_gate_passed"] if verdict else None,
            "failed_checks": {r["case"]: [k for k, v in r["checks"].items() if not v]
                for r in verdict["cases"]} if verdict else None,
            "mouth_psnr_db": {r["case"]: r["fidelity"]["mouth_psnr_db"]
                for r in verdict["cases"]} if verdict else None})
    aggregate = {"scope": "Short engineering screen; warm steady-state throughput, four paired 16s cases, no human review or sustained streaming validation",
        "resolution": [768, 512], "steps": [1000, 745, 498, 255],
        "decoder": "LightTAE v2 FP16", "attention": "SDPA", "rows": rows}
    (args.run / "comparison.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
