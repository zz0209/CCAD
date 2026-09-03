"""Independent integrity validation for the R008 paired corpus/code assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-run", required=True)
    parser.add_argument("--codes-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    corpus_dir = ROOT / "runs" / args.corpus_run
    codes_dir = ROOT / "runs" / args.codes_run
    corpus_status = json.loads((corpus_dir / "status.json").read_text(encoding="utf-8"))
    corpus_contract = json.loads((corpus_dir / "contract_validation.json").read_text(encoding="utf-8"))
    token_manifest_path = corpus_dir / "artifacts/token_manifest.json"
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    codes_config = json.loads((codes_dir / "config.resolved.json").read_text(encoding="utf-8"))
    codes_status = json.loads((codes_dir / "status.json").read_text(encoding="utf-8"))
    codes_contract = json.loads((codes_dir / "contract_validation.json").read_text(encoding="utf-8"))
    record = json.loads((codes_dir / "metrics.raw.jsonl").read_text(encoding="utf-8"))
    checks = {
        "source_runs_pass": corpus_status["status"] == "PASS" and codes_status["status"] == "PASS" and corpus_contract["ok"] and codes_contract["ok"],
        "token_manifest_binding": sha256(token_manifest_path) == codes_config["token_manifest_sha256"],
        "locked_split_set": set(token_manifest["outputs"]) == {"mean", "discovery", "calibration", "audit"},
        "audit_not_scored": record["checks"]["audit_metrics_not_computed"] is True and codes_config["audit_opened"] is False,
    }
    token_ok = True
    for split, info in token_manifest["outputs"].items():
        path = corpus_dir / info["path"]
        token_ok &= path.stat().st_size == 2 * info["tokens"] and sha256(path) == info["sha256"]
    checks["token_files_bound"] = bool(token_ok)
    output_ok = True
    index_range_ok = True
    finite_ok = True
    per_seed_hashes: dict[int, set[str]] = {}
    for split in record["splits"]:
        for item in split["files"]:
            path = Path(item["path"])
            width = 2 if item["dtype"] == "uint16" else 4
            expected_bytes = width * int(np.prod(item["shape"]))
            actual_hash = sha256(path)
            output_ok &= path.stat().st_size == expected_bytes and actual_hash == item["sha256"]
            per_seed_hashes.setdefault(item["seed"], set()).add(actual_hash)
            array = np.memmap(path, dtype="<u2" if item["dtype"] == "uint16" else "<f4", mode="r", shape=tuple(item["shape"]))
            if item["dtype"] == "uint16":
                index_range_ok &= int(array.max()) < codes_config["num_latents"]
            else:
                finite_ok &= bool(np.isfinite(array).all())
            del array
    decoder_ok = True
    for item in record["decoders"]:
        path = Path(item["path"])
        decoder_ok &= path.stat().st_size == 4 * int(np.prod(item["shape"])) and sha256(path) == item["sha256"]
        decoder = np.memmap(path, dtype="<f4", mode="r", shape=tuple(item["shape"]))
        decoder_ok &= bool(np.isfinite(decoder).all())
        del decoder
    checks.update({
        "all_sparse_output_hashes_and_sizes": bool(output_ok),
        "all_indices_in_range": bool(index_range_ok),
        "all_acts_finite": bool(finite_ok),
        "all_decoder_hashes_sizes_finite": bool(decoder_ok),
        "five_distinct_seed_assets": sorted(per_seed_hashes) == [1, 2, 3, 4, 5] and all(len(values) == 8 for values in per_seed_hashes.values()),
        "row_and_l0_contract": all(row["observed_rows"] == row["tokens"] and row["selected_l0"] == codes_config["k"] and row["nonzero_l0"] == codes_config["k"] for row in record["splits"]),
    })
    result = {"schema_version": "r008.paired_asset_validation.v1", "corpus_run": args.corpus_run, "codes_run": args.codes_run,
              "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks), "status": "PASS" if all(checks.values()) else "FAIL",
              "validator_sha256": sha256(Path(__file__).resolve())}
    write_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
