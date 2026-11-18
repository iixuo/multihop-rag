"""Freeze the two data-selected systems and reject stale evaluation reports."""
import argparse
from pathlib import Path

from common import atomic_json, file_hash, read_json, resolve_recorded_path, select_best


def make_selection(report):
    runs = report["runs"]
    required = {"rankerA", "rankerB", "rankerC", "rerankerA"}
    if set(runs) != required:
        raise ValueError(f"Selection requires all four runs: {sorted(required)}.")
    selected = select_best(runs)
    entries = []
    for key in selected:
        path = resolve_recorded_path(runs[key]["file"])
        if file_hash(path) != runs[key]["sha256"]:
            raise ValueError(f"Retrieval output changed after evaluation: {path}")
        entries.append({"id": key, "file": runs[key]["file"], "sha256": runs[key]["sha256"],
                        "metrics": runs[key]["groups"]["overall"]})
    return {
        "selected": entries, "queries_sha256": report["queries_sha256"],
        "corpus_sha256": report["corpus_sha256"], "staging": report["staging"],
        "policy": "FullEvidenceRecall@10, FactRecall@10, MAP@10; lexicographic ID for exact ties",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = make_selection(read_json(args.metrics))
    atomic_json(args.output, selection)
    print("Selected", ", ".join(entry["id"] for entry in selection["selected"]))


if __name__ == "__main__":
    main()
