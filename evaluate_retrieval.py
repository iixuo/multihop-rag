"""Evaluate the complete query set, then select two retrieval systems."""
import argparse
from pathlib import Path

from common import (
    align_predictions, atomic_json, chunk_records, display_path, file_hash,
    load_queries, object_hash, read_json, select_best, validate_retrieval,
)
from config import ROOT, dataset_paths, default_output
from metrics import RETRIEVAL_METRICS, RelevancePool, grouped_scores


def evaluate_file(path, gold, pool):
    pairs = align_predictions(read_json(path), gold)
    details = []
    for prediction, item in pairs:
        if prediction.get("status", "ok") != "ok":
            raise ValueError("A failed retrieval row remains; resume the run before evaluating it.")
        validate_retrieval(prediction)
        for chunk in prediction["retrieval_list"]:
            if chunk["text"] not in pool.valid_texts:
                raise ValueError(f"Retrieved text does not belong to the common chunk pool: {path}")
            if "chunk_id" in chunk and pool.by_id.get(chunk["chunk_id"]) != chunk["text"]:
                raise ValueError("chunk_id and text disagree.")
        if item["question_type"] == "null_query":
            continue
        score = pool.score(prediction["retrieval_list"], item["evidence_list"])
        details.append({"query": item["query"], "question_type": item["question_type"], **score})
    result = {
        "file": display_path(path), "sha256": file_hash(path),
        "evaluated_total": len(pairs), "evaluated_non_null": len(details),
        "groups": grouped_scores(details, RETRIEVAL_METRICS),
        "attainable_macro_fact_recall": (
            sum(d["attainable_fact_recall"] for d in details) / len(details) if details else None),
        "attainable_full_evidence_recall": (
            sum(d["all_facts_attainable"] for d in details) / len(details) if details else None),
    }
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunks", type=Path, help="A shared saved chunk table; otherwise reconstruct it.")
    args = parser.parse_args()
    out = args.output_dir or default_output(args.staging)
    inputs = args.inputs or [out / f"{key}.json" for key in
                             ("rankerA", "rankerB", "rankerC", "rerankerA")]
    corpus_path, query_path = dataset_paths(args.data_dir, args.staging)
    gold = load_queries(query_path)
    chunks = read_json(args.chunks) if args.chunks else chunk_records(
        read_json(corpus_path), args.chunk_size, args.overlap)
    pool = RelevancePool(chunks, gold)
    report = {
        "stage": "retrieval", "staging": args.staging,
        "corpus_sha256": file_hash(corpus_path), "queries_sha256": file_hash(query_path),
        "chunks_sha256": object_hash(chunks), "chunk_count": len(chunks),
        "aggregation": "macro over non-null queries; MAP@10 is mean AP@10 with denominator min(10, R); R counts all literal-match relevant chunks in the shared pool; zero if R=0",
        "runs": {},
    }
    all_details = {}
    for path in inputs:
        key = path.stem
        if key in report["runs"]:
            raise ValueError("Run filenames must have distinct stems.")
        run, details = evaluate_file(path, gold, pool)
        report["runs"][key], all_details[key] = run, details
        print(key, run["groups"]["overall"])
    if len(report["runs"]) >= 2:
        report["selected"] = select_best(report["runs"])
        report["selection_policy"] = "FullEvidenceRecall@10, FactRecall@10, MAP@10; ID breaks exact ties"
        print("Selected:", ", ".join(report["selected"]))
    output = args.output or out / "metrics_retrieval.json"
    atomic_json(output, report)
    atomic_json(output.with_name(output.stem + ".per_query.json"), all_details)
    print("Saved", output)


if __name__ == "__main__":
    main()
