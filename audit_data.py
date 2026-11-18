"""Audit query coverage and evidence lost at the fixed word-chunk boundaries."""
import argparse
from collections import Counter
from pathlib import Path

from common import atomic_json, chunk_records, file_hash, load_queries, object_hash, read_json
from config import ROOT, dataset_paths
from metrics import RelevancePool, fact_normalize


def audit(corpus_path, query_path):
    corpus = read_json(corpus_path)
    gold = load_queries(query_path)
    chunks = chunk_records(corpus)
    pool = RelevancePool(chunks, gold)
    non_null = [r for r in gold if r["question_type"] != "null_query"]
    missing_facts = {fact for fact, hits in pool.hits.items() if not hits}
    affected, attainable = [], []
    for row in non_null:
        facts = {fact_normalize(e["fact"]) for e in row["evidence_list"]}
        absent = facts & missing_facts
        attainable.append((len(facts) - len(absent)) / len(facts))
        if absent:
            affected.append({"query": row["query"], "unmatched_facts": [
                e["fact"] for e in row["evidence_list"] if fact_normalize(e["fact"]) in absent]})
    return {
        "kind": "dataset audit, not model experiment results",
        "corpus_sha256": file_hash(corpus_path), "queries_sha256": file_hash(query_path),
        "documents": len(corpus), "queries": len(gold),
        "question_types": dict(Counter(r["question_type"] for r in gold)),
        "non_null_queries": len(non_null), "unique_facts": len(pool.hits),
        "gold_answers_yes_no": sum(r["answer"].strip().lower() in ("yes", "no") for r in gold),
        "chunk_size_words": 256, "overlap_words": 20, "chunks": len(chunks),
        "chunks_sha256": object_hash(chunks), "facts_not_fully_present_in_any_chunk": len(missing_facts),
        "affected_queries": len(affected),
        "attainable_macro_fact_recall_without_top_k_limit": sum(attainable) / len(attainable) if attainable else None,
        "attainable_full_evidence_recall_without_top_k_limit": 1 - len(affected) / len(non_null) if non_null else None,
        "note": "Fact matching uses the supplied case-sensitive space/newline removal convention.",
        "affected_query_details": affected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = dataset_paths(args.data_dir, args.staging)
    result = audit(*paths)
    output = args.output or ROOT / "validation" / ("sample_data_audit.json" if args.staging else "data_audit.json")
    atomic_json(output, result)
    for key, value in result.items():
        if key != "affected_query_details":
            print(f"{key}: {value}")
    print("Saved", output)


if __name__ == "__main__":
    main()
