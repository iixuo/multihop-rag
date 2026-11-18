"""Audit saved BGE candidates and the pinned repository configuration, without inference."""
import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent

from common import atomic_json, file_hash, load_queries, read_json
from metrics import fact_normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "output")
    parser.add_argument("--refresh-model-config", action="store_true",
                        help="Download the two tiny repository JSON files at the recorded commit.")
    args = parser.parse_args()
    DATA, out = args.data_dir, args.output_dir
    gold = {row["query"]: row for row in load_queries(DATA / "rag.json")}
    candidates = read_json(out / "rankerB.candidates.json")
    by_query = {row["query"]: row for row in candidates}
    assert set(by_query) == set(gold)
    meta = read_json(out / "rankerB.meta.json")
    pool_scores = {}
    for cutoff in (10, 50):
        recalls, full, hits = [], [], []
        for query, row in by_query.items():
            assert len(row["retrieval_list"]) == 50
            if gold[query]["question_type"] == "null_query":
                continue
            facts = {fact_normalize(e["fact"]) for e in gold[query]["evidence_list"]}
            texts = [fact_normalize(item["text"]) for item in row["retrieval_list"][:cutoff]]
            covered = {fact for fact in facts if any(fact in text for text in texts)}
            recalls.append(len(covered) / len(facts))
            full.append(covered == facts)
            hits.append(bool(covered))
        pool_scores[str(cutoff)] = {
            "n": len(recalls), "FactRecall": statistics.mean(recalls),
            "FullEvidenceRecall": statistics.mean(full), "Hit": statistics.mean(hits),
            "full_count": sum(full),
        }
    for row in read_json(out / "rerankerA.json"):
        saved = {item["chunk_id"] for item in by_query[row["query"]]["retrieval_list"]}
        assert {item["chunk_id"] for item in row["retrieval_list"]} <= saved
    revision = meta["signature"]["resolved_revision"]
    cached = out / "report_analysis/candidate_pool_audit.json"
    if not args.refresh_model_config:
        if not cached.exists():
            raise SystemExit("No cached model configuration; rerun with --refresh-model-config.")
        previous = read_json(cached)
        assert previous["recorded_revision"] == revision
        configs = previous["official_repository_configs"]
    else:
        configs = {}
        for name in ("modules.json", "1_Pooling/config.json"):
            url = f"https://huggingface.co/BAAI/bge-m3/resolve/{revision}/{name}"
            with urllib.request.urlopen(url, timeout=25) as response:
                configs[name] = {"url": url, "value": json.load(response)}
    for name, config in configs.items():
        assert config["url"] == f"https://huggingface.co/BAAI/bge-m3/resolve/{revision}/{name}"
    pooling = configs["1_Pooling/config.json"]["value"]
    assert pooling["pooling_mode_cls_token"] and not pooling["pooling_mode_mean_tokens"]
    result = {
        "candidate_sha256": file_hash(out / "rankerB.candidates.json"),
        "query_sha256": file_hash(DATA / "rag.json"),
        "pool_scores": pool_scores,
        "all_reranked_ids_in_candidates": True,
        "recorded_revision": revision,
        "official_repository_configs": configs,
        "passage_truncation": meta["truncation"]["passages"],
        "scope": "Post-hoc audit of existing outputs and configured repository modules; not an FP32 or alternate-pooling inference ablation.",
    }
    atomic_json(out / "report_analysis/candidate_pool_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
