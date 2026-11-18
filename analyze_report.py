"""Recompute scores and derive report diagnostics without modifying GPU outputs."""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent

from common import (atomic_json, chunk_records, file_hash, format_chunk,
                    load_queries, read_json)
from evaluate_retrieval import evaluate_file
from evaluate_rag import evaluate_rows
from metrics import RelevancePool, answer_normalize, answer_tokens, fact_normalize
from paired_analysis import build_paired_analysis


def close_tree(actual, recorded):
    if isinstance(actual, dict):
        assert actual.keys() == recorded.keys(), (actual.keys(), recorded.keys())
        for key in actual:
            close_tree(actual[key], recorded[key])
    elif isinstance(actual, (float, int)):
        assert abs(actual - recorded) < 1e-12, (actual, recorded)
    else:
        assert actual == recorded, (actual, recorded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "output")
    args = parser.parse_args()
    DATA, out = args.data_dir, args.output_dir
    gold = load_queries(DATA / "rag.json")
    gm = {row["query"]: row for row in gold}
    chunks = chunk_records(read_json(DATA / "corpus.json"))
    pool = RelevancePool(chunks, gold)
    retrieval_report = read_json(out / "metrics_retrieval.json")
    rag_report = read_json(out / "metrics_rag.json")
    retrieval_details, predictions, rag_details = {}, {}, {}
    for key in retrieval_report["runs"]:
        run, details = evaluate_file(out / f"{key}.json", gold, pool)
        close_tree(run["groups"], retrieval_report["runs"][key]["groups"])
        assert file_hash(out / f"{key}.json") == retrieval_report["runs"][key]["sha256"]
        retrieval_details[key] = {row["query"]: row for row in details}
    for key in rag_report["runs"]:
        raw = read_json(out / f"{key}.json")
        run, details = evaluate_rows(raw, gold)
        close_tree(run["groups"], rag_report["runs"][key]["groups"])
        assert file_hash(out / f"{key}.json") == rag_report["runs"][key]["sha256"]
        predictions[key] = {row["query"]: row for row in raw}
        rag_details[key] = {row["query"]: row for row in details}
    result = {
        "recomputed_metrics_match": True,
        "corpus_sha256": file_hash(DATA / "corpus.json"),
        "queries_sha256": file_hash(DATA / "rag.json"),
        "counts": dict(Counter(row["question_type"] for row in gold)),
        "gold_answer_counts": dict(Counter(row["answer"] for row in gold).most_common(6)),
        "contexts": {}, "generation": {}, "retrieval": {}, "paired_retriever_change": {},
        "examples": [],
    }
    context_maps = {}
    for rid in ("rankerC", "rerankerA"):
        rows = read_json(out / f"{rid}.context.json")
        cm = {row["query"]: row for row in rows}
        context_maps[rid] = cm
        details = []
        for row in rows:
            item = gm[row["query"]]
            if item["question_type"] == "null_query":
                continue
            facts = {fact_normalize(e["fact"]) for e in item["evidence_list"]}
            normalized = fact_normalize(row["context"])
            covered = sum(fact in normalized for fact in facts)
            top10 = retrieval_details[rid][row["query"]]
            details.append({"query": row["query"], "covered": covered,
                            "fact_recall": covered / len(facts), "full": covered == len(facts),
                            "lost": covered < top10["covered_facts"],
                            "full_lost": bool(top10["FullEvidenceRecall@10"]) and covered < len(facts)})
        dm = {d["query"]: d for d in details}
        result["contexts"][rid] = {
            "truncated_queries": sum(row["truncated"] for row in rows), "total": len(rows),
            "mean_full_chunks": statistics.mean(row["full_chunks"] for row in rows),
            "mean_used_chunks": statistics.mean(len(row["used_chunk_ids"]) for row in rows),
            "packed_fact_recall": statistics.mean(d["fact_recall"] for d in details),
            "packed_full_evidence_recall": statistics.mean(d["full"] for d in details),
            "queries_losing_any_gold_fact": sum(d["lost"] for d in details),
            "top10_complete_queries_becoming_incomplete": sum(d["full_lost"] for d in details),
            "token_counts": {model: {"mean": statistics.mean(row["input_tokens"][model] for row in rows),
                                     "max": max(row["input_tokens"][model] for row in rows)}
                             for model in ("ragA", "ragB", "ragC", "ragD")},
        }
        for model in ("ragA", "ragB", "ragC", "ragD"):
            key = f"{model}__{rid}"
            raw = list(predictions[key].values())
            meta = read_json(out / f"{key}.meta.json")
            attempt = meta["attempts"][-1]
            conditional = {}
            for name, chosen in (("full_context", [q for q, d in dm.items() if d["full"]]),
                                 ("incomplete_context", [q for q, d in dm.items() if not d["full"]])):
                conditional[name] = {"n": len(chosen),
                    "EM": statistics.mean(rag_details[key][q]["ExactMatch"] for q in chosen),
                    "F1": statistics.mean(rag_details[key][q]["F1"] for q in chosen)}
            result["generation"][key] = {
                "model": meta["signature"]["model"], "seconds": attempt["seconds"],
                "shared_model_load_seconds": attempt["shared_model_load_seconds"],
                "seconds_plus_shared_load": attempt["seconds"] + attempt["shared_model_load_seconds"],
                "output_limit_count": sum(row["hit_output_limit"] for row in raw),
                "mean_generated_tokens": statistics.mean(row["generated_tokens"] for row in raw),
                "median_generated_tokens": statistics.median(row["generated_tokens"] for row in raw),
                "mean_normalized_answer_tokens": statistics.mean(rag_details[key][row["query"]]["predicted_tokens"] for row in raw),
                "conditional": conditional,
                "refused_count": sum(answer_normalize(row["model_answer"]) == "insufficient information" for row in raw),
                "prefix_refusal_null_count": sum(
                    row["question_type"] == "null_query" and
                    answer_tokens(row["model_answer"])[:2] == ["insufficient", "information"] for row in raw),
                "prefix_refusal_non_null_count": sum(
                    row["question_type"] != "null_query" and
                    answer_tokens(row["model_answer"])[:2] == ["insufficient", "information"] for row in raw),
                "yes_no_gold_with_literal_yes_no_prediction": sum(
                    answer_normalize(gm[row["query"]]["answer"]) in ("yes", "no") and
                    answer_normalize(row["model_answer"]) in ("yes", "no") for row in raw),
            }
    for key in retrieval_report["runs"]:
        meta = read_json(out / f"{key}.meta.json")
        result["retrieval"][key] = {
            "model": meta["signature"]["model"], "seconds": meta["attempts"][-1]["seconds"],
            "complete_evidence_queries": sum(row["FullEvidenceRecall@10"] for row in retrieval_details[key].values()),
            "ap10_mean": statistics.mean(row["AP@10"] for row in retrieval_details[key].values()),
            "ap10_median": statistics.median(row["AP@10"] for row in retrieval_details[key].values()),
            "ap10_zero_count": sum(row["AP@10"] == 0 for row in retrieval_details[key].values()),
            "ap10_values": [row["AP@10"] for row in retrieval_details[key].values()],
        }
    for model in ("ragA", "ragB", "ragC", "ragD"):
        left, right = rag_details[f"{model}__rankerC"], rag_details[f"{model}__rerankerA"]
        result["paired_retriever_change"][model] = {
            "EM_wins_reranker": sum(right[q]["ExactMatch"] > left[q]["ExactMatch"] for q in gm),
            "EM_losses_reranker": sum(right[q]["ExactMatch"] < left[q]["ExactMatch"] for q in gm),
            "mean_F1_delta": statistics.mean(right[q]["F1"] - left[q]["F1"] for q in gm),
        }
    # Choose short, directly inspectable contrasting cases from actual outputs.
    categories = {
        "phi_format_failure": lambda q: answer_normalize(gm[q]["answer"]) in ("yes", "no") and
            rag_details["ragB__rerankerA"][q]["ExactMatch"] == 1 and
            rag_details["ragD__rerankerA"][q]["AnswerContainment"] == 1 and
            rag_details["ragD__rerankerA"][q]["ExactMatch"] == 0,
        "qwen_false_refusal": lambda q: gm[q]["question_type"] == "temporal_query" and
            rag_details["ragB__rerankerA"][q]["ExactMatch"] == 1 and
            rag_details["ragC__rerankerA"][q]["false_refusal"] == 1,
        "retriever_reversal": lambda q: gm[q]["question_type"] != "null_query" and
            rag_details["ragC__rankerC"][q]["ExactMatch"] == 1 and
            rag_details["ragC__rerankerA"][q]["ExactMatch"] == 0,
        "null_refusal_contrast": lambda q: gm[q]["question_type"] == "null_query" and
            rag_details["ragC__rerankerA"][q]["ExactMatch"] == 1 and
            rag_details["ragB__rerankerA"][q]["ExactMatch"] == 0,
    }
    for label, predicate in categories.items():
        chosen = sorted((q for q in gm if predicate(q)), key=len)[:3]
        for query in chosen:
            item = gm[query]
            result["examples"].append({"category": label, "query": query, "gold": item["answer"],
                "type": item["question_type"], "facts": item["evidence_list"],
                "predictions": {key: predictions[key][query]["model_answer"] for key in predictions},
                "contexts": {rid: {"full_top10": retrieval_details[rid].get(query, {}).get("FullEvidenceRecall@10"),
                    "facts_present": [e["fact"] in context_maps[rid][query]["context"] for e in item["evidence_list"]]}
                    for rid in context_maps}})
    paired = build_paired_analysis(gm, predictions, rag_details, context_maps)
    atomic_json(out / "report_analysis" / "paired_query_analysis.json", paired)
    atomic_json(out / "report_analysis" / "results_analysis.json", result)
    print("Recomputation with the submitted scorers matched all seven metrics and question-type groups for all 12 runs.")
    print("Exclusive RR correct counts:", {m: row["count"] for m, row in paired["exclusive_RR"].items()})
    print("Exact McNemar tests:", json.dumps(paired["mcnemar_retriever_change"], indent=2))


if __name__ == "__main__":
    main()
