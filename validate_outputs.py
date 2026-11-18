"""Check full experiment completeness; optionally check submission artifacts."""
import argparse
from pathlib import Path

from common import (
    align_predictions, file_hash, load_queries, object_hash, read_json,
    select_best, validate_retrieval,
)
from config import LLMS, ROOT, dataset_paths, default_output
from context import make_prompt
from prepare_contexts import load_selection


def complete_file(path, gold, stage):
    rows = read_json(path)
    meta = read_json(path.with_suffix(".meta.json"))
    if meta.get("status") != "complete" or meta["signature"].get("stage") != stage:
        raise ValueError(f"Not a complete {stage} output: {path.name}")
    if meta["signature_hash"] != object_hash(meta["signature"]):
        raise ValueError(f"Metadata signature is corrupt: {path.name}")
    if meta.get("success_count") != len(gold) or meta.get("saved_count") != len(gold):
        raise ValueError(f"Incorrect checkpoint counts: {path.name}")
    pairs = align_predictions(rows, gold)
    if any(row.get("status") != "ok" for row, _ in pairs):
        raise ValueError(f"Failed rows remain: {path.name}")
    return rows, meta


def validate(out, data_dir, staging=False, submission=False):
    corpus_path, query_path = dataset_paths(data_dir, staging)
    gold = load_queries(query_path)
    hashes = {"queries_sha256": file_hash(query_path), "corpus_sha256": file_hash(corpus_path)}
    selection = load_selection(out / "selection.json", corpus_path, query_path, staging)
    retrieval_report = read_json(out / "metrics_retrieval.json")
    rag_report = read_json(out / "metrics_rag.json")
    for report in (retrieval_report, rag_report):
        if report["staging"] != staging or report["queries_sha256"] != hashes["queries_sha256"]:
            raise ValueError("An evaluation report refers to a different dataset.")
    required_retrievers = {"rankerA", "rankerB", "rankerC", "rerankerA"}
    if set(retrieval_report["runs"]) != required_retrievers:
        raise ValueError("Need evaluation results for all four retrieval systems.")
    if [entry["id"] for entry in selection["selected"]] != select_best(retrieval_report["runs"]):
        raise ValueError("The frozen selection differs from the reported selection policy.")
    for name in sorted(required_retrievers):
        path = out / f"{name}.json"
        rows, meta = complete_file(path, gold, "reranking" if name == "rerankerA" else "retrieval")
        if any(meta["signature"][key] != value for key, value in hashes.items()):
            raise ValueError(f"Dataset mismatch: {name}")
        if retrieval_report["runs"][name]["sha256"] != file_hash(path):
            raise ValueError(f"Stale retrieval evaluation: {name}")
        for row in rows:
            validate_retrieval(row)
            if len(row["retrieval_list"]) != 10:
                raise ValueError("Formal retrieval outputs must contain exactly 10 chunks.")
    candidate_path = out / "rankerB.candidates.json"
    candidates, candidate_meta = complete_file(candidate_path, gold, "retrieval")
    if candidate_meta["signature"].get("candidate_k") != 50:
        raise ValueError("rankerB must preserve the top-50 candidate pool.")
    reranker_meta = read_json((out / "rerankerA.json").with_suffix(".meta.json"))
    if reranker_meta["signature"].get("candidate_sha256") != file_hash(candidate_path):
        raise ValueError("The reranker was not run on the saved rankerB candidate file.")
    for row in candidates:
        validate_retrieval(row, minimum=50)
        if len(row["retrieval_list"]) != 50:
            raise ValueError("Candidate pools must contain exactly 50 chunks.")
    expected_rag = {f"{rag}__{entry['id']}" for rag in LLMS for entry in selection["selected"]}
    if set(rag_report["runs"]) != expected_rag:
        raise ValueError("Need exactly 4 LLMs x 2 selected retrievers = 8 evaluated RAG runs.")
    for entry in selection["selected"]:
        context_path = out / f"{entry['id']}.context.json"
        contexts, context_meta = complete_file(context_path, gold, "contexts")
        if context_meta["signature"]["retrieval_sha256"] != entry["sha256"]:
            raise ValueError("Contexts no longer match the selected retrieval output.")
        contexts_by_query = {row["query"]: row for row in contexts}
        for rag in LLMS:
            name = f"{rag}__{entry['id']}"
            path = out / f"{name}.json"
            rows, meta = complete_file(path, gold, "generation")
            signature = meta["signature"]
            if signature["model"] != LLMS[rag] or signature["rag_id"] != rag:
                raise ValueError("A generation file belongs to a different LLM.")
            if signature["context_sha256"] != file_hash(context_path):
                raise ValueError("Generation and shared context hashes disagree.")
            if rag_report["runs"][name]["sha256"] != file_hash(path):
                raise ValueError(f"Stale RAG evaluation: {name}")
            for row in rows:
                context = contexts_by_query[row["query"]]
                if row["prompt"] != context["prompt"] or row["prompt"] != make_prompt(row["query"], context["context"]):
                    raise ValueError("Two LLMs did not receive the same shared context prompt.")
                if not isinstance(row["model_answer"], str):
                    raise ValueError("model_answer must be a string.")
                if row["input_tokens"] != context["input_tokens"][rag]:
                    raise ValueError("Input token counts differ from context preparation.")
                if row["input_tokens"] + signature["max_new_tokens"] > signature["context_limit"]:
                    raise ValueError("A generation prompt exceeded its context limit.")
    if submission:
        if staging:
            raise ValueError("A five-query staging run is not a full submission.")
        missing = [name for name in ("report.pdf", "ai-declaration.pdf", "my-requirements.txt")
                   if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0]
        if missing:
            raise ValueError("Submission artifacts still missing: " + ", ".join(missing))
    return len(gold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()
    count = validate(args.output_dir or default_output(args.staging), args.data_dir,
                     args.staging, args.submission)
    print(f"PASS: 4 retrieval runs and 8 RAG runs each contain all {count} queries.")
    print("File checks do not certify report quality, declarations, or the GPU runtime limit.")


if __name__ == "__main__":
    main()
