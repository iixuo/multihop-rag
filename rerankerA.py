"""Cross-encode rankerB's top 50 candidates and keep the best 10."""
import argparse
import time
from pathlib import Path

from common import (
    RunStore, align_predictions, copied_labels, environment, file_hash,
    load_queries, now, object_hash, read_json, validate_retrieval,
)
from config import RERANKER, ROOT, TOP_K, dataset_paths, default_output
from retrieval import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("batch-size and max-length must be positive.")
    started, started_at = time.monotonic(), now()
    out = args.output_dir or default_output(args.staging)
    input_path = args.input or out / "rankerB.candidates.json"
    corpus_path, query_path = dataset_paths(args.data_dir, args.staging)
    gold = load_queries(query_path)
    pairs = align_predictions(read_json(input_path), gold)
    candidate_meta = read_json(input_path.with_suffix(".meta.json"))
    candidate_config = candidate_meta["signature"]
    if candidate_meta.get("status") != "complete":
        raise ValueError("Finish rankerB before starting the reranker.")
    if candidate_config.get("ranker_id") != "rankerB" or candidate_config.get("candidate_k") != 50:
        raise ValueError("rerankerA requires rankerB's top-50 candidate pool.")
    if candidate_config["queries_sha256"] != file_hash(query_path) or candidate_config["corpus_sha256"] != file_hash(corpus_path):
        raise ValueError("Candidate dataset hashes differ from this experiment.")
    for prediction, _ in pairs:
        if prediction.get("status", "ok") != "ok":
            raise ValueError("An unsuccessful candidate row remains.")
        validate_retrieval(prediction, minimum=50)
        if len(prediction["retrieval_list"]) != 50:
            raise ValueError("Candidate lists must contain exactly 50 chunks.")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(torch, args.device)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        RERANKER, revision=args.revision, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER, revision=args.revision, local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()
    signature = {
        "stage": "reranking", "ranker_id": "rerankerA", "model": RERANKER,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "candidate_sha256": file_hash(input_path),
        "corpus_sha256": file_hash(corpus_path), "queries_sha256": file_hash(query_path),
        "chunks_sha256": candidate_config["chunks_sha256"], "candidate_k": 50,
        "top_k": TOP_K, "max_length": args.max_length, "batch_size": args.batch_size,
        "device": device, "dtype": str(model.dtype), "seed": 0,
        "score": "raw single-label cross-encoder logit; larger is better",
        "implementation_sha256": object_hash({name: file_hash(ROOT / name) for name in
            ("rerankerA.py", "retrieval.py", "common.py", "config.py")}),
        "environment": environment(),
    }
    store = RunStore(out / "rerankerA.json", signature, [r["query"] for r in gold], args.resume)
    if all(store.done(r["query"]) for r in gold):
        store.complete()
        print("rerankerA: complete verified checkpoint; nothing to rerank.")
        return
    attempt = {"started_at": started_at, "resumed_count": store.meta["success_count"]}
    try:
        for prediction, item in pairs:
            if store.done(item["query"]):
                continue
            candidates = prediction["retrieval_list"]
            scores = []
            query_truncated = 0
            for start in range(0, len(candidates), args.batch_size):
                texts = [c["text"] for c in candidates[start:start + args.batch_size]]
                queries = [item["query"]] * len(texts)
                raw = tokenizer(queries, texts, truncation=False, padding=False,
                                return_attention_mask=False)
                query_truncated += sum(len(ids) > args.max_length for ids in raw["input_ids"])
                inputs = tokenizer(queries, texts, truncation=True, padding=True,
                                   max_length=args.max_length, return_tensors="pt").to(device)
                with torch.inference_mode():
                    logits = model(**inputs, return_dict=True).logits
                if logits.ndim != 2 or logits.shape[1] != 1 or not torch.isfinite(logits).all():
                    raise ValueError("Expected finite single-label reranker logits.")
                scores.extend(logits[:, 0].float().cpu().tolist())
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:TOP_K]
            items = [{**candidates[i], "retriever_score": candidates[i]["score"],
                      "score": scores[i]} for i in order]
            store.put({**copied_labels(item), "retrieval_list": items, "status": "ok",
                       "truncated_pairs": query_truncated})
            store.save()
            print(f"rerankerA: {store.meta['success_count']}/{len(gold)}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        attempt.update({"ended_at": now(), "seconds": time.monotonic() - started,
                        "status": "complete"})
        store.complete(attempts=store.meta.get("attempts", []) + [attempt],
                       chunk_file=candidate_meta.get("chunk_file"),
                       truncated_pairs=sum(r["truncated_pairs"] for r in store.rows.values()),
                       latest_attempt_seconds=attempt["seconds"])
    except BaseException as error:
        attempt.update({"ended_at": now(), "seconds": time.monotonic() - started,
                        "status": "failed", "error": f"{type(error).__name__}: {error}"})
        store.save("failed", attempts=store.meta.get("attempts", []) + [attempt])
        raise


if __name__ == "__main__":
    main()
