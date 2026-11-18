"""Dense retrieval with explicit model prefixes, caching, and top-50 candidates."""
import argparse
import os
import tempfile
import time
from pathlib import Path

from common import (
    RunStore, atomic_json, chunk_records, copied_labels, display_path,
    environment, file_hash, format_chunk, load_queries, now, object_hash, read_json,
)
from config import RETRIEVERS, ROOT, TOP_K, dataset_paths, default_output


def resolve_device(torch, requested):
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use the course GPU or explicitly choose --device cpu.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolved_model_revision(model):
    try:
        return model[0].auto_model.config._commit_hash
    except (AttributeError, TypeError, IndexError):
        return None


def save_numpy(path, vectors, np):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".vectors.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def count_truncated(tokenizer, texts, max_length):
    count = maximum = 0
    for start in range(0, len(texts), 64):
        batch = tokenizer(texts[start:start + 64], truncation=False, padding=False,
                          return_attention_mask=False, return_token_type_ids=False)
        lengths = [len(ids) for ids in batch["input_ids"]]
        count += sum(length > max_length for length in lengths)
        maximum = max([maximum] + lengths)
    return {"count": count, "total": len(texts), "maximum_tokens": maximum,
            "max_length": max_length}


def main(ranker_id):
    parser = argparse.ArgumentParser(description=f"{ranker_id}: {RETRIEVERS[ranker_id]['model']}")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--candidate-k", type=int, default=50 if ranker_id == "rankerB" else 10)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0 or args.candidate_k < TOP_K:
        parser.error("batch-size/max-length must be positive; candidate-k must be >= 10.")
    started, started_at = time.monotonic(), now()
    out = args.output_dir or default_output(args.staging)
    corpus_path, queries_path = dataset_paths(args.data_dir, args.staging)
    gold = load_queries(queries_path)
    chunks = chunk_records(read_json(corpus_path))
    if len(chunks) < args.candidate_k:
        raise ValueError("The corpus has fewer chunks than candidate-k.")
    chunk_hash = object_hash(chunks)
    chunk_file = out / f"chunks-{chunk_hash[:16]}.json"
    atomic_json(chunk_file, chunks)
    texts = [format_chunk(chunk) for chunk in chunks]
    spec = RETRIEVERS[ranker_id]
    passage_inputs = [spec["passage_prefix"] + text for text in texts]

    # Import torch before FAISS to avoid the observed macOS OpenMP conflict.
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer

    device = resolve_device(torch, args.device)
    torch.manual_seed(0)
    model = SentenceTransformer(spec["model"], device=device, revision=args.revision,
                                local_files_only=args.local_files_only)
    model.max_seq_length = args.max_length
    model.eval()
    if device.startswith("cuda"):
        model.half()
    config = {
        "stage": "retrieval", "ranker_id": ranker_id, **spec,
        "requested_revision": args.revision,
        "resolved_revision": resolved_model_revision(model),
        "corpus_sha256": file_hash(corpus_path), "queries_sha256": file_hash(queries_path),
        "chunks_sha256": chunk_hash, "chunk_size_words": 256, "overlap_words": 20,
        "max_length": args.max_length, "candidate_k": args.candidate_k, "top_k": TOP_K,
        "device": device, "dtype": "float16" if device.startswith("cuda") else "float32",
        "batch_size": args.batch_size, "seed": 0,
        "implementation_sha256": object_hash({
            name: file_hash(ROOT / name) for name in ("retrieval.py", "common.py", "config.py")}),
        "environment": environment(),
    }
    store = RunStore(out / f"{ranker_id}.json", config, [d["query"] for d in gold], args.resume)
    candidate_store = RunStore(
        out / f"{ranker_id}.candidates.json", {**config, "kind": "candidate_pool"},
        [d["query"] for d in gold], args.resume,
    )
    if all(store.done(d["query"]) and candidate_store.done(d["query"]) for d in gold):
        store.complete()
        candidate_store.complete()
        print(f"{ranker_id}: complete verified checkpoint; nothing to retrieve.")
        return
    attempt = {"started_at": started_at, "resumed_count": store.meta["success_count"]}
    cache_config = {key: config[key] for key in (
        "model", "query_prefix", "passage_prefix", "resolved_revision", "requested_revision",
        "corpus_sha256", "chunks_sha256", "max_length", "device", "dtype")}
    cache_config["environment"] = config["environment"]
    cache_path = args.cache_dir / f"embeddings-{object_hash(cache_config)}.npy"
    cache_meta = cache_path.with_suffix(".meta.json")
    try:
        if cache_path.exists() and cache_meta.exists() and read_json(cache_meta)["config"] == cache_config:
            vectors = np.load(cache_path, allow_pickle=False)
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
                raise ValueError("Invalid cached corpus embeddings.")
            print("Loaded corpus embedding cache.")
        else:
            vectors = model.encode(passage_inputs, batch_size=args.batch_size,
                                   convert_to_numpy=True, normalize_embeddings=True,
                                   show_progress_bar=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
                raise ValueError("Invalid corpus embeddings.")
            save_numpy(cache_path, vectors, np)
            atomic_json(cache_meta, {"config": cache_config})
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        import faiss
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        pending = [row for row in gold if not (
            store.done(row["query"]) and candidate_store.done(row["query"]))]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            inputs = [spec["query_prefix"] + row["query"] for row in batch]
            query_vectors = model.encode(inputs, batch_size=args.batch_size,
                                         convert_to_numpy=True, normalize_embeddings=True)
            if not np.isfinite(query_vectors).all():
                raise ValueError("A query embedding is not finite.")
            scores, ids = index.search(np.ascontiguousarray(query_vectors, dtype=np.float32),
                                       args.candidate_k)
            for row, row_scores, row_ids in zip(batch, scores, ids):
                items = [{
                    "chunk_id": chunks[int(idx)]["chunk_id"],
                    "text": texts[int(idx)], "score": float(score),
                } for score, idx in zip(row_scores, row_ids) if idx >= 0]
                if len(items) != args.candidate_k:
                    raise ValueError("FAISS did not return the expected candidate count.")
                candidate_store.put({**copied_labels(row), "retrieval_list": items, "status": "ok"})
                store.put({**copied_labels(row), "retrieval_list": items[:TOP_K], "status": "ok"})
            store.save()
            candidate_store.save()
            print(f"{ranker_id}: {store.meta['success_count']}/{len(gold)}", flush=True)
        truncation = {
            "passages": count_truncated(model.tokenizer, passage_inputs, args.max_length),
            "queries": count_truncated(model.tokenizer, [
                spec["query_prefix"] + row["query"] for row in gold], args.max_length),
        }
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        attempt.update({"ended_at": now(), "seconds": time.monotonic() - started, "status": "complete"})
        attempts = store.meta.get("attempts", []) + [attempt]
        metadata = {"attempts": attempts, "chunk_file": display_path(chunk_file),
                    "truncation": truncation, "latest_attempt_seconds": attempt["seconds"]}
        store.complete(**metadata)
        candidate_store.complete(**metadata)
    except BaseException as error:
        attempt.update({"ended_at": now(), "seconds": time.monotonic() - started,
                        "status": "failed", "error": f"{type(error).__name__}: {error}"})
        store.save("failed", attempts=store.meta.get("attempts", []) + [attempt])
        candidate_store.save("failed")
        raise
