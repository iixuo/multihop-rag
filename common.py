"""JSON contracts, chunking, provenance, and resumable output management."""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import ROOT


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    """Never delete an old output before a complete replacement is ready."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def object_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_recorded_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    versions = {}
    for name in ("torch", "transformers", "sentence-transformers", "numpy",
                 "faiss-cpu", "faiss-gpu", "accelerate", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def query_map(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty JSON list.")
    result = {}
    for row in rows:
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Every row must have a non-empty query string.")
        if query in result:
            raise ValueError(f"Duplicate query: {query[:120]}")
        result[query] = row
    return result


def load_queries(path):
    rows = read_json(path)
    query_map(rows)
    allowed = {"inference_query", "comparison_query", "temporal_query", "null_query"}
    for row in rows:
        if row.get("question_type") not in allowed:
            raise ValueError(f"Unknown question_type: {row.get('question_type')}")
        if not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError("A gold answer is missing or empty.")
        evidence = row.get("evidence_list")
        if not isinstance(evidence, list):
            raise ValueError("evidence_list must be a list.")
        if row["question_type"] != "null_query" and not evidence:
            raise ValueError("A non-null query has no evidence.")
        for item in evidence:
            if not isinstance(item.get("fact"), str) or not item["fact"].strip():
                raise ValueError("An evidence fact is empty.")
    return rows


def align_predictions(predictions, gold):
    """Reject missing, extra, or mismatched queries instead of skipping them."""
    pred_map, gold_map = query_map(predictions), query_map(gold)
    missing, extra = gold_map.keys() - pred_map.keys(), pred_map.keys() - gold_map.keys()
    if missing or extra:
        raise ValueError(f"Query mismatch: missing={len(missing)}, extra={len(extra)}.")
    result = []
    for item in gold:
        prediction = pred_map[item["query"]]
        if prediction.get("question_type") != item["question_type"]:
            raise ValueError("question_type differs from the gold dataset.")
        for key in ("answer", "gold_answer"):
            if key in prediction and prediction[key] != item["answer"]:
                raise ValueError(f"{key} differs from the gold dataset.")
        result.append((prediction, item))
    return result


def chunk_records(records, chunk_size=256, chunk_overlap=20):
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
    chunks = []
    for doc_index, record in enumerate(records):
        words = record["body"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append({
                "chunk_id": f"d{doc_index}:w{start}",
                "document_index": doc_index, "start_word": start, "end_word": end,
                "title": record["title"], "source": record["source"],
                "published_at": record["published_at"],
                "url": record.get("url"), "text": " ".join(words[start:end]),
            })
            if end == len(words):
                break
            start = end - chunk_overlap
    if not chunks:
        raise ValueError("The corpus produced no chunks.")
    return chunks


def format_chunk(chunk):
    return (f"title: {chunk['title']}\nsource: {chunk['source']}\n"
            f"published_at: {chunk['published_at']}\n\n{chunk['text']}")


def validate_retrieval(row, minimum=10):
    items = row.get("retrieval_list")
    if not isinstance(items, list) or len(items) < minimum:
        raise ValueError(f"Expected at least {minimum} retrieved chunks.")
    ids = []
    for item in items:
        if not isinstance(item.get("text"), str) or not item["text"]:
            raise ValueError("Retrieved chunk text is missing.")
        if not isinstance(item.get("score"), (int, float)) or not math.isfinite(item["score"]):
            raise ValueError("Retrieval scores must be finite numbers.")
        if "chunk_id" in item:
            ids.append(item["chunk_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("A retrieval list contains a duplicate chunk ID.")


def copied_labels(row):
    return {"query": row["query"], "answer": row["answer"],
            "question_type": row["question_type"],
            "gold_list": row.get("evidence_list", row.get("gold_list", []))}


class RunStore:
    """Persist results and verify the experiment signature on every resume."""
    def __init__(self, path, signature, expected_queries, resume=False):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.signature = signature
        self.expected = list(expected_queries)
        self.expected_set = set(self.expected)
        if len(set(self.expected)) != len(self.expected):
            raise ValueError("Expected query order contains duplicates.")
        self.rows = {}
        self.meta = {
            "signature": signature, "signature_hash": object_hash(signature),
            "created_at": now(), "status": "running", "attempts": [],
            "expected_queries_sha256": object_hash(self.expected),
            "environment": environment(),
        }
        if self.path.exists() or self.meta_path.exists():
            if not resume:
                raise FileExistsError(f"{self.path} exists; use --resume after checking its configuration.")
            if not self.meta_path.exists():
                raise ValueError("Cannot resume an output without provenance metadata.")
            self.meta = read_json(self.meta_path)
            if self.meta.get("signature_hash") != object_hash(signature):
                raise ValueError(f"Resume configuration mismatch: {self.path}")
            if object_hash(self.meta.get("signature")) != self.meta["signature_hash"]:
                raise ValueError("Checkpoint metadata signature is corrupt.")
            if self.meta.get("expected_queries_sha256") != object_hash(self.expected):
                raise ValueError("Checkpoint expected query order changed.")
            existing = read_json(self.path) if self.path.exists() else []
            self.rows = query_map(existing) if existing else {}
            if self.rows.keys() - self.expected_set:
                raise ValueError("Checkpoint contains queries outside this experiment.")
        self.save()

    def done(self, query):
        return query in self.rows and self.rows[query].get("status", "ok") == "ok"

    def put(self, row):
        if row["query"] not in self.expected_set:
            raise ValueError("Cannot save an unexpected query.")
        self.rows[row["query"]] = row

    def save(self, status=None, **metadata):
        if status:
            self.meta["status"] = status
        self.meta.update(metadata)
        self.meta["updated_at"] = now()
        ordered = [self.rows[q] for q in self.expected if q in self.rows]
        self.meta["saved_count"] = len(ordered)
        self.meta["success_count"] = sum(self.done(q) for q in self.expected)
        atomic_json(self.path, ordered)
        atomic_json(self.meta_path, self.meta)

    def complete(self, **metadata):
        if not all(self.done(q) for q in self.expected):
            raise ValueError("Cannot mark a partial or failed run complete.")
        self.save("complete", **metadata)


def selection_key(run):
    overall = run["groups"]["overall"]
    return (overall["FullEvidenceRecall@10"], overall["FactRecall@10"],
            overall["MAP@10"])


def select_best(runs, count=2):
    if len(runs) < count:
        raise ValueError(f"Need at least {count} evaluated retrieval runs.")
    return sorted(runs, key=lambda key: (
        -selection_key(runs[key])[0], -selection_key(runs[key])[1],
        -selection_key(runs[key])[2], key))[:count]
