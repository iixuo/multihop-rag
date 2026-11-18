"""Prepare identical, jointly budgeted context text for all four LLMs."""
import argparse
import time
from pathlib import Path

from common import (
    RunStore, align_predictions, copied_labels, environment, file_hash, load_queries,
    now, object_hash, read_json, resolve_recorded_path, validate_retrieval,
)
from config import (
    CONTEXT_LIMIT, LLMS, MAX_NEW_TOKENS, PREFIX, ROOT, dataset_paths, default_output,
)
from context import pack_context, tokenizer_signature


def load_selection(path, corpus_path, query_path, staging):
    selection = read_json(path)
    if selection["staging"] != staging:
        raise ValueError("Staging and full-data selections must not be mixed.")
    if selection["queries_sha256"] != file_hash(query_path) or selection["corpus_sha256"] != file_hash(corpus_path):
        raise ValueError("The selection refers to a different dataset.")
    if len(selection["selected"]) != 2 or len({e["id"] for e in selection["selected"]}) != 2:
        raise ValueError("Exactly two distinct selected retrievers are required.")
    for entry in selection["selected"]:
        source = resolve_recorded_path(entry["file"])
        if source.stem != entry["id"] or file_hash(source) != entry["sha256"]:
            raise ValueError("A selected retrieval output changed after selection.")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--context-limit", type=int, default=CONTEXT_LIMIT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.context_limit <= args.max_new_tokens or args.max_new_tokens <= 0:
        parser.error("Require context-limit > max-new-tokens > 0.")
    started = time.monotonic()
    out = args.output_dir or default_output(args.staging)
    corpus_path, query_path = dataset_paths(args.data_dir, args.staging)
    gold = load_queries(query_path)
    selection = load_selection(args.selection or out / "selection.json", corpus_path,
                               query_path, args.staging)
    # Only small tokenizer/config files are loaded here, not four sets of weights.
    from transformers import AutoConfig, AutoTokenizer

    tokenizers, model_specs, limits = {}, {}, [args.context_limit]
    for model_id, model_name in LLMS.items():
        config = AutoConfig.from_pretrained(model_name, revision=args.revision,
                                           local_files_only=args.local_files_only)
        revision = getattr(config, "_commit_hash", None) or args.revision
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision,
                                                 local_files_only=args.local_files_only)
        if not tokenizer.chat_template:
            raise ValueError(f"{model_name} has no chat template.")
        model_limit = getattr(config, "max_position_embeddings", None)
        if not isinstance(model_limit, int) or model_limit <= 0:
            raise ValueError(f"Cannot determine the context window: {model_name}")
        limits.append(model_limit)
        token_limit = tokenizer.model_max_length
        if isinstance(token_limit, int) and 0 < token_limit < 1_000_000:
            limits.append(token_limit)
        tokenizers[model_id] = tokenizer
        model_specs[model_id] = {
            "model": model_name, "requested_revision": args.revision,
            "resolved_revision": getattr(config, "_commit_hash", None),
            "tokenizer_sha256": tokenizer_signature(tokenizer),
            "model_context_limit": model_limit,
        }
        print(f"Loaded tokenizer: {model_id} ({model_name})", flush=True)
    limit = min(limits)
    if limit <= args.max_new_tokens:
        raise ValueError("No input budget remains after applying model context limits.")
    for entry in selection["selected"]:
        input_path = resolve_recorded_path(entry["file"])
        pairs = align_predictions(read_json(input_path), gold)
        source_meta = read_json(input_path.with_suffix(".meta.json"))
        if source_meta.get("status") != "complete":
            raise ValueError("Finish the selected retrieval run first.")
        signature = {
            "stage": "contexts", "retriever_id": entry["id"],
            "retrieval_sha256": file_hash(input_path),
            "corpus_sha256": file_hash(corpus_path), "queries_sha256": file_hash(query_path),
            "llms": model_specs, "requested_context_limit": args.context_limit,
            "context_limit": limit, "max_new_tokens": args.max_new_tokens,
            "prefix": PREFIX, "staging": args.staging,
            "implementation_sha256": object_hash({name: file_hash(ROOT / name) for name in
                ("prepare_contexts.py", "context.py", "common.py", "config.py")}),
            "environment": environment(),
        }
        store = RunStore(out / f"{entry['id']}.context.json", signature,
                         [row["query"] for row in gold], args.resume)
        if all(store.done(row["query"]) for row in gold):
            store.complete()
            print(f"{entry['id']}: contexts already complete.")
            continue
        attempt = {"started_at": now(), "resumed_count": store.meta["success_count"]}
        attempt_started = time.monotonic()
        try:
            for prediction, item in pairs:
                if store.done(item["query"]):
                    continue
                validate_retrieval(prediction)
                packed = pack_context(item["query"], prediction["retrieval_list"], tokenizers,
                                      limit, args.max_new_tokens)
                # Gold labels are stored for auditing, never passed to pack_context.
                store.put({**copied_labels(item), **packed, "status": "ok"})
                if len(store.rows) % 25 == 0:
                    store.save()
                    print(f"{entry['id']}: packed {len(store.rows)}/{len(gold)}", flush=True)
            attempt.update({"ended_at": now(), "seconds": time.monotonic() - attempt_started,
                            "status": "complete"})
            truncated = sum(row["truncated"] for row in store.rows.values())
            store.complete(attempts=store.meta.get("attempts", []) + [attempt],
                           truncated_queries=truncated, total_preparation_seconds=time.monotonic() - started)
        except BaseException as error:
            attempt.update({"ended_at": now(), "seconds": time.monotonic() - attempt_started,
                            "status": "failed", "error": f"{type(error).__name__}: {error}"})
            store.save("failed", attempts=store.meta.get("attempts", []) + [attempt])
            raise
        print(f"Saved {store.path}; {truncated}/{len(gold)} contexts truncated.")


if __name__ == "__main__":
    main()
