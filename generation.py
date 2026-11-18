"""Greedy RAG generation with model-native templates and verified checkpoints."""
import argparse
import copy
import time
from pathlib import Path

from common import (
    RunStore, align_predictions, environment, file_hash, load_queries, now,
    object_hash, read_json,
)
from config import LLMS, ROOT, dataset_paths, default_output
from context import chat_ids, make_prompt, tokenizer_signature
from retrieval import resolve_device


def verify_context_rows(rows, gold, signature, rag_id, tokenizer):
    pairs = align_predictions(rows, gold)
    budget = signature["context_limit"] - signature["max_new_tokens"]
    for row, _ in pairs:
        if row.get("status") != "ok":
            raise ValueError("Incomplete context preparation; resume prepare_contexts.py first.")
        if row.get("prompt") != make_prompt(row["query"], row["context"]):
            raise ValueError("Prompt and retrieved context disagree.")
        if row["context_sha256"] != object_hash(row["context"]):
            raise ValueError("Context hash mismatch.")
        size = len(chat_ids(tokenizer, row["prompt"]))
        if size != row["input_tokens"][rag_id] or size > budget:
            raise ValueError("Saved and actual chat token counts disagree or exceed the budget.")
    return pairs


def main(rag_id):
    parser = argparse.ArgumentParser(description=f"{rag_id}: {LLMS[rag_id]}")
    parser.add_argument("--contexts", nargs="+", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--quantization", choices=("none", "4bit"), default="none")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every <= 0:
        parser.error("checkpoint-every must be positive.")
    out = args.output_dir or default_output(args.staging)
    corpus_path, query_path = dataset_paths(args.data_dir, args.staging)
    gold = load_queries(query_path)
    context_runs = []
    for path in args.contexts:
        meta = read_json(path.with_suffix(".meta.json"))
        signature = meta["signature"]
        if meta.get("status") != "complete" or signature.get("stage") != "contexts":
            raise ValueError("A context input is not a complete preparation run.")
        if signature["staging"] != args.staging or signature["queries_sha256"] != file_hash(query_path) or signature["corpus_sha256"] != file_hash(corpus_path):
            raise ValueError("Context and generation datasets differ.")
        if signature["llms"][rag_id]["model"] != LLMS[rag_id]:
            raise ValueError("Context tokenizer belongs to a different LLM.")
        context_runs.append((path, signature, read_json(path)))
    retriever_ids = [signature["retriever_id"] for _, signature, _ in context_runs]
    if len(set(retriever_ids)) != len(retriever_ids):
        raise ValueError("Duplicate retriever context inputs.")
    model_spec = context_runs[0][1]["llms"][rag_id]
    if any(s["llms"][rag_id] != model_spec for _, s, _ in context_runs):
        raise ValueError("All inputs must use the same tokenizer/model revision.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = resolve_device(torch, args.device)
    if args.quantization == "4bit" and not device.startswith("cuda"):
        raise ValueError("This implementation supports bitsandbytes 4-bit loading on CUDA only.")
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.startswith("cuda") else "float32"
    if dtype_name == "bfloat16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support bfloat16.")
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    revision = model_spec["resolved_revision"] or model_spec["requested_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        LLMS[rag_id], revision=revision, local_files_only=args.local_files_only)
    if tokenizer_signature(tokenizer) != model_spec["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed since context preparation; prepare contexts again.")
    verified = [(path, signature, verify_context_rows(rows, gold, signature, rag_id, tokenizer))
                for path, signature, rows in context_runs]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_started = time.monotonic()
    kwargs = {
        "revision": revision, "local_files_only": args.local_files_only,
        "torch_dtype": dtype, "device_map": {"": device},
    }
    if args.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(LLMS[rag_id], **kwargs).eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_started
    actual_revision = getattr(model.config, "_commit_hash", None)
    if model_spec["resolved_revision"] and actual_revision != model_spec["resolved_revision"]:
        raise ValueError("Model revision differs from the context preparation revision.")
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.num_return_sequences = 1
    # Reset sampling defaults inherited from model repositories.
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.pad_token_id = tokenizer.pad_token_id
    model_limit = getattr(model.config, "max_position_embeddings", None)
    for context_path, context_config, pairs in verified:
        if not isinstance(model_limit, int) or context_config["context_limit"] > model_limit:
            raise ValueError("Prepared context limit exceeds the loaded model's position range.")
        signature = {
            "stage": "generation", "rag_id": rag_id, "model": LLMS[rag_id],
            "resolved_revision": actual_revision, "tokenizer_sha256": model_spec["tokenizer_sha256"],
            "retriever_id": context_config["retriever_id"], "context_sha256": file_hash(context_path),
            "queries_sha256": file_hash(query_path), "corpus_sha256": file_hash(corpus_path),
            "staging": args.staging, "context_limit": context_config["context_limit"],
            "max_new_tokens": context_config["max_new_tokens"],
            "decoding": "greedy", "num_beams": 1, "seed": args.seed,
            "device": device, "dtype": dtype_name, "quantization": args.quantization,
            "implementation_sha256": object_hash({name: file_hash(ROOT / name) for name in
                ("generation.py", "context.py", "common.py", "config.py", "retrieval.py")}),
            "environment": environment(),
        }
        output = out / f"{rag_id}__{context_config['retriever_id']}.json"
        store = RunStore(output, signature, [row["query"] for row in gold], args.resume)
        if all(store.done(row["query"]) for row in gold):
            store.complete()
            print(f"{output.name}: complete verified checkpoint; nothing to generate.")
            continue
        started = time.monotonic()
        attempt = {"started_at": now(), "resumed_count": store.meta["success_count"],
                   "shared_model_load_seconds": load_seconds}
        current_row = None
        try:
            for row, item in pairs:
                if store.done(item["query"]):
                    continue
                current_row = row
                ids = chat_ids(tokenizer, row["prompt"])
                if len(ids) + signature["max_new_tokens"] > signature["context_limit"]:
                    raise ValueError("Generation input exceeds its verified context budget.")
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                query_started = time.monotonic()
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.inference_mode():
                    sequences = model.generate(
                        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        generation_config=generation_config,
                        max_new_tokens=signature["max_new_tokens"],
                        return_dict_in_generate=False, output_scores=False, use_cache=True,
                    )
                generated_ids = sequences[0, len(ids):].detach().cpu().tolist()
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                eos = generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                hit_limit = len(generated_ids) >= signature["max_new_tokens"] and not (
                    generated_ids and generated_ids[-1] in eos_ids)
                store.put({
                    "query": item["query"], "question_type": item["question_type"],
                    "gold_answer": item["answer"], "prompt": row["prompt"],
                    "context_sha256": row["context_sha256"], "model_answer": answer,
                    "status": "ok", "input_tokens": len(ids),
                    "generated_tokens": len(generated_ids), "hit_output_limit": hit_limit,
                    "context_truncated": row["truncated"],
                    "seconds": time.monotonic() - query_started,
                })
                current_row = None
                if len(store.rows) % args.checkpoint_every == 0:
                    store.save()
                    print(f"{output.stem}: {store.meta['success_count']}/{len(gold)}", flush=True)
            attempt.update({"ended_at": now(), "seconds": time.monotonic() - started,
                            "status": "complete"})
            store.complete(attempts=store.meta.get("attempts", []) + [attempt],
                           latest_attempt_seconds=attempt["seconds"],
                           output_limit_count=sum(r["hit_output_limit"] for r in store.rows.values()))
        except BaseException as error:
            if current_row is not None:
                store.put({"query": current_row["query"], "question_type": current_row["question_type"],
                           "gold_answer": current_row["answer"], "status": "error",
                           "error": f"{type(error).__name__}: {error}"})
            attempt.update({"ended_at": now(), "seconds": time.monotonic() - started,
                            "status": "failed", "error": f"{type(error).__name__}: {error}"})
            store.save("failed", attempts=store.meta.get("attempts", []) + [attempt])
            raise
        print(f"Saved {output}; generation attempt {attempt['seconds']:.1f}s (shared load {load_seconds:.1f}s).")

