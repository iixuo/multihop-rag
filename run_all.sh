#!/usr/bin/env bash
# Run one stage at a time, or all stages, in the actual GPU environment.
set -euo pipefail
A2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
A2_PYTHON="${A2_PYTHON:-python3}"
A2_STAGE="${1:-help}"
A2_DEVICE="${A2_DEVICE:-cuda}"
A2_RAG_D_QUANTIZATION="${A2_RAG_D_QUANTIZATION:-none}"
A2_DATA_DIR="${A2_DATA_DIR:-$A2_ROOT/data}"
A2_OUT="${A2_OUTPUT_DIR:-$A2_ROOT/output}"
A2_STAGING=0
if [[ "${2:-}" == "--staging" ]]; then
    A2_STAGING=1
    A2_OUT="$A2_OUT/staging"
elif [[ -n "${2:-}" ]]; then
    echo "Second argument must be --staging or omitted." >&2
    exit 2
fi
A2_FLAGS=(--data-dir "$A2_DATA_DIR" --output-dir "$A2_OUT")
if [[ "$A2_STAGING" == "1" ]]; then A2_FLAGS+=(--staging); fi
if [[ "${A2_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
    A2_FLAGS+=(--local-files-only)
fi

stage1() {
    for ranker in rankerA rankerB rankerC; do
        "$A2_PYTHON" "$A2_ROOT/$ranker.py" --device "$A2_DEVICE" --resume "${A2_FLAGS[@]}"
    done
    "$A2_PYTHON" "$A2_ROOT/rerankerA.py" --device "$A2_DEVICE" --resume "${A2_FLAGS[@]}"
    retrieval_evaluation
}

retrieval_evaluation() {
    A2_EVAL_FLAGS=(--data-dir "$A2_DATA_DIR" --output-dir "$A2_OUT")
    if [[ "$A2_STAGING" == "1" ]]; then A2_EVAL_FLAGS+=(--staging); fi
    "$A2_PYTHON" "$A2_ROOT/evaluate_retrieval.py" "${A2_EVAL_FLAGS[@]}"
    "$A2_PYTHON" "$A2_ROOT/select_retrievers.py" --metrics "$A2_OUT/metrics_retrieval.json" --output "$A2_OUT/selection.json"
}

contexts() {
    "$A2_PYTHON" "$A2_ROOT/prepare_contexts.py" --resume "${A2_FLAGS[@]}"
}

stage2() {
    A2_CONTEXTS=()
    # Capture the checked IDs first so reader failures stop the pipeline.
    # This also avoids process-substitution quirks in macOS Bash 3.2.
    A2_SELECTED_IDS="$("$A2_PYTHON" - "$A2_OUT/selection.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    selection = json.load(handle)
if len(selection["selected"]) != 2 or len({entry["id"] for entry in selection["selected"]}) != 2:
    raise SystemExit("Exactly two selected retrievers are required.")
for entry in selection["selected"]:
    name = entry["id"]
    if name not in {"rankerA", "rankerB", "rankerC", "rerankerA"}:
        raise SystemExit("Unknown selected retriever.")
    print(name)
PY
)"
    while IFS= read -r context_id; do
        A2_CONTEXTS+=("$A2_OUT/$context_id.context.json")
    done <<< "$A2_SELECTED_IDS"
    if [[ "${#A2_CONTEXTS[@]}" != "2" ]]; then
        echo "Could not read two selected retrievers; run stage1 first." >&2
        exit 1
    fi
    for rag in ragA ragB ragC ragD; do
        A2_QUANTIZATION="none"
        if [[ "$rag" == "ragD" ]]; then A2_QUANTIZATION="$A2_RAG_D_QUANTIZATION"; fi
        "$A2_PYTHON" "$A2_ROOT/$rag.py" --contexts "${A2_CONTEXTS[@]}" --device "$A2_DEVICE" \
            --quantization "$A2_QUANTIZATION" --resume "${A2_FLAGS[@]}"
    done
    rag_evaluation
    # Capture new runs separately; preserve the submitted GPU environment record.
    "$A2_PYTHON" -m pip freeze > "$A2_OUT/environment-freeze.txt"
}

rag_evaluation() {
    A2_EVAL_FLAGS=(--data-dir "$A2_DATA_DIR" --output-dir "$A2_OUT")
    if [[ "$A2_STAGING" == "1" ]]; then A2_EVAL_FLAGS+=(--staging); fi
    "$A2_PYTHON" "$A2_ROOT/evaluate_rag.py" "${A2_EVAL_FLAGS[@]}"
}

case "$A2_STAGE" in
    stage1) stage1 ;;
    contexts) contexts ;;
    stage2) stage2 ;;
    evaluate) retrieval_evaluation; rag_evaluation ;;
    all) stage1; contexts; stage2 ;;
    help|-h|--help)
        echo "Usage: bash run_all.sh {stage1|contexts|stage2|evaluate|all} [--staging]"
        echo "Optional environment: A2_PYTHON, A2_DATA_DIR, A2_OUTPUT_DIR, A2_DEVICE, A2_LOCAL_FILES_ONLY=1"
        echo "A2_RAG_D_QUANTIZATION=4bit changes the reported (unquantized) experiment."
        ;;
    *) echo "Unknown stage: $A2_STAGE" >&2; exit 2 ;;
esac
