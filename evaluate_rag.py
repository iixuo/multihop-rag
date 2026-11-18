"""Seven answer metrics plus refusal/yes-no diagnostics; no ML packages needed."""
import argparse
from pathlib import Path

from common import align_predictions, atomic_json, display_path, file_hash, load_queries, read_json
from config import ROOT, dataset_paths, default_output
from metrics import RAG_METRICS, grouped_scores, score_answer


def evaluate_rows(predictions, gold):
    details = []
    for prediction, item in align_predictions(predictions, gold):
        if prediction.get("status", "ok") != "ok":
            raise ValueError("A failed generation remains in this run; resume it before formal evaluation.")
        if "model_answer" not in prediction:
            raise ValueError("model_answer is missing.")
        score = score_answer(prediction["model_answer"], item["answer"], item["question_type"])
        details.append({"query": item["query"], "question_type": item["question_type"],
                        "prediction": prediction["model_answer"], "gold": item["answer"], **score})
    return {"groups": grouped_scores(details, RAG_METRICS, rag=True)}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output_dir or default_output(args.staging)
    inputs = args.inputs or sorted(
        path for path in out.glob("rag?__*.json") if not path.name.endswith(".meta.json"))
    if not inputs:
        raise FileNotFoundError("No generation outputs found; pass --inputs explicitly.")
    _, query_path = dataset_paths(args.data_dir, args.staging)
    gold = load_queries(query_path)
    report = {
        "stage": "rag", "staging": args.staging, "queries_sha256": file_hash(query_path),
        "aggregation": "per-query metrics, then macro mean; strict normalized refusal",
        "normalization": "supplied ASCII alphanumeric normalization; no stemming or article removal",
        "runs": {}, "trivial_baselines": {},
    }
    all_details = {}
    for path in inputs:
        if path.stem in report["runs"]:
            raise ValueError("Run filenames must have distinct stems.")
        run, details = evaluate_rows(read_json(path), gold)
        run.update({"file": display_path(path), "sha256": file_hash(path)})
        report["runs"][path.stem], all_details[path.stem] = run, details
        print(path.stem, run["groups"]["overall"])
    for name, answer in (("always_yes", "Yes"),
                         ("always_insufficient_information", "Insufficient Information")):
        predictions = [{"query": item["query"], "question_type": item["question_type"],
                        "model_answer": answer} for item in gold]
        report["trivial_baselines"][name] = evaluate_rows(predictions, gold)[0]
    output = args.output or out / "metrics_rag.json"
    atomic_json(output, report)
    atomic_json(output.with_name(output.stem + ".per_query.json"), all_details)
    print("Saved", output)


if __name__ == "__main__":
    main()
