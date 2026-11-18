"""Draw three figures from actual evaluated outputs, never placeholder scores."""
import argparse
from pathlib import Path

from common import file_hash, read_json, resolve_recorded_path
from config import LLMS, default_output


def check_report(report):
    for name, run in report["runs"].items():
        if file_hash(resolve_recorded_path(run["file"])) != run["sha256"]:
            raise ValueError(f"Output changed after evaluation: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--staging", action="store_true")
    args = parser.parse_args()
    out = args.output_dir or default_output(args.staging)
    figures = args.figures_dir or out / "figures"
    retrieval = read_json(out / "metrics_retrieval.json")
    rag = read_json(out / "metrics_rag.json")
    selection = read_json(out / "selection.json")
    if retrieval["staging"] != args.staging or rag["staging"] != args.staging:
        raise ValueError("Staging and full-data reports must not be mixed.")
    check_report(retrieval)
    check_report(rag)
    selected = [entry["id"] for entry in selection["selected"]]
    systems = ["rankerA", "rankerB", "rankerC", "rerankerA"]
    expected = {f"{model}__{ranker}" for model in LLMS for ranker in selected}
    if set(retrieval["runs"]) != set(systems) or set(rag["runs"]) != expected:
        raise ValueError("Complete all four retrieval and eight RAG runs before plotting.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    label = "STAGING / five queries — " if args.staging else ""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("P@10", "FactRecall@10", "MAP@10", "FullEvidenceRecall@10")):
        values = [retrieval["runs"][s]["groups"]["overall"][metric] for s in systems]
        bars = ax.bar(systems, values, color=["#5b6eae" if s not in selected else "#00897b" for s in systems])
        ax.set_ylim(0, 1.1)
        ax.set_title(metric)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    fig.suptitle(label + "Retrieval quality (non-null queries; teal = selected)")
    fig.savefig(figures / "01_retrieval_comparison.png", dpi=200)
    plt.close(fig)

    names = [f"{model}__{ranker}" for ranker in selected for model in LLMS]
    metrics = ["Precision", "Recall", "F1", "ExactMatch", "ROUGE-L-F1", "AnswerContainment", "RefusalDecisionAccuracy"]
    matrix = np.array([[rag["runs"][name]["groups"]["overall"][metric] for metric in metrics] for name in names])
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(metrics)), ["Token P", "Token R", "Token F1", "EM", "ROUGE-L", "Containment", "Refusal acc."], rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > .6 else "black")
    ax.set_title(label + "Answer quality (macro mean over all queries)")
    fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    fig.savefig(figures / "02_rag_metrics_heatmap.png", dpi=200)
    plt.close(fig)

    types = ["inference_query", "comparison_query", "temporal_query", "null_query"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    for ax, ranker in zip(axes, selected):
        x = np.arange(len(types))
        for index, model in enumerate(LLMS):
            groups = rag["runs"][f"{model}__{ranker}"]["groups"]
            values = [groups[t]["F1"] if groups[t]["n"] else np.nan for t in types]
            ax.bar(x + (index - 1.5) * .19, values, .19, label=model)
        ax.set_xticks(x, ["Inference", "Comparison", "Temporal", "Null"])
        ax.set_ylim(0, 1.05)
        ax.set_title(ranker)
        ax.set_ylabel("Macro token F1")
    axes[-1].legend(ncol=2, fontsize=8)
    fig.suptitle(label + "Answer quality by question type (missing groups are not zero)")
    fig.savefig(figures / "03_f1_by_question_type.png", dpi=200)
    plt.close(fig)
    print("Saved three figures to", figures)
    print("Figures require your own interpretation in the report; staging is not full evaluation.")


if __name__ == "__main__":
    main()
