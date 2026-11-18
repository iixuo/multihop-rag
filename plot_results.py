"""Reproduce the three numbered report figures from saved, audited results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = {"ragA": "Llama 2", "ragB": "Llama 3.1", "ragC": "Qwen 2.5", "ragD": "Phi 4"}
COLORS = {"ragA": "#665f9d", "ragB": "#167d8d", "ragC": "#bb651f", "ragD": "#a43d50"}
RIDS = ("rankerC", "rerankerA")


def configure(output_dir=None, figures_dir=None):
    global ASSETS, ANALYSIS, CANDIDATES, RETRIEVAL, RAG
    out = Path(output_dir) if output_dir else ROOT / "output"
    ASSETS = Path(figures_dir) if figures_dir else out / "figures"
    ASSETS.mkdir(parents=True, exist_ok=True)
    ANALYSIS = json.loads((out / "report_analysis/results_analysis.json").read_text())
    CANDIDATES = json.loads((out / "report_analysis/candidate_pool_audit.json").read_text())
    RETRIEVAL = json.loads((out / "metrics_retrieval.json").read_text())
    RAG = json.loads((out / "metrics_rag.json").read_text())


def chart_retrieval_boxplot():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    systems = ("rankerA", "rankerB", "rankerC", "rerankerA")
    values = [ANALYSIS["retrieval"][key]["ap10_values"] for key in systems]
    fig, ax = plt.subplots(figsize=(7.1, 2.15))
    boxes = ax.boxplot(values, tick_labels=["LLM-Embedder", "BGE-M3", "E5", "BGE + rerank"],
                       widths=.48, whis=1.5, showmeans=True, patch_artist=True,
                       medianprops={"color": "#a04c00", "linewidth": 1.6},
                       meanprops={"marker": "D", "markerfacecolor": "#16546b", "markeredgecolor": "white", "markersize": 5},
                       flierprops={"marker": ".", "markersize": 2, "alpha": .25, "markeredgecolor": "#46505b"})
    for box in boxes["boxes"]:
        box.set_facecolor("#dce9ef")
    ax.set_ylim(-.05, 1.08)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel("Per-query AP@10", fontsize=10)
    ax.grid(axis="y", alpha=.15)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="#a04c00", label="Median"),
                       Line2D([0], [0], marker="D", linestyle="none", color="#16546b", label="Mean")],
              loc="upper center", bbox_to_anchor=(.5, 1.22), frameon=False, fontsize=10, ncol=2)
    fig.subplots_adjust(left=.10, right=.98, top=.83, bottom=.20)
    path = ASSETS / "retrieval_ap10_boxplot.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def chart_type_heatmap():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    types = ("inference_query", "comparison_query", "temporal_query", "null_query")
    names = [f"{m}__{r}" for m in MODELS for r in RIDS]
    values = np.array([[RAG["runs"][n]["groups"][t]["F1"] for t in types] for n in names])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(4), ["Inference\nn=816", "Comparison\nn=856", "Temporal\nn=583", "Null\nn=301"])
    ax.tick_params(length=0)
    labels = [MODELS[n.split("__")[0]] + (" / E5" if n.endswith("rankerC") else " / RR") for n in names]
    ax.set_yticks(range(8), labels, fontsize=9)
    for i in range(8):
        for j in range(4):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] > .59 else "black")
    for y in (1.5, 3.5, 5.5):
        ax.axhline(y, color="white", lw=1.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.03)
    colorbar.set_label("Macro token F1", fontsize=9)
    fig.subplots_adjust(left=.23, bottom=.18, right=.93, top=.98)
    path = ASSETS / "question_type_heatmap.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def chart_tradeoffs():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    for m in MODELS:
        keys = [f"{m}__{r}" for r in RIDS]
        x = [ANALYSIS["generation"][key]["seconds"] / 60 for key in keys]
        y = [RAG["runs"][key]["groups"]["overall"]["ExactMatch"] for key in keys]
        axes[0].plot(x, y, color=COLORS[m], lw=1)
        for i, key in enumerate(keys):
            axes[0].scatter(x[i], y[i], color=COLORS[m], marker="o" if i == 0 else "^", s=40)
        middle_x, middle_y = float(np.mean(x)), float(np.mean(y))
        offsets = {"ragA": (6, 4), "ragB": (6, 5), "ragC": (6, -16), "ragD": (-49, 8)}
        axes[0].annotate(MODELS[m], (middle_x, middle_y), xytext=offsets[m], textcoords="offset points", fontsize=8)
        fr = [RAG["runs"][key]["groups"]["overall"]["FalseRefusalRate"] for key in keys]
        nr = [RAG["runs"][key]["groups"]["overall"]["NullRefusalRecall"] for key in keys]
        axes[1].plot(fr, nr, color=COLORS[m], lw=1)
        for i in range(2):
            axes[1].scatter(fr[i], nr[i], color=COLORS[m], marker="o" if i == 0 else "^", s=40)
        if m in ("ragB", "ragC"):
            axes[1].annotate(MODELS[m], (np.mean(fr), np.mean(nr)), xytext=(7, -7 if m=="ragC" else 4),
                             textcoords="offset points", fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xticks([15, 30, 60, 120], ["15", "30", "60", "120"])
    from matplotlib.ticker import NullLocator
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_xlim(12, 142)
    axes[0].set_ylim(.1, .74)
    axes[0].set_xlabel("Generation and saving time (min, log scale)", fontsize=9)
    axes[0].set_ylabel("Exact match", fontsize=9)
    axes[0].set_title("A  Quality and processing cost", fontsize=10, loc="left", pad=10)
    axes[0].grid(alpha=.16)
    axes[1].set_xlim(-.035, .43)
    axes[1].set_ylim(-.08, 1.08)
    axes[1].set_xlabel("False refusal on non-null queries", fontsize=9)
    axes[1].set_ylabel("Null refusal recall", fontsize=9)
    axes[1].set_title("B  Strict refusal tradeoff", fontsize=10, loc="left", pad=10)
    axes[1].annotate("Llama 2 and Phi 4\nnear strict zero", (.001, 0), xytext=(.075, .12),
                     arrowprops={"arrowstyle": "->", "lw": .7}, fontsize=8)
    axes[1].grid(alpha=.16)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="none", color="#333333", label="E5"),
               Line2D([0], [0], marker="^", linestyle="none", color="#333333", label="RR")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, -.005))
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.25, wspace=.34)
    path = ASSETS / "quality_cost_refusal_tradeoffs.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args()
    configure(args.output_dir, args.figures_dir)
    for chart in (chart_retrieval_boxplot, chart_type_heatmap, chart_tradeoffs):
        print(chart())


if __name__ == "__main__":
    main()
