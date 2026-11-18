# Reproduction guide

Author: Haobo Yu

## Repository contents

- `report.pdf`: five body pages plus one references page, with three numbered figures and three tables.
- `config.py`, `ranker*.py`, `rerankerA.py`, `rag*.py` and supporting Python files: experimental code.
- `run_all.sh`: stage runner with configurable data and output paths.
- `output/`: metrics, selection, run metadata, analysis JSON and figures. Large retrieval/candidate/context/generation JSON files (~900 MB) are not included in this repository.
- `my-requirements.txt`: unchanged full `pip freeze` from the original GPU experiment.
- `requirements-runtime.txt` and `constraints-reported.txt`: portable runtime requirements and recoverable version constraints.
- `requirements-report.txt`: CPU-only plotting/report dependencies.
- `analyze_report.py`, `paired_analysis.py`, `audit_candidate_pool.py`, `plot_results.py`, `build_report.py`: report reproduction.
- `plot_baseline_figures.py`: original exploratory plotting script; its plots are not the three numbered report figures.

The original GPU predictions, candidate lists, contexts and inference configuration are retained. The 22 October revision replaces retrieval nDCG with AP/MAP, recomputes retrieval metrics and selection, and adds query-paired analysis, an AP box plot and exact McNemar tests. These are CPU analyses of existing outputs, not new model runs.

The course dataset, development validation files, staging outputs, caches and notebook checkpoints are deliberately excluded. Supply the original course data separately.

## Models (exactly as in config.py)

| ID | Hugging Face model | Role |
| --- | --- | --- |
| rankerA | BAAI/llm-embedder | Dense QA retrieval |
| rankerB | BAAI/bge-m3 | Dense-only retrieval, no query/passage prefix |
| rankerC | intfloat/e5-large-v2 | Dense retrieval with `query: ` / `passage: ` prefixes |
| rerankerA | BAAI/bge-reranker-base | Cross-encoder over rankerB top-50, retaining top-10 |
| ragA | meta-llama/Llama-2-7b-chat-hf | Answer generation |
| ragB | meta-llama/Llama-3.1-8B-Instruct | Answer generation |
| ragC | Qwen/Qwen2.5-7B-Instruct | Answer generation |
| ragD | microsoft/phi-4 | Answer generation, 14B, unquantized in the reported runs |

rankerA uses the QA instructions specified in `config.py`. BGE-M3 sparse and multi-vector modes are not enabled. Models are loaded at the resolved checkpoint commits recorded in the corresponding `*.meta.json` files during the reported runs. The selected systems remain rerankerA and rankerC, ordered by full-evidence recall, fact recall, then MAP@10. MAP-first and MRR-first ordering independently produce the same selection on these results. The Llama 2 and Llama 3.1 choices use the announced major-version exception; Llama, Qwen and Phi are three named families, not four unrelated families.

Per-query `AP@10` sums precision at the relevant ranks up to 10 and divides by `min(10, R)`, where `R` is the number of literal-match relevant chunks in the common corpus pool. If `R=0`, AP is zero. `MAP@10` in the aggregate reports is the mean of those AP scores; the two names deliberately distinguish a query score from its mean. The shared literal-match labels are proxies rather than exhaustive human relevance judgments.

## Environment and installation

Reported hardware: course NVIDIA L40S. Reported Python: 3.10.21. Key versions: PyTorch 2.14.0, Transformers 4.57.6, sentence-transformers 5.7.0 and NumPy 2.2.6. See the full freeze and metadata for the remaining details.

Use Linux with a compatible NVIDIA driver/CUDA-enabled PyTorch for a full rerun. Do not install the GPU requirements on macOS. The original freeze includes 88 local `file://` build paths from the course image, so it is an environment record, not a portable installation command. It has not been rewritten to invent missing package versions.

In a fresh Python 3.10 environment on the GPU host:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
python -m pip check
python -c "import torch, faiss, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

The constraints retain every explicit `name==version` from the original freeze; local-path entries without a recoverable version are not converted into guessed pins. This is not a complete lock of the original conda image. A fresh GPU installation and full inference rerun have not been performed for this packaging revision.

The code uses CPU `faiss.IndexFlatIP` while embeddings run on the GPU; `faiss-gpu` is retained to match the recorded installation. Do not install competing FAISS distributions together. bitsandbytes is available for an optional quantized rerun, but the reported Phi-4 result uses no quantization.

Both Llama checkpoints require appropriate access through the model provider. After accepting their terms, authenticate interactively with `huggingface-cli login`; never put an access token in the submission.

## Provide the course data

Run commands from this folder. Set `A2_DATA_DIR` to an existing directory containing the unmodified files:

```text
corpus.json
rag.json
sample-corpus.json
sample-rag.json
```

The two sample files are required only for `--staging`. Example (replace the path):

```bash
export A2_DATA_DIR="/absolute/path/to/course/data"
export A2_PYTHON=python
```

Without `A2_DATA_DIR`, the wrapper defaults to `./data`. Metadata validation rejects different datasets or a mixture of staging and full results. `corpus.json` is the document collection; gold answers/evidence in `rag.json` are used for scoring, not supplied in generation prompts.

## Check the submitted results without GPU inference

Python's standard library is sufficient for scoring, integrity checks and the two analysis scripts. No model download is needed.

```bash
python validate_outputs.py --data-dir "$A2_DATA_DIR" --submission
python analyze_report.py --data-dir "$A2_DATA_DIR"
python audit_candidate_pool.py --data-dir "$A2_DATA_DIR"
```

The analyzer recomputes all metric groups with the submitted scorers and checks them against the saved metrics and output hashes. It refreshes `output/report_analysis/results_analysis.json` and `paired_query_analysis.json`. The latter includes exclusive-correct queries under RR, all six model pairs by question type, selected cases with verbatim predictions, and two-sided exact McNemar tests for E5 versus RR. The exact test uses a standard-library binomial sum; no SciPy installation is required. Holm-adjusted p-values cover the four exploratory model-wise tests. These EM tests do not test F1 or prove equivalence; repeated entities may weaken the independent-query assumption.

The candidate audit checks BGE top-50 coverage and reranked candidate membership, using cached official pooling/module JSON at the recorded commit. Add `--refresh-model-config` to download those two small configuration files again; this does not run the model.

`bash run_all.sh evaluate` regenerates metric/detail files and the selection from existing results. Do not regenerate GPU outputs merely to inspect the report.

## Reproduce the three report figures and editable report

These commands are CPU-only. Use a separate reporting environment if necessary:

```bash
python -m pip install -r requirements-report.txt
python plot_results.py
python build_report.py --docx-output report.docx
```

`plot_results.py` creates:

1. `output/figures/retrieval_ap10_boxplot.png`
2. `output/figures/question_type_heatmap.png`
3. `output/figures/quality_cost_refusal_tradeoffs.png`

The saved analysis JSON permits figure generation without restoring the dataset. To recompute the analysis from raw results, first run the two analysis commands above.

`build_report.py` recreates the tables and figures from saved JSON and assembles five body pages followed by references on page six, following the supplied revision checklist. Body text is 11 pt; table text and captions are 10 pt. It is specific to the submitted experiment: its narrative is not automatically rewritten for new runs. Export the DOCX to PDF in Word or LibreOffice and inspect all pages; PDF pagination and fonts depend on the renderer. For a LibreOffice command-line export:

```bash
libreoffice --headless --convert-to pdf --outdir report-export report.docx
```

The supplied `report.pdf` is the checked export. No Word/LibreOffice installation is required just to read it or run the experimental code.

## Reproduce model inference

Keep the submitted `output/` unchanged. A new output directory also prevents resuming historical checkpoints under a changed environment:

```bash
export A2_OUTPUT_DIR="$PWD/output-reproduction"
export A2_DEVICE=cuda
bash run_all.sh all --staging
bash run_all.sh all
```

Or execute the full-data stages in order:

```bash
bash run_all.sh stage1
bash run_all.sh contexts
bash run_all.sh stage2
```

- `stage1`: three dense rankers, BGE cross-encoder reranking, retrieval evaluation and selection.
- `contexts`: construct shared text fitting all four tokenizers with a 4096-token total budget, reserving 64 output tokens.
- `stage2`: four LLMs over both selected contexts, followed by answer scoring.
- `evaluate`: scoring only.
- `--staging`: use sample files and `$A2_OUTPUT_DIR/staging`.

The wrapper passes the chosen data/output directory to every stage. It uses the core scripts' existing resume checks. A new freeze is saved as `$A2_OUTPUT_DIR/environment-freeze.txt`; the historical `my-requirements.txt` is not overwritten.

`A2_LOCAL_FILES_ONLY=1` disables checkpoint downloads when all required files are cached. `A2_RAG_D_QUANTIZATION=4bit` changes the experiment and must use a new output directory; it is not the reported configuration. Reproducing exact outputs may additionally require the recorded model revisions and the original GPU environment; a fresh default model download is not guaranteed to be the same revision.

## Timing boundary and interpretation

Phi-4 processed all 2556 queries in 116.27 min with rerankerA and 119.06 min with rankerC. The model-load timer was 76.55 s, incurred once before the paired runs. Charging that load to a standalone E5 run gives 120.33 min (RR: 117.55 min), but those sums are not independently measured cold-start end-to-end times. Downloads, context preparation and retrieval are outside these generation timings.

The saved results establish warm-model processing below two hours, not cold-start end-to-end compliance. The allowed boundary for the assignment's large-model runtime rule still requires tutor clarification; no approval is assumed.

BGE-M3's low top-10 score is documented rather than replaced. Its official pinned configuration uses CLS pooling; the saved top-50 pool provides substantially higher coverage. No FP32, alternate-pooling or E5-plus-reranker ablation is claimed.
