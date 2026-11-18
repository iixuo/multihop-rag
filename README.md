# Multi-hop QA RAG: Retrieval, Reranking and Generation Benchmark

An end-to-end Retrieval-Augmented Generation pipeline for multi-hop news question answering, built and evaluated on an NVIDIA L40S GPU. It compares three dense retrievers, a cross-encoder reranker and four open-weight LLMs on 2,556 questions, with per-query metrics and significance testing.

> 多跳问答 RAG 系统：完整实现“切分 → 稠密召回 → Cross-Encoder 重排 → LLM 生成 → 评测”流水线，对比 3 种向量模型与 4 种开源大模型。

Originally built as an individual project for COMP4703 (Natural Language Processing), The University of Queensland, 2026.

## Pipeline

```
corpus ──► chunking (256 words, 20 overlap → 4,743 chunks)
       ──► dense retrieval (FAISS inner product, top-50)
             llm-embedder | BGE-M3 | E5-large-v2
       ──► cross-encoder rerank (bge-reranker-base, top-50 → top-10)
       ──► context packing (shared 4,096-token budget across 4 tokenizers)
       ──► generation: Llama-2-7B | Llama-3.1-8B | Qwen2.5-7B | Phi-4 (14B)
       ──► evaluation: EM, F1, ROUGE-L, refusal accuracy, AP@10 / MAP / MRR, McNemar
```

## Key results

**Retrieval** (2,255 answerable queries, top-10):

| Retriever | MRR@10 | MAP@10 | Hit@10 | FactRecall@10 |
|---|---|---|---|---|
| llm-embedder | 0.376 | 0.190 | 0.734 | 0.396 |
| BGE-M3 (dense only) | 0.162 | 0.083 | 0.463 | 0.230 |
| E5-large-v2 | 0.437 | 0.246 | **0.809** | 0.480 |
| BGE-M3 top-50 + bge-reranker | **0.540** | **0.314** | 0.793 | **0.481** |

Reranking improved MRR@10 by about 24% and MAP@10 by about 28% over the best single-stage retriever (E5).

**Generation** (all 2,556 queries, reranked context):

| LLM | Exact Match | F1 | False refusal rate | Refusal recall on unanswerable |
|---|---|---|---|---|
| Llama-2-7B-chat | 0.309 | 0.337 | 0.2% | 0.0% |
| **Llama-3.1-8B-Instruct** | **0.651** | **0.660** | 0.2% | 48.8% |
| Qwen2.5-7B-Instruct | 0.590 | 0.595 | 29.7% | 93.7% |
| Phi-4 (14B) | 0.171 | 0.200 | 0.0% | 0.0% |

Findings:
- Llama-3.1-8B gives the best accuracy, about 2.1× the Llama-2 baseline.
- Qwen refuses most unanswerable questions but also wrongly refuses about 30% of answerable ones, a clear precision/caution trade-off.
- Phi-4 has high token recall but very verbose answers, so EM collapses. The bottleneck is output-format compliance, not knowledge.
- E5 vs reranked context differences were tested with exact McNemar tests (Holm-corrected); see `output/report_analysis/`.

Full write-up: [`report.pdf`](report.pdf). Figures: [`output/figures/`](output/figures/).

## Code map

| File | Purpose |
|---|---|
| `config.py` | Model choices, prompt, top-k and context budget |
| `common.py`, `retrieval.py` | Chunking, embedding and FAISS retrieval |
| `rankerA/B/C.py`, `rerankerA.py` | Retriever and reranker stages |
| `context.py`, `prepare_contexts.py` | Prompt construction and token-budgeted context packing |
| `generation.py`, `ragA-D.py` | LLM generation with resume support |
| `metrics.py`, `evaluate_retrieval.py`, `evaluate_rag.py` | Metrics and scoring |
| `paired_analysis.py`, `analyze_report.py` | Per-query paired analysis and McNemar tests |
| `validate_outputs.py` | Integrity checks (SHA-256, dataset consistency) |
| `run_all.sh` | Stage runner |

## Reproducing

See [`REPRODUCE.md`](REPRODUCE.md). The course dataset and large intermediate outputs (~900 MB) are not included. Retrieval, candidate, context and prediction files are excluded; metrics, metadata, analysis and figures are kept.

## Tech stack

Python 3.10 · PyTorch · Hugging Face Transformers · Sentence-Transformers · FlagEmbedding · FAISS · NVIDIA L40S
