"""Fixed model choices for the experiment; comments are intentionally English."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RETRIEVERS = {
    "rankerA": {
        "model": "BAAI/llm-embedder",
        "query_prefix": "Represent this query for retrieving relevant documents: ",
        "passage_prefix": "Represent this document for retrieval: ",
    },
    "rankerB": {
        "model": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "rankerC": {
        "model": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}
RERANKER = "BAAI/bge-reranker-base"
LLMS = {
    "ragA": "meta-llama/Llama-2-7b-chat-hf",
    "ragB": "meta-llama/Llama-3.1-8B-Instruct",
    "ragC": "Qwen/Qwen2.5-7B-Instruct",
    "ragD": "microsoft/phi-4",
}
PREFIX = (
    "Below is a question followed by some context from different sources.\n"
    "Please answer the question based on the context.\n"
    "The answer to the question is a word or entity, or Yes/No when appropriate.\n"
    "If the provided information is insufficient to answer the question, "
    "respond 'Insufficient Information'.\n"
    "Answer directly without explanation."
)
SEPARATOR = "\n\n--------------\n\n"
TOP_K = 10
CONTEXT_LIMIT = 4096
MAX_NEW_TOKENS = 64


def dataset_paths(data_dir, staging=False):
    data_dir = Path(data_dir)
    prefix = "sample-" if staging else ""
    return data_dir / f"{prefix}corpus.json", data_dir / f"{prefix}rag.json"


def default_output(staging=False):
    return ROOT / "output" / "staging" if staging else ROOT / "output"
