"""Seven retrieval and seven RAG metrics, with no ML dependencies."""
import re
from collections import Counter, defaultdict

from common import format_chunk

RETRIEVAL_METRICS = (
    "P@4", "P@10", "FactRecall@10", "MRR@10", "MAP@10",
    "Hit@10", "FullEvidenceRecall@10",
)
RAG_METRICS = (
    "Precision", "Recall", "F1", "ExactMatch", "ROUGE-L-F1",
    "AnswerContainment", "RefusalDecisionAccuracy",
)


def fact_normalize(text):
    # Preserve the exact matching convention of the supplied evaluator.
    return text.replace(" ", "").replace("\n", "")


def answer_tokens(text):
    return re.sub(r"[^A-Za-z0-9 ]+", "", text.lower()).split()


def answer_normalize(text):
    return " ".join(answer_tokens(text))


def harmonic(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def lcs_length(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if token == other
                           else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def contains_tokens(predicted, gold):
    return bool(gold) and any(predicted[i:i + len(gold)] == gold
                              for i in range(len(predicted) - len(gold) + 1))


def score_answer(predicted, gold, question_type):
    if not isinstance(predicted, str) or not isinstance(gold, str):
        raise ValueError("Predicted and gold answers must be strings.")
    p, g = answer_tokens(predicted), answer_tokens(gold)
    if not g:
        raise ValueError("Gold answer is empty after normalization.")
    overlap = sum((Counter(p) & Counter(g)).values())
    precision, recall = (overlap / len(p) if p else 0.0), overlap / len(g)
    longest = lcs_length(p, g)
    rouge_p = longest / len(p) if p else 0.0
    rouge_r = longest / len(g)
    abstained = " ".join(p) == "insufficient information"
    is_null = question_type == "null_query"
    return {
        "Precision": precision, "Recall": recall, "F1": harmonic(precision, recall),
        "ExactMatch": float(p == g), "ROUGE-L-F1": harmonic(rouge_p, rouge_r),
        "AnswerContainment": float(contains_tokens(p, g)),
        "RefusalDecisionAccuracy": float(abstained == is_null),
        "null_refusal": float(abstained) if is_null else None,
        "false_refusal": float(abstained) if not is_null else None,
        "yes_no_accuracy": float(p == g) if g in (["yes"], ["no"]) else None,
        "predicted_tokens": len(p), "gold_tokens": len(g),
    }


def average_precision_at(flags, relevant_count, k=10):
    """AP@k with denominator min(k, R), using the common pool's binary qrels.

    R counts all literally matched relevant chunks, not just retrieved hits.
    These qrels are reproducible proxies, not exhaustive human judgments.
    """
    if k < 1 or relevant_count < 0:
        raise ValueError("k must be positive and relevant_count non-negative.")
    if relevant_count == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, relevant in enumerate(flags[:k], 1):
        if relevant not in (0, 1):
            raise ValueError("AP requires binary relevance labels.")
        if relevant:
            hits += 1
            total += hits / rank
    if hits > relevant_count:
        raise ValueError("Retrieved relevant count exceeds the common qrels.")
    return total / min(k, relevant_count)


class RelevancePool:
    """Compute shared literal-match qrels; AP's denominator is run-independent."""
    def __init__(self, chunks, gold):
        self.texts = [fact_normalize(format_chunk(c)) for c in chunks]
        self.by_id = {c["chunk_id"]: format_chunk(c) for c in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError("Chunk IDs are not unique.")
        self.valid_texts = set(self.by_id.values())
        facts = {fact_normalize(e["fact"]) for row in gold for e in row["evidence_list"]}
        self.hits = {fact: {i for i, text in enumerate(self.texts) if fact in text}
                     for fact in facts}

    def score(self, retrieved, evidence):
        facts = set(fact_normalize(e["fact"]) for e in evidence)
        if not facts:
            raise ValueError("Use retrieval metrics only for non-null queries.")
        texts = [fact_normalize(item["text"]) for item in retrieved[:10]]
        flags = [int(any(fact in text for fact in facts)) for text in texts]
        covered = {fact for fact in facts if any(fact in text for text in texts)}
        relevant = set().union(*(self.hits[fact] for fact in facts))
        first = next((i + 1 for i, flag in enumerate(flags) if flag), None)
        available = sum(bool(self.hits[fact]) for fact in facts)
        return {
            "P@4": sum(flags[:4]) / 4, "P@10": sum(flags) / 10,
            "FactRecall@10": len(covered) / len(facts),
            "MRR@10": 1 / first if first else 0.0,
            "AP@10": average_precision_at(flags, len(relevant)),
            "Hit@10": float(bool(covered)),
            "FullEvidenceRecall@10": float(covered == facts),
            "gold_facts": len(facts), "covered_facts": len(covered),
            "pool_relevant_chunks": len(relevant),
            "attainable_fact_recall": available / len(facts),
            "all_facts_attainable": float(available == len(facts)),
        }


def aggregate(rows, names, rag=False):
    if not rows:
        result = {"n": 0, **{name: None for name in names}}
    else:
        result = {"n": len(rows), **{
            name: sum(row["AP@10" if name == "MAP@10" else name] for row in rows) / len(rows)
            for name in names}}
    if rag:
        for source, target in (
            ("null_refusal", "NullRefusalRecall"),
            ("false_refusal", "FalseRefusalRate"),
            ("yes_no_accuracy", "YesNoAccuracy"),
        ):
            applicable = [row[source] for row in rows if row[source] is not None]
            result[target] = sum(applicable) / len(applicable) if applicable else None
            result[f"{target}_n"] = len(applicable)
    return result


def grouped_scores(details, names, rag=False):
    groups = {"overall": aggregate(details, names, rag)}
    types = ("inference_query", "comparison_query", "temporal_query", "null_query")
    for question_type in types:
        groups[question_type] = aggregate(
            [row for row in details if row["question_type"] == question_type], names, rag)
    if rag:
        groups["non_null"] = aggregate(
            [row for row in details if row["question_type"] != "null_query"], names, rag)
    return groups
