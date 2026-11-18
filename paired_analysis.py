"""Query-paired diagnostics using saved predictions and exact-match correctness."""
from itertools import combinations
from math import comb

from metrics import answer_normalize, fact_normalize

MODELS = ("ragA", "ragB", "ragC", "ragD")
TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")


def exact_mcnemar(only_a, only_b):
    """Two-sided exact binomial McNemar test, without a chi-square approximation."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, j) for j in range(min(only_a, only_b) + 1))
    return min(1.0, (2 * tail) / (2 ** n))


def contingency(left, right, queries):
    counts = dict(n=len(queries), both_correct=0, only_A=0, only_B=0, both_wrong=0)
    for query in queries:
        a, b = bool(left[query]["ExactMatch"]), bool(right[query]["ExactMatch"])
        key = "both_correct" if a and b else "only_A" if a else "only_B" if b else "both_wrong"
        counts[key] += 1
    return counts


def build_paired_analysis(gold, predictions, scores, contexts):
    groups = {"overall": list(gold)}
    groups.update({kind: [q for q, g in gold.items() if g["question_type"] == kind]
                   for kind in TYPES})
    rr = {m: scores[f"{m}__rerankerA"] for m in MODELS}
    output = {"correctness": "normalized exact match; all four models use identical RR context",
              "pairwise_RR": {}, "exclusive_RR": {}, "exclusive_examples": {},
              "mcnemar_retriever_change": {},
              "mcnemar_method": "two-sided exact binomial, paired EM, four exploratory model-wise tests; Holm adjustment across four tests"}
    for a, b in combinations(MODELS, 2):
        output["pairwise_RR"][f"{a}__{b}"] = {
            "model_A": a, "model_B": b,
            "groups": {name: contingency(rr[a], rr[b], queries) for name, queries in groups.items()}}
    for model in MODELS:
        exclusive = [q for q in gold if rr[model][q]["ExactMatch"] == 1 and
                     sum(rr[m][q]["ExactMatch"] for m in MODELS) == 1]
        output["exclusive_RR"][model] = {
            "count": len(exclusive), "queries": exclusive,
            "by_type": {kind: sum(gold[q]["question_type"] == kind for q in exclusive) for kind in TYPES}}

        def facts_present(query):
            context = fact_normalize(contexts["rerankerA"][query]["context"])
            return [fact_normalize(e["fact"]) in context for e in gold[query]["evidence_list"]]

        # Prefer evidence-complete, short, inspectable cases; keep every exclusive ID above.
        chosen = sorted(exclusive, key=lambda q: (
            not (bool(facts_present(q)) and all(facts_present(q))), len(q), q))[:5]
        examples = []
        for q in chosen:
            answers, categories = {}, {}
            for m in MODELS:
                text = predictions[f"{m}__rerankerA"][q]["model_answer"]
                answers[m] = text
                score = rr[m][q]
                categories[m] = ("exact_correct" if score["ExactMatch"] else
                                 "canonical_refusal" if answer_normalize(text) == "insufficient information" else
                                 "contains_gold_but_not_exact" if score["AnswerContainment"] else "wrong_answer")
            examples.append({"query": q, "gold": gold[q]["answer"],
                             "type": gold[q]["question_type"], "exclusive_model": model,
                             "facts": gold[q]["evidence_list"], "facts_present": facts_present(q),
                             "context_sha256": contexts["rerankerA"][q]["context_sha256"],
                             "answers": answers, "error_categories": categories})
        output["exclusive_examples"][model] = examples
        counts = contingency(scores[f"{model}__rankerC"], rr[model], list(gold))
        output["mcnemar_retriever_change"][model] = {
            "A": "E5", "B": "RR", **counts,
            "p_exact_two_sided": exact_mcnemar(counts["only_A"], counts["only_B"])}
    ordered = sorted(output["mcnemar_retriever_change"],
                     key=lambda m: output["mcnemar_retriever_change"][m]["p_exact_two_sided"])
    running = 0.0
    for rank, m in enumerate(ordered):
        row = output["mcnemar_retriever_change"][m]
        running = max(running, min(1.0, (len(ordered) - rank) * row["p_exact_two_sided"]))
        row["p_holm_four_tests"] = running
    return output
