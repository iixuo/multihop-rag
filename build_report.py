"""Build five body pages and a separate references page from audited saved results."""
import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
import plot_results as plots
from plot_results import MODELS, RIDS

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=BASE / "output")
parser.add_argument("--docx-output", type=Path, default=BASE / "report.docx")
args = parser.parse_args()
plots.configure(args.output_dir)
RETRIEVAL, RAG, ANALYSIS = plots.RETRIEVAL, plots.RAG, plots.ANALYSIS
PAIRED = json.loads((args.output_dir / "report_analysis/paired_query_analysis.json").read_text())

doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin, section.bottom_margin = Inches(.65), Inches(.65)
section.left_margin, section.right_margin = Inches(.7), Inches(.7)
section.header_distance, section.footer_distance = Inches(.25), Inches(.28)
normal = doc.styles["Normal"]
normal.font.name, normal.font.size = "Times New Roman", Pt(11)
normal.paragraph_format.line_spacing = 1.02
normal.paragraph_format.space_after = Pt(5)
for name, size in (("Title", 18), ("Subtitle", 10), ("Heading 1", 12), ("Heading 2", 11), ("Caption", 10)):
    style = doc.styles[name]
    style.font.name, style.font.size, style.font.color.rgb = "Times New Roman", Pt(size), RGBColor(0, 0, 0)
    style.paragraph_format.space_before = Pt(7 if name.startswith("Heading") else 0)
    style.paragraph_format.space_after = Pt(5)
    if name.startswith("Heading"):
        style.font.bold = True
    if name == "Caption":
        style.font.italic = False
        style.font.bold = False
        style.paragraph_format.space_before = Pt(3)
    if style.element.pPr is not None:
        border = style.element.pPr.find(qn("w:pBdr"))
        if border is not None:
            style.element.pPr.remove(border)
doc.core_properties.title = "Retrieval and Generation Tradeoffs in Multi-Hop Question Answering"
doc.core_properties.author = "Haobo Yu"
doc.core_properties.subject = "COMP4703 Assignment 2 experimental report"
footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def p(text, style=None):
    return doc.add_paragraph(text, style)


def h(text, level=1, new_page=False):
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.page_break_before = new_page
    return paragraph


def table(headers, rows, widths, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement("w:" + edge)
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            element.set(qn("w:" + key), value)
        borders.append(element)
    t._tbl.tblPr.append(borders)
    for i, values in enumerate([headers] + rows):
        cells = t.rows[0].cells if i == 0 else t.add_row().cells
        for j, (cell, text, width) in enumerate(zip(cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            pr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for side, amount in (("top", "55"), ("bottom", "55"), ("left", "60"), ("right", "60")):
                child = OxmlElement("w:" + side)
                child.set(qn("w:w"), amount)
                child.set(qn("w:type"), "dxa")
                margins.append(child)
            pr.append(margins)
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), "DFE8EE" if i == 0 else ("F6F8FA" if i % 2 == 0 else "FFFFFF"))
            pr.append(shading)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(str(text))
            run.font.name, run.font.size = "Times New Roman", Pt(10)
            run.bold = i == 0 or (i - 1) in bold_rows
            run.font.color.rgb = RGBColor(0, 0, 0)
        row_props = t.rows[i]._tr.get_or_add_trPr()
        row_props.append(OxmlElement("w:cantSplit"))
        if i == 0:
            row_props.append(OxmlElement("w:tblHeader"))
    return t


def caption(text):
    return p(text, "Caption")


def picture(path, width=7.1):
    paragraph = p("")
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Inches(width))


def ref(number, text, url):
    paragraph = p(f"[{number}] {text} ")
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.keep_together = True
    for run in paragraph.runs:
        run.font.size = Pt(10)
    relationship = doc.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "20")
    props.append(size)
    run.append(props)
    text_element = OxmlElement("w:t")
    text_element.text = url
    run.append(text_element)
    link.append(run)
    paragraph._p.append(link)


# Body page 1
p("Retrieval and Generation Tradeoffs\nin Multi-Hop Question Answering", "Title")
p("COMP4703 Assignment 2   |   Haobo Yu", "Subtitle")
h("1 Introduction")
p("Multi-hop question answering needs evidence from several sources to support a short answer. I compare four retrieval pipelines and combine two selected pipelines with four language models, giving eight complete RAG runs. The experiment examines whether better retrieval improves generation, and how refusal, answer format and runtime affect system choice. BGE-M3 plus reranking with Llama-3.1-8B-Instruct has the best observed token F1 (0.660) and exact match (0.651), but paired query analysis reveals substantial differences hidden by these averages.")
h("2 Methodology")
p("The course MultiHop-RAG collection [10] contains 609 news articles and 2556 queries: 816 inference, 856 comparison, 583 temporal and 301 null queries. Each non-null query has two to four gold facts. The five-query sample is used only for staging. Splitting bodies into 256-word chunks with 20-word overlap yields 4743 chunks; title, source and publication date are retained for evidence attribution and temporal comparisons.")
p("Three dense rankers encode queries and chunks separately, normalize their vectors and search an exact FAISS inner-product index, equivalent to cosine ranking. LLM-Embedder uses its QA instructions [1]; BGE-M3 uses dense embeddings without instructions [2]; E5-large-v2 uses query: and passage: prefixes [3]. Encoder batches are 32, the input limit is 512 tokens and the output is ten chunks. The BGE-M3 checkpoint specifies CLS pooling [11], which the loader retains. Only 23/4743 passages exceed its encoder limit. I suspect dense-only scoring and task mismatch contribute to its weak ranking; a float32 or alternative-representation ablation would test that explanation.")
p("The fourth pipeline applies BGE-reranker-base [4] to BGE-M3's top 50 candidates, retaining ten. This cross-encoder scores each query-passage pair jointly, allowing closer interaction than independently encoded vectors, at 50 pairs per query and batch size 16. Holding the candidate retriever fixed isolates the effect of reranking; E5 top-50 plus reranking was not tested. All models use pretrained checkpoints without task-specific fine-tuning.")
p("The generators are Llama-2-7b-chat [5], Llama-3.1-8B-Instruct [6], Qwen2.5-7B-Instruct [7] and Phi-4 (14B) [8]. The four model choices span three named families, with Llama 2 and Llama 3.1 counted separately under the announced major-version exception. Runs used the course NVIDIA L40S, Python 3.10.21, PyTorch 2.14.0 and Transformers 4.57.6. Retrieval uses float16; generation uses bfloat16 without quantization, greedy decoding, batch size one and seed zero.")
p("Relative to the supplied prompt, I added 'or Yes/No when appropriate' because 1343 gold answers are binary. 'Answer directly without explanation' and the canonical 'Insufficient Information' refusal were already present and are retained. All four models receive identical question and context text through their native chat templates. Packing checks every tokenizer against a 4096-token total budget, reserving 64 output tokens; it keeps ranked full chunks and optionally part of the last chunk. Gold labels never enter prompts. Packing lowers full-evidence coverage from 0.184 to 0.165 for E5 and 0.205 to 0.190 for reranking, so retrieved and presented evidence differ.")

# Body page 2
h("3 Experiments", new_page=True)
h("3.1 Retrieval evaluation", level=2)
p("Retrieval metrics macro-average the 2255 non-null queries. A chunk is binary-relevant if it contains a complete gold fact after removing spaces/newlines while preserving case. P@4 and P@10 measure early precision; MRR@10 rewards the first relevant hit. For AP@10, I sum precision at each relevant rank in the first ten positions and divide by min(10, R), where R counts all matched relevant chunks in the shared pool; AP is zero when R is zero [12]. MAP@10 is the mean of these per-query AP values. Its denominator is independent of the retrieved hits. These literal-match labels are proxies and cannot supply exhaustive human relevance judgments.")
p("FactRecall@10 measures distinct gold-fact coverage; FullEvidenceRecall@10 requires every gold fact, and Hit@10 requires any. They supplement the ranking metrics by exposing missing hops. They are not estimates of recall over all semantically relevant documents.")
retrieval_metrics = ("P@4", "P@10", "MRR@10", "MAP@10", "FactRecall@10", "Hit@10", "FullEvidenceRecall@10")
rows = []
for key, name in (("rankerA", "LLM-Embedder"), ("rankerB", "BGE-M3"), ("rankerC", "E5"), ("rerankerA", "BGE + rerank")):
    rows.append([name] + [f"{RETRIEVAL['runs'][key]['groups']['overall'][metric]:.3f}" for metric in retrieval_metrics])
table(["System", "P@4", "P@10", "MRR", "MAP", "Fact\nrecall", "Hit", "Full\nevidence"], rows,
      [1.65, .7, .7, .7, .7, .85, .7, 1.1], bold_rows=(2, 3))
caption("Table 1. Seven retrieval metrics; all cutoffs are 10 except P@4. Bold rows are the two selected systems. n = 2255 for every row.")
p("RR has better early ordering than E5: MRR is 0.540 versus 0.437 and MAP is 0.314 versus 0.246, despite slightly lower P@10 (0.122 versus 0.124). The retained selection policy prioritizes full-evidence coverage, then fact coverage, then MAP, reflecting the need for multiple hops. Both MAP-first and MRR-first orderings independently select the same two systems, RR and E5; thus this selection is not an artifact of prioritizing coverage.")
picture(plots.chart_retrieval_boxplot())
caption("Figure 1. Per-query AP@10 on the same 2255 queries. Boxes show the interquartile range, lines medians, diamonds means, and whiskers 1.5 IQR; dots are outliers.")
stats = ANALYSIS["retrieval"]
p(f"Means exceed medians for every system: LLM-Embedder {stats['rankerA']['ap10_mean']:.3f}/{stats['rankerA']['ap10_median']:.3f}, BGE-M3 {stats['rankerB']['ap10_mean']:.3f}/{stats['rankerB']['ap10_median']:.3f}, E5 {stats['rankerC']['ap10_mean']:.3f}/{stats['rankerC']['ap10_median']:.3f} and RR {stats['rerankerA']['ap10_mean']:.3f}/{stats['rerankerA']['ap10_median']:.3f}. BGE-M3 scores zero on 1210 queries. RR's higher mean still conceals 467 zero-AP queries, versus E5's 431: stronger ordering does not guarantee broader success.")
p("BGE-M3's top-50 fact coverage reaches 0.609, versus 0.230 at top-10; reranking promotes available evidence to reach 0.481. It finds all facts for 463 queries versus E5's 415. However, BGE retrieval plus reranking takes about 18.0 minutes, versus 1.3 minutes for E5. Candidate availability, ranking quality and computation therefore favor different choices.")

# Body page 3
h("3.2 Answer quality and paired changes", level=2, new_page=True)
p("Seven answer metrics macro-average all 2556 queries. Token precision, recall and per-query F1 use multiset overlap after lowercase ASCII alphanumeric normalization. Exact match (EM) compares normalized sequences; ROUGE-L F1 uses token longest-common-subsequence overlap without stemming [9]. Containment requires the full gold token sequence contiguously. Refusal decision accuracy (RDA) checks whether a canonical refusal agrees with the null label. Only newly generated text is scored; explanations are not stripped.")
rag_metrics = ("Precision", "Recall", "F1", "ExactMatch", "ROUGE-L-F1", "AnswerContainment", "RefusalDecisionAccuracy")
rows = []
for model in MODELS:
    for rid in RIDS:
        key = f"{model}__{rid}"
        name = MODELS[model] + (" / E5" if rid == "rankerC" else " / RR")
        rows.append([name] + [f"{RAG['runs'][key]['groups']['overall'][metric]:.3f}" for metric in rag_metrics])
table(["LLM / retrieval", "Token\nP", "Token\nR", "Token\nF1", "EM", "ROUGE\nL F1", "Contain.", "RDA"], rows,
      [1.55, .75, .75, .75, .75, .8, .95, .8], bold_rows=(3,))
caption("Table 2. Seven answer metrics for eight runs, each on 2556 queries. E5 = rankerC; RR = rerankerA. Bold identifies the best observed overall F1/EM combination.")
p("Llama 3.1 has the highest overall F1 with either retriever. RR increases its F1 from 0.644 to 0.660, but decreases Qwen's from 0.605 to 0.595. Query types explain part of this reversal: Qwen/E5 leads inference F1 (0.890), while Llama 3.1/RR is stronger on comparison and temporal questions. Phi-4's zero comparison/temporal EM reflects its failure to return literal Yes/No answers.")
picture(plots.chart_type_heatmap())
caption("Figure 2. Token F1 by question type. Every row uses the same 816 inference, 856 comparison, 583 temporal and 301 null queries.")
mb = PAIRED["mcnemar_retriever_change"]["ragB"]
mc = PAIRED["mcnemar_retriever_change"]["ragC"]
p(f"Paired EM tells a more cautious story than aggregate gains. Switching E5 to RR makes 252 Llama 3.1 answers correct and 216 incorrect; Qwen gains 151 and loses 179. Two-sided exact McNemar tests [13] give p = {mb['p_exact_two_sided']:.4f} and {mc['p_exact_two_sided']:.4f}, respectively. Neither change is significant at 0.05, even before correction. All four exploratory model-wise tests remain non-significant after Holm correction (adjusted p = 0.3422 each). These EM tests do not test token-F1 differences or establish equivalence.")

# Body page 4
h("3.3 Refusal format and computational cost", level=2, new_page=True)
picture(plots.chart_tradeoffs())
caption("Figure 3. A: EM versus generation-and-saving time, excluding loading and retrieval. B: strict refusal tradeoff; upper left is preferable. Circles denote E5 and triangles RR; segments pair the same model across retrievers.")
p("Qwen/RR refuses correctly on 282/301 null queries (93.7%) but also refuses 669/2255 answerable queries (29.7%); its temporal false-refusal rate is 61.1%. Llama 3.1/RR refuses only five answerable queries but handles 147 null queries correctly (48.8%). I infer that Qwen's conservative response policy trades answerable-query coverage for null-query accuracy under this prompt. Its excellent null score does not imply stronger overall question answering.")
p("RDA is sensitive to class imbalance: always answering Yes scores 0.882 because 88.2% of queries are non-null, although its EM and F1 are only 0.306. Llama 3.1/RR exceeds this baseline by 0.345 EM; Llama 2/RR by only 0.003. An always-refuse predictor obtains F1 0.118. These controls expose why high refusal accuracy alone is insufficient.")
p("Phi-4/RR obtains recall 0.572 and containment 0.529, but precision 0.188 and EM 0.171. None of its 1343 binary-gold outputs is a literal normalized Yes/No. It often embeds the short answer in an explanation; 152/301 null outputs start with Insufficient Information but continue. The same concise-answer prompt therefore constrains the four models differently. I attribute part of Phi-4's strict-score deficit to answer format, while recognizing that containing the answer does not establish correct reasoning.")
p("Phi-4 reaches the 64-token cap on 1185 RR and 1208 E5 queries (46.4% and 47.3%) and generates about 49-50 tokens on average, versus roughly three for Llama 3.1 and Qwen. RR/E5 processing times are 116.27/119.06 minutes for Phi-4, 41.37/43.54 for Llama 2, 15.30/15.48 for Llama 3.1 and 14.77/15.13 for Qwen. Its longer output contributes to cost alongside parameter count.")
p("Phi-4's warm processing phases are below 120 minutes. The shared 76.55-second load occurred before RR; charging it to standalone E5 gives 120.33 minutes (RR: 117.55). These are reconstructed totals, not measured cold starts, so the allowed timing boundary still needs clarification. Downloads and context preparation are excluded. No shorter-prompt rerun was performed.")
p("Limitations: literal matching misses paraphrases, while overlapping chunks may reward redundant evidence. Twenty of 981 distinct facts occur in no fixed chunk, affecting 127 queries. Retriever selection uses the evaluation collection; repeated entities weaken the independent-query assumption behind the exploratory tests. There is no held-out selection set, repeated-run study or causal ablation. The findings apply to these checkpoints, prompt and benchmark.")

# Body page 5
h("3.4 Query level failure analysis", level=2, new_page=True)
counts = [PAIRED["exclusive_RR"][m]["count"] for m in MODELS]
p(f"With identical RR contexts, the numbers of queries answered correctly by exactly one of the four models are Llama 2: {counts[0]}, Llama 3.1: {counts[1]}, Qwen: {counts[2]} and Phi-4: {counts[3]}. Llama 3.1 has 167 exclusive temporal successes, whereas Qwen has 135 exclusive null successes. Table 3 shows complementarity beyond the averages: on inference, Qwen alone beats Llama 3.1 on 75 queries versus 45 in the opposite direction; on temporal queries, the balance reverses to 31 versus 227.")
short = {"ragA": "L2", "ragB": "L3", "ragC": "Q", "ragD": "P"}
groups = ("overall", "inference_query", "comparison_query", "temporal_query", "null_query")
rows = []
for pair in PAIRED["pairwise_RR"].values():
    values = [short[pair["model_A"]] + "/" + short[pair["model_B"]]]
    for group in groups:
        c = pair["groups"][group]
        values.append(f"{c['both_correct']} / {c['only_A']}\n{c['only_B']} / {c['both_wrong']}")
    rows.append(values)
table(["A/B", "Overall\nn=2556", "Inference\nn=816", "Comparison\nn=856", "Temporal\nn=583", "Null\nn=301"], rows,
      [.6, 1.3, 1.3, 1.3, 1.3, 1.3])
caption("Table 3. Paired EM outcomes under RR. Each cell reads: both correct / only A correct (first line); only B correct / both wrong (second line). L2 = Llama 2; L3 = Llama 3.1; Q = Qwen; P = Phi-4. Both wrong need not mean identical answers.")
p("Three illustrative exclusive-success cases follow; complete queries, gold facts, context hashes and all predictions are saved in paired_query_analysis.json. They distinguish missing evidence, failure to use available evidence and the effect of strict answer matching.")
p("First, did Engadget report the MacBook Air discount before The Verge reported the Galaxy Buds 2 discount? RR contains both matched facts and article dates (12 October versus 7 December 2023); only Llama 3.1 answers Yes, the gold answer. Qwen refuses, Phi-4 incorrectly claims the dates are unavailable, and Llama 2 gives a contradictory No/refusal: evidence is available but not used successfully.")
p("Second, did Nike's net income decrease while shipping-industry net income increased? Both gold facts are present: Nike falls from $1.47bn to $1.45bn, while carriers earned $364bn in 2021-22 after scant profits; only Qwen answers the gold Yes. Both Llama versions answer No and Phi-4 claims insufficient information, showing that the aggregate winner can miss a supported financial comparison.")
p("Third, which establishments adjust betting lines for news, injuries and rookie information and offer welcome bonuses? Phi-4 alone returns the gold Sportsbooks; Llama 2 says Yes, Llama 3.1 names DraftKings and Qwen gives Online betting sites, a plausible near-synonym rejected by EM. Only two of four matched facts survive in RR context, so this success also shows why an exact answer cannot certify complete multi-hop evidence use.")
h("4 Conclusions")
p("Llama 3.1/RR is the best observed F1/EM configuration, while Llama 3.1/E5 is a strong lower-cost alternative and its EM difference is not statistically significant here. Qwen contributes distinct inference and null-query successes; Phi-4's verbose format makes it a poor fit for this short-answer configuration despite its size. I would prioritize evidence-aware refusal calibration and per-model output constraints, then test them on held-out queries before deploying an ensemble.")

# References are separate from the five body pages, as requested in the revision checklist.
h("References", new_page=True)
p("Model sources were recorded with the original experiments. Evaluation-method sources [12-13] were checked on 22 October 2026.", "Caption")
ref(1, "BAAI. LLM-Embedder model and QA instructions.", "https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_embedder")
ref(2, "Chen, J., et al. 2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.", "https://arxiv.org/abs/2402.03216")
ref(3, "Wang, L., et al. 2022. Text Embeddings by Weakly-Supervised Contrastive Pre-training.", "https://arxiv.org/abs/2212.03533")
ref(4, "BAAI. BGE reranker base model card and cross-encoder usage.", "https://huggingface.co/BAAI/bge-reranker-base")
ref(5, "Touvron, H., et al. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models.", "https://arxiv.org/abs/2307.09288")
ref(6, "Grattafiori, A., et al. 2024. The Llama 3 Herd of Models.", "https://arxiv.org/abs/2407.21783")
ref(7, "Qwen Team. Qwen2.5 7B Instruct model card.", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct")
ref(8, "Microsoft. Phi 4 model card, architecture and chat format.", "https://huggingface.co/microsoft/phi-4")
ref(9, "Lin, C.-Y. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. pp. 74-81.", "https://aclanthology.org/W04-1013/")
ref(10, "Tang, Y., and Yang, Y. 2024. MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries.", "https://arxiv.org/abs/2401.15391")
ref(11, "BAAI. BGE-M3 pooling configuration, recorded checkpoint 5617a9f.", "https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json")
ref(12, "Apache Spark. RankingMetrics, version 3.5.6. AP at k uses min(k, number of relevant items) as denominator.", "https://github.com/apache/spark/blob/v3.5.6/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala")
ref(13, "Statsmodels. McNemar test documentation. Exact inference uses the binomial distribution.", "https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html")

args.docx_output.parent.mkdir(parents=True, exist_ok=True)
doc.save(args.docx_output)
print("Saved", args.docx_output)
print("Paragraph words:", sum(len(par.text.split()) for par in doc.paragraphs))
print("Figures:", len(doc.inline_shapes), "Tables:", len(doc.tables))
