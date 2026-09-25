# LEGACY builder that produced the first anonymous Word version. Do not run: it would
# overwrite the current Word file. Current files are built by build_all.py / tex2icaiit_docx.py.
"""Fill the official ICAIIT Template2026_Word.docx with Paper 2."""
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, Twips
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "_ICAIIT_official_template.docx"
OUT = ROOT / "Paper2_ICAIIT2027_Refusal" / "ICAIIT2027_paper.docx"
FIG = ROOT / "Paper2_ICAIIT2027_Refusal" / "figures"

TITLE = "Calibrated Refusal for Enterprise Retrieval-Augmented Generation over Conflicting Procedures"


def set_cell(cell, text, size=15, bold=False, center=True, font="Times New Roman"):
    cell.text = ""
    p = cell.paragraphs[0]
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold


def add_p(doc, text, style="Normal", first_indent=False, after=6):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.bold = False
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    if first_indent:
        pf.first_line_indent = Cm(0.5)
    else:
        pf.first_line_indent = Cm(0)
    return p


def add_h(doc, text, style="Heading 1"):
    p = doc.add_paragraph(text, style=style)
    return p


def add_fig(doc, path, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(7.4))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(12)
    r = cap.runs[0] if cap.runs else cap.add_run(caption)
    if cap.runs:
        r = cap.runs[0]
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    r.bold = False
    r.italic = False
    return cap


def add_table(doc, caption, rows):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(12)
    cap.paragraph_format.space_after = Pt(6)
    r = cap.add_run(caption)
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    r.bold = False
    tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
    if doc.tables:
        try:
            tbl.style = doc.tables[1].style
        except Exception:
            pass
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "Times New Roman"
            run.font.size = Pt(8)
            run.bold = i == 0
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
    return tbl


def add_ref(doc, text):
    p = doc.add_paragraph(style="references")
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(8)
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    p.paragraph_format.space_after = Pt(3)
    return p


def clear_after_section_break(doc):
    body = doc.element.body
    seen_break = False
    for child in list(body):
        tag = child.tag.split("}")[-1]
        if tag == "sectPr":
            continue
        if tag == "p":
            pPr = child.find(qn("w:pPr"))
            has_sect = pPr is not None and pPr.find(qn("w:sectPr")) is not None
            if has_sect:
                seen_break = True
                continue
        if seen_break:
            body.remove(child)


def build():
    doc = Document(str(TEMPLATE))
    t0 = doc.tables[0]
    set_cell(t0.cell(0, 0), TITLE, size=15, bold=True)
    set_cell(t0.cell(1, 0), "Anonymous Author", size=11, bold=False)
    set_cell(
        t0.cell(2, 0),
        "Affiliation omitted for double-blind review",
        size=9,
        bold=False,
    )

    # keywords + abstract are the first two paragraphs after the title table
    paras = [p for p in doc.paragraphs]
    # find keyword and abstract paras by prefix
    for p in paras:
        t = p.text.strip()
        if t.startswith("Keywords"):
            p.clear()
            r = p.add_run(
                "Keywords: Retrieval-Augmented Generation, Refusal, Enterprise Search, Document Conflict, BM25"
            )
            r.font.name = "Times New Roman"
            r.font.size = Pt(9)
        elif t.startswith("Abstract"):
            p.clear()
            r = p.add_run(
                "Abstract: Workplace retrieval-augmented generation systems answer operational "
                "questions from an intranet of procedures. When that intranet contains a current "
                "rule, a superseded copy, and marketing text that disagrees with both, a ranker "
                "that returns a lexically relevant document is not yet a safe answerer. This paper "
                "treats refusal as a first-class system action. We specify three inspectable "
                "policies on top of a BM25 retriever: always answer from the rank-1 record, refuse "
                "when extracted slot values in the top-3 disagree, and prefer a record marked "
                "current when one is retrieved. The policies are tested on an 18-document "
                "constructed retailer corpus and six desk queries covering factoid lookup, "
                "conflict, staleness, missing scope, customer wording, and repair. No language "
                "model wording is sampled and no user study is invented. Always answering from "
                "rank-1 is wrong on 4 of 6 tasks, including a 30-day refund window and a 15% "
                "restocking fee that are no longer in force. Conflict-based refusal produces 0 "
                "wrong answers but over-refuses when a governing value is recoverable. Preferring "
                "the current record recovers the 14-day window and the 8% fee and still refuses "
                "an international warranty that the corpus does not license. The result is a small, "
                "reproducible refusal gate for enterprise RAG, not a new neural architecture."
            )
            r.font.name = "Times New Roman"
            r.font.size = Pt(9)

    clear_after_section_break(doc)

    add_h(doc, "1\tINTRODUCTION")
    add_p(
        doc,
        "Enterprise assistants that retrieve internal procedures and then generate an answer "
        "are now a standard applied-IT pattern [1]. The usual evaluation asks whether the "
        "generated sentence is faithful to the retrieved text. That test is incomplete when "
        "the retrieved set itself disagrees. A superseded standard operating procedure can "
        "outrank the current one because the two files repeat the same terms. Marketing copy "
        "can outrank a governing rule because it mentions the customer and the refund. In "
        "those cases a fluent answer is a wrong business action.",
    )
    add_p(
        doc,
        "This paper asks a narrower systems question than answer quality. When should a "
        "workplace RAG stack refuse to state a number? We keep the generator out of the "
        "loop on purpose. The object of study is the gate that sits on retrieved things: "
        "extract a slot, test for conflict, and either emit a value or refuse. Three "
        "policies are compared on the same ranking. The contribution is an inspectable "
        "refusal design and a corpus-level measurement, not a claim about employee trust "
        "or about large-language-model wording.",
        first_indent=True,
    )
    add_p(
        doc,
        "A running example makes the failure concrete. The current refund procedure says "
        "14 calendar days. A superseded copy still says 30. The public website still says "
        "30. A worker types what is the refund window for an unopened item. BM25 returns "
        "the superseded file first because unopened, refund, and days occur in both texts. "
        "A stack that speaks rank-1 tells the worker 30 days. That sentence is lexically "
        "grounded and operationally false. Refusal is the action that stops that sentence "
        "before a customer hears it.",
        first_indent=True,
    )
    add_p(
        doc,
        "The rest of the paper is organized as follows. Section 2 reviews retrieval-augmented "
        "generation and abstention. Section 3 describes the corpus, the slot extractor, and "
        "the three policies. Section 4 reports retrieval and refusal outcomes. Section 5 "
        "discusses implications and limits. Section 6 concludes.",
        first_indent=True,
    )

    add_h(doc, "2\tRELATED WORK")
    add_p(
        doc,
        "This section separates two lines of work that are often collapsed in enterprise "
        "demos: retrieving passages for generation, and deciding not to answer.",
    )
    add_h(doc, "2.1\tRetrieval-Augmented Generation", style="Heading 2")
    add_p(
        doc,
        "Lewis et al. introduced retrieval-augmented generation as a way to condition a "
        "parametric model on non-parametric memory [1]. Dense retrievers such as DPR [5] "
        "and fusion-in-decoder architectures [6] improved open-domain question answering "
        "on Wikipedia-scale collections. Later surveys document the spread of RAG into "
        "assistants that cite internal files [8]. Retrieval also reduces some classes of "
        "conversational hallucination [4]. Those results assume that a relevant passage is "
        "close to a correct passage. Workplace intranets violate that assumption: relevance "
        "and currency are different properties.",
    )
    add_p(
        doc,
        "BM25 remains a strong and inspectable lexical baseline [2]. We use it so that a "
        "reviewer can see why a superseded file wins: it repeats the query terms. Nothing "
        "in the refusal policies requires BM25. A dense index can be swapped without "
        "changing the slot tests.",
        first_indent=True,
    )
    add_h(doc, "2.2\tRefusal and Abstention", style="Heading 2")
    add_p(
        doc,
        "Self-RAG trains a model to retrieve on demand and to critique its own generations "
        "with reflection tokens [3]. Language models can also report uncertainty about "
        "their own answers [7]. Those methods operate on generated text. They do not, by "
        "themselves, encode the institutional fact that two retrieved procedures disagree "
        "on a refund window. Corrective and iterative RAG rewrite the query when the first "
        "set looks weak [9], [10]. Black-box retrieval wrappers and hallucination corpora "
        "make the same point from the generator side [11], [12]. That is useful when the "
        "right document is missing from the top-k. It is the wrong move when the right "
        "document and the wrong document are both present and the wrong one is ranked first.",
    )
    add_p(
        doc,
        "The gap we address is therefore small and applied. Enterprise RAG needs a gate "
        "that can refuse a number because the retrieved things conflict, not only because "
        "the generator is unsure. The gate should be readable by an auditor.",
        first_indent=True,
    )

    add_h(doc, "3\tMETHOD")
    add_p(
        doc,
        "The method has four parts: a constructed intranet, six desk queries, a lexical "
        "retriever with a slot extractor, and three refusal policies. Figure 1 shows the "
        "pipeline. The generator is omitted so that wording effects do not hide ranking "
        "and conflict.",
    )
    add_fig(
        doc,
        FIG / "fig1_refusal_pipeline.png",
        "Figure 1. Refusal gate on top of BM25 retrieval and slot extraction.",
    )

    add_h(doc, "3.1\tCorpus and Tasks", style="Heading 2")
    add_p(
        doc,
        "A live company corpus cannot be published and would mix unknown missing files "
        "into the measurement. The evaluation fixture is an 18-record English intranet "
        "for a fictional retailer, Northline Parcels. The set is written so that four "
        "situations exist as files, not only as story: a current 14-day refund rule, a "
        "superseded 30-day rule, public marketing copy that still promises 30 days, and "
        "a domestic warranty that forbids quoting an international period. Table 1 lists "
        "the records that instantiate those situations.",
    )
    add_table(
        doc,
        "Table 1. Records that instantiate conflict, staleness, and missing scope.",
        [
            ["Record", "Status", "Slot"],
            ["SOP-REFUND-UNOPENED-v3", "Current, 1 Mar 2026", "14 days"],
            ["SOP-REFUND-UNOPENED-v2", "Superseded, 1 Jun 2024", "30 days"],
            ["POLICY-WEB-RETURNS", "Marketing, 12 Nov 2025", "30 days"],
            ["PRICE-LIST-2026 / 2024", "Current 8% / old 15%", "Restock fee"],
            ["SOP-WARRANTY-DOMESTIC", "Domestic only", "No international number"],
            ["CHANGELOG-REFUNDS", "1 Mar 2026", "30 days withdrawn"],
        ],
    )
    add_p(
        doc,
        "Six queries force six actions a desk system must get right. T1 asks for the "
        "unopened refund window. T2 asks whether a day-20 return is allowed. T3 asks for "
        "the restocking fee on an opened unused return. T4 asks for an international "
        "warranty period. T5 asks for one customer-facing sentence about the day-20 case. "
        "T6 asks whether a 30-day answer is still correct. Gold values are taken from the "
        "current governing record: 14 days, 14 days, 8%, refuse, 14 days, and 14 days.",
        first_indent=True,
    )

    add_h(doc, "3.2\tRetriever and Slot Extraction", style="Heading 2")
    add_p(
        doc,
        "Ranking uses BM25 with k1 = 1.5 and b = 0.75 [2]. For each query the top-3 "
        "records are kept. A regular-expression extractor then reads days, restocking "
        "percentages, a superseded flag, a current flag, and an explicit ban on quoting "
        "an international warranty. Mentions that sit next to previous, from, or withdrawn "
        "are ignored so that a current file that names the old 30-day window is not read "
        "as a 30-day rule. The extractor is deterministic. The same rules are published "
        "with the corpus.",
    )

    add_h(doc, "3.3\tRefusal Policies", style="Heading 2")
    add_p(
        doc,
        "Three policies consume the same top-3 extract. Always rank-1 emits the slot "
        "value of the first record, or 12 months when the first record is the domestic "
        "warranty. Conflict refuse emits REFUSE when the distinct extracted values for "
        "the asked slot in the top-3 are more than one, or when the query asks for an "
        "international warranty. Current override first looks for a record marked current "
        "in that top-3 and emits its value; if none is found it falls back to conflict "
        "refuse. No policy averages 14 and 30.",
    )
    add_h(doc, "3.4\tEvaluation Metrics", style="Heading 2")
    add_p(
        doc,
        "We score four outcomes. A correct answer is a predicted number that equals the "
        "gold slot. A wrong answer is a predicted number that does not. A correct refuse "
        "is REFUSE on T4. A false refuse is REFUSE on a task whose gold is a number. "
        "Safe means not wrong: a correct answer, a correct refuse, or a false refuse. "
        "False refuse is safe for a desk because it sends the worker to a human instead "
        "of a withdrawn number. It is still a cost, which is why we report it separately.",
    )

    add_h(doc, "4\tRESULTS")
    add_p(
        doc,
        "Results are reported in two layers. The first layer is what BM25 actually "
        "returns. The second layer is what each policy does with that set. Table 2 "
        "summarizes both.",
    )
    add_table(
        doc,
        "Table 2. Policy outcomes on six desk queries. Gold is the current governing value or REFUSE.",
        [
            ["Task", "Gold", "Rank-1", "Conflict refuse", "Current override"],
            ["T1 Factoid", "14", "30 wrong", "REFUSE", "14 correct"],
            ["T2 Conflict", "14", "30 wrong", "REFUSE", "14 correct"],
            ["T3 Stale fee", "8", "15 wrong", "REFUSE", "8 correct"],
            ["T4 Missing", "REFUSE", "12 wrong", "REFUSE correct", "REFUSE correct"],
            ["T5 Inform", "14", "14 correct", "14 correct", "14 correct"],
            ["T6 Repair", "14", "14 correct", "REFUSE", "14 correct"],
        ],
    )

    add_h(doc, "4.1\tRetrieval Evidence", style="Heading 2")
    add_p(
        doc,
        "On T1 the superseded 30-day procedure is ranked first and the current 14-day "
        "procedure is second. On T2 the marketing page that promises 30 days is ranked "
        "first and the governing procedure is third. On T3 the 2024 price list that "
        "charges 15% is ranked first and the 2026 list that charges 8% is second. Figure 2 "
        "plots those extracted numbers against the governing value. Rank-1 is not a rare "
        "error on this corpus. It is the default lexical winner when the old file is a "
        "close paraphrase of the new file.",
    )
    add_p(
        doc,
        "T4 is different. The only warranty record is domestic and tells staff not to "
        "quote a period for a foreign address. It is ranked first. Absence here is a "
        "retrieved refusal instruction, not an empty index. T5 and T6 show that query "
        "wording can already surface the current procedure or the changelog. That is why "
        "a rank-1 baseline is not wrong on every task. It is wrong on the ordinary ones.",
        first_indent=True,
    )
    add_fig(
        doc,
        FIG / "fig3_slot_values.png",
        "Figure 2. Rank-1 extracted slot versus the governing value. T4 is omitted because gold is REFUSE.",
    )

    add_h(doc, "4.2\tRefusal Outcomes", style="Heading 2")
    add_p(
        doc,
        "Always answering from rank-1 is wrong on 4 of 6 tasks. The wrong numbers are "
        "exactly the ones a desk must not say: 30 days, 30 days again, 15%, and a 12-month "
        "international warranty. Conflict refuse produces 0 wrong answers. It refuses T1, "
        "T2, T3, and T6 because 14 and 30, or 8 and 15, both appear in the top-3, and it "
        "correctly refuses T4. The cost is over-refusal: T1, T3, and T6 have a recoverable "
        "governing value. Current override recovers those values and still refuses T4. "
        "Figure 3 compares the wrong-answer counts.",
    )
    add_fig(
        doc,
        FIG / "fig2_wrong_answers.png",
        "Figure 3. Wrong answers out of six tasks for the three policies.",
    )
    add_p(
        doc,
        "Counted that way, always rank-1 is safe on 2 of 6 tasks and wrong on 4. Conflict "
        "refuse is safe on all 6 and wrong on 0, with 4 false refuses. Current override is "
        "safe on all 6, wrong on 0, and false-refuses only T4, which is the intended refuse. "
        "The interesting comparison is therefore not accuracy against a generator. It is "
        "wrong-answer rate against an inspectable gate.",
        first_indent=True,
    )
    add_p(
        doc,
        "Task by task, the mechanism is visible. T1 is a near-paraphrase fight: v2 and v3 "
        "share the same lexical frame, so BM25 cannot see effective date. Conflict refuse "
        "sees 30 and 14 and stops. Current override reads the replaces line on v3 and "
        "emits 14. T2 is institutional rather than temporal: marketing copy is supposed "
        "to be friendlier than the desk rule. Rank-1 follows the public page. Current "
        "override ignores that page because it is not marked current. T3 repeats T1 for a "
        "percentage. T4 is the only task whose gold is REFUSE; both refusal policies get "
        "it and the rank-1 baseline does not. T5 already retrieves v3 first, so all three "
        "policies emit 14. T6 retrieves the changelog first; a last-number read of that "
        "file already yields 14, which is why rank-1 is correct there, but the same "
        "changelog still contains 30, so conflict refuse over-fires until the current "
        "file is preferred.",
        first_indent=True,
    )
    add_p(
        doc,
        "The current-override policy is not magic. It works because the corpus marks "
        "currency in the file: a replaces line, a 2026 price-list name, or an explicit "
        "SUPERSEDED stamp. A company that never marks those fields cannot run this gate. "
        "That is an information-organization requirement, not a model requirement.",
        first_indent=True,
    )

    add_h(doc, "5\tDISCUSSION")
    add_p(
        doc,
        "The measurement supports a practical design rule. Do not emit a slot from rank-1 "
        "when the top-k contains more than one value for that slot, unless a record in "
        "that set is marked current. The rule is cheap, auditable, and independent of "
        "the generator. It also shows why accuracy-only RAG scores mislead: a system that "
        "always talks can look useful while stating a withdrawn 30-day window.",
    )
    add_h(doc, "5.1\tImplications", style="Heading 2")
    add_p(
        doc,
        "Three implementation implications follow. First, dates and version status must "
        "be first-class fields in the snippet, not characters buried in a filename. "
        "Second, when two retrieved things disagree on days or percent, the interface "
        "should refuse or show both; it should not average them. Third, changelogs should "
        "stay retrievable, because they are the repair cue on T6. None of these is a new "
        "neural architecture.",
    )
    add_h(doc, "5.2\tLimitations", style="Heading 2")
    add_p(
        doc,
        "The policies are also a specification for a later language-model stage. A "
        "generator can be added on top of current override without changing the gold "
        "labels. The generator would then be allowed to write a sentence only when the "
        "gate emits a number, and would be required to produce a refusal plus the "
        "conflicting snippets when the gate emits REFUSE. That experiment is future work. "
        "It is not reported here because an extra sampler would add an uncontrolled "
        "source of wording to a paper whose claim is about the gate.",
    )
    add_p(
        doc,
        "The corpus is fictional and English-only. Eighteen documents are enough to "
        "instantiate the conflicts and small enough to audit by hand; they are not a "
        "company. BM25 is one retriever. A dense or hybrid ranker might promote the "
        "current file more often; that is an open measurement. No language-model wording "
        "is reported, so fluency effects are out of scope. No employees were observed. "
        "Anyone citing this paper as a user study would be mis-citing it. The work is "
        "also distinct from an information-behavior analysis of the same fixture: here "
        "the unit of evaluation is the system action, not the worker's knowledge state. "
        "Reuse of the constructed files does not make the two papers the same claim.",
    )
    add_h(doc, "5.3\tThreats to Validity", style="Heading 2")
    add_p(
        doc,
        "Construct validity depends on the slot extractor. A missed 30-day hyphen or a "
        "withdrawn window read as current would change Table 2. We publish the extractor "
        "with the corpus so that threat can be audited. Internal validity is limited by "
        "the designed conflicts: we planted a 14-versus-30 pair and then measured whether "
        "the policies see it. External validity is limited to English SOP-like prose. "
        "The paper does not claim that every intranet will rank a superseded file first. "
        "It claims that when that ranking happens, a current-override gate is a cheap "
        "and testable response.",
    )

    add_h(doc, "6\tCONCLUSION")
    add_p(
        doc,
        "Enterprise RAG fails in a boring way: the first retrieved thing is often the "
        "old thing. A gate that always speaks that thing is wrong on the ordinary refund "
        "and fee questions in this fixture. A gate that refuses when extracted slots "
        "disagree stops the wrong numbers and then over-refuses. Preferring a record "
        "marked current recovers the governing 14-day window and the 8% fee and still "
        "refuses an unlicensed international warranty. The next useful measurement is "
        "the same gate on a dense index and, separately, on recorded desk sessions. This "
        "paper stops at the inspectable policies and the corpus walkthrough.",
    )

    add_h(doc, "REFERENCES")
    refs = [
        '[1] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 9459-9474.',
        '[2] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333-389, 2009, doi: 10.1561/1500000019.',
        '[3] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in Proc. Int. Conf. Learning Representations, 2024.',
        '[4] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston, "Retrieval augmentation reduces hallucination in conversation," in Findings of the Association for Computational Linguistics: EMNLP, 2021, pp. 3784-3803.',
        '[5] V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in Proc. EMNLP, 2020, pp. 6769-6781.',
        '[6] G. Izacard and E. Grave, "Leveraging passage retrieval with generative models for open domain question answering," in Proc. EACL, 2021, pp. 874-880.',
        '[7] S. Kadavath et al., "Language models (mostly) know what they know," arXiv:2207.05221, 2022.',
        '[8] Y. Gao et al., "Retrieval-augmented generation for large language models: A survey," arXiv:2312.10997, 2023.',
        '[9] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, and G. Neubig, "Active retrieval augmented generation," in Proc. EMNLP, 2023, pp. 7969-7992.',
        '[10] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, "Corrective retrieval augmented generation," arXiv:2401.15884, 2024.',
        '[11] W. Shi et al., "REPLUG: Retrieval-augmented black-box language models," in Proc. NAACL, 2024, pp. 8371-8384.',
        '[12] C. Niu et al., "RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models," in Proc. ACL, 2024, pp. 10862-10878.',
    ]
    for r in refs:
        add_ref(doc, r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print("WROTE", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    build()
