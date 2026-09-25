# Five conference papers: Ahmed Ali and Asher Mehfooz

| # | Folder | Venue | Deadline | Format | Pages |
|---|--------|-------|----------|--------|-------|
| 1 | `Paper1_ICAIIT2027_Agent` | ICAIIT 2027, Koethen (Track 8) | 10 Jan 2027 | Word template only, double-blind | 5-7 |
| 2 | `Paper2_ICAIIT2027_Refusal` | ICAIIT 2027, Koethen (Track 4/8) | 10 Jan 2027 | Word template only, double-blind | 5-7 |
| 3 | `Paper3_ISI2027_Informed` | ISI 2027, HAW Hamburg | metadata 5 Oct 2026, PDF 12 Oct 2026 | Word template (LaTeX copy included), double-blind, APA | max 7,000 words |
| 4 | `Paper4_TRR318_SocialXAI` | 4th TRR 318 Conference, Paderborn | 1 Nov 2026 | official Word or LaTeX template, authors visible | 2 incl. references |
| 5 | `Paper5_ICAIIT2027_Intent` | ICAIIT 2027, Koethen (Track 4) | 10 Jan 2027 | Word template only, double-blind | 5-7 |

## What is in each paper folder

- `main.tex`, `references.tex` / `references.bib`, class file, `figures/`: the LaTeX source
- `main.pdf`: compiled paper **with author names** (camera-ready version)
- `main_ANONYMOUS.pdf`: the same paper without names, for double-blind review (ICAIIT and ISI)
- `*.docx` / `*_ANONYMOUS.docx`: the Word version. ICAIIT and ISI ask for the Word file, so these are the ones to upload.

**For EasyChair at ICAIIT and ISI, upload the `_ANONYMOUS` file.** Both conferences are double-blind, and a paper with names on it can be desk-rejected. Use the named version after acceptance. TRR 318 asks for authors to be visible, so Paper 4 has only the named version.

## Compiling

WinEdt 11 / MiKTeX: open `main.tex` and run **PDFLaTeX** twice. For Paper 4, run PDFLaTeX, then **Biber**, then PDFLaTeX twice.
For the anonymous version of an ICAIIT or ISI paper, change the first line to `\documentclass[review]{icaiit2027}` (or `[review]{isi2027}`).

Everything in one go (all PDFs, both versions, and all ICAIIT Word files):

    python scripts/build_all.py

## Experiments (all numbers in Papers 1 and 5 come from these runs)

- `experiments/paper1_agent/`: synthetic store, 100 tickets, local LLM planner, validator, evaluation, audit logs.
  `python store.py`, `python planner.py ../_models/phi35 phi35`, `python evaluate.py phi35 phi3mini`, `python make_figures.py`
- `experiments/paper5_intent/`: Bitext customer-support data, TF-IDF / MiniLM / few-shot LLM, leave-intents-out thresholds.
  `python run_intent.py`, `python llm_fewshot.py`, `python make_figures.py`
- Model weights are not in the repository. To rerun, download Phi-3.5-mini / Phi-3-mini ONNX (cpu-int4) and all-MiniLM-L6-v2 ONNX from Hugging Face into `experiments/_models/phi35`, `phi3mini`, `minilm`.
- Models ran on the CPU through ONNX Runtime (Windows Application Control blocks the llama.cpp DLL on this PC).

## Other folders

- `plans/`: the original planning documents for all five papers
- `templates/`: official ICAIIT and ISI Word templates, official TRR 318 Word and LaTeX templates, `latex/icaiit2027.cls`
- `northline_corpus/`: the 18-document corpus shared by Papers 2 and 3
- `scripts/`: `build_all.py`, `tex2icaiit_docx.py` (LaTeX to ICAIIT Word), `check_icaiit.py` (formal rule check)

## Check before submitting

- Confirm the affiliation "University of Kotli Azad Jammu and Kashmir, Kotli 11100" (e-mails: rajaahmedalikhan97@gmail.com, info.hasher@gmail.com). ICAIIT asks for the full postal address (street and number) of each author.
- ICAIIT allows at most 4 papers per author; Papers 1, 2 and 5 make 3.
