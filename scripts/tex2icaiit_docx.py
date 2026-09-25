"""Build an ICAIIT Word submission from a paper's main.tex + references.tex.

ICAIIT accepts papers only in the official Word template. The LaTeX file is the
single source; this script fills templates/_ICAIIT_official_template.docx with the
same title, authors, keywords, abstract, sections, figures, tables, algorithm and
references, so the Word and LaTeX versions cannot drift apart.

Supports the LaTeX subset used by the ICAIIT papers in this folder.
Usage:  python tex2icaiit_docx.py <paper_dir> <out.docx> [--review]
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).parent))
from _build_icaiit import TEMPLATE, add_h, add_ref, clear_after_section_break, set_cell  # noqa: E402

ACCENTS = {r'\"{u}': "ü", r'\"{a}': "ä", r'\"{o}': "ö", r"\'{c}": "ć", r"\v{c}": "č", r"\u{g}": "ğ",
           r"\'{e}": "é", r'\"u': "ü", r'\"a': "ä", r'\"o': "ö"}


def braced(s, i):
    """Return (content, index after closing brace) for the group starting at s[i] == '{'."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced")


def arg(src, name):
    i = src.index("\\" + name + "{")
    return braced(src, i + len(name) + 1)[0]


class Ctx:
    def __init__(self, cites):
        self.cites = cites
        self.labels = {}


def clean(s, ctx):
    """LaTeX inline markup -> list of (text, italic, bold) runs."""
    for k, v in ACCENTS.items():
        s = s.replace(k, v)
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: ", ".join(f"[{ctx.cites[k.strip()]}]" for k in m.group(1).split(",")), s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: str(ctx.labels.get(m.group(1), "?")), s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\LLM\{([^}]*)\}", r"\1", s)
    s = s.replace("\\\\", " ").replace("~", " ").replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
    s = s.replace("---", "\u2014").replace("--", "\u2013").replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("\\ ", " ").replace("\\allowbreak", "").replace("\\quad", "    ")
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("\\geq", "\u2265").replace("\\neq", "\u2260")
               .replace("\\times", "\u00d7").replace("_1", "1").replace("\\", "").replace("{", "").replace("}", ""), s)
    runs, i, buf = [], 0, ""
    while i < len(s):
        m = re.match(r"\\(emph|textit|textbf)\{", s[i:])
        if m:
            if buf:
                runs.append((buf, False, False)); buf = ""
            inner, j = braced(s, i + len(m.group(0)) - 1)
            runs.append((inner, m.group(1) != "textbf", m.group(1) == "textbf"))
            i = j
            continue
        buf += s[i]
        i += 1
    if buf:
        runs.append((buf, False, False))
    out = []
    for t, it, bd in runs:
        t = re.sub(r"\\[a-zA-Z]+\*?", "", t).replace("{", "").replace("}", "")
        t = re.sub(r"\s+", " ", t)
        out.append((t, it, bd))
    return out


def para(doc, runs, indent, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, font="Times New Roman", after=0):
    p = doc.add_paragraph(style="Normal")
    p.alignment = align
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0.5) if indent else Cm(0)
    pf.space_after = Pt(after)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.0
    first = font != "Courier New"
    for t, it, bd in runs:
        if first:
            t = t.lstrip(); first = False
        r = p.add_run(t)
        r.font.name = font
        r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), font)
        r.font.size = Pt(size)
        r.italic = it
        r.bold = bd
    return p


def caption(doc, text, before, after):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(text) < 80 else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    return p


def table_rows(body, ctx):
    body = body.split("\\toprule", 1)[1].split("\\bottomrule", 1)[0]
    rows = []
    for line in body.split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if not line:
            continue
        m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$", line, re.S)
        if m:
            n = int(m.group(1))
            rows.append(["".join(t for t, _, _ in clean(m.group(2), ctx))] + [""] * (n - 1))
            continue
        rows.append(["".join(t for t, _, _ in clean(c, ctx)).strip() for c in line.split("&")])
    return rows


def add_table(doc, rows):
    tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        tbl.style = doc.styles["Table Grid"]
    except KeyError:
        pass
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(8)
            r.bold = i == 0
    return tbl


def build(paper_dir, out, review=False):
    paper_dir = Path(paper_dir)
    src = (paper_dir / "main.tex").read_text(encoding="utf-8")
    src = re.sub(r"(?m)(?<!\\)%.*$", "", src)
    refs_src = (paper_dir / "references.tex").read_text(encoding="utf-8")
    items = re.split(r"\\bibitem\{([^}]*)\}", refs_src.split("\\end{thebibliography}")[0])[1:]
    keys = items[0::2]
    ctx = Ctx({k: i + 1 for i, k in enumerate(keys)})

    body = src.split("\\maketitle", 1)[1].split("\\input{references.tex}", 1)[0]
    # pre-number floats and sections so \ref works in any order
    fig = tab = alg = 0
    sec = 0
    for m in re.finditer(r"\\begin\{(figure\*?|table\*?|algorithm)\}.*?\\end\{\1\}|\\section\{", body, re.S):
        if m.group(0).startswith("\\section"):
            sec += 1
            continue
        env = m.group(1).rstrip("*")
        lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
        if env == "figure":
            fig += 1; n = fig
        elif env == "table":
            tab += 1; n = tab
        else:
            alg += 1; n = alg
        if lab:
            ctx.labels[lab.group(1)] = n

    doc = Document(str(TEMPLATE))
    t0 = doc.tables[0]
    set_cell(t0.cell(0, 0), "".join(t for t, _, _ in clean(arg(src, "title"), ctx)), size=15, bold=True)
    if review:
        set_cell(t0.cell(1, 0), "Anonymous Author(s)", size=11)
        set_cell(t0.cell(2, 0), "Affiliation and e-mail omitted for double-blind review", size=9)
    else:
        set_cell(t0.cell(1, 0), arg(src, "icaiitauthors"), size=11)
        aff = arg(src, "icaiitaffiliations").replace("\\\\", "\n").replace("\n ", "\n").strip()
        c = t0.cell(2, 0)
        c.text = ""
        for k, line in enumerate(l for l in aff.split("\n") if l.strip()):
            p = c.paragraphs[0] if k == 0 else c.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(line.strip())
            r.font.name = "Times New Roman"; r.font.size = Pt(9); r.italic = True
    kw = "".join(t for t, _, _ in clean(arg(src, "keywords"), ctx))
    ab = "".join(t for t, _, _ in clean(arg(src, "abstract"), ctx))
    for p in doc.paragraphs:
        t = p.text.strip()
        for lead, text in (("Keywords", kw), ("Abstract", ab)):
            if t.startswith(lead):
                p.clear()
                r = p.add_run(f"{lead}:\t")
                r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(9)
                r = p.add_run(text)
                r.font.name = "Times New Roman"; r.font.size = Pt(9)
    clear_after_section_break(doc)

    sec = sub = 0
    indent_next = False
    fig = tab = alg = 0
    tokens = re.split(r"(\\section\*?\{[^}]*\}|\\subsection\{[^}]*\}|\\begin\{(?:figure\*?|table\*?|algorithm|itemize)\}.*?\\end\{(?:figure\*?|table\*?|algorithm|itemize)\})",
                      body, flags=re.S)
    for tok in tokens:
        if not tok or not tok.strip():
            continue
        if tok.startswith("\\section*"):
            add_h(doc, tok[tok.index("{") + 1:-1]); indent_next = False
        elif tok.startswith("\\section"):
            sec += 1; sub = 0
            add_h(doc, f"{sec}\t{tok[9:-1].upper()}"); indent_next = False
        elif tok.startswith("\\subsection"):
            sub += 1
            add_h(doc, f"{sec}.{sub}\t{tok[12:-1]}", style="Heading 2"); indent_next = False
        elif tok.startswith("\\begin{figure"):
            fig += 1
            path = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", tok).group(1)
            cap = "".join(t for t, _, _ in clean(re.search(r"\\caption\{(.*)\}\s*\\label", tok, re.S).group(1), ctx))
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.add_run().add_picture(str(paper_dir / path), width=Cm(7.4))
            caption(doc, f"Figure {fig}: {cap.strip()}", 6, 12)
        elif tok.startswith("\\begin{table"):
            tab += 1
            capm = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", tok, re.S)
            cap = "".join(t for t, _, _ in clean(capm.group(1), ctx))
            caption(doc, f"Table {tab}: {cap.strip()}", 12, 6)
            add_table(doc, table_rows(tok, ctx))
            doc.add_paragraph().paragraph_format.space_after = Pt(6)
        elif tok.startswith("\\begin{algorithm"):
            alg += 1
            lines = tok.split("\\begin{algobody}", 1)[1].split("\\end{algobody}", 1)[0].strip().split("\n")
            for line in lines:
                line = line.replace("\\ind", "    ")
                para(doc, [(line, False, False)], False, size=9, align=WD_ALIGN_PARAGRAPH.LEFT, font="Courier New")
            cap = "".join(t for t, _, _ in clean(re.search(r"\\caption\{(.*?)\}", tok, re.S).group(1), ctx))
            caption(doc, f"Algorithm {alg}: {cap.strip()}", 6, 12)
        elif tok.startswith("\\begin{itemize"):
            for it in tok.split("\\item")[1:]:
                it = it.replace("\\end{itemize}", "").strip()
                p = para(doc, [("\u2022\t", False, False)] + clean(it, ctx), False)
                p.paragraph_format.left_indent = Cm(0.5)
                p.paragraph_format.first_line_indent = Cm(-0.5)
            doc.paragraphs[-1].paragraph_format.space_after = Pt(6)
            indent_next = False
        else:
            for chunk in re.split(r"\n\s*\n", tok):
                chunk = chunk.strip()
                if not chunk or chunk.startswith("\\newcommand"):
                    continue
                para(doc, clean(chunk, ctx), indent_next)
                indent_next = True

    add_h(doc, "References")
    for k, txt in zip(items[0::2], items[1::2]):
        t = "".join(x for x, _, _ in clean(txt.strip(), ctx))
        add_ref(doc, f"[{ctx.cites[k]}] {t.strip()}")
    doc.save(str(out))
    print("WROTE", out)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], review="--review" in sys.argv)
