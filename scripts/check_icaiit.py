"""Check ICAIIT formal rules on the LaTeX sources: abstract 150-300 words,
at least 4 keywords, at least 10 references, every reference cited, 5-7 pages."""
import re
import sys
from pathlib import Path

import pypdf

BS = "\\"
ROOT = Path(__file__).resolve().parent.parent


def group(s, start):
    depth, j = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[start:j - 1]


for name in sys.argv[1:] or ["Paper1_ICAIIT2027_Agent", "Paper2_ICAIIT2027_Refusal", "Paper5_ICAIIT2027_Intent"]:
    d = ROOT / name
    s = (d / "main.tex").read_text(encoding="utf-8")
    ab = group(s, s.index(BS + "abstract{") + len("abstract{") + 1)
    kw = group(s, s.index(BS + "keywords{") + len("keywords{") + 1)
    refs = (d / "references.tex").read_text(encoding="utf-8")
    keys = re.findall(BS * 2 + r"bibitem\{([^}]*)\}", refs)
    cited = {k.strip() for m in re.findall(BS * 2 + r"cite\{([^}]*)\}", s) for k in m.split(",")}
    pages = len(pypdf.PdfReader(str(d / "main.pdf")).pages)
    ok = 150 <= len(ab.split()) <= 300 and len(kw.split(",")) >= 4 and len(keys) >= 10 \
        and not (set(keys) - cited) and not (cited - set(keys)) and 5 <= pages <= 7
    print(f"{name}: abstract {len(ab.split())} words, {len(kw.split(','))} keywords, {len(keys)} refs, "
          f"uncited {sorted(set(keys) - cited)}, undefined {sorted(cited - set(keys))}, {pages} pages -> {'OK' if ok else 'CHECK'}")
