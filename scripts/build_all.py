"""Compile all five papers with MiKTeX/TeX Live and build the ICAIIT Word files.

For each paper this produces
  main.pdf              version with author names (camera-ready)
  main_ANONYMOUS.pdf    double-blind version for ICAIIT and ISI submission
ICAIIT papers also get <name>.docx and <name>_ANONYMOUS.docx from the same LaTeX.
Usage: python scripts/build_all.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS = [
    # folder, class name or None, needs biber, docx name (ICAIIT only)
    ("Paper1_ICAIIT2027_Agent", "icaiit2027", False, "ICAIIT2027_Paper1_Agent.docx"),
    ("Paper2_ICAIIT2027_Refusal", "icaiit2027", False, "ICAIIT2027_Paper2_Refusal.docx"),
    ("Paper3_ISI2027_Informed", "isi2027", False, None),
    ("Paper4_TRR318_SocialXAI", None, True, None),
    ("Paper5_ICAIIT2027_Intent", "icaiit2027", False, "ICAIIT2027_Paper5_Intent.docx"),
]


def latex(folder, job, biber):
    def run(*cmd):
        r = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, errors="replace")
        return r.returncode
    run("pdflatex", "-interaction=nonstopmode", f"{job}.tex")
    if biber:
        if (folder / f"{job}.bcf").exists():
            run("biber", job)
        else:
            run("bibtex", job)
    run("pdflatex", "-interaction=nonstopmode", f"{job}.tex")
    run("pdflatex", "-interaction=nonstopmode", f"{job}.tex")
    log = (folder / f"{job}.log").read_text(encoding="latin-1")
    errors = [l for l in log.splitlines() if l.startswith("!")]
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    return errors, pages.group(1) if pages else "?"


def main():
    ok = True
    for name, cls, biber, docx in PAPERS:
        folder = ROOT / name
        errs, pages = latex(folder, "main", biber)
        print(f"{name}: main.pdf {pages} pages {'OK' if not errs else errs[:3]}")
        ok &= not errs
        if cls:
            src = (folder / "main.tex").read_text(encoding="utf-8")
            anon = src.replace(f"\\documentclass{{{cls}}}", f"\\documentclass[review]{{{cls}}}", 1)
            (folder / "main_ANONYMOUS.tex").write_text(anon, encoding="utf-8")
            errs, pages = latex(folder, "main_ANONYMOUS", biber)
            print(f"{name}: main_ANONYMOUS.pdf {pages} pages {'OK' if not errs else errs[:3]}")
            ok &= not errs
        if docx:
            for extra, suffix in (([], ""), (["--review"], "_ANONYMOUS")):
                out = folder / docx.replace(".docx", f"{suffix}.docx")
                subprocess.run([sys.executable, str(ROOT / "scripts" / "tex2icaiit_docx.py"), str(folder), str(out)] + extra,
                               check=True, capture_output=True)
            print(f"{name}: {docx} (+ _ANONYMOUS) written")
        for ext in (".aux", ".log", ".out", ".bbl", ".blg", ".bcf", ".run.xml"):
            for job in ("main", "main_ANONYMOUS"):
                p = folder / f"{job}{ext}"
                if p.exists():
                    p.unlink()
        a = folder / "main_ANONYMOUS.tex"
        if a.exists():
            a.unlink()
    print("ALL OK" if ok else "SOME ERRORS")


if __name__ == "__main__":
    if not shutil.which("pdflatex"):
        sys.exit("pdflatex not found; install MiKTeX or TeX Live")
    main()
