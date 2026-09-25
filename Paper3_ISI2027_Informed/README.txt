ISI 2027 submission package
19. Internationales Symposium fuer Informationswissenschaft
HAW Hamburg, 16-18 March 2027

Paper (double-blind Forschungsbeitrag, English, APA)
Becoming Informed at Work: Information Behavior of Employees
Using Retrieval-Augmented Assistants

OFFICIAL RULES CHECKED
- EasyChair: https://easychair.org/conferences/?conf=isi2027
- Metadata deadline: 5 October 2026
- PDF deadline: 12 October 2026
- Official CFP: https://isi2027.informationswissenschaft.org/cfp/
- Official Word template: https://isi2027.informationswissenschaft.org/template/
- Full paper: max 7,000 words excluding references; anonymised; APA
- Official CFP asks for the Word template. Submit ISI2027_Paper3_Informed_ANONYMOUS.docx
  (built from that template) to EasyChair. main.pdf is the same
  paper in a LaTeX reconstruction of the same layout
  (A4, 2.54 cm / 1.00 in margins, top 2.76 cm / 1.09 in,
  Calibri/Helvetica 17 pt title, 14 pt Heading 1, Times 11 pt body,
  table captions above, figure captions below).
- ISI 2025 notes said LaTeX only after agreement. If the chairs
  insist on .docx, use ISI2027_Paper3_Informed_ANONYMOUS.docx.

WHAT TO UPLOAD
  Preferred: ISI2027_Paper3_Informed_ANONYMOUS.docx
  Optional extra: main.pdf
  Do not upload author names. The files are already anonymous.

BUILD THE PDF (optional)
  pdflatex main.tex
  pdflatex main.tex
  or: tectonic main.tex

CONTENTS
  ISI2027_Paper3_Informed_ANONYMOUS.docx       official Word-template version (submit this)
  main.tex                 LaTeX source
  isi2027.cls              official-layout class
  main.pdf                 compiled paper
  figures/                 five figures, NO titles inside the images
  corpus/                  18 Northline records used in Section 5
  retrieval_results.json   BM25 scores reported in Table 2
  run_bm25.py              reproduces the lexical ranking
  README.txt               this file

FIGURES
  Titles are not drawn on the images. Captions sit only under
  the figure in the paper (Word style Bildunterschrift /
  LaTeX \caption). Table captions sit above the table
  (Word style Tabellenueberschrift).

WHAT THE RESULTS ARE
  Real BM25 walkthrough of a constructed corpus.
  No invented interviews, quotes, or kappa.

AUTHORS AND ANONYMITY
  Ahmed Ali and Asher Mehfooz.
  Upload the _ANONYMOUS files for double-blind review.
  ISI2027_Paper3_Informed.docx and main.pdf carry the names (camera-ready).
  For an anonymous LaTeX PDF use \documentclass[review]{isi2027}.
