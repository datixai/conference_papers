%% ISI 2027 layout reconstructed from the official Word template
%% (isi2027.informationswissenschaft.org/template/) and the publisher
%% formatting notes (Trebuchet/Calibri headings, Times body, APA).
%% A4, ~1 inch margins. No running headers (publisher adds them).
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{isi2027}[2026/09/22 ISI 2027 submission class]
\newif\ifisireview \isireviewfalse
\DeclareOption{review}{\isireviewtrue}
\ProcessOptions\relax
\LoadClass[11pt,a4paper]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}
\RequirePackage[margin=2.54cm,top=2.76cm]{geometry}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{enumitem}
\RequirePackage[english]{babel}
\RequirePackage{microtype}
\RequirePackage[round,authoryear,sort]{natbib}
\RequirePackage[hidelinks]{hyperref}

\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setstretch{1.08}

%% Headings: sans (Calibri/Trebuchet analogue) 14 pt bold; body Times 11 pt
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}

\titleformat{\section}
  {\normalfont\sffamily\bfseries\fontsize{14}{17}\selectfont}
  {\thesection}{0.7em}{}
\titlespacing*{\section}{0pt}{1.15em}{0.45em}

\titleformat{\subsection}
  {\normalfont\bfseries\fontsize{12}{15}\selectfont}
  {\thesubsection}{0.6em}{}
\titlespacing*{\subsection}{0pt}{0.95em}{0.30em}

%% Table caption above, italic 11 pt; figure caption below, 10 pt
\captionsetup{
  labelfont={it},
  textfont={it},
  skip=6pt,
  justification=raggedright,
  singlelinecheck=false
}
\captionsetup[table]{font=normalsize, position=top, skip=5pt}
\captionsetup[figure]{font=small, position=bottom, skip=6pt}

\setlist[itemize]{leftmargin=1.3em,itemsep=0.2em,topsep=0.25em}
\setlist[enumerate]{leftmargin=1.4em,itemsep=0.25em,topsep=0.3em}

\newcommand{\isiTitle}[1]{%
  {\centering\sffamily\bfseries\fontsize{17}{21}\selectfont #1\par}%
  \vspace{0.7em}%
}
\newcommand{\isiAnonymous}{%
  {\centering\fontsize{11}{14}\selectfont Anonymous author(s)\par}%
  {\centering\fontsize{10}{13}\selectfont Affiliation omitted for double-blind review\par}%
  \vspace{0.9em}%
}
\newcommand{\isiAuthors}[2]{%
  \ifisireview\isiAnonymous\else
  {\centering\fontsize{11}{14}\selectfont #1\par}%
  {\centering\fontsize{10}{13}\selectfont #2\par}%
  \vspace{0.9em}\fi
}
\newcommand{\isiAbstract}[1]{%
  {\noindent\bfseries Abstract.\ }{\fontsize{10.5}{13}\selectfont #1\par}%
  \vspace{0.35em}%
}
\newcommand{\isiKeywords}[1]{%
  {\noindent\fontsize{11}{14}\selectfont\textit{Keywords:} #1\par}%
  \vspace{0.7em}%
}

\renewcommand{\refname}{References}
\renewcommand{\bibfont}{\fontsize{10}{12}\selectfont}
