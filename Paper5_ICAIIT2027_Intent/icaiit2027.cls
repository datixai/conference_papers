%% icaiit2027.cls
%% LaTeX reconstruction of the official ICAIIT "Template2026_Word.docx"
%% (15th International Conference on Applied Innovations in IT, Koethen 2027).
%% ICAIIT accepts the Word template for submission; this class reproduces its
%% measurements so the LaTeX and Word versions of a paper look the same:
%%   A4; margins top 3.3 cm, bottom 4.2 cm, left/right 2.6 cm; header 1.4 cm,
%%   footer 2.5 cm; no running heads or page numbers.
%%   Title 15 pt bold centred (17 pt leading); authors 11/13 pt, 24 pt before;
%%   affiliations 9/11 pt italic; keywords 9/11 pt, 2 cm hanging, 48 pt before;
%%   abstract 9/11 pt, 2 cm hanging, 12 pt before, 30 pt after.
%%   Body in two 7.5 cm columns with 0.8 cm gap, Times 10 pt, single spacing,
%%   first paragraph unindented, later paragraphs 0.5 cm indent.
%%   Section 13 pt bold capitals, 0.7 cm hanging, 24 pt before, 12 pt after;
%%   subsection 12 pt bold, 1.0 cm hanging, 11 pt before/after;
%%   subsubsection 11 pt bold, 1.0 cm hanging, 10 pt before/after.
%%   Captions 9 pt, "Table n:" above tables, "Figure n:" below figures,
%%   one-line captions centred, longer captions justified.
%%
%% Options:  review  -> anonymous version for double-blind submission.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{icaiit2027}[2026/09/25 ICAIIT 2027 paper class]

\newif\if@icaiitreview \@icaiitreviewfalse
\DeclareOption{review}{\@icaiitreviewtrue}
\ProcessOptions\relax

\LoadClass[10pt,a4paper,twocolumn]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{newtxtext,newtxmath}
\RequirePackage[a4paper,top=3.3cm,bottom=4.2cm,left=2.6cm,right=2.6cm,
  headsep=1.4cm,footskip=2.5cm,columnsep=0.8cm]{geometry}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{enumitem}
\RequirePackage{titlesec}
\RequirePackage[font=footnotesize,labelsep=colon,justification=justified,
  singlelinecheck=true,skip=6pt]{caption}
\RequirePackage{xcolor}
\RequirePackage{microtype}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{cite}

\pagestyle{empty}
\AtBeginDocument{\thispagestyle{empty}}
\setlength{\parindent}{0.5cm}
\setlength{\parskip}{0pt}
\renewcommand{\footnotesize}{\fontsize{9}{11}\selectfont}
\renewcommand{\small}{\fontsize{9}{11}\selectfont}

%% Captions: 9 pt, table above, figure below
\captionsetup[table]{position=top,skip=4pt}
\captionsetup[figure]{position=bottom,skip=6pt}
\setlength{\abovecaptionskip}{6pt}
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}

%% Headings (no dot after the number; titlesec suppresses first indent)
\titleformat{\section}{\fontsize{13}{15}\selectfont\bfseries}
  {\makebox[0.7cm][l]{\thesection}}{0pt}{\MakeUppercase}
\titlespacing*{\section}{0pt}{24pt}{12pt}
\titleformat{\subsection}{\fontsize{12}{13}\selectfont\bfseries}
  {\makebox[1.0cm][l]{\thesubsection}}{0pt}{}
\titlespacing*{\subsection}{0pt}{11pt}{11pt}
\titleformat{\subsubsection}{\fontsize{11}{12}\selectfont\bfseries}
  {\makebox[1.0cm][l]{\thesubsubsection}}{0pt}{}
\titlespacing*{\subsubsection}{0pt}{10pt}{10pt}

%% Lists: 6 pt after
\setlist{itemsep=1pt,topsep=3pt,partopsep=0pt,after=\vspace{3pt}}
\setlist[itemize]{leftmargin=1.2em}
\setlist[enumerate]{leftmargin=1.5em}

%% ------------------------------------------------------------ front matter
\def\@icaiitauthors{}\def\@icaiitaffil{}\def\@icaiitkeywords{}\def\@icaiitabstract{}
\newcommand{\icaiitauthors}[1]{\def\@icaiitauthors{#1}}
\newcommand{\icaiitaffiliations}[1]{\def\@icaiitaffil{#1}}
\newcommand{\keywords}[1]{\def\@icaiitkeywords{#1}}
\renewcommand{\abstract}[1]{\def\@icaiitabstract{#1}}

\newcommand{\@hangpar}[2]{%
  \par\noindent\hangindent=2cm\hangafter=1
  \fontsize{9}{11}\selectfont
  \makebox[2cm][l]{\textbf{#1}}#2\par}

\renewcommand{\maketitle}{%
  \twocolumn[{%
    \centering
    {\fontsize{15}{17}\selectfont\bfseries \@title\par}
    \vspace{24pt}
    \if@icaiitreview
      {\fontsize{11}{13}\selectfont Anonymous Author(s)\par}
      \vspace{4pt}
      {\fontsize{9}{11}\selectfont\itshape Affiliation and e-mail omitted for double-blind review\par}
    \else
      {\fontsize{11}{13}\selectfont \@icaiitauthors\par}
      \vspace{4pt}
      {\fontsize{9}{11}\selectfont\itshape \@icaiitaffil\par}
    \fi
    \raggedright
    \vspace{48pt}
    \begin{minipage}{\textwidth}
      \setlength{\parskip}{0pt}\justifying@icaiit
      \@hangpar{Keywords:}{\@icaiitkeywords}
      \vspace{12pt}
      \@hangpar{Abstract:}{\@icaiitabstract}
    \end{minipage}
    \vspace{30pt}
  }]%
}
\newcommand{\justifying@icaiit}{\rightskip=0pt\leftskip=0pt\parfillskip=0pt plus 1fil}

%% Unnumbered back-matter headings in the same style as sections
\newcommand{\acknowledgments}{\section*{Acknowledgments}}

%% IEEE-style reference list at 9 pt
\renewcommand{\refname}{References}
\let\@oldthebibliography\thebibliography
\renewcommand{\thebibliography}[1]{%
  \section*{References}%
  \fontsize{9}{11}\selectfont
  \list{\@biblabel{\@arabic\c@enumiv}}%
    {\settowidth\labelwidth{\@biblabel{#1}}%
     \leftmargin\labelwidth \advance\leftmargin\labelsep
     \itemsep 1pt \parsep 0pt
     \usecounter{enumiv}\let\p@enumiv\@empty
     \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy\clubpenalty4000\widowpenalty4000\sfcode`\.\@m}
\let\endthebibliography\endlist

%% Listings / code in Courier
\renewcommand{\ttdefault}{pcr}

%% Algorithm block (template 2.3.9): Courier 9 pt, numbered steps, 0.5 cm
%% indent per nesting level, caption below, 6 pt before / 12 pt after.
\RequirePackage{float}
\newfloat{algorithm}{tbp}{loa}
\floatname{algorithm}{Algorithm}
\captionsetup[algorithm]{position=bottom,skip=6pt}
\newenvironment{algobody}
  {\par\vspace{6pt}\noindent\ttfamily\fontsize{9}{11}\selectfont
   \setlength{\parindent}{0pt}\obeylines\raggedright}
  {\par\vspace{6pt}}
\newcommand{\ind}{\hspace*{0.5cm}}

%% Listing block (template 2.3.8): Courier 9 pt, left aligned, caption below.
\newfloat{listing}{tbp}{lol}
\floatname{listing}{Listing}
\captionsetup[listing]{position=bottom,skip=6pt}

\endinput
