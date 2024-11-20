\ProvidesClass{BachelorBUI}[2022/10/01]

\LoadClass{scrartcl}
\KOMAoptions{
fontsize=11pt,
parskip=false,
abstract=true,
twoside,
captions=tableheading
}

\RedeclareSectionCommand[
  beforeskip=1em,
  afterskip=0.5em,
  indent=0em,
  runin=false,
  afterindent=false
]{paragraph}

\RedeclareSectionCommand[
  beforeskip=1em,
  afterskip=0.5em,
  indent=2em,
  runin=false,
  afterindent=false
]{subparagraph}

% \RedeclareSectionCommand[
%     beforeskip=1em,
%     afterskip=0.5em,
%     indent=0em,
%     runin=false,
%     afterindent=false,
%     font=\normalfont\bfseries
% ]{paragraph}

% \RedeclareSectionCommand[
%     beforeskip=1em,
%     afterskip=0.5em,
%     indent=2em,
%     runin=false,
%     afterindent=false,
%     font=\normalfont\itshape
% ]{subparagraph}

\addtokomafont{captionlabel}{\bfseries}

\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage[babel]{microtype}
\RequirePackage[english]{babel}


\input{glyphtounicode.tex}
\input{glyphtounicode-cmr.tex}
\pdfgentounicode=1
\pdfinterwordspaceon
\pdfomitcharset 1
\usepackage{colorprofiles}
\usepackage{cmap}
\usepackage{hyperref}

\hypersetup{
 bookmarksnumbered,
%   % pdfdisplaydoctitle = true,
  colorlinks,
  linkcolor={black},
  citecolor={black},
  urlcolor={black}
}

\clubpenalty10000
\widowpenalty10000
\displaywidowpenalty=10000

\RequirePackage{graphicx}
\RequirePackage[left=2.5cm,right=2.5cm,top=1.5cm,bottom=2.5cm,includehead]{geometry}
\RequirePackage[
headsepline,
automark
]{scrlayer-scrpage}

\pagestyle{scrheadings}
\clearpairofpagestyles
\ihead{\headmark}
\ohead{\thepage}

\automark[section]{section}
\automark*[subsection]{}

\RequirePackage{setspace}

\RequirePackage[list=true]{subcaption}

\RequirePackage{mathtools}
\RequirePackage{amssymb}

\ifdefined\pdffontattr
  \immediate\pdfobj stream file {umsa.cmap}
  {\usefont{U}{msa}{m}{n}\pdffontattr\font{/ToUnicode \the\pdflastobj\space 0 R}}
  \immediate\pdfobj stream file {umsb.cmap}
  {\usefont{U}{msb}{m}{n}\pdffontattr\font{/ToUnicode \the\pdflastobj\space 0 R}}
\fi
\RequirePackage{xfrac}
\RequirePackage{siunitx}

\RequirePackage{booktabs}

\RequirePackage[newcommands]{ragged2e}
\RequirePackage{tabularx}
\RequirePackage{longtable}
\RequirePackage[para,flushleft]{threeparttable}
\RequirePackage{multirow}

\PassOptionsToPackage{table}{xcolor}
\RequirePackage{url}

\RequirePackage{pdfpages}

\RequirePackage{marvosym}
\RequirePackage{pifont}
\RequirePackage{eurosym}

\RequirePackage{xspace}

\RequirePackage{enumitem}

\RequirePackage{listings}
\RequirePackage{scrhack}

\renewcaptionname{english}{\bibname}{References}
\renewcaptionname{english}{\abstractname}{Abstract}
\renewcaptionname{english}{\tablename}{Tab.}
\renewcaptionname{english}{\figurename}{Fig.}

\subject{Bachelor's Thesis}
\addtokomafont{subject}{\normalfont\sffamily\huge}
\newcommand{\authorname}[1]{\renewcommand\@authorname{#1}}
\newcommand\@authorname{\@latex@error{No \noexpand\authorname specified}\@ehc}
\newcommand{\email}[1]{\renewcommand\@email{#1}}
\newcommand\@email{\@latex@error{No \noexpand\email specified}\@ehc}
\newcommand{\MatrNr}[1]{\renewcommand\@MatrNr{#1}}
\newcommand\@MatrNr{\@latex@error{No \noexpand\MatrNr specified}\@ehc}
\newcommand{\thesislanguage}[1]{\renewcommand\@thesislanguage{#1}}
\newcommand\@thesislanguage{\@latex@error{No \noexpand\thesislanguage specified}\@ehc}
\newcommand{\keywords}[1]{\renewcommand\@keywords{#1}}
\newcommand\@keywords{\@latex@error{No \noexpand\keywords specified}\@ehc}

\renewcommand{\floatpagefraction}{0.6}
\renewcommand{\textfraction}{0.15}
\renewcommand{\topfraction}{0.8}
\renewcommand{\bottomfraction}{0.5}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}


\author{{\normalsize \sffamily \@authorname} \\ [2mm]
 {\small \@email} \\ [-2mm]
 {\small Student ID: \@MatrNr} \\ [-2mm]
 {\small Date: \today} }
\addtokomafont{author}{\sffamily}
\date{}
%

%% Customize Listing
\lstset{
	basicstyle = \footnotesize\ttfamily,
	keywordstyle=\color{blue}\bfseries,
	tabsize=3,
	breaklines=true,
	showstringspaces = false,
	upquote = true,
	xleftmargin=4mm,
	xrightmargin=.25in,
	aboveskip=15pt,
	belowskip=12pt,
	belowcaptionskip=12pt,
	numbers=left,
	numberstyle=\tiny,
	frame=tb,
	moredelim=**[is][\color{blue}]{@}{@}
}
\lstset{literate=%
   {Ö}{{\"O}}1
   {Ä}{{\"A}}1
   {Ü}{{\"U}}1
   {ß}{{\ss}}1
   {ü}{{\"u}}1
   {ä}{{\"a}}1
   {ö}{{\"o}}1
}
\renewcommand{\lstlistingname}{Program Code}
