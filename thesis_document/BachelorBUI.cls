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

\addtokomafont{captionlabel}{\bfseries}

\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage[babel]{microtype}     % improves the micro-typographic layout
\RequirePackage[english,ngerman]{babel} %% englisch and austrian language


\input{glyphtounicode.tex}    %Part of modern distribution
\input{glyphtounicode-cmr.tex}     %Additionnal glyph: You must grab it from pdfx package
\pdfgentounicode=1
\pdfinterwordspaceon
\pdfomitcharset 1
\usepackage{colorprofiles}
\usepackage{cmap}
\usepackage{hyperref}

\hypersetup{ % Setzt einige Werte die in den Eigenschaften des PDF gespeichert sind.
 bookmarksnumbered,
%   % pdfdisplaydoctitle = true,
  colorlinks, % Für Druck auf "false" setzen!
  linkcolor={black}, % auch keine grellen Rahmen im View
  citecolor={black},
  urlcolor={black}
}

%% lines at - see: http://www.komascript.de/chapterwithlines
%\renewcommand{\chapterlineswithprefixformat}[3]{%
%  {\raggedleft #2}\nobreak
%  \rule[.5\baselineskip]{\textwidth}{1pt}\\*
%  #3\par\nobreak
%  \rule[-.5\baselineskip]{\textwidth}{1pt}%
%}
%% Linien bei Überschriften - Ende

%% Absatzkontrolle
\clubpenalty10000
\widowpenalty10000
\displaywidowpenalty=10000

\RequirePackage{graphicx} 
%Erweiterung graphic paket, optionale Parameter für \includegraphics
% Seitenrand neu definieren
\RequirePackage[left=2.5cm,right=2.5cm,top=1.5cm,bottom=2.5cm,includehead]{geometry}
%Seitenlayout anpassen
\RequirePackage[
headsepline,
automark
]{scrlayer-scrpage} %Kopf und Fußzeile

\pagestyle{scrheadings}
\clearpairofpagestyles
\ihead{\headmark}
\ohead{\thepage}

\automark[section]{section}
\automark*[subsection]{}

\RequirePackage{setspace}

\RequirePackage[list=true]{subcaption}

\RequirePackage{mathtools} %% extension of amsmath (also loads amstext)
\RequirePackage{amssymb}   %% for complex formulas
%Correct UTF8 mapping for ams fonts
\ifdefined\pdffontattr% \ifdefined is part of the e-TeX extension, which is part of any modern LaTeX compiler.
  \immediate\pdfobj stream file {umsa.cmap}
  {\usefont{U}{msa}{m}{n}\pdffontattr\font{/ToUnicode \the\pdflastobj\space 0 R}}
  \immediate\pdfobj stream file {umsb.cmap}
  {\usefont{U}{msb}{m}{n}\pdffontattr\font{/ToUnicode \the\pdflastobj\space 0 R}}
\fi
\RequirePackage{xfrac}  %% diagonals fractions - e.g. numbers
\RequirePackage{siunitx} %% SI Units

\RequirePackage{booktabs} %% for better looking tables 

\RequirePackage[newcommands]{ragged2e}
\RequirePackage{tabularx}
\RequirePackage{longtable}
\RequirePackage[para,flushleft]{threeparttable}
\RequirePackage{multirow}

\PassOptionsToPackage{table}{xcolor}
% \RequirePackage{xcolor} %% improvement to the color package, included by pdfx 
% muss jetzt vor pdfx geladen werden
\RequirePackage{url} %% links inside the document as well as to websites

\RequirePackage{pdfpages} %% to include pdf pages

\RequirePackage{marvosym} %% additional symbols
\RequirePackage{pifont}   %% additional symbols
\RequirePackage{eurosym} %% for correct Euro-Symbol with \officialeuro{}

\RequirePackage{xspace}  %% for intelligent spacing after commands \xspace

\RequirePackage{enumitem}  %% to customize itemize, enumerate, description

\RequirePackage{listings} %% syntax highlighting - for program code
\RequirePackage{scrhack} %% solve some problems with float of listings

% Eigene Sprachabhängige Begriffe
\renewcaptionname{ngerman}{\tablename}{Tab.}
\renewcaptionname{ngerman}{\figurename}{Abb.}
\renewcaptionname{ngerman}{\bibname}{Literaturverzeichnis}
\renewcaptionname{ngerman}{\abstractname}{Kurzfassung}

\renewcaptionname{english}{\tablename}{Tab.}
\renewcaptionname{english}{\figurename}{Fig.}

%% Title Format
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
%\title{\@thesistitle}

%% Typografische Einstellungen zu Gleitobjekten
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