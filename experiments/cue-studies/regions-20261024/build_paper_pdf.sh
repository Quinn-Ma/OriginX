#!/usr/bin/env bash
# Build main.pdf in the given paper directory with the WSL TinyTeX toolchain.
set -euo pipefail
cd "$1"
export PATH="$HOME/.local/bin:$PATH"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
grep -E "Output written|undefined|Overfull|LaTeX Warning" main.log | head -30 || true
