# ELEC 594 Capstone Midterm Report (Fall 2026)

**Robot Learning for Household Tasks: SO101 Demonstrations, OriginX Adaptation, and Training-Free Execution**
Qinzhen Ma, MECE Program, Rice University

- Report PDF: [`ELEC594_Midterm_Report_Qinzhen_Ma.pdf`](ELEC594_Midterm_Report_Qinzhen_Ma.pdf)
- LaTeX source: `main.tex` (IEEE conference style), bibliography `refs.bib`
- `IEEEtran.cls` and `IEEEtran.bst` are included so the report builds without extra packages.
- Figures: `figures/so101_frames.jpg` (Fig. 1); Figs. 2–4 are drawn with TikZ/pgfplots inside `main.tex`.

## Rebuild

```bash
latexmk -pdf main.tex
```

or, without latexmk: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
Required packages (all in a standard TeX Live install): tikz, pgfplots, booktabs, xcolor, microtype, cite, hyperref.

## Regenerate Fig. 1

`scripts/extract_so101_frames.py` reads a local LeRobot v3 dataset and writes the frame montage:

```bash
python scripts/extract_so101_frames.py <path/to/so101_pick_place_v3> 0 figures/so101_frames.jpg
```

It needs `av`, `pandas`, `pyarrow` and `Pillow` (all present in a LeRobot environment). The dataset itself is not stored in this repository.
