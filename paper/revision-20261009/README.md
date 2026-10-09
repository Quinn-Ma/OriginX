# OriginX revised manuscript source

Research manuscript, revised October 9, 2026. No new training, rollout, or model call was performed for this revision.

## Build

The main paper and supplement use the unmodified official `cvpr-org/author-kit` style downloaded October 9, 2026. The official kit currently identifies its annual update as CVPR 2026; final CVPR 2027 compliance is not asserted. Main text ends on page 8; page 9 contains references only. The supplement has 5 pages.

From this directory, use Tectonic 0.17.0 (the compiler used for these PDFs):

```
tectonic --keep-logs main.tex
tectonic --keep-logs supplement.tex
```

The source also uses standard LaTeX packages. Do not change template margins or shrink body fonts to meet a page limit. `cvpr.sty` is the original upstream file; `fontenc` ensures Times-family fonts resolve under the XeTeX/Tectonic engine.

## Files and evidence boundaries

- `main.tex`, `references.tex`: revised argument, all 17 verified references, and Future Directions.
- `supplement.tex`, `supplement_body.tex`, `task_table.tex`: complete methods and all 50 original task rows.
- `analysis_macros.tex`, `task_analysis.tex`: frozen numerical summaries, generated from the retrospective audit.
- `figures/`: protocol diagram, descriptive rescue-yield plot, and actual intervention-input camera views.
- `analysis_statistics.json`: computed summaries without identifying absolute evidence paths; these statistics do not substitute for raw records.
- `audit_analysis.py`: read-only raw-evidence verifier, retained with its documented workspace layout. Re-running it requires the original campaign archives in that layout and does not invoke models.

The raw source records remain unchanged: historical success 1496/2500, rescue classification 56/813/28/107 over 1004 unique failures. Task resampling is exploratory task-mixture sensitivity, not causal significance or a confidence interval for a finite census. The four development IDs remain disclosed. The paper does not report new full-benchmark or organizer-accepted scores.

These PDFs omit personal author names, project links, and identifying filesystem paths. Full submission anonymization still requires human review of the public project name, citations, source and supplementary artifacts under the final conference policy. This source package is not a submitted or accepted conference paper.

Template source: https://github.com/cvpr-org/author-kit
2027 call: https://cvpr.thecvf.com/Conferences/2027/CallForPapers
