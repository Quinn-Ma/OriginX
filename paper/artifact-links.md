# OriginX artifact links (identifying; exclude from anonymous submission)

Author: Qinzhen Ma. Updated October 9, 2026.

- Model weights only: https://huggingface.co/Qinzhen3/OriginX
- Public repository: https://github.com/Quinn-Ma/OriginX
- Original benchmark release: https://github.com/Quinn-Ma/OriginX/releases/tag/v1.0.0
- Astra rescue evidence release: https://github.com/Quinn-Ma/OriginX/releases/tag/v1.1.0-astra-rescue
- Published project website: https://robocasa.originxairobotics.com
- Official RoboCasa365 submission pull request: https://github.com/robocasa-benchmark/leaderboard/pull/26 — submitted and pending review; not accepted.
- Audited source release commit: `e04001d5a6ccb7836bdd8b18be8c11ac28d5aee7`

The CVPR 2027 draft is an internal research manuscript, not a submitted or accepted paper. Its default LaTeX author is anonymous, and this identifying note is not included or imported into the PDF. Before any formal submission, check the actual CVPR 2027 instructions, author/affiliation details, anonymization and preprint rules, formatting, reproducibility claims, and the external-assistance disclosure policy. Public project availability does not establish leaderboard acceptance. The per-episode results in the paper remain fixed regardless of release status.

The manuscript currently describes a completed author-run 59.84% evaluation and a paired confirmation that did not establish reliable gain. The completed conditional Astra replay now reports 56 strict rescues / 1,004 historical B2000 failures, 813 not rescued, 28 unknown, and 107 initial-state deviations; all attempts are terminal, but valid outcomes are incomplete. This does not create a new 2,500-episode score. The draft does not claim first place or a conference award.


## Revised manuscript, October 9, 2026

- [Main paper](OriginX_CVPR2027_revised.pdf): One-Shot Language Assistance under Matched Robot Replays; Future Directions in Section 7.
- [Supplement](OriginX_CVPR2027_supplement.pdf): full methods and task-level record.
- [Editable LaTeX](revision-20261009/) and [source bundle](OriginX_CVPR2027_source.zip).
- [Revision QA](OriginX_CVPR2027_revision_qa.json).

The earlier draft is preserved as a historical version. No training, rollout or assistant call was added for this revision.
