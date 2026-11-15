# MVTec AD 2 access + evaluation checklist

Do these by hand (form is gated, not automatable). Update the status column,
never delete rows.

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Download via request form (mvtec.com/company/company-research/datasets) | DONE 2026-11-15 | can, vial, sheet_metal; CC BY-NC-SA 4.0 |
| 2 | Cite dataset: Heckler-Kram et al., arXiv:2503.21622, 2025 | TODO | goes in the paper repo and README |
| 3 | Register account on benchmark.mvtec.com evaluation server | TODO | before week 11 |
| 4 | Read submission rules: quota per account/tool/day | TODO | record the official quota here |
| 5 | Decide submission budget (aim: <= 6 tuning submissions, keep headroom) | TODO | reserve 2 for the final report runs |
| 6 | Confirm which metrics the server reports (SegF1 official; AUROC?) | TODO | |
| 7 | Decide calibration-normal policy: validation split only (QR6) | DONE | written into docs/hypotheses.md QR6 |
| 8 | Never upload memory-bank or calibration files anywhere; images only to server | DONE | policy, standing |
| 9 | Keep test_private images out of any public repo / notebook screenshot | DONE | policy, standing |

Local splits (verified 2026-11-15, on-disk layout):

| category | train | validation | test_public good/bad(+GT) | test_private | test_private_mixed |
|----------|-------|------------|--------------------------|--------------|--------------------|
| can | 412 | 46 | 72 / 90 (+90) | 321 | 321 |
| vial | — | — | — | 276 | 276 |
| sheet_metal | — | — | — | 142 | 142 |

(to fill: vial/sheet_metal counts — same command pattern as can.)
