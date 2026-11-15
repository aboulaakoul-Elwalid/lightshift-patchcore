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

## Finding — 2026-11-15: vanilla PatchCore on AD2 can is near-random at image level (expected)
Fitted anomalib PatchCore (WRN-50, layer2+3, coreset 0.25, 256px) on TESTpub can:
image_AUROC 0.45–0.46, pixel_AUROC ~0.71. Verified NOT a bug:
- WideResNet-50 ImageNet weights confirmed loading (timm HF cache, allclose vs fresh create_model).
- Direct minimal reimplementation outside anomalib reproduces AUROC ~0.38 (30v30, pretrained features).
- Per-lighting-condition AUROC also ~0.4; alternative score stats (p99/top10/mean) don't help; argmax lands on mask 1/30.
Root cause is the dataset itself: can defects are pin-prick sized (median fg fraction 5.6e-5 of a 2232x1024 image),
which vanish at 256px; plus large pose variability. Consistent with the MVTec AD 2 paper (arXiv:2503.21622):
PatchCore AU-PRO0.05 can = 4.7% TESTpriv / 4.6% mix (Table 7), F1 0.3% (Table 8), mean AU-PRO 28.8;
half-resolution input lifts PatchCore mean to 62.3 (Table 10). Official protocol resizes to 256x256 (confirmed).
Conclusion: do NOT use AD2 can for a naive image-AUROC sanity gate; reproduction gate runs on MVTec AD.
AD2 fits are still useful for the robustness study but expect low absolute numbers (paper-matched).
