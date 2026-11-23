# Registered hypotheses and research questions

Registered before running any experiment. Each entry is dated and falsifiable.
Later results must reference these IDs; hedges and amendments get a new dated
section, never an in-place edit.

Registration: 2026-11-15 (week 0, before any LightShift experiment ran).
Prior in-house evidence (single category, clean + 2 shifts, raw-vs-clipped
scores) is listed at the bottom as context, not as confirmation.

## QR1 — Does per-channel feature-statistic drift predict accuracy drop?

Hypothesis: across lighting shifts (gamma, brightness, contrast, colour
temperature, directional shading; 3-5 severities) and layers (layer2, layer3),
the Cohen's-d-style shift of per-channel patch-embedding statistics
(||mu_shift - mu_ref||, mean over channels) correlates with AUROC drop with
Spearman rho >= 0.6 per category.

Falsified if median rho across categories/shift families < 0.35 or if any
modifier (B2 variant) breaks the rank ordering implied by measured drift.

## QR2 — AUROC vs F1 at fixed threshold disagree under shift

Hypothesis: at severity >= 2 shifts, measuring AUROC only overstates usable
quality: F1 (image level) at the source-lighting threshold drops by >= 15
percentage points more than AUROC, on at least one of the three categories.

Falsified if median (AUROC drop - F1@source-threshold drop) over
categories x shifts x severities < 5 points.

## QR3 — B3 conformal threshold retains coverage under shift

Hypothesis: with n >= 20 target normals and alpha = 0.1, the B3 threshold keeps
realized per-shift FPR <= alpha + one small-sample quadratic band implied by
Beta(n+1-k,k), without retraining, for every lighting shift family, in >= 2 of
3 categories. A naive source threshold does not satisfy this at severity >= 2.

Falsified if realized FPR exceeds alpha + band in >= 2 of 3 categories.

## QR4 — B2 alignment recovers accuracy with small n

Hypothesis: AdaBN-style alignment from k=10 target normals recovers >= 50% of
the AUROC drop measured between n=0 and n=50 (the "n-curve"), averaged over
categories and moderate-severity shifts; no variant erases discoloured defects
(negative control: defect visible at reference, colour-matched at target, must
stay detectable at >= 80% of reference AU-PRO).

## QR5 — Cost budget

Hypothesis: the full serving path (predict + capacitate-to-n<=50 calibration)
runs within 2x vanilla latency, p95, on the hardware used for the report, with
bank size held at the reproduction config.

## QR6 — External validity on MVTec AD 2

Hypothesis: gains transfer to the MVTec AD 2 can/vial/sheet_metal public test
splits; at least the sign of (proposed - vanilla) image-level SegF1 /
AUROC delta is preserved for moderate shifts, using calibration normals drawn
from validation split only.

## Prior in-house evidence (context only, 2026-11-12 measurements)

- Raw (unclamped) anomalies scores are required: clamped post-processed scores
  collapse the score distribution used by B2/B3. Seed the raw-score forward-hook
  pattern from `apps/vision-inspection/raw_score_sweep.py` of smartex-lab-p0.
- The sujet's B3 wording ("k-th largest") is backwards; correct reading is
  k-th smallest, else n=50, alpha=0.1 rejects ~90% of good parts. Fix
  documented, tests to be written here.
- Pilot drift-vs-AUROC table exists for one category and two shifts; did not
  yet implement the QR1 correlation proper (no severity ladder, no per-layer).

---

## Amendment 2026-11-23 — corrections and results (appended; nothing above was edited)

### Corrections to the earlier pilot (scripts/pilot_shift_eval.py)
- **B2a oracle.** The pilot estimated the mean gap as mu(shifted target photos) - mu(the same photos, clean). A deployment has no clean twins.
  The deployable estimator compares the k target photos with the memory-bank images' own moments (src/lightshift/b2.py).
- **B2b bug.** The pilot multiplied centred features by the target std instead of rescaling by sd_ref / sd_target. The corrected B2b is unit-tested.
  The pilot's "B2b fails" is withdrawn; it was an implementation error, not a finding.
- **Gamma direction.** x**g with g < 1 *brightens*; the comment in shift.py said "darkens". Comment fixed; parameters unchanged.

### QR4 (protocol: scripts/qr4_ncurve.py; data: results/qr4/; summary: results/qr4/qr4_summary.md)
Seeded split of train/good: 60% bank, 10% source calibration, 30% target pool. 5 families x severities {2,3,4} + clean control,
k in {5,10,25,50}, 10 draws per k, 3 categories (bottle, hazelnut, leather).
- Reading of the registered wording used: gain(k=10)/gain(k=50) >= 0.5 on conditions with gain(50) > 0.005.
  Result: median ratio 1.00 for B2a and B2b; >= 0.5 in 100% of conditions. B2a saturates already at k = 5; B2b needs k >= 25 for a stable false-alarm rate.
- Share of the vanilla AUROC drop recovered (7 of 45 conditions with drop >= 0.01): B2a ~ 0.48, corrected B2b ~ 0.84.
- B2a at k=10 raises TPR at the conformal threshold from 96.1% (B3 only) to 98.8%; deployable B2a matches the paired-twin oracle (AUROC 0.996 both).
- **Negative control (discolouration) was NOT run.** A related check, calibration contamination, is in results/contamination/ (see report Part D).
- Verdict: partially supported (see report D3 for the caveats).

### QR3 (same runs)
- Source threshold at severities 2/3/4: mean FPR 31% / 36% / 52% (vs ~12% clean).
- Conformal threshold from k photos: mean FPR 17.6% (k=5), 9.6% (10), 7.1% (25), 9.2% (50). Theory (n+1-k)/(n+1): 16.7%, 9.1%, 7.7%, 9.8%.
- Per category at k=25: hazelnut 2%, leather 7%, bottle 12% (k=50: 18%). Two of three categories at or below alpha: not falsified under the registered rule.
- Guarantee needs n >= (1-alpha)/alpha = 9 photos for alpha = 0.1; k=5 shows the predicted excess.

### QR2 (computed 2026-11-23 from results/qr4/qr4_rows.csv; vanilla, source threshold; 45 shifted conditions)
F1 derived from stored FPR/TPR and test-set sizes. Median (F1 drop - AUROC drop) = 1.4 points (bottle 2.3, hazelnut 0.0, leather 1.4) < 5: **falsified by the registered rule.**
The >= 15-point effect occurs in 4 of 45 conditions, all hazelnut (max 20.6). Cause: defect-heavy test sets (63/83, 70/110, 92/124 defective): flagging everything still gives
F1 = 2*prevalence/(1+prevalence) ~ 0.78-0.86. FPR is prevalence-independent and is the metric used from here on.

### QR1 (scripts/qr1_drift.py; results/qr1/)
Spearman rho(drift_all, AUROC drop) = 0.52 bottle, 0.77 hazelnut, 0.57 leather (median 0.57): above the falsification line (0.35), below the 0.6 target in two of three categories: **partially supported.**
rho(drift_all, FPR at source threshold) = 0.76-0.89; rho(drift_all, mean good score) = 0.87-0.95.

### QR5 (scripts/qr5_latency.py; results/qr5/)
Serving path (B2a + B3) p95 / vanilla p95 = 0.92 (CPU), 1.00 (GPU); one-off calibration with k=10 photos 0.6 s (GPU). **Supported.** CPU absolute numbers were measured under heavy machine load; only the ratio is reliable.

### QR6 (scripts/qr6_natural.py; results/qr6/): REAL lighting in AD 2 test_public
AD 2 test_public contains each scene under regular / overexposed / underexposed / shift_1..4, which gives a natural lighting-shift test (train and validation are all regular).
Vanilla source-threshold FPR: vial overexposed 80%, underexposed 100%; sheet_metal shift_2 50%; AUROC vial 0.95 (regular) -> 0.72 (overexposed).
B3 (k photos): FPR = 33/25/20% for k=2/3/4 on vial and 8.9/8.3% for k=8/11 on can, matching 1/(k+1). B2a neutral-to-slightly-negative on vial, +0.01..0.02 on sheet_metal; B2b collapses on sheet_metal at k=2-3. Can baseline at chance.
**Verdict: B3 transfers; B2 not demonstrated** (k <= 4, tiny samples). The earlier synthetic-shift-on-AD2 plan was dropped in favour of this.

### B1 (scripts/run_extra.py b1; results/b1/): augmented bank at equal memory
Source-threshold FPR 39.7% (vanilla) -> 19.7% (all families augmented), but unseen families do not improve (shading 90%, colour_temp 28%). AUROC under shading 0.973 -> 0.998 (seen) vs B2a 0.981.
B1 gives no coverage guarantee; B3 on top gives 9.6%. The "seen" arm uses the same generator/seeds as the test shifts (near-oracle).

### QR5 re-measurement and calibration contamination (2026-11-23)
- QR5 repeated (results/qr5/qr5.json; first run kept as qr5_loaded_run1.json): p95 ratio serving/vanilla 0.97 (CPU), 0.95 (GPU). CPU p50 ~ 307-327 ms under load average 4-8 on 6 cores (threads 10): absolute CPU numbers are not benchmark quality; the earlier informal 84 ms figure was on a quieter machine.
- Contamination (scripts/contamination.py; results/contamination/): m defective photos among the k calibration photos. k=10 (threshold = max): m=1 drops TPR 91% -> 52%, m=3 -> 29%; B2a gives no protection.
  k=50 (k_rank=46): TPR flat at 91% for m = 0, 2, 4, then 79% at m=5 and 57% at m=8: cliff at m > n-k, as the rank argument predicts. Discolouration negative control still NOT run.
