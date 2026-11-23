"""QR6 (AD2 transfer) and B1 (augmented bank at equal memory) drivers.

  python scripts/run_extra.py qr6 [cats...]     -> results/qr6/qr6_rows.csv
  python scripts/run_extra.py b1  [cats...]     -> results/b1/b1_rows.csv

QR6 protocol (AD2, policy: calibration normals only from the validation split):
  bank = train/good, source calibration + target pool = validation/good,
  test = test_public good/bad. Same synthetic shifts, same B2/B3 code as QR4.
B1: bank built from randomly shifted images (same bank size as vanilla).
  b1_seen   augments with all five families
  b1_heldout augments with gamma/brightness/contrast only; color_temp & shading are unseen
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr4_ncurve as q  # noqa: E402

AD2 = Path("/mnt/data-ssd/mvtec-ad-2")


def ad2_split(cat):
    r = AD2 / cat
    ls = lambda d: sorted(p for p in (r / d).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))  # noqa: E731
    val = ls("validation/good")
    return ls("train/good"), val, val, ls("test_public/good"), ls("test_public/bad")


def dump(rows, path):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    mode, cats = sys.argv[1], sys.argv[2:]
    if mode == "qr6":
        out = q.REPO / "results" / "qr6"
        q.OUT = out
        out.mkdir(exist_ok=True)
        rows = []
        for c in cats or ["vial", "sheet_metal", "can"]:
            rows += q.run_cat(c, splitter=ad2_split, ks=(5, 10, 25), variants=("b2a", "b2b"), tag="ad2_")
            dump(rows, out / "qr6_rows.csv")
    elif mode == "b1":
        out = q.REPO / "results" / "b1"
        q.OUT = out
        out.mkdir(exist_ok=True)
        rows = []
        for c in cats or ["bottle", "hazelnut", "leather"]:
            for arm, fams in (("b1_seen", list(q.SHIFT_FAMILIES)), ("b1_heldout", ["gamma", "brightness", "contrast"])):
                r = q.run_cat(c, ks=(10,), variants=("b2a",), aug_families=fams, tag=arm + "_")
                for x in r:
                    x["arm"] = arm
                rows += r
                dump(rows, out / "b1_rows.csv")
    else:
        raise SystemExit("usage: run_extra.py qr6|b1 [cats]")
    print("done")
