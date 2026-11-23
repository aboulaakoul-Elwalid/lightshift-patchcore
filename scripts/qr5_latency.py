"""QR5: cost of the serving path vs vanilla (CPU and GPU, 256 px, bottle bank).

vanilla : backbone -> L2 norm -> NN distance -> max
serving : same + B2a mean subtraction + threshold compare  (per image)
calibration (one-off, k target normals): k forward passes + moments + conformal threshold
p50/p95 over 200 timed images after 20 warmup. Writes results/qr5/qr5.json
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr4_ncurve as q  # noqa: E402
from lightshift import b2  # noqa: E402
from lightshift.b3 import conformal_threshold  # noqa: E402
from lightshift.patchcore import Extractor, build_bank, l2norm  # noqa: E402

OUT = q.REPO / "results" / "qr5"
OUT.mkdir(exist_ok=True)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def pct(xs, p):
    return float(np.percentile(xs, p))


def main():
    res = {}
    bank_f, cal_f, pool_f, good_f, bad_f = q.split("bottle")
    imgs = [Image.open(p).convert("RGB") for p in (good_f + bad_f)[:60]]
    for dev_name in ("cpu", "cuda"):
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        dev = torch.device(dev_name)
        ext = Extractor(dev)
        raw = q.feats(ext, bank_f)
        ref = b2.pool([b2.image_moments(x) for x in raw])
        bank = build_bank(list(raw), 0.25, 0)
        del raw
        tgt = b2.pool([b2.image_moments(ext.raw(i)) for i in imgs[:10]])
        thr = 0.5

        def vanilla(img):
            f = l2norm(ext.raw(img))
            return float((2 - 2 * f @ bank.T).min(1).values.max())

        def serving(img):
            f = l2norm(b2.align(ext.raw(img), "mean", tgt, ref))
            s = float((2 - 2 * f @ bank.T).min(1).values.max())
            return s > thr

        out = {}
        for name, fn in (("vanilla", vanilla), ("serving_b2a_b3", serving)):
            for i in range(20):
                fn(imgs[i % len(imgs)])
            ts = []
            for i in range(200):
                sync(dev)
                t0 = time.perf_counter()
                fn(imgs[i % len(imgs)])
                sync(dev)
                ts.append((time.perf_counter() - t0) * 1000)
            out[name] = {"p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95), "mean_ms": st.mean(ts)}
        out["p95_ratio"] = out["serving_b2a_b3"]["p95_ms"] / out["vanilla"]["p95_ms"]
        cal = {}
        for k in (10, 50):
            pool = [Image.open(p).convert("RGB") for p in pool_f[:k]]
            sync(dev)
            t0 = time.perf_counter()
            mom = [b2.image_moments(ext.raw(i)) for i in pool]
            m = b2.pool(mom)
            sc = []
            for i in pool:
                f = l2norm(b2.align(ext.raw(i), "mean", m, ref))
                sc.append(float((2 - 2 * f @ bank.T).min(1).values.max()))
            conformal_threshold(np.array(sc), 0.1)
            sync(dev)
            cal[f"k={k}_total_s"] = time.perf_counter() - t0
        out["calibration"] = cal
        out["bank_shape"] = list(bank.shape)
        out["loadavg_at_start"] = list(__import__("os").getloadavg())
        out["torch_threads"] = torch.get_num_threads()
        res[dev_name] = out
        print(dev_name, json.dumps(out, indent=1), flush=True)
    res["peak_rss_mb"] = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss // 1024
    (OUT / "qr5.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
