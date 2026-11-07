#!/usr/bin/env python
"""Paired per-shape analysis for Option B (bootstrap over TEST SHAPES, seeds averaged per shape)."""
import argparse, json, os
from collections import defaultdict
import numpy as np
from halfhop.stats_utils import paired_bootstrap, fmt_ci

METRICS = ["miou", "b_miou", "b_acc", "acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--oracle_gate_pts", type=float, default=1.0, help="oracle must beat random by this many boundary-mIoU points")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.dir, "runs.jsonl"))]
    gk = lambda r: (r["category"], r["backbone"], r["depth"], r["feat"], r["dense"], r["residual"])
    ak = lambda r: (r["arm"], r["budget"], r["alpha"])
    shp = defaultdict(lambda: defaultdict(list))        # group -> armkey -> [npz dict per seed]
    for r in rows:
        shp[gk(r)][ak(r)].append(dict(np.load(os.path.join(a.dir, "per_shape", r["run_id"] + ".npz"))))

    def mean_over_seeds(lst, m):
        with np.errstate(all="ignore"):
            return np.nanmean(np.stack([d[m] for d in lst]), axis=0)

    out = ["# Option B summary", "",
           "Δ in points, mean over test shapes (seeds averaged per shape), 95% bootstrap CI over shapes; * = CI excludes 0. "
           "`oracle_*` uses GT labels (upper bound). Boundary = point with a differently-labelled kNN neighbour.", ""]
    gate = []
    for g, arms in sorted(shp.items()):
        cat, bb, depth, feat, dense, res = g
        out += [f"## {cat} | {bb} | depth {depth} | feat={feat} | dense={dense} residual={res}", ""]
        base = arms.get(("none", None, None))
        hdr = ["arm", "budget", "α", "slowed", "mIoU"] + [f"Δ{m} vs none" for m in ("miou", "b_miou")] + [f"Δ{m} vs rand" for m in ("miou", "b_miou", "b_acc")]
        out += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for ak_, lst in sorted(arms.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2] or 0)):
            arm, budget, alpha = ak_
            rnd = None
            if "_" in arm:
                rnd = arms.get(("rand_" + arm.split("_")[1], budget, alpha))
            cells = []
            for ref, ms in ((base, ("miou", "b_miou")), (rnd, ("miou", "b_miou", "b_acc"))):
                for m in ms:
                    if ref is None or ref is lst or (ref is rnd and arm.startswith("rand_")):
                        cells.append("—")
                    else:
                        cells.append(fmt_ci(paired_bootstrap(mean_over_seeds(lst, m) - mean_over_seeds(ref, m))))
            frac = np.mean([r["frac_edges_slowed"] for r in rows if gk(r) == g and ak(r) == ak_])
            out.append(f"| {arm} | {budget} | {alpha} | {frac:.3f} | {100 * np.nanmean(mean_over_seeds(lst, 'miou')):.2f} | " + " | ".join(cells) + " |")
        out.append("")
        # oracle gate + recovery fraction (geometry gain as a share of oracle gain, both over random)
        for lv in ("node", "edge"):
            for (arm, budget, alpha), lst in arms.items():
                if arm != f"oracle_{lv}":
                    continue
                rnd = arms.get((f"rand_{lv}", budget, alpha))
                if rnd is None:
                    continue
                d = paired_bootstrap(mean_over_seeds(lst, "b_miou") - mean_over_seeds(rnd, "b_miou"))
                ok = d["lo"] > 0 and 100 * d["mean"] >= a.oracle_gate_pts
                gate.append((g, lv, budget, alpha, d, ok))
                geo = arms.get((f"geom_{lv}", budget, alpha))
                if geo is not None:
                    og = np.nanmean(mean_over_seeds(lst, "b_miou") - mean_over_seeds(rnd, "b_miou"))
                    gg = np.nanmean(mean_over_seeds(geo, "b_miou") - mean_over_seeds(rnd, "b_miou"))
                    out.append(f"- level={lv} b={budget} α={alpha}: geometry recovers {100 * gg / og:.0f}% of the oracle's boundary-mIoU gain over random "
                               f"(oracle {100 * og:+.2f} pts, geom {100 * gg:+.2f} pts)." if abs(og) > 1e-9 else f"- level={lv}: oracle gain ~0.")
        out.append("")
    out += ["## Oracle gate (kill criterion)", "",
            f"Oracle must beat matched-random on boundary mIoU by >= {a.oracle_gate_pts} pt with a CI above 0.", "",
            "| group | level | budget | α | oracle − random (b-mIoU pts) | gate |", "|---|---|---|---|---|---|"]
    for g, lv, b, al, d, ok in gate:
        out.append(f"| {g[0]}/{g[1]}/L{g[2]} | {lv} | {b} | {al} | {fmt_ci(d)} | {'PASS' if ok else 'FAIL'} |")
    out += ["", "If the gate FAILS on the categories you care about, selective slowing is not the lever: stop and report the negative result. "
            "Only if it passes is a geometry-vs-random comparison worth reading.", ""]
    path = a.out or os.path.join(a.dir, "summary.md")
    open(path, "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()

