#!/usr/bin/env python
"""Analyse Option A. Selection of every arm's config uses VALIDATION accuracy only.

Reports test accuracy (mean +- std over splits, seeds averaged within split), then paired
differences (per split) with bootstrap CIs, then the kill-criterion check.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
from halfhop.stats_utils import paired_bootstrap, fmt_ci

REFS = ["gcn", "hh_all", "gcn_selfloop", "appnp", "hh_random"]


def key_of(r):
    return (r["arm"], r.get("level"), r.get("budget"), r.get("alpha"), r.get("sl_w"), r.get("appnp_alpha"))


def key_str(k):
    _, lv, b, al, w, ta = k
    parts = [f"{n}={v}" for n, v in (("level", lv), ("budget", b), ("alpha", al), ("sl_w", w), ("appnp_a", ta)) if v is not None]
    return ",".join(parts) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/a/runs.jsonl")
    ap.add_argument("--out", default="results/a/summary.md")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.runs)]
    agg = defaultdict(lambda: defaultdict(lambda: ([], [])))   # (ds,key)->split->(vals,tests)
    for r in rows:
        v, t = agg[(r["dataset"], key_of(r))][r["split"]]
        v.append(r["val_acc"]); t.append(r["test_acc"])
    datasets = sorted({r["dataset"] for r in rows})
    out = ["# Option A summary", "",
           "Config per arm chosen by mean VALIDATION accuracy over splits. `oracle_full` uses test labels: upper bound only, not a deployable method.", ""]
    decisions = []
    for ds in datasets:
        keys = [k for (d, k) in agg if d == ds]
        splits = sorted(set.intersection(*[set(agg[(ds, k)].keys()) for k in keys]))
        best, series = {}, {}
        for arm in sorted({k[0] for k in keys}):
            cand = [k for k in keys if k[0] == arm]
            val = {k: np.mean([np.mean(agg[(ds, k)][s][0]) for s in splits]) for k in cand}
            kb = max(val, key=val.get)
            best[arm] = (kb, val[kb])
            series[arm] = np.array([np.mean(agg[(ds, kb)][s][1]) for s in splits])
        out += [f"## {ds}  ({len(splits)} splits)", "",
                "| arm | selected config | val acc | test acc (mean ± std over splits) |", "|---|---|---|---|"]
        for arm, (kb, v) in best.items():
            tag = " (UPPER BOUND)" if arm == "hh_oracle_full" else ""
            out.append(f"| {arm}{tag} | {key_str(kb)} | {100 * v:.2f} | {100 * series[arm].mean():.2f} ± {100 * series[arm].std():.2f} |")
        out += ["", "Paired test-accuracy difference in points, arm minus reference, per split (* = 95% CI excludes 0):", "",
                "| arm | " + " | ".join(f"vs {r}" for r in REFS if r in series) + " |",
                "|---|" + "---|" * len([r for r in REFS if r in series])]
        for arm in series:
            cells = []
            for ref in REFS:
                if ref not in series:
                    continue
                cells.append("—" if ref == arm else fmt_ci(paired_bootstrap(series[arm] - series[ref])))
            out.append(f"| {arm} | " + " | ".join(cells) + " |")
        out.append("")
        if "hh_oracle_train" in series and "hh_all" in series:
            decisions.append((ds, "oracle_train − hh_all", paired_bootstrap(series["hh_oracle_train"] - series["hh_all"])))
        if "hh_all" in series and "gcn_selfloop" in series:
            decisions.append((ds, "hh_all − gcn_selfloop", paired_bootstrap(series["hh_all"] - series["gcn_selfloop"])))
        for s in ("hh_feat", "hh_mlp"):
            if s in series and "hh_random" in series:
                decisions.append((ds, f"{s} − hh_random", paired_bootstrap(series[s] - series["hh_random"])))
    out += ["## Kill-criterion check", "",
            "| dataset | comparison | Δ acc (pts) [95% CI] | verdict |", "|---|---|---|---|"]
    for ds, name, r in decisions:
        v = "clear gain" if r["lo"] > 0 else ("clear loss" if r["hi"] < 0 else "inconclusive")
        out.append(f"| {ds} | {name} | {fmt_ci(r)} | {v} |")
    out += ["", "Reading guide:",
            "- `oracle_train − hh_all` has no clear gain on >=2 datasets: selection is not the lever; write the negative result.",
            "- `hh_all − gcn_selfloop` / `appnp` not positive: Half-Hop's gain is mostly ego preservation; say so.",
            "- `hh_feat/hh_mlp − hh_random` positive AND oracle_train positive: you have a selection story.", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()

