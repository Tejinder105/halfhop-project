import json
import numpy as np
from scipy import stats

datasets = ['texas', 'cornell', 'wisconsin', 'chameleon']

for ds in datasets:
    with open(f'experiments/results/phase1_controls/{ds}_phase1_controls.json') as f:
        data = json.load(f)
    print(f"{'='*65}")
    print(f"DATASET: {ds.upper()}")
    print(f"{'='*65}")
    res = data['results']
    hh_splits = np.array(res['hh_config']['per_split'])
    hh_mean = res['hh_config']['mean']

    for name, v in res.items():
        m = v['mean']
        s = v['std']
        delta = m - hh_mean
        splits = np.array(v['per_split'])
        diff = splits - hh_splits
        # Paired t-test and Wilcoxon
        if np.all(diff == 0):
            p_val = 1.0
        else:
            try:
                _, p_val = stats.wilcoxon(splits, hh_splits)
            except Exception:
                _, p_val = stats.ttest_rel(splits, hh_splits)
        
        sig = "*" if p_val < 0.05 else ("+" if p_val < 0.10 else " ")
        print(f"  {name:22s}: {m:5.2f}% ± {s:4.2f}% | Δ vs HH: {delta:+5.2f}% (p={p_val:.4f}{sig})")
    print()

