import json

for ds in ["wisconsin", "chameleon"]:
    with open(f"experiments/results/phase1_controls/{ds}_phase1_controls.json") as f:
        d = json.load(f)
    print(f"\n=== {ds.upper()} ===")
    for k, v in d.items():
        if isinstance(v, dict) and "scores" in v:
            mean = sum(v["scores"]) / len(v["scores"]) * 100
            print(f"  {k}: {mean:.2f}%")
        elif isinstance(v, dict) and "mean" in v:
            print(f"  {k}: {v['mean']*100:.2f}%")
        else:
            print(f"  {k}: {v}")
