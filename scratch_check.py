import json

for ds in ['wisconsin', 'chameleon', 'texas', 'cornell']:
    with open(f'experiments/results/sensitivity/sensitivity_{ds}_hh-gcn.jsonl') as f:
        lines = [json.loads(l) for l in f]
    lines_sorted = sorted(lines, key=lambda r: r.get('mean', 0), reverse=True)
    print(f"=== {ds} top 5 ===")
    for r in lines_sorted[:5]:
        a = r.get("alpha")
        p = r.get("p")
        m = r.get("mean") * 100
        s = r.get("std") * 100
        print(f"  alpha={a} p={p} mean={m:.2f}% std={s:.2f}%")
