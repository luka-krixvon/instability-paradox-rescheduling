"""
multi_stats_tnow.py — 跨 t_now 的多實例/多擾動統計（鞏固一般性）

實例與靜態解和 t_now 無關(t_now 只是凍結/可變的切點)，故沿用 _mi/ 的隨機實例。
對 t_now∈{2,4,6,8,10} × 實例 × 缺席擾動 × {A1,A2} × w4 求解(只計 q≥t_now 的未來時段)。
輸出每個 t_now 的：A1 完全無效比例、A2 穩定加班代價 mean±95%CI、A2 偏差降幅；並彙總 pooled。
圖：左=A2 加班代價 vs t_now(誤差棒)、右=A1 無效% vs t_now。
無需 API。建議背景執行。
    cd Code && python multi_stats_tnow.py [--instances 5] [--max-absent 5]
"""
from __future__ import annotations
import os, sys, subprocess, argparse, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

from SubgroupOptimizer import HHCPFullParser
from displacement_milp import solve_slot, load_static

HERE = os.path.dirname(os.path.abspath(__file__)); MI = os.path.join(HERE, "_mi"); os.makedirs(MI, exist_ok=True)
PY = sys.executable
W4 = [0, 5, 10, 20, 50, 100]
TNOW_LIST = [2, 4, 6, 8, 10]
TARGET_N = 16


def ci95(a):
    a = np.asarray(a, float)
    return 1.96 * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0


def ensure_instance(g):
    H = os.path.join(MI, f"H{g}.csv"); S = os.path.join(MI, f"S{g}.csv")
    if not (os.path.exists(H) and os.path.exists(S)):
        subprocess.run([PY, "make_tight_instance.py", "--target-nurses", str(TARGET_N),
                        "--pack", "tight", "--seed", str(g), "--out-hhcp", H, "--out-sol", S],
                       cwd=HERE, capture_output=True, text=True)
    return H, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--max-absent", type=int, default=5)
    args = ap.parse_args()

    recs = []
    insts = []
    for g in range(1, args.instances + 1):
        H, S = ensure_instance(g)
        if not os.path.exists(S):
            print(f"inst g={g} gen failed"); continue
        insts.append((g, HHCPFullParser(H), load_static(S)))
    print(f"loaded {len(insts)} instances")

    for t_now in TNOW_LIST:
        for g, parser, slot_assign in insts:
            Sset = set(parser.nurses.keys())
            active = sorted({n for q, m in slot_assign.items() if q >= t_now for n in m.values()})
            active = active[:args.max_absent]
            for a in active:
                for regime in ["A1", "A2"]:
                    for w4 in W4:
                        dev = ot = 0.0
                        for q in sorted(slot_assign):
                            if q < t_now: continue
                            out = solve_slot(parser, q, slot_assign[q], a, float(w4), Sset, regime=regime)
                            if out: dev += out[0]; ot += out[1]
                        recs.append(dict(t_now=t_now, g=g, absent=a, regime=regime, w4=w4,
                                         deviation=int(dev), overtime=round(ot, 1)))
        nd = len({(r['g'], r['absent']) for r in recs if r['t_now'] == t_now})
        print(f"t_now={t_now}: {nd} disruptions done")

    df = pd.DataFrame(recs)
    df.to_csv("multi_stats_tnow_raw.csv", index=False)

    # 每個 t_now 的統計
    rows = []
    for t_now in TNOW_LIST:
        d = df[df.t_now == t_now]
        a1 = d[d.regime == "A1"]; a2 = d[d.regime == "A2"]
        # A1 無效比例
        tot = inert = 0
        for _, gdf in a1.groupby(["g", "absent"]):
            tot += 1
            if (gdf.deviation.max() - gdf.deviation.min() == 0) and (gdf.overtime.max() - gdf.overtime.min() == 0):
                inert += 1
        # A2 cost / dev_drop
        cost = a2.groupby(["g", "absent"]).apply(lambda x: x.overtime.max() - x.overtime.min()).values
        ddrop = a2.groupby(["g", "absent"]).apply(lambda x: x.deviation.max() - x.deviation.min()).values
        mono = sum(1 for _, x in a2.groupby(["g", "absent"])
                   if (x.sort_values("w4").deviation.diff().dropna() <= 0).all())
        rows.append(dict(t_now=t_now, n_disrupt=tot, A1_inert_pct=round(100 * inert / max(tot, 1)),
                         A2_cost_mean=round(np.mean(cost), 1), A2_cost_ci=round(ci95(cost), 1),
                         A2_devdrop_mean=round(np.mean(ddrop), 1), A2_devdrop_ci=round(ci95(ddrop), 1),
                         A2_dev_monotone=f"{mono}/{tot}"))
    res = pd.DataFrame(rows)
    res.to_csv("multi_stats_tnow_summary.csv", index=False)
    print("\n=== 跨 t_now 統計 ===\n" + res.to_string(index=False))

    # pooled
    a1all = df[df.regime == "A1"]; a2all = df[df.regime == "A2"]
    tot = inert = 0
    for _, gdf in a1all.groupby(["t_now", "g", "absent"]):
        tot += 1
        if (gdf.deviation.max() - gdf.deviation.min() == 0) and (gdf.overtime.max() - gdf.overtime.min() == 0): inert += 1
    cost_all = a2all.groupby(["t_now", "g", "absent"]).apply(lambda x: x.overtime.max() - x.overtime.min()).values
    print(f"\nPOOLED (所有 t_now): {tot} 擾動樣本")
    print(f"  A1 完全無效: {inert}/{tot} = {100*inert/max(tot,1):.0f}%")
    print(f"  A2 穩定加班代價: mean={np.mean(cost_all):.1f} ± {ci95(cost_all):.1f} (95%CI), n={len(cost_all)}")

    # 圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.errorbar(res.t_now, res.A2_cost_mean, yerr=res.A2_cost_ci, fmt="o-",
                 color="#1f4e79", lw=2, ms=8, capsize=4)
    ax1.set_xlabel("t_now (disruption epoch)"); ax1.set_ylabel("A2 overtime cost of stability (min)\nmean ± 95% CI")
    ax1.set_title("Cost of stability is robust across disruption epochs"); ax1.grid(alpha=.3)
    ax2.bar(res.t_now.astype(str), res.A1_inert_pct, color="#d62728", width=0.6)
    for x, v in zip(res.t_now.astype(str), res.A1_inert_pct):
        ax2.annotate(f"{v}%", (x, v), ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("t_now"); ax2.set_ylabel("A1: % disruptions with w4 fully inert"); ax2.set_ylim(0, 105)
    ax2.set_title("Assignment-only inertness across epochs"); ax2.grid(alpha=.3, axis="y")
    fig.suptitle("Generality across disruption epochs (t_now)", fontsize=12)
    fig.tight_layout(); fig.savefig("fig_tnow.png", dpi=150)
    print("\n已輸出: multi_stats_tnow_raw.csv, multi_stats_tnow_summary.csv, fig_tnow.png")


if __name__ == "__main__":
    main()
