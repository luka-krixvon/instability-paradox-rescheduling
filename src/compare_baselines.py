"""
compare_baselines.py — 與標準 baselines 對照（cold-start / VNS / ALNS）vs 精確 A1/A2

層級：指派級重排(非路由)。所有方法在「同一 per-slot 指派問題」上比較，目標一致：
  obj(w4) = W2·overtime + w4·deviation + W5·unserved
  - deviation = Σ_{i,t}[x(1-x̄)+(1-x)x̄]（與 MILP 一致：保留=0、移動=2、丟棄孤兒前指派=1、孤兒安置=1）
方法：
  - cold_start : 只最小化 overtime(+unserved)，忽略偏差 → 「Solve Smart/盲目重排」極端(w4 無關)
  - VNS        : 貪心起始 + 變鄰域下降(relocate/swap) + shaking，最小化 obj(w4)
  - ALNS       : 貪心起始 + destroy(隨機移除 k)–greedy repair，最小化 obj(w4)
  - A1 / A2    : 精確 MILP(solve_slot, regime A1/A2)
正確性驗證：VNS/ALNS 對精確 A2 的 optimality gap 應很小。
彙總多實例×多擾動(t_now=4)，輸出每法 mean±95%CI 與 gap，及 Pareto 疊圖。
    cd Code && python compare_baselines.py [--instances 6] [--max-absent 6]
"""
from __future__ import annotations
import os, sys, subprocess, argparse, math, random
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

from SubgroupOptimizer import HHCPFullParser, H_STD, H_MAX
from displacement_milp import solve_slot, load_static, TRAVEL_PER_TASK, W2, W5

HERE = os.path.dirname(os.path.abspath(__file__)); MI = os.path.join(HERE, "_mi"); os.makedirs(MI, exist_ok=True)
PY = sys.executable
W4 = [0, 5, 10, 20, 50, 100]
T_NOW = 4; TARGET_N = 16


def ci95(a):
    a = np.asarray(a, float)
    return 1.96 * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0


# ---------- per-slot 實例 + 評估 ----------
def build_slot(parser, q, task_to_nurse, absent, S):
    tasks = [t for t in task_to_nurse if t in parser.tasks]
    avail = sorted({n for n in S if n != absent and q in parser.nurses[n].slots})
    cand = {t: [n for n in parser.tasks[t].candidate_nurses if n in avail] for t in tasks}
    dur = {t: parser.tasks[t].duration + TRAVEL_PER_TASK for t in tasks}
    xbar = {t: task_to_nurse[t] for t in tasks
            if task_to_nurse[t] != absent and task_to_nurse[t] in avail}
    return tasks, avail, cand, dur, xbar


def evaluate(assign, tasks, dur, xbar, w4):
    load = defaultdict(float); served = 0
    for t in tasks:
        w = assign.get(t)
        if w is not None:
            load[w] += dur[t]; served += 1
    ot = sum(max(0.0, l - H_STD) for l in load.values())
    feas = all(l <= H_MAX + 1e-6 for l in load.values())
    dev = 0
    for t in tasks:
        w = assign.get(t); xb = xbar.get(t)
        if xb is None:
            dev += 1 if w is not None else 0
        else:
            dev += 0 if w == xb else (1 if w is None else 2)
    uns = len(tasks) - served
    obj = W2 * ot + w4 * dev + W5 * uns
    return obj, dev, round(ot, 1), served, feas


def marginal_ok(assign, load, t, w, dur):
    return w is None or load[w] + dur[t] <= H_MAX + 1e-6


def greedy_init(tasks, cand, dur, xbar, w4, ignore_dev=False):
    """逐任務貪心：選使 obj 邊際增量最小且可行的 worker(或 None)。"""
    assign = {}; load = defaultdict(float)
    for t in sorted(tasks, key=lambda t: -dur[t]):
        best_w, best_c = None, None
        for w in cand[t] + [None]:
            if w is not None and load[w] + dur[t] > H_MAX + 1e-6:
                continue
            if w is None:
                dot = 0.0; dd = (1 if xbar.get(t) is not None else 0); pen = W5
            else:
                dot = max(0.0, load[w] + dur[t] - H_STD) - max(0.0, load[w] - H_STD)
                xb = xbar.get(t); dd = 0 if w == xb else (2 if xb is not None else 1)
                pen = 0.0
            c = W2 * dot + (0 if ignore_dev else w4 * dd) + pen
            if best_c is None or c < best_c - 1e-9:
                best_c, best_w = c, w
        assign[t] = best_w
        if best_w is not None: load[best_w] += dur[t]
    return assign


def cold_start(tasks, cand, dur, xbar):
    """只最小化 overtime(+unserved)，忽略偏差。"""
    return greedy_init(tasks, cand, dur, xbar, 0.0, ignore_dev=True)


def _local_search(assign, tasks, cand, dur, xbar, w4):
    """relocate + swap 的 best-improvement 下降。"""
    def obj(a): return evaluate(a, tasks, dur, xbar, w4)[0]
    cur = dict(assign); curo = obj(cur)
    improved = True
    while improved:
        improved = False
        # relocate
        for t in tasks:
            for w in cand[t] + [None]:
                if w == cur[t]: continue
                cand_a = dict(cur); cand_a[t] = w
                _, _, _, _, feas = evaluate(cand_a, tasks, dur, xbar, w4)
                if not feas: continue
                o = obj(cand_a)
                if o < curo - 1e-9:
                    cur, curo, improved = cand_a, o, True
        # swap
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                t1, t2 = tasks[i], tasks[j]
                w1, w2 = cur[t1], cur[t2]
                if w1 == w2: continue
                if w2 is not None and w2 not in cand[t1]: continue
                if w1 is not None and w1 not in cand[t2]: continue
                cand_a = dict(cur); cand_a[t1], cand_a[t2] = w2, w1
                _, _, _, _, feas = evaluate(cand_a, tasks, dur, xbar, w4)
                if not feas: continue
                o = obj(cand_a)
                if o < curo - 1e-9:
                    cur, curo, improved = cand_a, o, True
    return cur, curo


def vns(tasks, cand, dur, xbar, w4, rng, iters=120):
    if not tasks: return {}
    best = greedy_init(tasks, cand, dur, xbar, w4)
    best, besto = _local_search(best, tasks, cand, dur, xbar, w4)
    noimp = 0
    for _ in range(iters):
        if noimp >= 25: break
        # shake: 隨機重指派 1-2 個任務
        cand_a = dict(best)
        for _ in range(rng.randint(1, 2)):
            t = rng.choice(tasks)
            opts = cand[t] + [None]
            cand_a[t] = rng.choice(opts)
        cand_a, o = _local_search(cand_a, tasks, cand, dur, xbar, w4)
        _, _, _, _, feas = evaluate(cand_a, tasks, dur, xbar, w4)
        if feas and o < besto - 1e-9:
            best, besto, noimp = cand_a, o, 0
        else:
            noimp += 1
    return best


def alns(tasks, cand, dur, xbar, w4, rng, iters=300):
    if not tasks: return {}
    def obj(a): return evaluate(a, tasks, dur, xbar, w4)[0]
    best = greedy_init(tasks, cand, dur, xbar, w4)
    best, besto = _local_search(best, tasks, cand, dur, xbar, w4)   # 起始也拋光
    cur, curo = dict(best), besto
    for it in range(iters):
        if it % 60 == 59: cur, curo = dict(best), besto   # 週期性回到 best
        k = max(1, min(len(tasks), rng.randint(1, 4)))
        removed = rng.sample(tasks, k)
        partial = dict(cur)
        for t in removed: partial[t] = None
        # greedy repair（含偏差）
        load = defaultdict(float)
        for t in tasks:
            if partial[t] is not None: load[partial[t]] += dur[t]
        for t in sorted(removed, key=lambda t: -dur[t]):
            best_w, best_c = None, None
            for w in cand[t] + [None]:
                if w is not None and load[w] + dur[t] > H_MAX + 1e-6: continue
                if w is None:
                    dot = 0.0; dd = (1 if xbar.get(t) is not None else 0); pen = W5
                else:
                    dot = max(0.0, load[w] + dur[t] - H_STD) - max(0.0, load[w] - H_STD)
                    xb = xbar.get(t); dd = 0 if w == xb else (2 if xb is not None else 1); pen = 0.0
                c = W2 * dot + w4 * dd + pen
                if best_c is None or c < best_c - 1e-9: best_c, best_w = c, w
            partial[t] = best_w
            if best_w is not None: load[best_w] += dur[t]
        partial, o = _local_search(partial, tasks, cand, dur, xbar, w4)  # 拋光修復解
        if o < curo - 1e-9 or rng.random() < 0.05:   # 接受改進或少量探索
            cur, curo = partial, o
            if o < besto - 1e-9: best, besto = dict(partial), o
    return best


def run_method(method, parser, slot_assign, absent, Sset, w4, rng):
    """彙總一個擾動在未來時段的 (deviation, overtime, obj)。"""
    dev = ot = obj = 0.0
    for q in sorted(slot_assign):
        if q < T_NOW: continue
        if method in ("A1", "A2"):
            out = solve_slot(parser, q, slot_assign[q], absent, w4, Sset, regime=method)
            if out: dev += out[0]; ot += out[1]
            # obj 重建
            tasks, avail, cand, dur, xbar = build_slot(parser, q, slot_assign[q], absent, Sset)
            continue
        tasks, avail, cand, dur, xbar = build_slot(parser, q, slot_assign[q], absent, Sset)
        if method == "cold": a = cold_start(tasks, cand, dur, xbar)
        elif method == "VNS": a = vns(tasks, cand, dur, xbar, w4, rng)
        elif method == "ALNS": a = alns(tasks, cand, dur, xbar, w4, rng)
        o, d, ov, served, feas = evaluate(a, tasks, dur, xbar, w4)
        dev += d; ot += ov; obj += o
    return dev, round(ot, 1)


def a2_obj(parser, slot_assign, absent, Sset, w4):
    """A2 精確解的 obj 總和(供 gap 計算)：用 solve_slot 的 dev/ot 重算 obj。"""
    dev = ot = 0.0
    for q in sorted(slot_assign):
        if q < T_NOW: continue
        out = solve_slot(parser, q, slot_assign[q], absent, w4, Sset, regime="A2")
        if out: dev += out[0]; ot += out[1]
    return W2 * ot + w4 * dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--max-absent", type=int, default=6)
    args = ap.parse_args()
    rng = random.Random(20260628)

    insts = []
    for g in range(1, args.instances + 1):
        H = os.path.join(MI, f"H{g}.csv"); S = os.path.join(MI, f"S{g}.csv")
        if not (os.path.exists(H) and os.path.exists(S)):
            subprocess.run([PY, "make_tight_instance.py", "--target-nurses", str(TARGET_N),
                            "--pack", "tight", "--seed", str(g), "--out-hhcp", H, "--out-sol", S],
                           cwd=HERE, capture_output=True, text=True)
        if os.path.exists(S):
            insts.append((g, HHCPFullParser(H), load_static(S)))
    print(f"loaded {len(insts)} instances")

    recs = []; gaps = []
    for g, parser, slot_assign in insts:
        Sset = set(parser.nurses.keys())
        active = sorted({n for q, m in slot_assign.items() if q >= T_NOW for n in m.values()})[:args.max_absent]
        for a in active:
            # cold-start (w4 無關)，記在每個 w4 以便畫點
            cdev, cot = run_method("cold", parser, slot_assign, a, Sset, 0.0, rng)
            for w4 in W4:
                for method in ["A1", "A2", "VNS", "ALNS"]:
                    dev, ot = run_method(method, parser, slot_assign, a, Sset, float(w4), rng)
                    recs.append(dict(g=g, absent=a, method=method, w4=w4, deviation=dev, overtime=ot))
                    if method in ("VNS", "ALNS"):
                        # gap vs A2 exact obj
                        aobj = a2_obj(parser, slot_assign, a, Sset, float(w4))
                        mobj = W2 * ot + w4 * dev
                        if aobj > 1e-6:
                            gaps.append((method, w4, max(0.0, (mobj - aobj) / aobj * 100)))
                recs.append(dict(g=g, absent=a, method="cold", w4=w4, deviation=cdev, overtime=cot))
        print(f"instance g={g}: done ({len(active)} disruptions)")

    df = pd.DataFrame(recs); df.to_csv("baselines_raw.csv", index=False)
    ndis = df[["g", "absent"]].drop_duplicates().shape[0]
    print(f"\n總擾動樣本: {ndis}")

    # gap 檢驗
    gdf = pd.DataFrame(gaps, columns=["method", "w4", "gap_pct"])
    print("\n=== metaheuristic optimality gap vs 精確 A2 (%) ===")
    print(gdf.groupby("method")["gap_pct"].agg(["mean", "max"]).round(2).to_string())

    # 每法每 w4 的 mean±CI
    rows = []
    for method in ["A1", "A2", "VNS", "ALNS", "cold"]:
        for w4 in W4:
            sub = df[(df.method == method) & (df.w4 == w4)]
            rows.append(dict(method=method, w4=w4,
                             dev_mean=round(sub.deviation.mean(), 1), dev_ci=round(ci95(sub.deviation), 1),
                             ot_mean=round(sub.overtime.mean(), 1), ot_ci=round(ci95(sub.overtime), 1)))
    summ = pd.DataFrame(rows); summ.to_csv("baselines_summary.csv", index=False)
    print("\n=== 各法 Pareto (mean±95%CI) — 摘錄 w4∈{0,100} ===")
    print(summ[summ.w4.isin([0, 100])].to_string(index=False))

    # Pareto 疊圖
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {"A2": ("#1f4e79", "o-", "A2 displacement (exact)"),
              "VNS": ("#2ca02c", "s--", "VNS"),
              "ALNS": ("#9467bd", "^--", "ALNS"),
              "A1": ("#d62728", "D-", "A1 assignment-only (exact)")}
    for m, (col, st, lab) in styles.items():
        s = summ[summ.method == m].sort_values("w4")
        ax.errorbar(s.ot_mean, s.dev_mean, xerr=s.ot_ci, yerr=s.dev_ci, fmt=st,
                    color=col, lw=1.8, ms=6, capsize=3, label=lab, alpha=.9)
    cs = summ[summ.method == "cold"].iloc[0]
    ax.scatter([cs.ot_mean], [cs.dev_mean], marker="*", s=220, color="#ff7f0e",
               zorder=5, label="cold-start (efficiency-only)")
    ax.set_xlabel("Total overtime (min)  →  efficiency cost")
    ax.set_ylabel("Total deviation  →  instability")
    ax.set_title(f"Baselines vs exact A1/A2 (mean ± 95% CI, {ndis} disruptions)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_baselines.png", dpi=150)
    print("\n已輸出: baselines_raw.csv, baselines_summary.csv, fig_baselines.png")


if __name__ == "__main__":
    main()
