"""
tradeoff_demo.py — 最小可控 MILP：證明「穩定—效率權衡」在容量緊時會出現、鬆時不會

忠實複製論文 Phase-1 目標（簡化版，聚焦 w2·overtime + w4·deviation）：
  minimize  w2·Σ O_i + w4·Σ[x(1-x̄)+(1-x)x̄] + w5·Σ u_j
  s.t. 服務履行、加班線性化 O_i≥Σdur·x−H_std、容量 Σdur·x≤H_max
求解器：scipy.optimize.milp（HiGHS），與 StabilityReRosterer 同。

兩個小實例：
  TIGHT — 容量緊：要避免加班就必須「位移」既有任務（造成偏差）→ w4 vs w2 真權衡
  LOOSE — 容量鬆：有閒置護理師可直接吸收孤兒（偏差固定）→ w4 無效（= 目前 benchmark 的病灶）

預期：TIGHT 下排班隨 w4 翻轉（dev 3→1、OT 0→10）；LOOSE 下 w4 完全無效。
無需 API。
    cd Code && python tradeoff_demo.py
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

H_STD, H_MAX = 240.0, 360.0
W2, W5 = 5.0, 1000.0


def solve(nurses, tasks, dur, candidates, xbar, w4):
    """回傳 (chosen assignments set, deviation, overtime, objective)。"""
    xpairs = [(i, j) for j in tasks for i in nurses if i in candidates[j]]
    xidx = {p: k for k, p in enumerate(xpairs)}
    nO = {i: len(xpairs) + a for a, i in enumerate(nurses)}
    nu = {j: len(xpairs) + len(nurses) + b for b, j in enumerate(tasks)}
    N = len(xpairs) + len(nurses) + len(tasks)

    c = np.zeros(N)
    for (i, j), k in xidx.items():
        xb = 1.0 if (i, j) in xbar else 0.0
        c[k] = w4 * (1.0 - 2.0 * xb)         # deviation 線性化係數
    for i in nurses:
        c[nO[i]] = W2                         # overtime
    for j in tasks:
        c[nu[j]] = W5                         # unserved penalty
    dev_const = w4 * sum(1 for p in xbar if p[0] in nurses)  # 事後補回常數項

    rows, lb, ub = [], [], []
    # 服務履行：Σ_i x[i,j] + u[j] = 1
    for j in tasks:
        r = np.zeros(N)
        for i in nurses:
            if (i, j) in xidx: r[xidx[(i, j)]] = 1.0
        r[nu[j]] = 1.0
        rows.append(r); lb.append(1.0); ub.append(1.0)
    # 加班 LB：Σ dur·x − O ≤ H_std
    for i in nurses:
        r = np.zeros(N)
        for j in tasks:
            if (i, j) in xidx: r[xidx[(i, j)]] = dur[j]
        r[nO[i]] = -1.0
        rows.append(r); lb.append(-np.inf); ub.append(H_STD)
    # 容量：Σ dur·x ≤ H_max
    for i in nurses:
        r = np.zeros(N)
        for j in tasks:
            if (i, j) in xidx: r[xidx[(i, j)]] = dur[j]
        rows.append(r); lb.append(-np.inf); ub.append(H_MAX)

    A = np.array(rows)
    cons = LinearConstraint(A, np.array(lb), np.array(ub))
    integ = np.zeros(N)
    for (i, j), k in xidx.items(): integ[k] = 1
    for j in tasks: integ[nu[j]] = 1
    lo = np.zeros(N); hi = np.ones(N)
    for i in nurses: hi[nO[i]] = np.inf       # O 連續 ≥0
    res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(lo, hi))
    x = res.x
    chosen = sorted([(i, j) for (i, j), k in xidx.items() if x[k] > 0.5])
    deviation = int(round(sum(1 for (i, j) in chosen if (i, j) not in xbar)
                          + sum(1 for (i, j) in xbar if i in nurses and (i, j) not in chosen)))
    overtime = sum(max(0.0, x[nO[i]]) for i in nurses)
    return chosen, deviation, round(overtime, 1), round(res.fun + dev_const, 2)


def run_case(name, nurses, tasks, dur, candidates, xbar):
    print(f"\n=== {name} ===")
    print(f"  可用護理師={nurses}  任務={tasks}  時長={dur}")
    print(f"  候選={candidates}  基線指派 x̄={sorted(xbar)}")
    print(f"  {'w4':>4} | {'deviation':>9} {'overtime':>9} | chosen")
    prev = None
    for w4 in [0, 5, 10, 20, 25, 30, 40, 50]:
        chosen, dev, ot, obj = solve(nurses, tasks, dur, candidates, xbar, float(w4))
        flip = "  ← 翻轉!" if (prev is not None and chosen != prev) else ""
        print(f"  {w4:>4} | {dev:>9} {ot:>9} | {chosen}{flip}")
        prev = chosen


if __name__ == "__main__":
    # 共同：T1,T2,T3 各 125 分鐘；兩任務=250(OT10)，三任務=375>360 不可行
    dur = {"T1": 125.0, "T2": 125.0, "T3": 125.0}
    # N2 缺席（已移除）；基線 N1-T1, N3-T3（T2 原屬 N2，現為孤兒）
    xbar = {("N1", "T1"), ("N3", "T3")}

    # TIGHT：T2 只能去 N1 或 N3（皆已有任務）；要無加班服務 T2 必須把 T1 位移到 N4
    run_case("TIGHT 容量緊（避免加班需位移既有任務）",
             nurses=["N1", "N3", "N4"], tasks=["T1", "T2", "T3"], dur=dur,
             candidates={"T1": ["N1", "N4"], "T2": ["N1", "N3"], "T3": ["N3"]},
             xbar=xbar)

    # LOOSE：T2 也能直接給閒置的 N4（無需位移、無加班）→ 偏差固定 → w4 無效
    run_case("LOOSE 容量鬆（閒置護理師直接吸收孤兒）= 目前 benchmark 的病灶",
             nurses=["N1", "N3", "N4"], tasks=["T1", "T2", "T3"], dur=dur,
             candidates={"T1": ["N1", "N4"], "T2": ["N1", "N3", "N4"], "T3": ["N3"]},
             xbar=xbar)

    print("\n結論：TIGHT 下排班隨 w4 翻轉（穩定—效率權衡成立）；LOOSE 下 w4 無效（與現有 benchmark 相同）。")
    print("→ 方法本身會產生權衡訊號；現有 benchmark 因『容量過鬆』而無訊號。把實例變緊即可救活論文。")
