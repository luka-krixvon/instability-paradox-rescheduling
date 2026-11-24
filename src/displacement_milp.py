"""
displacement_milp.py — make-or-break：允許「位移既有指派」的重排班 MILP，測 w4 是否恢復 Pareto

對照現行 StabilityReRosterer（只丟棄缺席指派 + 安置孤兒 → 偏差為事件常數 → w4 無效），
本變體在「每個未來時段內」允許所有任務在可用候選護理師間自由重分派（含位移既有任務）：
  min  Σ w2·O_i + Σ w4·[x(1-x̄)+(1-x)x̄] + Σ w5·u_j
  s.t. 服務履行、技能(候選)、容量(Σdur·x ≤ H_max)、加班 O_i ≥ Σdur·x − H_std
逐未來時段(slot-decomposed)求解並彙總，掃 w4∈{0,5,10,20,50,100}。
若 deviation/overtime 隨 w4 改變 → 權衡(Pareto)在真實緊實例上恢復 → 救援路線 C/V2 成立。

工作量說明：overtime 以「服務時間」為主（travel 抽象化為每任務常數 TRAVEL_PER_TASK），
聚焦展示「集中(加班) vs 平衡(位移=偏差)」的權衡機制本身。
無需 API。
    cd Code && python displacement_milp.py [--instance tight|orig]
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from SubgroupOptimizer import HHCPFullParser, H_STD, H_MAX

W2, W5 = 5.0, 1000.0
TRAVEL_PER_TASK = 15.0          # 每個 stop 的固定旅行估計(分鐘)，抽象化 travel
W4_SWEEP = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # 去掉 w4=0(退化:偏差任意)，加密
T_NOW = 4
SLOTS_PER_DAY = 3


def parse_tids(s):
    return [int(x) for x in str(s).replace(" ", "").strip(";").split(";") if x != ""]


def load_static(sol_path):
    import pandas as pd
    df = pd.read_csv(sol_path)
    if "GlobalSlot" not in df.columns:
        df["GlobalSlot"] = (df["Day"].astype(int) - 1) * SLOTS_PER_DAY + df["Shift"].astype(int)
    # slot -> {task: nurse}
    slot_assign = defaultdict(dict)
    for _, r in df.iterrows():
        q = int(r["GlobalSlot"])
        for t in parse_tids(r["TaskIDs"]):
            slot_assign[q][t] = int(r["NurseID"])
    return slot_assign


def solve_slot(parser, q, task_to_nurse, absent, w4, S, h_std=H_STD, h_max=H_MAX, regime="A2"):
    """單一時段的指派 MILP。regime='A2' 允許位移(所有任務自由重分派)；
    'A1' assignment-only(既有指派固定到原護理師,只有孤兒任務自由)。
    回傳 (deviation, overtime, served, total)。h_std/h_max 可覆寫。"""
    tasks = [t for t in task_to_nurse if t in parser.tasks]
    if not tasks:
        return 0, 0.0, 0, 0
    # 可用候選護理師：在 S、該時段可出勤、非缺席
    avail = {n for n in S if n != absent and q in parser.nurses[n].slots}
    # 每個 task 的候選：A2=全候選∩avail；A1=既有指派固定(僅 1 選項)、孤兒自由
    cand = {}
    for t in tasks:
        full = [n for n in parser.tasks[t].candidate_nurses if n in avail]
        st = task_to_nurse.get(t)
        if regime == "A1" and st is not None and st != absent and st in avail:
            cand[t] = [st]          # incumbent 固定到原護理師(不位移)
        else:
            cand[t] = full          # orphan(或 A2): 自由
    dur = {t: parser.tasks[t].duration + TRAVEL_PER_TASK for t in tasks}
    # 基線 x̄：靜態指派(缺席者的任務視為無 x̄ → 孤兒)
    xbar = {}
    for t in tasks:
        n = task_to_nurse[t]
        if n != absent and n in avail:
            xbar[(n, t)] = 1
    nurses = sorted(avail)

    xpairs = [(i, t) for t in tasks for i in cand[t]]
    if not xpairs:
        return 0, 0.0, 0, len(tasks)
    xidx = {p: k for k, p in enumerate(xpairs)}
    nO = {i: len(xpairs) + a for a, i in enumerate(nurses)}
    nu = {t: len(xpairs) + len(nurses) + b for b, t in enumerate(tasks)}
    N = len(xpairs) + len(nurses) + len(tasks)

    c = np.zeros(N)
    for (i, t), k in xidx.items():
        xb = 1.0 if (i, t) in xbar else 0.0
        c[k] = w4 * (1.0 - 2.0 * xb)
    for i in nurses:
        c[nO[i]] = W2
    for t in tasks:
        c[nu[t]] = W5

    rows, lb, ub = [], [], []
    # 服務履行
    for t in tasks:
        r = np.zeros(N)
        for i in cand[t]:
            r[xidx[(i, t)]] = 1.0
        r[nu[t]] = 1.0
        rows.append(r); lb.append(1.0); ub.append(1.0)
    # 加班 LB：Σ dur·x − O ≤ H_std
    for i in nurses:
        r = np.zeros(N)
        for t in tasks:
            if (i, t) in xidx: r[xidx[(i, t)]] = dur[t]
        r[nO[i]] = -1.0
        rows.append(r); lb.append(-np.inf); ub.append(h_std)
    # 容量
    for i in nurses:
        r = np.zeros(N)
        for t in tasks:
            if (i, t) in xidx: r[xidx[(i, t)]] = dur[t]
        rows.append(r); lb.append(-np.inf); ub.append(h_max)

    cons = LinearConstraint(np.array(rows), np.array(lb), np.array(ub))
    integ = np.zeros(N)
    for (i, t), k in xidx.items(): integ[k] = 1
    for t in tasks: integ[nu[t]] = 1
    lo = np.zeros(N); hi = np.ones(N)
    for i in nurses: hi[nO[i]] = np.inf
    res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(lo, hi),
               options={"time_limit": 30})
    if not res.success or res.x is None:
        return None
    x = res.x
    chosen = {(i, t) for (i, t), k in xidx.items() if x[k] > 0.5}
    deviation = sum(1 for (i, t) in chosen if (i, t) not in xbar) \
              + sum(1 for (i, t) in xbar if (i, t) not in chosen)
    overtime = sum(max(0.0, x[nO[i]]) for i in nurses)
    served = sum(1 for t in tasks if x[nu[t]] < 0.5)
    return deviation, round(overtime, 1), served, len(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", choices=["tight", "orig"], default="tight")
    ap.add_argument("--hhcp", type=str, default=None)
    ap.add_argument("--sol", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--hstd", type=float, default=H_STD)
    ap.add_argument("--hmax", type=float, default=H_MAX)
    args = ap.parse_args()
    if args.hhcp and args.sol:
        HHCP, SOL = args.hhcp, args.sol
    elif args.instance == "tight":
        HHCP, SOL = "HHCP-tight.csv", "solution_schedule_tight.csv"
    else:
        HHCP, SOL = "HHCP-patient-65-task-256-nurse-59.csv", "solution_schedule.csv"

    parser = HHCPFullParser(HHCP)
    S = set(parser.nurses.keys())
    slot_assign = load_static(SOL)

    # 最忙護理師(未來任務最多)缺席
    fut_tasks = defaultdict(int)
    for q, m in slot_assign.items():
        if q >= T_NOW:
            for t, n in m.items():
                fut_tasks[n] += 1
    absent = max(fut_tasks, key=lambda n: fut_tasks[n])
    print(f"實例={args.instance}  缺席最忙護理師=nurse {absent} ({fut_tasks[absent]} 未來任務)")
    print(f"{'w4':>6} | {'deviation':>9} {'overtime':>9} {'unserved':>8}")

    rows = []
    for w4 in W4_SWEEP:
        tot_dev = tot_ot = 0.0
        tot_served = tot_tasks = 0
        for q in sorted(slot_assign):
            if q < T_NOW:
                continue
            out = solve_slot(parser, q, slot_assign[q], absent, w4, S, args.hstd, args.hmax)
            if out is None:
                print(f"  slot {q}: solve failed"); continue
            d, o, s, n = out
            tot_dev += d; tot_ot += o; tot_served += s; tot_tasks += n
        unserved = tot_tasks - tot_served
        rows.append((w4, int(tot_dev), round(tot_ot, 1), unserved))
        print(f"  {w4:>6.1f} | {int(tot_dev):>9} {tot_ot:>9.1f} {unserved:>8}")

    devs = {d for _, d, _, _ in rows}; ots = {o for _, _, o, _ in rows}
    print()
    if len(devs) > 1 or len(ots) > 1:
        print("✅ deviation/overtime 隨 w4 改變 → 位移變體在真實緊實例上【恢復了穩定—效率權衡】！")
        print("   → 救援路線 C/V2 成立：現行模型無權衡是『只安置孤兒』所致，允許位移即可修。")
    else:
        print("⚠️ 仍恆定 → 即使允許位移也無權衡（可能容量過飽和/服務時間主導），需再調實例緊度。")
    import pandas as pd
    out = args.out or f"displacement_w4_{args.instance}.csv"
    pd.DataFrame(rows, columns=["w4", "deviation", "overtime", "unserved"]).to_csv(out, index=False)
    print(f"已輸出: {out}")


if __name__ == "__main__":
    main()
