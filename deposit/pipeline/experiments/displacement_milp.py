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
from dataclasses import dataclass
import hashlib
import json
import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from SubgroupOptimizer import HHCPFullParser, H_STD, H_MAX

W2, W5 = 5.0, 1000.0
TRAVEL_PER_TASK = 15.0          # 每個 stop 的固定旅行估計(分鐘)，抽象化 travel
W4_SWEEP = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # 去掉 w4=0(退化:偏差任意)，加密
T_NOW = 4
SLOTS_PER_DAY = 3


class DisplacementSolveError(RuntimeError):
    """Fatal, keyed failure from one displacement solve.

    The exception always carries the complete scientific cell key.  Callers must not catch it
    in order to continue with another slot: doing that would turn a missing solve into a zero.
    """

    def __init__(self, message, *, stage, slot, absent, regime, w4):
        self.key = dict(stage=stage, slot=slot, absent=absent, regime=regime, w4=w4)
        super().__init__(
            f"{message} | stage={stage!r} slot={slot!r} absent={absent!r} "
            f"regime={regime!r} w4={w4!r}")


@dataclass(frozen=True)
class SlotSolveResult:
    """Metrics plus the complete solver/feasibility evidence for one slot.

    Iteration yields the historical four metrics so old read-only analysis helpers can unpack the
    object, while formal R-stage callers persist :meth:`evidence_row`.
    """

    deviation: int
    overtime: float
    served: int
    total: int
    assignment: dict
    status_code: int
    status_name: str
    raw_message: str
    primal: float
    dual_bound: float
    full_objective_gap: object
    full_objective_gap_undefined_reason: object
    stop_reason: str
    assignment_hash: str
    feasibility_verified: bool
    wall_time_s: float
    time_limit_s: float
    presolve: bool
    deviation_constant_C: float

    def __iter__(self):
        yield self.deviation
        yield self.overtime
        yield self.served
        yield self.total

    def __getitem__(self, index):
        return (self.deviation, self.overtime, self.served, self.total)[index]

    def __len__(self):
        return 4

    def evidence_row(self):
        return dict(
            deviation=self.deviation,
            overtime=self.overtime,
            served=self.served,
            total=self.total,
            unserved=self.total - self.served,
            solver_status_code=self.status_code,
            solver_status_name=self.status_name,
            solver_raw_message=self.raw_message,
            obj_incumbent=self.primal,
            mip_dual_bound=self.dual_bound,
            gap_full_objective=self.full_objective_gap,
            gap_full_objective_undefined_reason=self.full_objective_gap_undefined_reason,
            stop_reason=self.stop_reason,
            assignment_hash=self.assignment_hash,
            feasibility_verified=self.feasibility_verified,
            wall_time_s=self.wall_time_s,
            time_limit_s=self.time_limit_s,
            solver_presolve=self.presolve,
            deviation_constant_C=self.deviation_constant_C,
            assignment_json=json.dumps(
                sorted((str(t), None if n is None else int(n))
                       for t, n in self.assignment.items()),
                separators=(",", ":")),
        )


def _assignment_hash(assign):
    payload = json.dumps(
        sorted((repr(t), repr(assign[t])) for t in assign), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def solve_slot(parser, q, task_to_nurse, absent, w4, S, h_std=H_STD, h_max=H_MAX, regime="A2",
               *, solver_options=None, admissible_statuses=(0,), status_names=None,
               stage="UNREGISTERED"):
    """單一時段的指派 MILP。regime='A2' 允許位移(所有任務自由重分派)；
    'A1' assignment-only(既有指派固定到原護理師,只有孤兒任務自由)。
    回傳 :class:`SlotSolveResult`（可向後相容地解包成
    ``deviation, overtime, served, total``）。h_std/h_max 可覆寫。

    Formal callers must pass solver options and admissible status/name mappings loaded from their
    registered design manifest.  The default is retained only for non-formal legacy utilities.
    """
    key = dict(stage=stage, slot=q, absent=absent, regime=regime, w4=w4)
    if regime not in ("A1", "A2"):
        raise DisplacementSolveError("unknown recovery regime", **key)
    opts = dict({"time_limit": 30.0, "presolve": False}
                if solver_options is None else solver_options)
    if opts.get("presolve") is not False:
        raise DisplacementSolveError(
            "presolve must be false because the registered feasible-fixture defect is unresolved "
            "with HiGHS presolve enabled", **key)
    if "time_limit" not in opts or not math.isfinite(float(opts["time_limit"])) \
            or float(opts["time_limit"]) <= 0:
        raise DisplacementSolveError("solver time_limit must be a positive finite manifest value",
                                     **key)
    status_names = dict({0: "Optimal"} if status_names is None else status_names)
    admissible_statuses = frozenset(int(x) for x in admissible_statuses)
    if admissible_statuses != frozenset({0}):
        raise DisplacementSolveError(
            f"this exact recovery path is code-0-only, got {sorted(admissible_statuses)}", **key)
    tasks = [t for t in task_to_nurse if t in parser.tasks]
    if not tasks:
        empty = {}
        return SlotSolveResult(
            deviation=0, overtime=0.0, served=0, total=0, assignment=empty,
            status_code=0, status_name="EmptyInstance", raw_message="no in-model tasks",
            primal=0.0, dual_bound=0.0, full_objective_gap=0.0,
            full_objective_gap_undefined_reason=None,
            stop_reason="empty_instance", assignment_hash=_assignment_hash(empty),
            feasibility_verified=True, wall_time_s=0.0,
            time_limit_s=float(opts["time_limit"]), presolve=False,
            deviation_constant_C=0.0)
    # 可用候選護理師：在 S、該時段可出勤、非缺席
    avail = {n for n in S if n != absent and q in parser.nurses[n].slots}
    # 每個 task 的候選：A2=全候選∩avail；A1=既有指派固定(僅 1 選項)、孤兒自由
    cand = {}
    for t in tasks:
        full = sorted(n for n in parser.tasks[t].candidate_nurses if n in avail)
        st = task_to_nurse.get(t)
        incumbent_available = st is not None and st != absent and st in avail
        if incumbent_available and st not in full:
            raise DisplacementSolveError(
                f"incumbent task {t!r}->worker {st!r} is available but absent from that task's "
                "eligible candidate list; A1 cannot pin an ineligible edge", **key)
        if regime == "A1" and incumbent_available:
            cand[t] = [st]          # incumbent 固定到原護理師(不位移)
        else:
            cand[t] = full          # orphan(或 A2): 自由
        if regime == "A1" and not set(cand[t]).issubset(set(full)):
            raise DisplacementSolveError(
                f"A1 candidate set is not a subset of A2 for task {t!r}: "
                f"A1={cand[t]!r}, A2={full!r}", **key)
    dur = {t: parser.tasks[t].duration + TRAVEL_PER_TASK for t in tasks}
    # 基線 x̄：靜態指派(缺席者的任務視為無 x̄ → 孤兒)
    xbar = {}
    for t in tasks:
        n = task_to_nurse[t]
        if n != absent and n in avail:
            xbar[(n, t)] = 1
    nurses = sorted(avail)

    xpairs = [(i, t) for t in tasks for i in cand[t]]
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
    started = time.monotonic()
    try:
        res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(lo, hi),
                   options=opts)
    except Exception as exc:
        raise DisplacementSolveError(
            f"scipy.optimize.milp raised {type(exc).__name__}: {exc}", **key) from exc
    wall = time.monotonic() - started
    status = int(getattr(res, "status", -1))
    if status not in admissible_statuses:
        raise DisplacementSolveError(
            f"non-admissible solver status={status}, message={getattr(res, 'message', None)!r}; "
            f"admissible={sorted(admissible_statuses)}", **key)
    if not bool(getattr(res, "success", False)) or res.x is None:
        raise DisplacementSolveError(
            f"code-0 solve has no successful incumbent: status={status}, "
            f"message={getattr(res, 'message', None)!r}", **key)
    x = res.x
    if not np.all(np.isfinite(x)):
        raise DisplacementSolveError("solver returned a non-finite solution vector", **key)
    chosen = {(i, t) for (i, t), k in xidx.items() if x[k] > 0.5}
    assignment = {}
    for t in tasks:
        picked = [i for i in cand[t] if x[xidx[(i, t)]] > 0.5]
        is_unserved = x[nu[t]] > 0.5
        if len(picked) + int(is_unserved) != 1:
            raise DisplacementSolveError(
                f"task {t!r} violates service closure: picked={picked!r}, "
                f"unserved_value={x[nu[t]]!r}", **key)
        assignment[t] = None if is_unserved else picked[0]
    loads = defaultdict(float)
    for t, worker in assignment.items():
        if worker is None:
            continue
        if worker not in cand[t]:
            raise DisplacementSolveError(
                f"task {t!r} assigned to worker {worker!r} outside candidate set {cand[t]!r}",
                **key)
        loads[worker] += dur[t]
    allowance = math.ulp(float(h_max)) * 8
    overloaded = {i: value for i, value in loads.items() if value > h_max + allowance}
    if overloaded:
        raise DisplacementSolveError(
            f"schedule feasibility check found capacity overloads {overloaded!r} "
            f"(h_max={h_max!r}, allowance={allowance!r})", **key)
    deviation = sum(1 for (i, t) in chosen if (i, t) not in xbar) \
              + sum(1 for (i, t) in xbar if (i, t) not in chosen)
    overtime = sum(max(0.0, loads.get(i, 0.0) - h_std) for i in nurses)
    solver_ot = sum(max(0.0, float(x[nO[i]])) for i in nurses)
    if abs(overtime - solver_ot) > max(1.0, abs(overtime)) * 1e-8:
        raise DisplacementSolveError(
            f"solver overtime {solver_ot!r} does not close to schedule load overtime "
            f"{overtime!r}", **key)
    served = sum(1 for t in tasks if assignment[t] is not None)
    primal = float(getattr(res, "fun", float("nan")))
    dual = float(getattr(res, "mip_dual_bound", float("nan")))
    if not math.isfinite(primal) or not math.isfinite(dual):
        raise DisplacementSolveError(
            f"code-0 solve lacks finite primal/dual bounds: P={primal!r}, B={dual!r}", **key)
    constant = float(w4) * len(xbar)
    denom = abs(primal + constant)
    if denom < 1e-12:
        if abs(primal - dual) >= 1e-12:
            raise DisplacementSolveError(
                f"zero full-objective denominator but unequal primal/dual: "
                f"P={primal!r}, B={dual!r}, C={constant!r}", **key)
        full_gap = None
        gap_undefined = "zero_full_objective_denominator"
    else:
        full_gap = (primal - dual) / denom
        gap_undefined = None
    if full_gap is not None and not math.isfinite(full_gap):
        raise DisplacementSolveError(
            f"full-objective gap is non-finite: P={primal!r}, B={dual!r}, C={constant!r}",
            **key)
    return SlotSolveResult(
        deviation=int(deviation), overtime=float(overtime), served=int(served),
        total=len(tasks), assignment=assignment, status_code=status,
        status_name=str(status_names.get(status, f"status_{status}")),
        raw_message=str(getattr(res, "message", "")), primal=primal, dual_bound=dual,
        full_objective_gap=(None if full_gap is None else float(full_gap)),
        full_objective_gap_undefined_reason=gap_undefined,
        stop_reason="proven_optimal",
        assignment_hash=_assignment_hash(assignment), feasibility_verified=True,
        wall_time_s=float(wall), time_limit_s=float(opts["time_limit"]),
        presolve=False, deviation_constant_C=constant)


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
