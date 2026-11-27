"""
make_tight_instance.py — 產生「更緊(高利用率)」的實例 + 可行貪心靜態解

目的：現有實例護理師過多、容量太鬆 → 擾動不級聯 → w4/路由無訊號。
本腳本把護理師子集化（貪心集合覆蓋，確保每個 task 仍有候選），並 best-fit 緊密打包
產生可行靜態解，使每時段利用率大幅提高 → 擾動會逼出位移/級聯 → 可望出現 w4 權衡訊號。

輸出：
  HHCP-tight.csv             (緊實例，保留原始護理師 ID 的子集)
  solution_schedule_tight.csv(對應靜態解 Π_static)
驗證：重解析、所有 task 有候選、所有 task 被服務、每(nurse,slot)工時<=H_MAX、利用率報告。
無需 API。
    cd Code && python make_tight_instance.py [--target-nurses 16]
"""
from __future__ import annotations
import argparse
import random
from collections import defaultdict
import pandas as pd

from SubgroupOptimizer import HHCPFullParser, RoutePhysicsEngine, H_MAX

SRC = "HHCP-patient-65-task-256-nurse-59.csv"
OUT_HHCP = "HHCP-tight.csv"
OUT_SOL = "solution_schedule_tight.csv"


def greedy_set_cover(tasks, target, rng=None):
    """選護理師子集覆蓋所有 task（每 task 至少一候選在子集內），至少 target 位。
    rng 給定時隨機化 tie-break 與 padding → 產生不同實例(供多實例統計)。"""
    task_cands = {tid: set(t.candidate_nurses) for tid, t in tasks.items()}
    uncovered = set(tasks.keys())
    nurse_to_tasks = defaultdict(set)
    for tid, cs in task_cands.items():
        for n in cs:
            nurse_to_tasks[n].add(tid)
    S = set()
    while uncovered:
        best_cov = max(len(nurse_to_tasks[n] & uncovered) for n in nurse_to_tasks)
        cands = [n for n in nurse_to_tasks if len(nurse_to_tasks[n] & uncovered) == best_cov]
        best = rng.choice(cands) if rng else cands[0]
        S.add(best)
        uncovered -= nurse_to_tasks[best]
    if len(S) < target:
        rest = [n for n in nurse_to_tasks if n not in S]
        if rng:
            rng.shuffle(rest)
        else:
            rest = sorted(rest, key=lambda n: -len(nurse_to_tasks[n]))
        for n in rest:
            if len(S) >= target:
                break
            S.add(n)
    return S


def build_schedule(parser, physics, S, pack="tight"):
    """指派每位病患的任務給單一(候選∩可用∩有容量)護理師-時段。
    pack='tight'：best-fit(先填最滿)→高利用率；pack='spread'：worst-fit(先填最空)→低利用率。
    容量檢查用 compute_workload(完整序列) 以「正確含 travel」，保證工時<=H_MAX。"""
    nurses = {n: parser.nurses[n] for n in S}
    route_order = defaultdict(list)   # (nid,slot) -> 依序的唯一 pid
    route_tasks = defaultdict(dict)   # (nid,slot) -> {pid: [tids]}
    unplaced = []

    def seq_of(nid, slot):
        return [t for pid in route_order[(nid, slot)] for t in route_tasks[(nid, slot)][pid]]

    def workload_with(nid, slot, extra_tids):
        return physics.compute_workload(nid, seq_of(nid, slot) + list(extra_tids))["workload"]

    def cur_wl(nid, slot):
        return physics.compute_workload(nid, seq_of(nid, slot))["workload"]

    patients = sorted(parser.patients.values(), key=lambda p: -len(p.task_list))
    for p in patients:
        pid = p.patient_id
        tids = [t for t in p.task_list if t in parser.tasks]
        if not tids:
            continue
        cand_sets = [set(parser.tasks[t].candidate_nurses) for t in tids]
        full_cover = set.intersection(*cand_sets) & S if cand_sets else set()
        options = []
        for nid in full_cover:
            for slot in nurses[nid].slots:
                if workload_with(nid, slot, tids) <= H_MAX:   # 含 travel 的正確檢查
                    options.append((cur_wl(nid, slot), nid, slot))
        if options:
            options.sort(key=lambda x: (-x[0] if pack == "tight" else x[0]))  # tight=best-fit / spread=worst-fit
            _, nid, slot = options[0]
            route_order[(nid, slot)].append(pid)
            route_tasks[(nid, slot)][pid] = list(tids)
        else:
            # 退化：逐任務指派（仍以含 travel 的工時檢查）
            for t in tids:
                placed = False
                cand = sorted(set(parser.tasks[t].candidate_nurses) & S,
                              key=lambda nn: -max((cur_wl(nn, s) for s in nurses[nn].slots), default=0))
                for nid in cand:
                    for slot in nurses[nid].slots:
                        if workload_with(nid, slot, [t]) <= H_MAX:
                            cell = (nid, slot)
                            if pid not in route_order[cell]:
                                route_order[cell].append(pid)
                                route_tasks[cell][pid] = []
                            route_tasks[cell][pid].append(t)
                            placed = True; break
                    if placed:
                        break
                if not placed:
                    unplaced.append(t)
    return route_order, route_tasks, unplaced


def emit_schedule(parser, physics, route_order, route_tasks):
    rows = []
    for (nid, slot), pids in route_order.items():
        if not pids:
            continue
        day, shift = slot // 3 + 1, slot % 3
        base = physics.nurse_base(nid)
        cx, cy = base
        acc = 0.0
        route_rows = []
        for seq, pid in enumerate(pids, start=1):
            tids = route_tasks[(nid, slot)][pid]
            svc = sum(parser.tasks[t].duration for t in tids)
            loc = physics.task_location(tids[0])
            tt = physics.travel_time(physics.dist((cx, cy), loc))
            arr = acc + tt
            dep = arr + svc
            acc = dep
            cx, cy = loc
            route_rows.append(dict(NurseID=nid, Day=day, Shift=shift, Sequence=seq,
                                   PatientID=pid, TaskIDs=";".join(str(t) for t in tids) + ";",
                                   ArrivalTime=round(arr, 2), ServiceDuration=svc,
                                   DepartureTime=round(dep, 2), TravelTime=round(tt, 2),
                                   AccumulatedTime=0.0))
        final = acc + physics.travel_time(physics.dist((cx, cy), base))   # 含返航
        for r in route_rows:
            r["AccumulatedTime"] = round(final, 2)   # metrics 取 max，全列填最終值
        rows.extend(route_rows)
    return pd.DataFrame(rows)


def emit_hhcp(src_lines, S, parser):
    """重寫 HHCP：保留 patient/header；nurse 只留子集；task 候選過濾；candidate score 重建。"""
    out = []
    # 區段標頭行號
    def hdr(prefix):
        for i, l in enumerate(src_lines):
            if l.lower().startswith(prefix):
                return i
        return None
    pi, ni, ti, ci = (hdr("patient,"), hdr("nurse,"), hdr("task,"),
                      next(i for i, l in enumerate(src_lines) if l.lower().startswith("candidate solution")))
    # header 區塊（0..patient hdr-1）+ patient 區段（patient hdr .. nurse hdr-1）原樣保留
    out.extend(src_lines[0:ni])
    # nurse 區段：header + 子集列（原樣）
    out.append(src_lines[ni])
    for l in src_lines[ni + 1:ti]:
        parts = l.split(',')
        if parts and parts[0].strip().isdigit() and int(parts[0]) in S:
            out.append(l)
    # task 區段：header + 重建列（候選過濾）
    out.append(src_lines[ti])
    flat_pairs = []
    for tid in sorted(parser.tasks.keys()):
        t = parser.tasks[tid]
        kept = [n for n in t.candidate_nurses if n in S]   # 保留原順序
        dur = int(t.duration) if float(t.duration).is_integer() else t.duration
        row = [str(tid), str(t.required_skill), str(dur), str(len(kept))] + [str(n) for n in kept]
        out.append(",".join(row) + ",")
        for n in kept:
            flat_pairs.append((tid, n))
    # candidate score 區段：header + 重建（依新 flat 順序）
    out.append(src_lines[ci])
    for idx, (tid, nid) in enumerate(flat_pairs):
        sc = parser.task_nurse_score.get((tid, nid), 0.5)
        out.append(f"{idx},{sc}")
    return out, len(flat_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-nurses", type=int, default=16)
    ap.add_argument("--out-hhcp", type=str, default=OUT_HHCP)
    ap.add_argument("--out-sol", type=str, default=OUT_SOL)
    ap.add_argument("--pack", choices=["tight", "spread"], default="tight")
    ap.add_argument("--seed", type=int, default=None, help="隨機化 set-cover 以產生不同實例")
    args = ap.parse_args()
    out_hhcp, out_sol = args.out_hhcp, args.out_sol
    rng = random.Random(args.seed) if args.seed is not None else None

    parser = HHCPFullParser(SRC)
    physics = RoutePhysicsEngine(parser)
    src_lines = open(SRC, "r", encoding="utf-8", errors="ignore").read().splitlines()

    S = greedy_set_cover(parser.tasks, args.target_nurses, rng)
    print(f"選定護理師子集: {len(S)} 位 (原 {len(parser.nurses)})  →  {sorted(S)}")

    # 覆蓋檢查
    bad = [tid for tid, t in parser.tasks.items() if not (set(t.candidate_nurses) & S)]
    print(f"無候選的 task 數: {len(bad)}  (應為 0)")

    # 靜態解
    route, route_tasks, unplaced = build_schedule(parser, physics, S, pack=args.pack)
    sol = emit_schedule(parser, physics, route, route_tasks)
    served = sum(len(v) for d in route_tasks.values() for v in d.values())
    print(f"靜態解: {len(route)} 條路線, 服務 {served} 個任務 (共 {len(parser.tasks)}), 未放置 {len(unplaced)}")

    # 利用率（每 nurse-slot 工時 / H_MAX）
    wl = []
    for (nid, slot) in route:
        seq = [t for pid in route[(nid, slot)] for t in route_tasks[(nid, slot)][pid]]
        wl.append(physics.compute_workload(nid, seq)["workload"])
    if wl:
        import statistics as st
        print(f"利用率: 平均工時 {st.mean(wl):.0f}/{H_MAX:.0f}min ({st.mean(wl)/H_MAX*100:.0f}%) "
              f"max={max(wl):.0f} 飽和(>300)路線數={sum(1 for w in wl if w>300)}/{len(wl)}")
    # 路線長度分布
    rl = sol.groupby(["NurseID", "Day", "Shift"]).size()
    print(f"路線長度: n=1:{(rl==1).sum()} n=2:{(rl==2).sum()} n=3:{(rl==3).sum()} n>=4:{(rl>=4).sum()} max={rl.max()}")

    # 寫出
    new_lines, npairs = emit_hhcp(src_lines, S, parser)
    open(out_hhcp, "w", encoding="utf-8").write("\n".join(new_lines) + "\n")
    sol.to_csv(out_sol, index=False)
    print(f"已輸出: {out_hhcp} ({npairs} candidate scores), {out_sol} ({len(sol)} rows)")

    # 驗證重解析
    p2 = HHCPFullParser(out_hhcp)
    print(f"重解析驗證: nurses={len(p2.nurses)} tasks={len(p2.tasks)} scores={len(p2.task_nurse_score)}")
    bad2 = [tid for tid, t in p2.tasks.items() if not t.candidate_nurses]
    print(f"重解析後無候選的 task: {len(bad2)} (應為 0)")


if __name__ == "__main__":
    main()
