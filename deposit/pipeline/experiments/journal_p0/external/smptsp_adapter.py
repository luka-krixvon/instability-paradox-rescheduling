"""
external/smptsp_adapter.py (E2) — adapter for the Personnel Task (Re)Scheduling
benchmark that Borgonjon & Maenhout (2021, C&OR, doi:10.1016/j.cor.2021.105624)
build on: the Shift-Minimisation Personnel Task Scheduling Problem (SMPTSP /
OR-Library "ptask"). Data + format:
  - OR-Library:      http://people.brunel.ac.uk/~mastjjb/jeb/orlib/ptaskinfo.html
  - KU Leuven (Smet): https://people.cs.kuleuven.be/~pieter.smet/smptsp.html
  - PTASPLIB:         https://sites.google.com/site/ptsplib/smptsp/instances

Verified file layout (KU Leuven / OR-Library ".dat"; lines beginning with '#' are comments):
    # ... comment lines (tightness, multi-skilling level, seed) ...
    Type = 1                                          (1 = shift/worker minimisation)
    Jobs = T
    start_1 end_1                                     (T lines)
    ...
    Qualifications = W
    c_1: j_1 j_2 ... j_{c_1}                          (W lines; c_w qualified jobs, 1-based indices)
    ...

Modeling map to the paper's assignment-level recovery model (see external/README.md):
  task j     : duration = end_j - start_j (service minutes); candidates = workers qualified for j;
               slot = time bucket of start_j (tasks close in time compete for a worker's slot capacity).
  worker w   : available in every slot (personnel span the horizon).
The temporal non-overlap of SMPTSP is abstracted to per-slot capacity (total assigned duration
<= H_max), consistent with the paper's flat-workload abstraction; this tests the A1/A2 paradox
on real qualification/duration structure rather than reproducing SMPTSP's routing.

    cd Code && python3 journal_p0/external/smptsp_adapter.py path/to/instance.txt [--slots K]
"""
from __future__ import annotations
import argparse, re, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instance_spec import InstanceSpec, TaskSpec, WorkerSpec


class SMPTSPFormatError(ValueError):
    """Malformed or lossy SMPTSP input; the adapter must never warn-and-proceed."""


class ExternalDesignUnregistered(ValueError):
    """A required R0-D external-design choice is still HUMAN-BLOCKED."""


ON_UNBUILDABLE_TASK = ("error", "drop_and_record")


def _data_lines(path):
    """Non-comment, non-blank lines (comments begin with '#')."""
    out = []
    for ln in open(path, "r", encoding="utf-8", errors="ignore"):
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def parse_smptsp(path, one_based=True):
    """Parse a ptask/SMPTSP ".dat" into (jobs, quals): jobs=[(start,end)], quals=[set(job idx, 0-based)]."""
    lines = _data_lines(path)
    i = 0
    def find(label):
        nonlocal i
        while i < len(lines):
            m = re.match(rf"{label}\s*=\s*(\d+)", lines[i], re.I)
            i += 1
            if m:
                return int(m.group(1))
        raise ValueError(f"'{label} = ...' not found in {path}")
    _type = find("Type")
    if _type != 1:
        raise SMPTSPFormatError(
            f"Type={_type}; this adapter is registered only for Type=1 SMPTSP instances")
    T = find("Jobs")
    jobs = []
    for job_id in range(T):
        if i >= len(lines):
            raise SMPTSPFormatError(
                f"job table ended at row {job_id}, expected {T} rows")
        toks = re.findall(r"-?\d+", lines[i])
        i += 1
        if len(toks) != 2:
            raise SMPTSPFormatError(
                f"job row {job_id} has {len(toks)} integer fields, expected 2: {toks!r}")
        a, b = toks
        jobs.append((int(a), int(b)))
    W = find("Qualifications")
    quals = []
    for _ in range(W):
        if i >= len(lines):
            raise SMPTSPFormatError(
                f"qualification table ended at worker {len(quals)}, expected {W} rows")
        toks = [int(x) for x in re.findall(r"-?\d+", lines[i])]; i += 1
        if not toks:
            raise SMPTSPFormatError(
                f"worker {len(quals)} qualification row has no integer fields")
        cnt, idx = toks[0], toks[1:]                 # "c: j1 j2 ... jc"
        if len(idx) != cnt:
            raise SMPTSPFormatError(
                f"worker {len(quals)} declares {cnt} qualified jobs but provides {len(idx)}")
        if one_based:
            idx = [j - 1 for j in idx]
        bad = [j for j in idx if not (0 <= j < T)]
        if bad:
            raise SMPTSPFormatError(
                f"worker {len(quals)} references job index outside [0,{T}): {bad[:10]}")
        quals.append(set(idx))
    return jobs, quals


def to_spec(jobs, quals, num_slots=None, name="smptsp", on_unbuildable_task=None):
    """Build an InstanceSpec. Slots partition [min start, max end] into `num_slots` buckets."""
    if num_slots is None:
        raise ExternalDesignUnregistered(
            "num_slots is UNSET. The former tasks-per-slot heuristic is not a registration; "
            "R0-D must record a positive integer.")
    if isinstance(num_slots, bool) or int(num_slots) != num_slots or int(num_slots) < 1:
        raise ExternalDesignUnregistered(
            f"num_slots must be a registered positive integer, got {num_slots!r}")
    num_slots = int(num_slots)
    if on_unbuildable_task is None:
        raise ExternalDesignUnregistered(
            "on_unbuildable_task is UNSET; R0-D must register 'error' or 'drop_and_record'")
    if on_unbuildable_task not in ON_UNBUILDABLE_TASK:
        raise ExternalDesignUnregistered(
            f"on_unbuildable_task={on_unbuildable_task!r}; expected {ON_UNBUILDABLE_TASK}")
    T = len(jobs)
    if T < 1:
        raise SMPTSPFormatError("instance contains no jobs")
    for j, (start, end) in enumerate(jobs):
        if end <= start:
            raise SMPTSPFormatError(
                f"job {j} has non-positive interval [{start},{end}]")
    starts = [s for s, _ in jobs]; ends = [e for _, e in jobs]
    lo, hi = min(starts), max(ends)
    span = max(1, hi - lo)
    def bucket(s):
        return min(num_slots - 1, int((s - lo) * num_slots / span))
    cand_of = [set() for _ in range(T)]
    for w, js in enumerate(quals):
        for j in js:
            if 0 <= j < T:
                cand_of[j].add(w)
    tasks = []
    unqualifiable = []
    for j, (s, e) in enumerate(jobs):
        dur = max(1.0, float(e - s))
        if not cand_of[j]:
            unqualifiable.append(j)
            continue
        tasks.append(TaskSpec(id=j, duration=dur, slot=bucket(s), candidates=frozenset(cand_of[j])))
    if unqualifiable and on_unbuildable_task == "error":
        raise SMPTSPFormatError(
            f"{len(unqualifiable)} of {T} jobs have no qualified worker under registered "
            f"policy 'error'; first ids={unqualifiable[:10]}")
    if not tasks:
        raise SMPTSPFormatError("no buildable tasks remain after qualification validation")
    slots_all = frozenset(range(num_slots))
    workers = [WorkerSpec(id=w, slots=slots_all) for w in range(len(quals))]
    return InstanceSpec(name, tasks, workers, num_slots,
                        meta=dict(source="smptsp", n_jobs=T, n_workers=len(quals),
                                  horizon=(lo, hi), num_slots_registered=num_slots,
                                  on_unbuildable_task=on_unbuildable_task,
                                  tasks_required=T, tasks_built=len(tasks),
                                  dropped_no_qualified_worker=len(unqualifiable),
                                  dropped_task_keys=tuple(unqualifiable))).validate()


def load(path, num_slots=None, one_based=True, on_unbuildable_task=None):
    jobs, quals = parse_smptsp(path, one_based=one_based)
    return to_spec(jobs, quals, num_slots=num_slots, name=os.path.basename(path),
                   on_unbuildable_task=on_unbuildable_task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--on-unbuildable-task", required=True,
                    choices=list(ON_UNBUILDABLE_TASK))
    ap.add_argument("--zero-based", action="store_true", help="job indices are 0-based")
    args = ap.parse_args()
    spec = load(args.path, num_slots=args.slots, one_based=not args.zero_based,
                on_unbuildable_task=args.on_unbuildable_task)
    print(f"[smptsp] {spec.name}: {len(spec.tasks)} tasks, {len(spec.workers)} workers, "
          f"{spec.num_slots} slots  (meta={spec.meta})")
    tps = {}
    for t in spec.tasks:
        tps[t.slot] = tps.get(t.slot, 0) + 1
    print(f"  tasks/slot: {dict(sorted(tps.items()))}")
    print(f"  candidate-set size: min={min(len(t.candidates) for t in spec.tasks)} "
          f"max={max(len(t.candidates) for t in spec.tasks)}")
    print("  -> pass this instance to journal_p0/external/run_external.py")


if __name__ == "__main__":
    main()
