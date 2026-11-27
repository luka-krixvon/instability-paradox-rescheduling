"""
external/hhcrsp_adapter.py (E1) — adapter for the Mankowska, Meisel & Bierwirth (2014)
Home Health Care Routing and Scheduling benchmark.
  Data + format PDF: https://prodlog.wiwi.uni-halle.de/forschung/research_data/hhcrsp/

VERIFIED format (from the official "Description of file format.pdf" and the instance files):
a MATLAB-style `key=[ ... ];` text. Nodes are 1..nbNodes; **node 1 and node nbNodes are
depots**, nodes 2..nbNodes-1 are patients. Keys:
    nbNodes= ;  nbVehi= (staff) ;  nbServi= (service types) ;
    r=[ nbNodes x nbServi ]      service requirement of each node (0/1)
    a=[ nbVehi  x nbServi ]      staff qualification (0/1)
    d=[ nbNodes x nbNodes ]      distance matrix (unused: travel is abstracted)
    p=[ nbNodes blocks of (nbVehi x nbServi) ]   processing time of service s at node i by staff k
    e=[ nbNodes ] , l=[ nbNodes ]                time windows

Map to the paper's assignment model: task (i,s) with r[i][s]=1 becomes a TaskSpec with
duration = mean over qualified staff of p[i][k][s] (staff-dependent times collapsed to one
d_j; travel abstracted, as in the paper), candidates = {k : a[k][s]=1}, slot = time-window
bucket of node i. Because the released benchmark is over-provisioned (as the paper found for
its own instance), tightness is induced with the same G1 treatment: reduce to a covering
worker subset and best-fit pack the incumbent at cap = rho*H_max (see run_external.py --cover).

Only the `InstanzCPLEX_*` files are in this MATLAB-style format. The `InstanzVNS_*` files in the
same distribution use a different, whitespace-delimited layout with `nbCust=` instead of
`nbNodes=`; this adapter does not parse them and now says so with a diagnostic instead of
failing obscurely.

===============================================================================================
S2 REVISION (2026-07-29) — the decimal-splitting parser defect, and what is NOT fixed here
===============================================================================================
Scope of this revision: **the parser, its shape assertions and its fail-fast behaviour, and
nothing else.** No external solve was run, no result was produced, no parameter was tuned, and
no number in the manuscript is validated by anything in this file (`rerun_plan_v2_1.md` ground
rule E1).

**The defect.** `_rows()` extracted numbers with the integer pattern `re.findall(r"-?\\d+", ...)`.
The `p=` blocks in the released instances are written with a decimal point (`14.0`), so that
pattern split every value into TWO integers: the row `[14.0,14.0,...]` of `nbServi` values became
`2*nbServi` integers alternating `14, 0, 14, 0, ...`. Nothing detected it, because the only shape
assertion counted ROWS (`len(p_rows) == N*K`) and never the width of a row. The consequences were:

  * `p[i][k][s]` for an EVEN service index returned the integer part of the value, and for an ODD
    service index returned its fractional digits — `0` for every `x.0` value in the shipped files;
  * `to_spec()` then filtered durations with `p[i][k][s] > 0` and silently `continue`d when the
    filtered list was empty, so **every task whose required service has an odd index was dropped
    without a trace**;
  * where the fractional part was non-zero the duration was not dropped but silently WRONG
    (e.g. `20.5` parsed as `20` at s=0 and `5` at s=1).

Because the previous behaviour is what produced the external numbers now on disk, those numbers
are invalidated pending stage R2 — see `rerun_plan_v2_1.md` REQ-9 "Invalidates". This file does
not re-run anything and must not be cited as evidence about what the corrected numbers are.

**The fix.**
  1. one float-aware number pattern (`_NUM`) used everywhere, with `int` enforced only where the
     format guarantees integers (`nbNodes`, `nbVehi`, `nbServi`, and the 0/1 matrices `r` and `a`);
  2. key lookup anchored to the start of a line and required to match EXACTLY ONCE, so a key can
     no longer be found inside another token and a duplicated key is an error rather than a
     silent first-match;
  3. full shape assertions: row COUNT *and* row WIDTH for `r`, `a` and `p`, length of `e` and `l`,
     `r`/`a` restricted to {0,1}, `p` finite and non-negative, `e[i] <= l[i]`;
  4. every violation raises `HHCRSPFormatError` naming the offending key and index — `assert` is
     no longer used for validation, because `assert` disappears under `python -O`.

**Two design parameters are deliberately UNSET here** and must be pre-registered in R0-D
(`rerun_plan_v2_1.md` REQ-1, REQ-9 item 1):
  * `num_slots` — the pre-S2 code silently defaulted to `round(n_tasks/12)` clipped to [1,15];
    a silent default defeats pre-registration, so `None` now raises;
  * `on_unbuildable_task` — whether a required service with no qualified staff, or with no
    positive processing time, ABORTS the load or is dropped-and-recorded. Both are defensible;
    choosing one here would be an unregistered design decision, so `None` raises. Either way the
    counts are recorded in `spec.meta` so the evidence survives (REQ-9 item 4).

    python3 journal_p0/external/hhcrsp_adapter.py path/to/InstanzCPLEX_HCSRP_50_1.txt \\
        --slots 3 --on-unbuildable-task drop_and_record
"""
from __future__ import annotations
import argparse, math, re, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instance_spec import InstanceSpec, TaskSpec, WorkerSpec


class HHCRSPFormatError(AssertionError, ValueError):
    """The file is not the MATLAB-style HHCRSP layout this adapter documents, or violates it.

    Inherits from both AssertionError and ValueError so that callers written against the pre-S2
    behaviour (which raised both) keep working.
    """


class ExternalDesignUnregistered(ValueError):
    """A design parameter that R0-D must pre-register was not supplied (REQ-1 / REQ-9)."""


ON_UNBUILDABLE_TASK = ("error", "drop_and_record")

#: One number pattern for the whole file: optional sign, optional decimal point, optional exponent.
_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_INNER_GROUP = re.compile(r"\[([^\[\]]*)\]")


def _find_key(txt, key):
    """Locate `key=` anchored at the start of a line. Exactly one match is required."""
    hits = list(re.finditer(r"(?m)^[ \t]*%s[ \t]*=" % re.escape(key), txt))
    if not hits:
        raise HHCRSPFormatError(
            f"key '{key}=' not found at the start of any line. If this is an `InstanzVNS_*` file, "
            f"it uses a different layout (`nbCust=`, whitespace-delimited) that this adapter does "
            f"not parse.")
    if len(hits) > 1:
        raise HHCRSPFormatError(f"key '{key}=' appears {len(hits)} times; the format defines it once")
    return hits[0]


def _scalar(txt, key):
    m = _find_key(txt, key)
    m2 = re.match(r"\s*(%s)" % _NUM, txt[m.end():])
    if not m2:
        raise HHCRSPFormatError(f"scalar '{key}=' is not followed by a number")
    raw = m2.group(1)
    try:
        val = int(raw)
    except ValueError:
        raise HHCRSPFormatError(f"scalar '{key}={raw}' is not an integer")
    if val <= 0:
        raise HHCRSPFormatError(f"scalar '{key}={val}' must be positive")
    return val


def _block(txt, key):
    """Return the substring of key's value: the bracket-balanced `[ ... ]` following `key=`."""
    m = _find_key(txt, key)
    i = txt.find("[", m.end())
    if i < 0:
        raise HHCRSPFormatError(f"block '{key}=' is not followed by '['")
    depth = 0
    for j in range(i, len(txt)):
        if txt[j] == "[":
            depth += 1
        elif txt[j] == "]":
            depth -= 1
            if depth == 0:
                return txt[i:j + 1]
    raise HHCRSPFormatError(f"unterminated block '{key}'")


def _rows(block, key, n_rows, n_cols):
    """Innermost `[ ... ]` groups, each parsed as floats. Row COUNT and row WIDTH are both checked
    — the missing width check is what let the decimal-splitting defect through."""
    groups = _INNER_GROUP.findall(block)
    rows = [[float(x) for x in re.findall(_NUM, g)] for g in groups]
    if len(rows) != n_rows:
        raise HHCRSPFormatError(f"block '{key}' has {len(rows)} rows, expected {n_rows}")
    for idx, row in enumerate(rows):
        if len(row) != n_cols:
            raise HHCRSPFormatError(
                f"block '{key}' row {idx} has {len(row)} values, expected {n_cols}: {row[:8]}"
                f"{'...' if len(row) > 8 else ''}")
        for c, v in enumerate(row):
            if not math.isfinite(v):
                raise HHCRSPFormatError(f"block '{key}' [{idx}][{c}] is not finite: {v!r}")
    return rows


def _vector(block, key, n):
    vals = [float(x) for x in re.findall(_NUM, block)]
    if len(vals) != n:
        raise HHCRSPFormatError(f"vector '{key}' has {len(vals)} values, expected {n}")
    for i, v in enumerate(vals):
        if not math.isfinite(v):
            raise HHCRSPFormatError(f"vector '{key}'[{i}] is not finite: {v!r}")
    return vals


def _binary(rows, key):
    out = []
    for i, row in enumerate(rows):
        new = []
        for j, v in enumerate(row):
            iv = int(v)
            if iv != v or iv not in (0, 1):
                raise HHCRSPFormatError(f"matrix '{key}'[{i}][{j}] = {v!r} is not 0 or 1")
            new.append(iv)
        out.append(new)
    return out


def parse_hhcrsp(path):
    """Parse one `InstanzCPLEX_*` file. Every violation raises `HHCRSPFormatError`."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    N, K, S = _scalar(txt, "nbNodes"), _scalar(txt, "nbVehi"), _scalar(txt, "nbServi")
    if N < 3:
        raise HHCRSPFormatError(f"nbNodes={N}: need at least two depots and one patient node")
    r = _binary(_rows(_block(txt, "r"), "r", N, S), "r")
    a = _binary(_rows(_block(txt, "a"), "a", K, S), "a")
    p_rows = _rows(_block(txt, "p"), "p", N * K, S)          # block per node = K staff rows
    for i, row in enumerate(p_rows):
        for j, v in enumerate(row):
            if v < 0:
                raise HHCRSPFormatError(f"processing time p row {i} col {j} is negative: {v!r}")
    e = _vector(_block(txt, "e"), "e", N)
    l = _vector(_block(txt, "l"), "l", N)
    for i in range(N):
        if e[i] > l[i]:
            raise HHCRSPFormatError(f"time window inverted at node {i}: e={e[i]} > l={l[i]}")
    p = [p_rows[i * K:(i + 1) * K] for i in range(N)]        # p[node][staff][service]
    return dict(N=N, K=K, S=S, r=r, a=a, p=p, e=e, l=l)


def to_spec(hh, num_slots=None, name="hhcrsp", on_unbuildable_task=None):
    """Map a parsed instance onto the paper's assignment model.

    `num_slots` and `on_unbuildable_task` have no defaults on purpose: both are R0-D
    pre-registration parameters (see the module docstring).
    """
    if num_slots is None:
        raise ExternalDesignUnregistered(
            "num_slots is UNSET. The pre-S2 code silently used round(n_tasks/12) clipped to "
            "[1,15]; a silent default defeats the R0-D pre-registration REQ-9 item 1 requires. "
            "Pass the registered slot count explicitly.")
    if int(num_slots) != num_slots or num_slots < 1:
        raise ExternalDesignUnregistered(f"num_slots must be a positive integer, got {num_slots!r}")
    num_slots = int(num_slots)
    if on_unbuildable_task is None:
        raise ExternalDesignUnregistered(
            "on_unbuildable_task is UNSET. Whether a required service with no qualified staff (or "
            f"no positive processing time) aborts the load or is dropped-and-recorded is an R0-D "
            f"decision; pass one of {ON_UNBUILDABLE_TASK}.")
    if on_unbuildable_task not in ON_UNBUILDABLE_TASK:
        raise ExternalDesignUnregistered(
            f"on_unbuildable_task={on_unbuildable_task!r}; expected one of {ON_UNBUILDABLE_TASK}")

    N, K, S, r, a, p, e, l = (hh[k] for k in ("N", "K", "S", "r", "a", "p", "e", "l"))
    qual = {s: {k for k in range(K) if a[k][s] == 1} for s in range(S)}
    patients = list(range(1, N - 1))            # 0-based nodes 1..N-2 (i.e. format nodes 2..N-1)
    if not patients:
        raise HHCRSPFormatError("no patient nodes between the two depots")
    lo = min(e[i] for i in patients); hi = max(l[i] for i in patients)
    span = max(1.0, hi - lo)

    def bucket(i):
        mid = 0.5 * (e[i] + l[i])
        return min(num_slots - 1, max(0, int((mid - lo) * num_slots / span)))

    tasks, tid = [], 0
    required = 0
    no_staff, no_positive_time = [], []
    for i in patients:
        for s in range(S):
            if r[i][s] != 1:
                continue
            required += 1
            cands = qual.get(s, set())
            if not cands:
                no_staff.append((i, s)); continue
            durs = [p[i][k][s] for k in cands if p[i][k][s] > 0]
            if not durs:
                no_positive_time.append((i, s)); continue
            dur = sum(durs) / len(durs)         # staff-dependent times -> one d_j (mean over qualified)
            tasks.append(TaskSpec(id=tid, duration=float(dur), slot=bucket(i),
                                  candidates=frozenset(cands)))
            tid += 1
    dropped = len(no_staff) + len(no_positive_time)
    if dropped and on_unbuildable_task == "error":
        raise HHCRSPFormatError(
            f"{dropped} of {required} required services cannot be built into a task "
            f"({len(no_staff)} with no qualified staff, {len(no_positive_time)} with no positive "
            f"processing time). Under the registered policy 'error' this aborts (REQ-2). First "
            f"offenders (node,service): no_staff={no_staff[:5]} "
            f"no_positive_time={no_positive_time[:5]}.")
    if not tasks:
        raise HHCRSPFormatError("no tasks could be built from this instance")
    workers = [WorkerSpec(id=k, slots=frozenset(range(num_slots))) for k in range(K)]
    return InstanceSpec(name, tasks, workers, num_slots,
                        meta=dict(source="hhcrsp", patients=len(patients), services=S,
                                  caregivers=K, horizon=(lo, hi),
                                  num_slots_registered=num_slots,
                                  on_unbuildable_task=on_unbuildable_task,
                                  tasks_required=required, tasks_built=len(tasks),
                                  dropped_no_qualified_staff=len(no_staff),
                                  dropped_no_positive_time=len(no_positive_time),
                                  dropped_task_keys=tuple(no_staff + no_positive_time),
                                  )).validate()


def load(path, num_slots=None, on_unbuildable_task=None):
    return to_spec(parse_hhcrsp(path), num_slots=num_slots, name=os.path.basename(path),
                   on_unbuildable_task=on_unbuildable_task)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--slots", type=int, required=True,
                    help="registered slot count (R0-D); there is deliberately no default")
    ap.add_argument("--on-unbuildable-task", required=True, choices=list(ON_UNBUILDABLE_TASK),
                    help="registered policy (R0-D) for required services that cannot be built")
    args = ap.parse_args(argv)
    hh = parse_hhcrsp(args.path)
    print(f"[hhcrsp] {os.path.basename(args.path)}: nbNodes={hh['N']} caregivers={hh['K']} "
          f"services={hh['S']} patients={hh['N']-2}")
    spec = to_spec(hh, num_slots=args.slots, name=os.path.basename(args.path),
                   on_unbuildable_task=args.on_unbuildable_task)
    ds = [t.duration for t in spec.tasks]
    tps = {}
    for t in spec.tasks:
        tps[t.slot] = tps.get(t.slot, 0) + 1
    print(f"  -> {len(spec.tasks)} tasks, {len(spec.workers)} workers, {spec.num_slots} slots; "
          f"duration min/mean/max={min(ds):.2f}/{sum(ds)/len(ds):.2f}/{max(ds):.2f}")
    print(f"  tasks/slot={dict(sorted(tps.items()))}")
    print(f"  candidate-set size min/max={min(len(t.candidates) for t in spec.tasks)}/"
          f"{max(len(t.candidates) for t in spec.tasks)}")
    print(f"  required={spec.meta['tasks_required']} built={spec.meta['tasks_built']} "
          f"dropped_no_qualified_staff={spec.meta['dropped_no_qualified_staff']} "
          f"dropped_no_positive_time={spec.meta['dropped_no_positive_time']}")


if __name__ == "__main__":
    main()
