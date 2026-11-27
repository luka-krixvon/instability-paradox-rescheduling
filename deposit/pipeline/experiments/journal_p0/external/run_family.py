"""
external/run_family.py (E1/E2) — run the A1/A2 study over an EXTERNAL instance FAMILY
across a tightness axis, and emit the E4-style summary the manuscript uses.

Tightness is induced (benchmarks are over-provisioned, as the paper found for its own
instance): we reduce to a redundant worker cover (min_coverage>=2 so an absence never
orphans a task lacking any other qualified worker) and vary the number of time slots
(fewer slots -> more tasks per worker-slot cell -> tighter).

===============================================================================================
S2 REVISION (2026-07-29) — fail-fast and UNSET parameters only; nothing was run
===============================================================================================
This driver executes stage **R2**, whose hard predecessors are R0 passed, **R0-D** signed off and
**SNAP** closed (`rerun_plan_v2_1.md` REQ-1). None of those exists. **No external solve was run to
produce this revision**, no parameter was tuned, and no number here is validated by anything in
this file. `main()` therefore refuses to run unless `--stage R2` is passed explicitly.

Changes:

1. **Every design parameter is now manifest-only rather than silently defaulted.** The pre-S2 defaults
   (`--max-absent 5` here versus `6` in `run_external.py`; `--min-coverage 2`; `--rho 0.99`;
   `--slots 3`) were undisclosed design choices that pre-registration exists to prevent, and the
   two drivers disagreed with each other. A HUMAN-REGISTERED R0-D manifest is now mandatory and
   `run_instance()` raises `ExternalDesignUnregistered` if any scientific field is omitted.

2. **The primary absence rule is UNSET and must be pre-registered (REQ-9 item 3).** The pre-S2
   code made `safe()` — keep only workers all of whose tasks retain an alternative qualified
   worker — the primary axis, silently, with `or ranked[:max_absent]` as an invisible fallback.
   `safe()` is now available ONLY as `--absence-rule safe_filter_sensitivity`, and the module
   docstring states that it may never be the primary analysis. **Which rule is primary is not
   decided here**: `--absence-rule` is required and every offered rule is listed as a candidate
   for R0-D registration, not as a recommendation. The number of workers the filter rejects is
   returned so it can be persisted rather than discarded.

3. **Fail-fast (REQ-2).** `if o:` silently contributed 0 deviation, 0 overtime and 0 unserved for
   a failed solve; it now raises. An unplaced task or a non-flat-feasible incumbent aborts via
   `build_flat_feasible_incumbent(strict=True)` instead of being computed and thrown away.

The shipped `external_design_manifest_TEMPLATE.json` remains HUMAN-BLOCKED. It cannot run until
R0-D registers every field and its registration SHA256. Formal invocation is:

    python3 journal_p0/external/run_family.py --stage R2 \
      --manifest REGISTERED_EXTERNAL_FAMILY.json --snap-id <archive-sha256> --out-dir <dir>
"""
from __future__ import annotations
import argparse, glob as globmod, json, math, os, shutil, sys, tempfile
from collections import defaultdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Code/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # journal_p0/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_core
from SubgroupOptimizer import H_MAX
from displacement_milp import solve_slot
from instance_spec import MiniParser, build_flat_feasible_incumbent, greedy_worker_cover
from hhcrsp_adapter import ExternalDesignUnregistered, ON_UNBUILDABLE_TASK

W4_SWEEP = [0, 5, 10, 20, 50, 100]
REGIMES = ("A1", "A2")
HERE = os.path.dirname(os.path.abspath(__file__))

#: Candidate absence rules for R0-D registration. This list is NOT a recommendation and contains no
#: default: REQ-9 item 3 requires the primary rule to be pre-specified by a human, and it requires
#: `safe_filter_sensitivity` never to be the primary analysis.
ABSENCE_RULES = ("all_task_holders", "busiest_task_holders", "safe_filter_sensitivity")


class ExternalSolveFailure(RuntimeError):
    """A recovery solve failed. REQ-2 forbids contributing zeros and continuing."""


def load_external_design(path, snap_id, *, expected_mode):
    """Load a HUMAN-REGISTERED R2 design; a template can never start a solve.

    ``snap_id`` is supplied by the outer runner after independently hashing the immutable archive.
    The manifest contains only the non-self-referential binding rule.
    """
    if not path or not os.path.isfile(path):
        raise ExternalDesignUnregistered(f"R2 design manifest not found: {path!r}")
    if not isinstance(snap_id, str) or len(snap_id) != 64 \
            or any(ch not in "0123456789abcdef" for ch in snap_id):
        raise ExternalDesignUnregistered(
            "--snap-id must be the 64-lowercase-hex SHA256 of the immutable SNAP archive")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            manifest = json.load(fh)
        except ValueError as exc:
            raise ExternalDesignUnregistered(
                f"R2 design manifest is invalid JSON: {exc}") from exc
    if manifest.get("stage") != "R2":
        raise ExternalDesignUnregistered(
            f"external manifest stage={manifest.get('stage')!r}, expected 'R2'")
    if manifest.get("registration_status") != "REGISTERED":
        raise ExternalDesignUnregistered(
            "R2 external design remains HUMAN-BLOCKED. Copy the template, complete R0-D, set "
            "registration_status='REGISTERED', and record its registration hash before solving.")
    if manifest.get("snap_binding") != "cli_content_addressed_archive_sha256":
        raise ExternalDesignUnregistered(
            "external manifest must register "
            "snap_binding='cli_content_addressed_archive_sha256'")
    registration_hash = manifest.get("r0d_registration_sha256")
    if not isinstance(registration_hash, str) or len(registration_hash) != 64 \
            or any(ch not in "0123456789abcdef" for ch in registration_hash):
        raise ExternalDesignUnregistered(
            "r0d_registration_sha256 must be a non-empty 64-lowercase-hex human registration hash")
    if manifest.get("mode") != expected_mode:
        raise ExternalDesignUnregistered(
            f"external manifest mode={manifest.get('mode')!r}, runner expects {expected_mode!r}")
    if manifest.get("analysis_role") not in ("primary", "sensitivity"):
        raise ExternalDesignUnregistered(
            "analysis_role must be human-registered as 'primary' or 'sensitivity'")
    policy = s2_core.StageSolverPolicy.from_manifest(
        manifest, stage="R2", manifest_path=path,
        manifest_sha256=s2_core.sha256_file(path))
    if policy.admissible_codes != {0} or policy.optimal_codes != {0} \
            or policy.accepts_time_limit_incumbent:
        raise ExternalDesignUnregistered(
            "R2 external exact solves are code-0-only; time-limit incumbents are not admissible")
    design = manifest.get("design")
    if not isinstance(design, dict):
        raise ExternalDesignUnregistered("external manifest.design must be an object")
    common = ("benchmark", "slots", "rho", "min_coverage", "max_absent",
              "absence_rule", "on_unbuildable_task", "dur_scale", "w4_levels",
              "solver_options")
    missing = [key for key in common if key not in design or design[key] is None]
    if missing:
        raise ExternalDesignUnregistered(
            f"external R0-D registration has UNSET fields: {missing}")
    if design["benchmark"] not in ("hhcrsp", "smptsp"):
        raise ExternalDesignUnregistered(
            f"benchmark={design['benchmark']!r}; expected hhcrsp or smptsp")
    numeric_int = ("slots", "min_coverage", "max_absent")
    for key in numeric_int:
        value = design[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ExternalDesignUnregistered(
                f"design.{key} must be a registered positive integer, got {value!r}")
    for key in ("rho", "dur_scale"):
        value = float(design[key])
        if not math.isfinite(value) or value <= 0:
            raise ExternalDesignUnregistered(
                f"design.{key} must be positive and finite, got {design[key]!r}")
    if design["absence_rule"] not in ABSENCE_RULES:
        raise ExternalDesignUnregistered(
            f"absence_rule={design['absence_rule']!r}; expected {ABSENCE_RULES}")
    if design["absence_rule"] == "safe_filter_sensitivity" \
            and manifest.get("analysis_role") == "primary":
        raise ExternalDesignUnregistered(
            "safe_filter_sensitivity is selection-biased and may not be the primary analysis")
    if design["on_unbuildable_task"] not in ON_UNBUILDABLE_TASK:
        raise ExternalDesignUnregistered(
            f"on_unbuildable_task={design['on_unbuildable_task']!r}; "
            f"expected {ON_UNBUILDABLE_TASK}")
    options = design["solver_options"]
    if not isinstance(options, dict) or options.get("presolve") is not False:
        raise ExternalDesignUnregistered(
            "solver_options must be an object with presolve=false")
    if float(options.get("time_limit", -1)) != float(policy.time_limit_s):
        raise ExternalDesignUnregistered(
            "solver_options.time_limit must equal solver.time_limit_s")
    if set(options) != {"time_limit", "presolve", "mip_rel_gap"}:
        raise ExternalDesignUnregistered(
            "solver_options must declare exactly time_limit, presolve and mip_rel_gap; "
            f"got {sorted(options)}")
    mip_rel_gap = float(options["mip_rel_gap"])
    if not math.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ExternalDesignUnregistered(
            f"solver_options.mip_rel_gap must be finite and non-negative, got {mip_rel_gap!r}")
    try:
        levels = tuple(float(x) for x in design["w4_levels"])
    except (TypeError, ValueError) as exc:
        raise ExternalDesignUnregistered("w4_levels must be a non-empty numeric list") from exc
    if not levels or len(set(levels)) != len(levels) or not all(math.isfinite(x) for x in levels):
        raise ExternalDesignUnregistered(
            f"w4_levels must be finite, non-empty and unique, got {levels!r}")
    if 0.0 not in levels or 100.0 not in levels:
        raise ExternalDesignUnregistered(
            "registered w4_levels must contain 0 and 100 because the persisted "
            "cost-of-stability contrast is explicitly OT(w4=100)-OT(w4=0)")
    return dict(
        manifest=manifest, design=design, policy=policy, snap_id=snap_id,
        manifest_sha256=policy.manifest_sha256,
        registration_sha256=registration_hash, w4_levels=levels,
        solver_options=dict(options))


def ci95(a):
    a = [float(x) for x in a]
    if len(a) < 2:
        return 0.0
    m = sum(a) / len(a); sd = (sum((x - m) ** 2 for x in a) / (len(a) - 1)) ** 0.5
    return 1.96 * sd / math.sqrt(len(a))


def load_one(benchmark, path, slots, dur_scale, on_unbuildable_task=None):
    if benchmark == "hhcrsp":
        import hhcrsp_adapter
        return hhcrsp_adapter.load(path, num_slots=slots,
                                   on_unbuildable_task=on_unbuildable_task)
    if benchmark == "smptsp":
        import smptsp_adapter
        spec = smptsp_adapter.load(path, num_slots=slots,
                                   on_unbuildable_task=on_unbuildable_task)
        if dur_scale != 1.0:                      # rescale SMPTSP interval lengths to the paper's minute scale
            from instance_spec import TaskSpec, InstanceSpec
            ts = [TaskSpec(t.id, max(1.0, t.duration * dur_scale), t.slot, t.candidates) for t in spec.tasks]
            spec = InstanceSpec(spec.name, ts, spec.workers, spec.num_slots, spec.meta)
        return spec
    raise ValueError(benchmark)


def run_instance(spec, rho=None, min_coverage=None, max_absent=None, absence_rule=None, *,
                 w4_levels=None, solver_options=None, solver_policy=None,
                 evidence_sink=None):
    """Run the A1/A2 sweep on one instance.

    Every parameter is required. Returns `(DataFrame, report)` where `report` is the incumbent
    report augmented with the absence-axis bookkeeping REQ-9 item 1a needs persisted:
    `active_task_holders`, `absence_rule`, `n_rejected_by_safe_filter`, `n_disruptions`.
    """
    for name, val in (("rho", rho), ("min_coverage", min_coverage), ("max_absent", max_absent),
                      ("absence_rule", absence_rule)):
        if val is None:
            raise ExternalDesignUnregistered(
                f"{name} is UNSET. REQ-9 item 1 requires it to be pre-registered in "
                f"audit_outputs/external_design_registration.md before the first external solve; "
                f"the pre-S2 silent default is not a registration.")
    if absence_rule not in ABSENCE_RULES:
        raise ExternalDesignUnregistered(
            f"absence_rule={absence_rule!r}; expected one of {ABSENCE_RULES}")
    levels = tuple(W4_SWEEP if w4_levels is None else w4_levels)
    if not levels:
        raise ExternalDesignUnregistered("w4_levels is empty")
    spec = greedy_worker_cover(spec, min_coverage=min_coverage)
    # REQ-2: an unplaced task or a non-flat-feasible incumbent aborts here instead of being
    # computed into `rep` and then ignored by the caller.
    sa, rep = build_flat_feasible_incumbent(spec, cap=rho * H_MAX, quiet=True, strict=True)
    parser = MiniParser(spec); S = {w.id for w in spec.workers}
    cand_of = {t.id: set(t.candidates) for t in spec.tasks}
    # tasks each worker holds in the incumbent
    holds = defaultdict(list)
    for q, m in sa.items():
        for t, w in m.items():
            holds[w].append(t)

    # "safely disruptable": every task the worker holds has an alternative qualified kept worker,
    # so recovery need not force a drop. This filter is a SENSITIVITY analysis only (REQ-9 item 3):
    # it selects on the very property the A1/A2 contrast is about, so making it primary would bias
    # the axis. It is never applied unless it was explicitly registered as the rule.
    def safe(w):
        return all(len((cand_of.get(t, set()) & S) - {w}) >= 1 for t in holds[w])

    ranked = sorted(holds, key=lambda w: (-len(holds[w]), w))
    n_rejected_by_safe = sum(1 for w in ranked if not safe(w))
    if absence_rule == "all_task_holders":
        # The unranked population axis: deterministic worker-id order, capped only by the
        # registered max_absent.  It must not silently become the busiest-worker axis.
        actives = sorted(holds)[:max_absent]
    elif absence_rule == "busiest_task_holders":
        actives = ranked[:max_absent]                  # `ranked` is already busiest-first
    else:                                              # safe_filter_sensitivity
        actives = [w for w in ranked if safe(w)][:max_absent]
        # No `or ranked[:max_absent]` fallback: the pre-S2 code silently swapped in the unfiltered
        # axis when the filter emptied it, which made the reported axis unrecoverable.
        if not actives:
            raise ExternalDesignUnregistered(
                "the safe-filter sensitivity axis is empty; REQ-9 item 2 forbids silently falling "
                "back to the unfiltered axis. Record the failure and register a new configuration.")
    rows = []
    for a in actives:
        for regime in REGIMES:
            for w4 in levels:
                dev = ot = uns = 0
                for q in sorted(sa):
                    try:
                        o = solve_slot(
                            parser, q, sa[q], a, w4, S, regime=regime,
                            solver_options=solver_options,
                            admissible_statuses=(
                                (0,) if solver_policy is None
                                else solver_policy.admissible_codes),
                            status_names=(
                                None if solver_policy is None else
                                {x.code: x.name for x in solver_policy.admissible}),
                            stage=("UNREGISTERED" if solver_policy is None
                                   else solver_policy.stage))
                    except Exception as exc:
                        raise ExternalSolveFailure(
                            f"external solve aborted for instance={spec.name!r}, slot={q!r}, "
                            f"absent={a!r}, regime={regime!r}, w4={w4!r}: "
                            f"{type(exc).__name__}: {exc}") from exc
                    if evidence_sink is not None:
                        evidence_sink.append(dict(
                            inst=spec.name, absent=a, regime=regime, w4=float(w4), slot=q,
                            **o.evidence_row()))
                    dev += o[0]; ot += o[1]; uns += o[3] - o[2]
                rows.append(dict(absent=a, regime=regime, w4=w4, deviation=dev,
                                 overtime=round(ot, 1), unserved=uns))
    rep = {**spec.meta, **rep,
           "active_task_holders": len(ranked), "absence_rule": absence_rule,
           "n_rejected_by_safe_filter": n_rejected_by_safe,
           "n_disruptions": len(actives), "max_absent": max_absent,
           "min_coverage": min_coverage,
           "expected_rows": len(actives) * len(REGIMES) * len(levels)}
    return pd.DataFrame(rows), rep


def summarize(df):
    keys = ["inst", "absent"] if "inst" in df.columns else ["absent"]
    def inert(a):
        n = ok = 0
        for _, g in a.groupby(keys):
            n += 1; ok += int(g.deviation.nunique() == 1 and g.overtime.round(1).nunique() == 1)
        return 100.0 * ok / n if n else float("nan")
    a1, a2 = df[df.regime == "A1"], df[df.regime == "A2"]
    cos = []
    for _, g in a2.groupby(["inst", "absent"]) if "inst" in a2 else a2.groupby("absent"):
        o0 = g[g.w4 == 0].overtime.values; o1 = g[g.w4 == 100].overtime.values
        if len(o0) and len(o1):
            cos.append(o1[0] - o0[0])
    mono = tot = 0
    for _, g in (a2.groupby(["inst", "absent"]) if "inst" in a2 else a2.groupby("absent")):
        tot += 1
        d = g.sort_values("w4").deviation.values
        mono += int(all(d[i] >= d[i + 1] - 1e-9 for i in range(len(d) - 1)))
    return dict(a1_inert=round(inert(a1), 1), a1_ot=round(a1.overtime.mean(), 1),
                a2_ot=round(a2.overtime.mean(), 1),
                cost_of_stability=round(sum(cos) / len(cos), 1) if cos else float("nan"),
                cos_ci95=round(ci95(cos), 1),
                a2_dev_monotone_pct=round(100.0 * mono / tot, 1) if tot else float("nan"),
                a1_unserved=round(a1.unserved.mean(), 1), a2_unserved=round(a2.unserved.mean(), 1))


def validate_external_closure(raw_df, solver_df, *, family):
    """Close every persisted disruption row to its per-slot solver/schedule evidence."""
    if raw_df.empty or solver_df.empty:
        raise ExternalSolveFailure("R2 produced an empty raw or solver-evidence artifact")
    required = {
        "solver_status_code", "solver_status_name", "solver_raw_message",
        "obj_incumbent", "mip_dual_bound", "gap_full_objective",
        "gap_full_objective_undefined_reason", "stop_reason", "assignment_hash",
        "assignment_json", "feasibility_verified", "deviation", "overtime", "unserved",
        "solver_presolve",
    }
    if not required <= set(solver_df.columns):
        raise ExternalSolveFailure(
            f"solver sidecar missing columns {sorted(required - set(solver_df.columns))}")
    if set(pd.to_numeric(solver_df.solver_status_code)) != {0}:
        raise ExternalSolveFailure("R2 solver sidecar contains a non-zero status")
    if not solver_df.feasibility_verified.astype(bool).all():
        raise ExternalSolveFailure("R2 solver sidecar contains an unverified schedule")
    if solver_df.solver_presolve.astype(bool).any():
        raise ExternalSolveFailure("R2 solver sidecar says presolve was enabled")
    if set(solver_df.stop_reason.astype(str)) != {"proven_optimal"}:
        raise ExternalSolveFailure("R2 code-0 sidecar has a non-optimal stop reason")
    for rec in solver_df.to_dict("records"):
        where = (rec["inst"], rec["absent"], rec["regime"], rec["w4"], rec["slot"])
        if not str(rec["solver_status_name"]).strip() \
                or not str(rec["solver_raw_message"]).strip():
            raise ExternalSolveFailure(f"{where}: empty solver status name/raw message")
        for name in ("obj_incumbent", "mip_dual_bound"):
            if not math.isfinite(float(rec[name])):
                raise ExternalSolveFailure(f"{where}: non-finite {name}={rec[name]!r}")
        gap = rec["gap_full_objective"]
        reason = rec["gap_full_objective_undefined_reason"]
        if pd.isna(gap):
            if str(reason) != "zero_full_objective_denominator":
                raise ExternalSolveFailure(
                    f"{where}: empty full-objective gap without zero-denominator reason")
            if abs(float(rec["obj_incumbent"]) - float(rec["mip_dual_bound"])) > 1e-12:
                raise ExternalSolveFailure(
                    f"{where}: undefined zero-denominator gap but primal != dual")
        elif not math.isfinite(float(gap)):
            raise ExternalSolveFailure(f"{where}: non-finite full-objective gap")
        try:
            pairs = json.loads(rec["assignment_json"])
            replay = {int(task): (None if worker is None else int(worker))
                      for task, worker in pairs}
        except Exception as exc:
            raise ExternalSolveFailure(
                f"{where}: assignment_json cannot be replayed: {type(exc).__name__}: {exc}") from exc
        if s2_core.assignment_hash(replay) != rec["assignment_hash"]:
            raise ExternalSolveFailure(f"{where}: assignment hash replay mismatch")

    raw_key = ["inst", "absent", "regime", "w4"]
    side = solver_df.copy()
    if family:
        raw_key = ["inst", "slots", "cap_rho", "absent", "regime", "w4"]
        side = side.rename(columns={"config_slots": "slots"})
    agg = side.groupby(raw_key).agg(
        deviation_side=("deviation", "sum"),
        overtime_side=("overtime", "sum"),
        unserved_side=("unserved", "sum")).reset_index()
    closed = raw_df.merge(agg, on=raw_key, how="outer", indicator=True)
    if set(closed["_merge"]) != {"both"}:
        raise ExternalSolveFailure(
            f"raw/sidecar key mismatch: {closed[closed['_merge'] != 'both'][raw_key + ['_merge']].head().to_dict('records')}")
    if (closed.deviation.astype(float) != closed.deviation_side.astype(float)).any() \
            or (closed.unserved.astype(float) != closed.unserved_side.astype(float)).any():
        raise ExternalSolveFailure("raw deviation/unserved does not close to per-slot evidence")
    delta = (closed.overtime.astype(float) -
             closed.overtime_side.astype(float).round(1)).abs()
    if (delta > 1e-8).any():
        raise ExternalSolveFailure("raw overtime does not close to schedule-derived per-slot overtime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["R2"],
                    help="formal stage this invocation belongs to. R2's hard predecessors are R0, "
                         "R0-D and SNAP; required here so a bare invocation cannot overwrite a "
                         "canonical artifact")
    ap.add_argument("--manifest", required=True,
                    help="human-registered R2 family design JSON; templates are rejected")
    ap.add_argument("--snap-id", required=True,
                    help="64-hex SHA256 independently recomputed from immutable SNAP archive")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    registered = load_external_design(args.manifest, args.snap_id, expected_mode="family")
    design = registered["design"]
    if not os.path.isdir(args.out_dir):
        raise ExternalDesignUnregistered(f"--out-dir does not exist: {args.out_dir!r}")
    input_glob = design.get("input_glob")
    if not isinstance(input_glob, str) or not input_glob:
        raise ExternalDesignUnregistered("family design.input_glob is UNSET")
    if not os.path.isabs(input_glob):
        input_glob = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), input_glob)
    paths = sorted(globmod.glob(input_glob))
    if not paths:
        raise ExternalDesignUnregistered(f"no files match registered input_glob={input_glob!r}")
    axis = design.get("tightness_axis")
    if not isinstance(axis, dict) or axis.get("mode") not in ("rho_sweep", "slots_sweep") \
            or not isinstance(axis.get("values"), list) or not axis["values"]:
        raise ExternalDesignUnregistered(
            "design.tightness_axis must register mode=rho_sweep|slots_sweep and non-empty values")
    if axis["mode"] == "rho_sweep":
        configs = [(int(design["slots"]), float(value)) for value in axis["values"]]
    else:
        configs = [(int(value), float(design["rho"])) for value in axis["values"]]
    if any(slots < 1 or not math.isfinite(rho) or rho <= 0 for slots, rho in configs):
        raise ExternalDesignUnregistered(f"invalid registered (slots,rho) grid: {configs!r}")
    print(f"[{design['benchmark']}] {len(paths)} instances; "
          f"registered configs (slots,rho)={configs}; snap={registered['snap_id']}")
    summary_rows, raw_parts, solver_rows = [], [], []
    for slots, rho in configs:
        parts = []
        rhos = []
        for p in paths:
            before = len(solver_rows)
            spec = load_one(
                design["benchmark"], p, slots, float(design["dur_scale"]),
                on_unbuildable_task=design["on_unbuildable_task"])
            df, rep = run_instance(
                spec, rho=rho, min_coverage=int(design["min_coverage"]),
                max_absent=int(design["max_absent"]),
                absence_rule=design["absence_rule"],
                w4_levels=registered["w4_levels"],
                solver_options=registered["solver_options"],
                solver_policy=registered["policy"], evidence_sink=solver_rows)
            if len(df) != rep["expected_rows"]:
                raise ExternalDesignUnregistered(
                    f"{os.path.basename(p)}: row count {len(df)} != "
                    f"registered realized disruptions={rep['n_disruptions']} x "
                    f"|regimes|={len(REGIMES)} x |weights|={len(registered['w4_levels'])} "
                    f"= {rep['expected_rows']} (REQ-9 item 8)")
            df["inst"] = os.path.basename(p)
            df["slots"] = slots
            df["cap_rho"] = rho
            for name in ("active_task_holders", "absence_rule",
                         "n_rejected_by_safe_filter", "n_disruptions", "max_absent",
                         "min_coverage", "unplaced", "feasible", "rho_used_mean",
                         "cover_tasks_dropped", "cover_tasks_under_covered",
                         "dropped_no_qualified_staff", "dropped_no_positive_time",
                         "dropped_no_qualified_worker"):
                df[name] = rep.get(name, spec.meta.get(name, 0))
            parts.append(df)
            raw_parts.append(df.copy())
            rhos.append(rep["rho_used_mean"])
            for evidence in solver_rows[before:]:
                evidence["config_slots"] = slots
                evidence["cap_rho"] = rho
        alldf = pd.concat(parts, ignore_index=True)
        s = summarize(alldf)
        s["slots"] = slots; s["cap_rho"] = rho
        s["rho_used"] = round(sum(rhos) / len(rhos), 2); s["instances"] = len(paths)
        summary_rows.append(s)
        print(f"  slots={slots} cap_rho={rho} rho_used={s['rho_used']:.2f} | A1 inert {s['a1_inert']:.0f}% "
              f"ot {s['a1_ot']:.0f} | A2 ot {s['a2_ot']:.0f} cost_of_stab {s['cost_of_stability']:.0f}"
              f"+-{s['cos_ci95']:.0f} dev-mono {s['a2_dev_monotone_pct']:.0f}% | "
              f"unserved A1={s['a1_unserved']:.1f} A2={s['a2_unserved']:.1f}")
    summary_df = pd.DataFrame(summary_rows)
    raw_df = pd.concat(raw_parts, ignore_index=True)
    solver_df = pd.DataFrame(solver_rows)
    validate_external_closure(raw_df, solver_df, family=True)
    metadata = dict(
        stage="R2", snap_id=registered["snap_id"],
        design_manifest_sha256=registered["manifest_sha256"],
        r0d_registration_sha256=registered["registration_sha256"],
        solver_options_json=json.dumps(registered["solver_options"], sort_keys=True,
                                       separators=(",", ":")))
    for frame in (summary_df, raw_df, solver_df):
        for key, value in metadata.items():
            frame[key] = value
    if solver_df.duplicated(
            subset=["inst", "config_slots", "cap_rho", "absent", "regime", "w4", "slot"]).any():
        raise ExternalSolveFailure("duplicate R2 per-slot solver evidence key")
    prefix = f"external_{design['benchmark']}_family"
    frames = {
        f"{prefix}.csv": summary_df,
        f"{prefix}_raw.csv": raw_df,
        f"{prefix}_solver_evidence.csv": solver_df,
    }
    tmpdir = tempfile.mkdtemp(prefix="external_r2_", dir=args.out_dir)
    try:
        staged = {}
        for base, frame in frames.items():
            path = os.path.join(tmpdir, base)
            frame.to_csv(path, index=False)
            landed = pd.read_csv(path)
            if len(landed) != len(frame):
                raise ExternalSolveFailure(
                    f"post-write row-count mismatch for {base}: {len(landed)} != {len(frame)}")
            for key, value in metadata.items():
                if key not in landed or set(landed[key].astype(str)) != {str(value)}:
                    raise ExternalSolveFailure(
                        f"post-write metadata mismatch for {base}.{key}")
            staged[base] = path
        for base, path in staged.items():
            os.replace(path, os.path.join(args.out_dir, base))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    for base in frames:
        path = os.path.join(args.out_dir, base)
        print(f"[hashes] {base} sha256={s2_core.sha256_file(path)}")


if __name__ == "__main__":
    main()
