"""
external/run_external.py (E1-E2) — run the A1-vs-A2 disruption study on one EXTERNAL
benchmark instance, reusing the paper's exact recovery
solver (displacement_milp.solve_slot) via MiniParser. Produces the same fair
comparison as E4: is the stability weight inert under A1, and does displacement (A2)
recover the trade-off, on real qualification/duration data?

Formal R2 has no scientific command-line defaults.  It requires a HUMAN-REGISTERED design
manifest plus the independently recomputed 64-hex SNAP archive hash:

    python3 journal_p0/external/run_external.py --stage R2 \
      --manifest REGISTERED_EXTERNAL_SINGLE.json --snap-id <archive-sha256> --out-dir <dir>

`external_design_manifest_TEMPLATE.json` is deliberately HUMAN-BLOCKED and cannot run.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, math, tempfile
from collections import defaultdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Code/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SubgroupOptimizer import H_MAX
from displacement_milp import solve_slot
from instance_spec import (MiniParser, build_flat_feasible_incumbent, _synthetic_spec,
                           greedy_worker_cover)
from run_family import (load_external_design, run_instance, summarize,
                        validate_external_closure)
import s2_core

W4_SWEEP = [0, 5, 10, 20, 50, 100]


def ci95(a):
    a = [float(x) for x in a]
    if len(a) < 2:
        return 0.0
    m = sum(a) / len(a)
    sd = (sum((x - m) ** 2 for x in a) / (len(a) - 1)) ** 0.5
    return 1.96 * sd / math.sqrt(len(a))


def load_spec(benchmark, path, num_slots, on_unbuildable_task):
    if benchmark == "smptsp":
        import smptsp_adapter
        return smptsp_adapter.load(
            path, num_slots=num_slots, on_unbuildable_task=on_unbuildable_task)
    if benchmark == "hhcrsp":
        import hhcrsp_adapter
        return hhcrsp_adapter.load(
            path, num_slots=num_slots, on_unbuildable_task=on_unbuildable_task)
    raise ValueError(benchmark)


def run_disruption(parser, slot_assign, absent, S, regime, w4, *,
                   solver_options=None, solver_policy=None, evidence_sink=None,
                   instance_name=""):
    """Sum solve_slot over all slots for one (absent worker, regime, w4). -> (dev, ot, unserved)."""
    dev = ot = served = total = 0
    for q in sorted(slot_assign):
        out = solve_slot(
            parser, q, slot_assign[q], absent, w4, S, regime=regime,
            solver_options=solver_options,
            admissible_statuses=(
                (0,) if solver_policy is None else solver_policy.admissible_codes),
            status_names=(None if solver_policy is None else
                          {x.code: x.name for x in solver_policy.admissible}),
            stage=("UNREGISTERED" if solver_policy is None else solver_policy.stage))
        if evidence_sink is not None:
            evidence_sink.append(dict(
                inst=instance_name, absent=absent, regime=regime, w4=float(w4), slot=q,
                **out.evidence_row()))
        d, o, s, n = out
        dev += d; ot += o; served += s; total += n
    return dev, round(ot, 1), total - served


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["R2"])
    ap.add_argument("--manifest", required=True,
                    help="human-registered R2 single-instance design JSON; templates are rejected")
    ap.add_argument("--snap-id", required=True,
                    help="64-hex SHA256 independently recomputed from immutable SNAP archive")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    registered = load_external_design(args.manifest, args.snap_id, expected_mode="single")
    design = registered["design"]
    if not os.path.isdir(args.out_dir):
        raise ValueError(f"--out-dir does not exist: {args.out_dir!r}")
    path = design.get("input_path")
    if not isinstance(path, str) or not path:
        raise ValueError("single-instance design.input_path is UNSET")
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), path)
    if not os.path.isfile(path):
        raise ValueError(f"registered input_path does not exist: {path!r}")
    spec = load_spec(design["benchmark"], path, int(design["slots"]),
                     design["on_unbuildable_task"]).validate()
    solver_rows = []
    df, rep = run_instance(
        spec, rho=float(design["rho"]), min_coverage=int(design["min_coverage"]),
        max_absent=int(design["max_absent"]), absence_rule=design["absence_rule"],
        w4_levels=registered["w4_levels"],
        solver_options=registered["solver_options"],
        solver_policy=registered["policy"], evidence_sink=solver_rows)
    if len(df) != rep["expected_rows"]:
        raise ValueError(
            f"raw rows {len(df)} != registered realized design rows {rep['expected_rows']}")
    df["inst"] = spec.name
    for name, value in rep.items():
        if isinstance(value, (str, int, float, bool)):
            df[name] = value
    summary_df = pd.DataFrame([{**summarize(df), "inst": spec.name}])
    solver_df = pd.DataFrame(solver_rows)
    validate_external_closure(df, solver_df, family=False)
    metadata = dict(
        stage="R2", snap_id=registered["snap_id"],
        design_manifest_sha256=registered["manifest_sha256"],
        r0d_registration_sha256=registered["registration_sha256"],
        solver_options_json=json.dumps(registered["solver_options"], sort_keys=True,
                                       separators=(",", ":")))
    for frame in (df, summary_df, solver_df):
        for key, value in metadata.items():
            frame[key] = value
    prefix = f"external_{design['benchmark']}_{os.path.basename(path)}".replace("/", "_")
    frames = {
        f"{prefix}_raw.csv": df,
        f"{prefix}_summary.csv": summary_df,
        f"{prefix}_solver_evidence.csv": solver_df,
    }
    tmpdir = tempfile.mkdtemp(prefix="external_r2_single_", dir=args.out_dir)
    try:
        for base, frame in frames.items():
            staged = os.path.join(tmpdir, base)
            frame.to_csv(staged, index=False)
            landed = pd.read_csv(staged)
            if len(landed) != len(frame):
                raise ValueError(f"post-write row count mismatch for {base}")
            os.replace(staged, os.path.join(args.out_dir, base))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    for base in frames:
        output = os.path.join(args.out_dir, base)
        print(f"[hashes] {base} sha256={s2_core.sha256_file(output)}")


if __name__ == "__main__":
    main()
