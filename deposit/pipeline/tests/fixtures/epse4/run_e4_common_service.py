#!/usr/bin/env python3
"""Focused R1b common-service regression and negative-control fixture.

This is synthetic fixture evidence only.  It never promotes a canonical result,
never invents a SNAP receipt, and never claims scientific validation.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
JP0 = os.path.join(REPO, "experiments", "journal_p0")
for path in (JP0, os.path.join(REPO, "experiments")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_e4 as E4  # noqa: E402
from s2_core import PipelineFailure, sha256_file  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("[PASS] " + message)


def expect_failure(fn, token, label):
    tokens = (token,) if isinstance(token, str) else tuple(token)
    try:
        fn()
    except PipelineFailure as exc:
        message = str(exc)
        missing = [value for value in tokens if value not in message]
        check(not missing,
              f"{label}: failed loudly with every expected context token")
        return message
    raise AssertionError(f"{label}: expected PipelineFailure")


def run_fixture():
    manifest = os.path.join(THIS, "manifest_epse4_A.csv")
    contract = E4.load_design_contract()
    policy = E4.solver_policy()
    check(E4.SOLVER_TIME_LIMIT_S == 1800.0
          and policy.time_limit_s == 1800.0
          and contract["runtime_contract"]["solver_time_limit_s"] == 1800.0,
          "R1b uniformly pre-registers the 1800-second per-solve limit")
    check(policy.solver_options
          == {"presolve": False, "time_limit": 1800.0},
          "R1b keeps the exact code0-only presolve=false solver options")
    with tempfile.TemporaryDirectory(prefix="r1b-common-service-") as out_dir:
        raw, summary, provenance, design, _hashes, _reports, _log = E4.run(
            manifest, out_dir, mode="horizon", max_absent=None,
            canonical_assignment=False, verbose=False)

        check(len(raw) == 24, "all two active fixture caregivers x 2 regimes x 6 w4 cells")
        check(set(raw["absent"].astype(int)) == {0, 1},
              "default uncapped axis contains every active future task holder")
        check(set(raw["eligible_disruptions_in_scenario"].astype(int)) == {2}
              and set(raw["active_caregiver_count"].astype(int)) == {3}
              and set(raw["scenario_N"].astype(int)) == {3},
              "caregiver pool N=3 is retained separately from two eligible disruptions")
        check(bool((raw["active_caregiver_count"].astype(int)
                    != raw["eligible_disruptions_in_scenario"].astype(int)).all()),
              "fixture explicitly exercises N != eligible-disruption count")
        check(bool((raw["U"] == raw["U_common"]).all())
              and bool((raw["U_common"] == raw["U_min_A1"]).all()),
              "every A1/A2/w4 row enforces U=U_common=U_min_A1")
        check(bool((raw["U_min_A2"] <= raw["U_min_A1"]).all()),
              "independent minima respect checked A1 subset A2 containment")
        check(bool((raw["U_min_A2"] < raw["U_common"]).any()),
              "fixture exercises a strict U_min_A2<U_common case without exclusion")
        check(set(raw["decomposition"]) == {"horizon"}
              and set(raw["decomposition_groups"].astype(int)) == {1},
              "every emitted result is one horizon solve")
        check(set(raw["solver_presolve"].astype(str).str.lower()) == {"false"},
              "every row records the pre-registered presolve=false policy")
        time_columns = [
            "time_limit_s", "weighted_time_limit_s", "w_refine_time_limit_s",
            "tie_break_time_limit_s",
            "umin_A1_time_limit_s", "umin_A2_time_limit_s"]
        check(all(set(raw[column].astype(float)) == {1800.0}
                  for column in time_columns),
              "every R1b solver-certificate role records the uniform 1800-second limit")
        check(set(raw["u_min_solver_status_chain"]) == {"A1:0|A2:0"},
              "both independent U-min solves have code-0 evidence")
        check(bool(raw["umin_A1_proven_optimal"].astype(bool).all())
              and bool(raw["umin_A2_proven_optimal"].astype(bool).all())
              and bool(raw["weighted_proven_optimal"].astype(bool).all())
              and bool(raw["w_refine_proven_optimal"].astype(bool).all())
              and bool(raw["tie_break_proven_optimal"].astype(bool).all()),
              "U-min, weighted-Z, W-refine and final-D certificates are separately optimal")
        role_stages = {
            "weighted_": ("Z", 5.0),
            "w_refine_": ("W", 1.0),
            "tie_break_": ("D", 1.0),
            "umin_A1_": ("U", 1.0),
            "umin_A2_": ("U", 1.0),
        }
        check(all(
                  set(raw[prefix + "exact_certificate_method"])
                  <= set(E4.EXACT_CERTIFICATE_METHODS)
                  and bool(set(raw[prefix + "exact_certificate_method"]))
                  and set(raw[prefix + "exact_certificate_objective_stage"])
                  == {stage}
                  and set(raw[prefix + "lattice_step"].astype(float)) == {step}
                  and all(
                      E4._cell_bool(
                          proof,
                          field=prefix + "lattice_proves_exact_optimality",
                          where="common-service fixture")
                      is (str(method) == E4.STRUCTURAL_LOWER_BOUND_METHOD)
                      for method, proof in zip(
                          raw[prefix + "exact_certificate_method"],
                          raw[prefix + "lattice_proves_exact_optimality"]))
                  for prefix, (stage, step) in role_stages.items()),
              "all five solver roles pair an authorized replayable method with "
              "its exact-proof truth value (structural=true, status0=false)")
        check("A2_cos_ci95" not in summary.columns
              and set(summary["ci95_method"]) == {E4.CI95_METHOD_NOTE},
              "R1b computes no disruption-IID confidence interval")
        check(set(provenance["source_file"]) == {"e4_raw.csv"},
              "summary provenance cites only the R1b raw artifact")
        check(set(raw["h_sha256"]) == {
                  sha256_file(os.path.join(THIS, "H_epse4_A.csv"))}
              and set(raw["s_sha256"]) == {
                  sha256_file(os.path.join(THIS, "S_epse4_A.csv"))},
              "raw rows carry rehashed H/S source receipts")

        parser = E4.HHCPFullParser(os.path.join(THIS, "H_epse4_A.csv"))
        plan = E4.load_static(os.path.join(THIS, "S_epse4_A.csv"))
        model = E4.WeightedModel(
            parser, plan, set(parser.nurses), 0, "A2", E4.future_slots(plan),
            policy, rho=0.95, seed=901)
        bounds = model._variable_bounds()
        overtime_upper_bounds = {
            float(bounds.ub[index]) for index in model.oidx.values()}
        check(overtime_upper_bounds == {E4.OVERTIME_SLACK_UPPER_BOUND}
              and E4.OVERTIME_SLACK_UPPER_BOUND
              == float(E4.H_MAX) - float(E4.H_STD),
              "every overtime slack has the exact finite H_MAX-H_STD upper bound")

        calls = []

        def forced_time_limit(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status=1, x=None, message="Time limit reached",
                fun=17.0, mip_dual_bound=16.0, mip_gap=0.0625,
                mip_node_count=123)

        original_milp = E4.milp
        E4.milp = forced_time_limit
        try:
            expect_failure(
                lambda: model._solve(
                    "Z", w4=50, u_eq=0,
                    stage_label="forced_timeout_context"),
                (
                    "non-admissible status; R1b aborts without retry",
                    "rho=0.95", "seed=901", "absent=0", "regime='A2'",
                    "lex_stage='forced_timeout_context'",
                    "objective_stage='Z'", "w4=50", "status_code=1",
                    "solver_message='Time limit reached'",
                    "mip_node_count=123", "time_limit_s=1800.0",
                    "presolve=False", "retry_attempted=False",
                ),
                "negative non-admissible solver-status injection")
        finally:
            E4.milp = original_milp
        check(len(calls) == 1,
              "a non-admissible solver status aborts after one call with no retry")
        passed_bounds = calls[0]["bounds"]
        check({float(passed_bounds.ub[index]) for index in model.oidx.values()}
              == {E4.OVERTIME_SLACK_UPPER_BOUND},
              "the finite overtime bound is passed to scipy.optimize.milp")

        mutated = raw.copy()
        mutated.loc[mutated.index[0], "U"] += 1
        expect_failure(
            lambda: E4.extra_checks(
                mutated, summary, provenance, design, verbose=False),
            "every row must satisfy U == U_common == U_min_A1",
            "negative common-service mutation")

        mutated = raw.copy()
        mutated.loc[mutated.index[0], "solver_presolve"] = True
        expect_failure(
            lambda: E4.extra_checks(
                mutated, summary, provenance, design, verbose=False),
            "does not record presolve=false",
            "negative presolve-policy mutation")

        mutated = raw.copy()
        mutated["active_caregiver_count"] = \
            mutated["eligible_disruptions_in_scenario"]
        expect_failure(
            lambda: E4.extra_checks(
                mutated, summary, provenance, design, verbose=False),
            "active_caregiver_count/scenario_N does not match",
            "negative caregiver-pool/eligible-count conflation mutation")

        mutated = raw.copy()
        mutated.loc[mutated.index[0], "w_refine_lattice_step"] = 5.0
        expect_failure(
            lambda: E4.extra_checks(
                mutated, summary, provenance, design, verbose=False),
            "persisted exact-certificate float differs from independent replay",
            "negative W-refine lattice-proof mutation")

        gap_rows = raw.index[raw["weighted_gap_full_objective"].notna()]
        check(len(gap_rows) > 0,
              "fixture contains a defined weighted raw gap for tamper testing")
        mutated = raw.copy()
        gap_row = gap_rows[0]
        mutated.loc[gap_row, "weighted_gap_full_objective"] = (
            float(mutated.loc[gap_row, "weighted_gap_full_objective"]) + 0.125)
        expect_failure(
            lambda: E4.extra_checks(
                mutated, summary, provenance, design, verbose=False),
            "raw full-objective gap was altered instead of preserved",
            "negative weighted raw-gap mutation")

    fake_snap = "a" * 64
    with tempfile.TemporaryDirectory(prefix="r1b-formal-mode-negative-") as out_dir:
        expect_failure(
            lambda: E4.run(
                manifest, out_dir, mode="per-slot", max_absent=None,
                formal=True, snap_id=fake_snap,
                cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S,
                canonical_assignment=False, verbose=False),
            "formal R1b is horizon-only", "negative formal per-slot request")
    with tempfile.TemporaryDirectory(prefix="r1b-formal-cap-negative-") as out_dir:
        expect_failure(
            lambda: E4.run(
                manifest, out_dir, mode="horizon", max_absent=1,
                formal=True, snap_id=fake_snap,
                cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S,
                canonical_assignment=False, verbose=False),
            "forbids an absentee cap", "negative formal absentee cap")

    print("\nR1b COMMON-SERVICE FIXTURE: PASS")


def run_shard_merge_fixture():
    """Build six tiny formal fixture shards and exercise the official merger."""
    snap = "a" * 64
    mixed_snap = "b" * 64
    hp = os.path.join(THIS, "H_epse4_A.csv")
    sp = os.path.join(THIS, "S_epse4_A.csv")
    generator = os.path.join(JP0, "make_rho_instance.py")
    family_runner = os.path.join(JP0, "gen_rho_family.py")
    h_sha, s_sha = sha256_file(hp), sha256_file(sp)
    with tempfile.TemporaryDirectory(prefix="r1b-shard-merge-") as root:
        manifest = os.path.join(root, "rho_manifest_fixture.csv")
        rows = []
        for rho in E4.FORMAL_RHOS:
            rows.append({
                "rho_target": rho, "seed": 901, "hhcp": hp, "sol": sp,
                "feasible": True, "rho_global": rho,
                "rho_used_mean": rho, "rho_used_max": rho,
                "N": 3, "cap": 360.0, "src": hp,
                "src_sha256": h_sha, "h_sha256": h_sha,
                "s_sha256": s_sha, "generator": generator,
                "generator_sha256": sha256_file(generator),
                "family_runner": family_runner,
                "family_runner_sha256": sha256_file(family_runner),
            })
        pd.DataFrame(rows).to_csv(manifest, index=False)

        wrong_n_manifest = os.path.join(root, "rho_manifest_wrong_N.csv")
        wrong_n_rows = pd.DataFrame(rows)
        wrong_n_rows["N"] = 2
        wrong_n_rows.to_csv(wrong_n_manifest, index=False)
        expect_failure(
            lambda: E4.run(
                wrong_n_manifest, os.path.join(root, "wrong_N_must_fail"),
                mode="horizon", max_absent=None,
                rho_filter=E4.FORMAL_RHOS[0], canonical_assignment=False,
                formal=True, snap_id=snap,
                cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S,
                verbose=False),
            "manifest N differs from the parsed caregiver pool",
            "negative formal manifest-N mutation")

        shard_dirs = []
        for index, rho in enumerate(E4.FORMAL_RHOS):
            shard_dir = os.path.join(root, f"shard_{index}")
            E4.run(
                manifest, shard_dir, mode="horizon", max_absent=None,
                rho_filter=rho, canonical_assignment=False, formal=True,
                snap_id=snap, cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S,
                verbose=False)
            shard_dirs.append(shard_dir)

        merged_dir = os.path.join(root, "merged")
        result = E4.merge_formal_shards(
            manifest, merged_dir, shard_dirs, snap_id=snap,
            cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S, verbose=False,
            allow_fixture_manifest=True)
        raw, summary, provenance, diagnostics, receipts = result[:5]
        check(len(raw) == 6 * 24
              and set(round(float(v), 8) for v in raw["rho"])
              == set(E4.FORMAL_RHOS),
              "formal merger accepts exactly six complete mutually exclusive rho shards")
        check(len(summary) == 6 and len(receipts) == 6,
              "combined summary and shard-receipt sidecar cover all six rho levels")
        check(os.path.isfile(os.path.join(merged_dir, "e4_shard_receipts.csv")),
              "landed merge preserves the six shard artifact SHA256 receipts")
        check(set(raw["snap_id"]) == {snap}
              and set(diagnostics["snap_id"]) == {snap},
              "combined raw and diagnostics retain one common SNAP")
        check(set(raw["eligible_disruptions_in_scenario"].astype(int)) == {2}
              and set(raw["active_caregiver_count"].astype(int)) == {3}
              and set(raw["scenario_N"].astype(int)) == {3},
              "formal shard merge preserves caregiver-pool N separately from the disruption axis")
        check(set(provenance["source_file"]) == {"e4_raw.csv"},
              "combined summary/provenance are rebuilt from combined landed raw")

        expect_failure(
            lambda: E4.merge_formal_shards(
                manifest, os.path.join(root, "missing_must_fail"),
                shard_dirs[:-1], snap_id=snap,
                cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S,
                verbose=False, allow_fixture_manifest=True),
            "exactly six shard directories", "negative missing-rho shard family")
        expect_failure(
            lambda: E4.merge_formal_shards(
                manifest, os.path.join(root, "duplicate_must_fail"),
                shard_dirs[:-1] + [shard_dirs[0]], snap_id=snap,
                cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S, verbose=False,
                allow_fixture_manifest=True),
            "duplicate shard directory", "negative duplicate-rho shard family")

        mixed_dir = os.path.join(root, "shard_5_mixed_snap")
        E4.run(
            manifest, mixed_dir, mode="horizon", max_absent=None,
            rho_filter=E4.FORMAL_RHOS[-1], canonical_assignment=False,
            formal=True, snap_id=mixed_snap,
            cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S,
            verbose=False)
        expect_failure(
            lambda: E4.merge_formal_shards(
                manifest, os.path.join(root, "mixed_snap_must_fail"),
                shard_dirs[:-1] + [mixed_dir], snap_id=snap,
                cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S, verbose=False,
                allow_fixture_manifest=True),
            "SNAP", "negative mixed-SNAP shard family")

    print("R1b SIX-SHARD MERGE FIXTURE: PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true",
        help="retained for explicit harnesses; the fixture runs by default")
    parser.add_argument(
        "--skip-shard-merge", action="store_true",
        help="run only the single-shard common-service checks")
    args = parser.parse_args(argv)
    run_fixture()
    if not args.skip_shard_merge:
        run_shard_merge_fixture()
    return 0


if __name__ == "__main__":
    sys.exit(main())
