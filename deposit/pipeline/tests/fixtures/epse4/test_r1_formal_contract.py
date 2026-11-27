#!/usr/bin/env python3
"""Regression checks for R1's all-eligible formal contract.

These are tiny-fixture and input-axis checks, not scientific results.  They
write only to a temporary directory and never touch canonical artifacts.
"""
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JOURNAL = os.path.join(ROOT, "experiments", "journal_p0")
EXPERIMENTS = os.path.join(ROOT, "experiments")
for path in (JOURNAL, EXPERIMENTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import eps_constraint as R1  # noqa: E402
from displacement_milp import T_NOW, load_static  # noqa: E402
from s2_core import PipelineFailure  # noqa: E402


def main() -> int:
    manifest = os.path.join(HERE, "manifest_epse4_A_formal.csv")
    out = tempfile.mkdtemp(prefix="r1-formal-contract-")
    progress = StringIO()
    with redirect_stdout(progress):
        rng, front, design, *_ = R1.run(
            manifest,
            out,
            formal=True,
            snap_id="a" * 64,
            cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S,
            canonical_assignment=False,
            verbose=False,
        )
    assert set(rng["absent"].astype(int)) == {0, 1}
    assert set(rng["absence_rule"]) == {R1.ABSENCE_RULE}
    assert set(rng["absence_cap"].astype(int)) == {-1}
    assert (front["U"].astype(int) == front["U_common"].astype(int)).all()
    proof_columns = {
        "effective_epsilon", "deviation_parity", "epsilon_tightened_by",
        "solve_anchor_epsilon", "solver_executed_for_requested_epsilon",
        "epsilon_solution_source", "epsilon_reuse_proof",
        "epsilon_equivalence_rule", "n_incumbent_pairs_available",
    }
    assert proof_columns <= set(front.columns)
    requested = front["eps"].astype(int)
    expected_parity = (
        front["n_incumbent_pairs_available"].astype(int)
        + front["n_tasks_future"].astype(int)
        - front["U_common"].astype(int)
    ) % 2
    expected_effective = [
        int(eps) - ((int(eps) - int(parity)) % 2)
        for eps, parity in zip(requested, expected_parity)
    ]
    assert front["deviation_parity"].astype(int).tolist() \
        == expected_parity.astype(int).tolist()
    assert front["effective_epsilon"].astype(int).tolist() \
        == expected_effective
    assert (front["D"].astype(int)
            <= front["effective_epsilon"].astype(int)).all()
    sources = set(front["epsilon_solution_source"].astype(str))
    assert R1.EPSILON_PARITY_REUSE_SOURCE in sources
    assert R1.EPSILON_W_FLOOR_REUSE_SOURCE in sources
    diagnostics = pd.read_csv(os.path.join(out, "eps_solver_diagnostics.csv"))
    reused = diagnostics[
        diagnostics["solver_executed"].astype(str).str.lower().isin(
            ["false", "0"])]
    assert not reused.empty
    assert (reused["wall_time_s"].astype(float) == 0.0).all()
    assert set(reused["solver_status_code"].astype(int)) == {0}
    mutant = front.copy()
    mutant.loc[mutant.index[0], "effective_epsilon"] = (
        int(mutant.loc[mutant.index[0], "effective_epsilon"]) + 1)
    try:
        R1.extra_checks(rng, mutant, design, verbose=False, formal=True)
    except PipelineFailure as exc:
        assert "epsilon-proof" in str(exc)
    else:
        raise AssertionError("tampered effective_epsilon passed the R1 proof gate")
    placeable_a1 = front["placeable_A1"].astype(str).str.lower().isin(["true", "1"])
    assert (placeable_a1
            == (front["U_min_A1"].astype(int) == 0)).all()
    assert "R1_DISRUPTION_START rho=0.5 seed=901 absent=0" in progress.getvalue()
    assert "R1_DISRUPTION_START rho=0.5 seed=901 absent=1" in progress.getvalue()
    assert R1.EXPECTED_RUNTIME_CONTRACT["solver_time_limit_s"] == 1800.0
    assert (R1.EXPECTED_RUNTIME_CONTRACT[
        "overtime_variable_upper_bound_minutes"]
        == R1.H_MAX - R1.H_STD == 120.0)

    try:
        R1.run(
            manifest,
            tempfile.mkdtemp(prefix="r1-formal-cap-must-fail-"),
            max_absent=1,
            formal=True,
            snap_id="b" * 64,
            cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S,
            canonical_assignment=False,
            verbose=False,
        )
    except PipelineFailure as exc:
        assert "every eligible disruption" in str(exc)
    else:
        raise AssertionError("formal R1 accepted a debug --max-absent cap")

    official = pd.read_csv(os.path.join(JOURNAL, "rho_manifest.csv"))
    total = 0
    for _, row in official.iterrows():
        plan = load_static(os.path.join(JOURNAL, str(row["sol"])))
        direct = sorted({n for q, assignments in plan.items() if q >= T_NOW
                         for n in assignments.values()})
        via_r1 = R1.active_absentees(None, plan, None)
        assert via_r1 == direct
        assert len(via_r1) > 6  # every current official scenario exposes the old truncation bug
        total += len(via_r1)
    assert len(official) == 30
    print("R1_FORMAL_CONTRACT_PASS scenarios={} eligible_total={} min_per_scenario_gt_6=true".format(
        len(official), total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
