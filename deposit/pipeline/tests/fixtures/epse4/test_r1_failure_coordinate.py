#!/usr/bin/env python3
"""Focused regression for R1's fail-fast solver-coordinate evidence.

This injects one status-1 result before HiGHS is called.  It is diagnostic
evidence only: no scientific result is generated and no canonical path is
written.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JOURNAL = os.path.join(ROOT, "experiments", "journal_p0")
EXPERIMENTS = os.path.join(ROOT, "experiments")
for path in (JOURNAL, EXPERIMENTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import eps_constraint as R1  # noqa: E402
import eps_oracle_independent as ORACLE  # noqa: E402
from displacement_milp import load_static  # noqa: E402
from s2_core import SolverStatusFailure  # noqa: E402


def main() -> int:
    parser = R1.HHCPFullParser(os.path.join(HERE, "H_epse4_A.csv"))
    incumbent = load_static(os.path.join(HERE, "S_epse4_A.csv"))
    policy = R1.solver_policy(
        cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S,
        require_cli_time_limit=True,
    )
    model = R1.HorizonModel(
        parser, incumbent, set(parser.nurses), 0, "A1", policy,
        label="rho=0.5 seed=901 absent=0",
    )

    captured = {"calls": 0}

    def forced_status_one(*, c, constraints, integrality, bounds, options):
        captured.update(
            calls=captured["calls"] + 1,
            bounds=bounds,
            constraints=constraints,
            options=dict(options),
        )
        return SimpleNamespace(
            status=1,
            x=None,
            message="forced iteration or time limit",
            mip_node_count=17,
        )

    original_milp = R1.milp
    stderr = StringIO()
    R1.milp = forced_status_one
    try:
        with redirect_stderr(stderr):
            try:
                model._solve(
                    "W", u_eq=0, eps=1, stage_label="lex2_minW")
            except SolverStatusFailure as exc:
                assert exc.context["status_code"] == 1
                assert exc.context["where"] == "R1:lex2_minW"
            else:
                raise AssertionError("R1 accepted an injected status-1 solve")
    finally:
        R1.milp = original_milp

    lines = [line for line in stderr.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1, lines
    record = json.loads(lines[0])
    assert lines[0] == json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert record["event"] == "R1_SOLVER_STATUS_FAILURE"
    assert record["stage"] == "R1"
    assert record["rho"] == 0.5
    assert record["seed"] == 901
    assert record["absent"] == 0
    assert record["regime"] == "A1"
    assert record["epsilon"] == 1
    assert record["effective_epsilon"] == 0
    assert record["deviation_parity"] == 0
    assert record["epsilon_tightened_by"] == 1
    assert record["lex_stage"] == "lex2_minW"
    assert record["solver_status_code"] == 1
    assert record["solver_message"] == "forced iteration or time limit"
    assert record["wall_time_s"] >= 0.0
    assert record["time_limit_s"] == R1.FORMAL_SOLVER_TIME_LIMIT_S == 1800.0
    assert record["presolve"] is False
    assert captured["calls"] == 1             # no changed-option retry
    assert captured["options"] == {"presolve": False, "time_limit": 1800.0}
    assert policy.admissible_codes == {0}      # status 1 remains forbidden
    assert float(np.asarray(captured["constraints"].ub)[-1]) \
        == -float(model.dev_const)

    upper = np.asarray(captured["bounds"].ub, dtype=float)
    for index in model.oidx.values():
        assert upper[index] == R1.OVERTIME_VARIABLE_UPPER_BOUND == 120.0
    assert np.isfinite(upper[list(model.oidx.values())]).all()

    oracle_incumbent = ORACLE.read_incumbent_csv(
        os.path.join(HERE, "S_epse4_A.csv"))
    oracle_model = ORACLE.OracleHorizonModel(
        parser, oracle_incumbent, 0, "A1",
        time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S, presolve=False,
        rho=0.5, seed=901)
    oracle_captured = {"calls": 0}

    def forced_oracle_status_one(*, c, constraints, integrality, bounds, options):
        oracle_captured.update(
            calls=oracle_captured["calls"] + 1,
            bounds=bounds,
            constraints=constraints,
            options=dict(options),
        )
        return SimpleNamespace(
            status=1, x=None, message="forced oracle time limit")

    original_oracle_milp = ORACLE.milp
    oracle_stderr = StringIO()
    ORACLE.milp = forced_oracle_status_one
    try:
        with redirect_stderr(oracle_stderr):
            try:
                oracle_model._solve("W", u_eq=0, epsilon=1)
            except ORACLE.OracleFailure as exc:
                assert "did not return code-0 Optimal" in str(exc)
            else:
                raise AssertionError("oracle accepted an injected status-1 solve")
    finally:
        ORACLE.milp = original_oracle_milp
    oracle_lines = [
        line for line in oracle_stderr.getvalue().splitlines() if line.strip()]
    assert len(oracle_lines) == 1, oracle_lines
    oracle_record = json.loads(oracle_lines[0])
    assert oracle_lines[0] == json.dumps(
        oracle_record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert oracle_record == {
        "absent": 0,
        "component": "independent_oracle",
        "deviation_parity": 0,
        "effective_epsilon": 0,
        "epsilon": 1,
        "epsilon_tightened_by": 1,
        "event": "R1_ORACLE_SOLVER_STATUS_FAILURE",
        "lex_stage": "lex2_minW",
        "objective_stage": "W",
        "presolve": False,
        "regime": "A1",
        "rho": 0.5,
        "seed": 901,
        "solver_message": "forced oracle time limit",
        "solver_status_code": 1,
        "stage": "R1",
        "time_limit_s": 1800.0,
    }
    oracle_upper = np.asarray(oracle_captured["bounds"].ub, dtype=float)
    for index in oracle_model.oindex.values():
        assert (oracle_upper[index]
                == ORACLE.OVERTIME_VARIABLE_UPPER_BOUND_ORACLE == 120.0)
    assert np.isfinite(oracle_upper[list(oracle_model.oindex.values())]).all()
    assert oracle_captured["calls"] == 1
    assert float(np.asarray(oracle_captured["constraints"].ub)[-1]) \
        == -float(oracle_model.dev_constant)
    assert oracle_captured["options"] == {
        "presolve": False, "time_limit": 1800.0}

    print("R1_FAILURE_COORDINATE_PASS status1_rejected=true retry=false "
          "time_limit_s=1800 production_and_oracle_overtime_upper_bound=120")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
