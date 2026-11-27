#!/usr/bin/env python3
"""Focused real-instance regression for the former rho=.85/epsilon=69 timeout.

This is an operational regression over the registered input family, not a
scientific result.  It emits no CSV and never promotes an artifact.  The test
proves the parity identity from the loaded model, obtains one code-0 anchor at
epsilon 68, and verifies that requested epsilon 69 reuses that exact anchor
without another solver call.
"""
from __future__ import annotations

import json
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JOURNAL = os.path.join(ROOT, "experiments", "journal_p0")
EXPERIMENTS = os.path.join(ROOT, "experiments")
for path in (JOURNAL, EXPERIMENTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import eps_constraint as R1  # noqa: E402
from displacement_milp import load_static  # noqa: E402


def main() -> int:
    hhcp = os.path.join(JOURNAL, "H_rho085_n16_s1.csv")
    incumbent_path = os.path.join(JOURNAL, "S_rho085_n16_s1.csv")
    parser = R1.HHCPFullParser(hhcp)
    incumbent = load_static(incumbent_path)
    workforce = set(parser.nurses)
    policy = R1.solver_policy(
        cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S,
        require_cli_time_limit=True,
    )
    models = {
        regime: R1.HorizonModel(
            parser, incumbent, workforce, 25, regime, policy,
            label="rho=0.85 seed=1 absent=25",
        )
        for regime in ("A1", "A2")
    }
    u_min = {regime: models[regime].min_U()["U"] for regime in models}
    u_common = int(u_min["A1"])
    model = models["A2"]
    assert u_min == {"A1": 0, "A2": 0}, u_min
    assert model.n_tasks == 186, model.n_tasks
    assert len(model.xbar) == 178, len(model.xbar)

    budget68 = model.epsilon_budget(u_common, 68)
    budget69 = model.epsilon_budget(u_common, 69)
    assert budget68["effective"] == budget69["effective"] == 68
    assert budget68["parity"] == budget69["parity"] == 0

    started = time.perf_counter()
    anchor68 = model.solve_lex(u_common, eps=68,
                               canonical_assignment=False)
    anchor_wall_s = time.perf_counter() - started
    assert anchor68["status_chain"] == "0|0"
    assert int(anchor68["D"]) <= 68
    before = len(model.diagnostics)
    point69 = model.frontier_point(
        u_common, 69,
        # Only D is inspected before the effective-cache branch.  This sentinel
        # deliberately prevents the unrelated W-floor reuse branch here.
        w_floor_point={"D": 10 ** 9},
        effective_cache={68: anchor68},
        canonical_assignment=False,
    )
    added = model.diagnostics[before:]
    assert len(added) == 2
    assert all(record["solver_executed"] is False for record in added)
    assert all(record["epsilon"] == 69 for record in added)
    assert all(record["effective_epsilon"] == 68 for record in added)
    assert point69["epsilon_solution_source"] \
        == R1.EPSILON_PARITY_REUSE_SOURCE
    assert point69["solver_executed_for_requested_epsilon"] is False
    assert point69["solve_anchor_epsilon"] == 68
    assert (point69["U"], point69["W"], point69["D"]) \
        == (anchor68["U"], anchor68["W"], anchor68["D"])
    assert point69["assignment"] == anchor68["assignment"]

    print(json.dumps({
        "status": "R1_REAL_PARITY_REGRESSION_PASS",
        "rho": 0.85,
        "seed": 1,
        "absent": 25,
        "regime": "A2",
        "n_tasks": model.n_tasks,
        "xbar_pairs": len(model.xbar),
        "U_common": u_common,
        "requested_epsilon": 69,
        "effective_epsilon": 68,
        "anchor_status_chain": anchor68["status_chain"],
        "anchor_UWD": [anchor68["U"], anchor68["W"], anchor68["D"]],
        "anchor_wall_s": round(anchor_wall_s, 6),
        "additional_solver_calls_for_69": 0,
        "proof": R1.EPSILON_PARITY_REUSE_PROOF,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
