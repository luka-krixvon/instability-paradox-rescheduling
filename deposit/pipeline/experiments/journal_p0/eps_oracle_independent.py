"""Independent monolithic model oracle for formal R1 epsilon validation.

This module deliberately does not import ``eps_constraint`` or ``s2_core`` and
does not call any production model/constraint/objective helper.  It does share
``SubgroupOptimizer.HHCPFullParser`` as the registered HHCP *data parsing layer*;
that parser is not an independently implemented part of this oracle.  The oracle
independently parses the incumbent assignment CSV, constructs its own horizon
MILP, derives its own complete epsilon grid, reads back an integral assignment,
and recomputes U/W/D.  Formal R1 compares every
``(disruption, regime, epsilon)`` production point with this oracle:
``|W_prod-W_oracle| <= 1e-9`` and exact equality for D and U.

The numerical constants below are independently frozen oracle inputs for the
registered study design.  A production-code change to one of these constants
therefore causes a visible disagreement rather than silently moving both sides.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from SubgroupOptimizer import HHCPFullParser


T_NOW_ORACLE = 4
SLOTS_PER_DAY_ORACLE = 3
TRAVEL_PER_TASK_ORACLE = 15.0
H_STD_ORACLE = 240.0
H_MAX_ORACLE = 360.0
INT_TOL_ORACLE = 1e-6
LEX_W_ROW_SLACK_ORACLE = 1e-4
LEX_W_TOL_ORACLE = 1e-9
OVERTIME_VARIABLE_UPPER_BOUND_ORACLE = H_MAX_ORACLE - H_STD_ORACLE
EPSILON_PARITY_RULE_ORACLE = (
    "D=|xbar|+(n_tasks-U_common)-2|x_intersection_xbar|; "
    "effective_epsilon_is_largest_le_requested_with_that_parity"
)
EPSILON_DIRECT_SOURCE_ORACLE = "direct_effective_budget_solve"
EPSILON_PARITY_REUSE_SOURCE_ORACLE = "parity_equivalent_anchor_reuse"
EPSILON_W_FLOOR_REUSE_SOURCE_ORACLE = "unconstrained_w_floor_anchor_reuse"
EPSILON_DIRECT_PROOF_ORACLE = "code0_optimal_solve_at_effective_epsilon"
EPSILON_PARITY_REUSE_PROOF_ORACLE = (
    "requested_and_anchor_budgets_define_the_same_integer_feasible_set_by_D_parity"
)
EPSILON_W_FLOOR_REUSE_PROOF_ORACLE = (
    "unconstrained_lexicographic_W_then_D_optimum_remains_optimal_for_every_"
    "epsilon_that_contains_its_D_value"
)


class OracleFailure(RuntimeError):
    """The independent oracle could not establish a required exact result."""


def _need(condition, message, **context):
    if not condition:
        suffix = " ".join(
            "{}={!r}".format(k, context[k]) for k in sorted(context))
        raise OracleFailure(message + (" | " + suffix if suffix else ""))


def _task_ids(text):
    return [int(x) for x in str(text).replace(" ", "").strip(";").split(";")
            if x != ""]


def read_incumbent_csv(path):
    """Independent parser for ``S_*.csv``; no production loader is called."""
    out = defaultdict(dict)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _need(reader.fieldnames is not None, "incumbent CSV has no header",
              path=path)
        fields = set(reader.fieldnames)
        _need({"NurseID", "TaskIDs"} <= fields,
              "incumbent CSV lacks NurseID/TaskIDs", path=path,
              fields=sorted(fields))
        for line, row in enumerate(reader, 2):
            if "GlobalSlot" in fields and str(row["GlobalSlot"]).strip() != "":
                q = int(row["GlobalSlot"])
            else:
                _need({"Day", "Shift"} <= fields,
                      "incumbent CSV needs GlobalSlot or Day+Shift",
                      path=path, line=line)
                q = ((int(row["Day"]) - 1) * SLOTS_PER_DAY_ORACLE
                     + int(row["Shift"]))
            nurse = int(row["NurseID"])
            for task in _task_ids(row["TaskIDs"]):
                _need(task not in out[q],
                      "task appears twice in one incumbent slot",
                      path=path, line=line, slot=q, task=task)
                out[q][task] = nurse
    return dict(out)


def _slot_allowed(q, slots):
    return int(q) in {int(v) for v in slots}


class OracleHorizonModel:
    """Independent whole-horizon U/W/D model for one absence and regime."""

    def __init__(self, parser, incumbent, absent, regime, *,
                 time_limit_s, presolve, rho=None, seed=None):
        _need(regime in ("A1", "A2"), "unknown oracle regime", regime=regime)
        self.parser = parser
        self.incumbent = incumbent
        self.absent = int(absent)
        self.regime = regime
        self.rho = None if rho is None else float(rho)
        self.seed = None if seed is None else int(seed)
        self.time_limit_s = float(time_limit_s)
        self.presolve = bool(presolve)
        _need(OVERTIME_VARIABLE_UPPER_BOUND_ORACLE >= 0.0,
              "oracle H_MAX-H_STD overtime bound is negative",
              h_max=H_MAX_ORACLE, h_std=H_STD_ORACLE)
        self.slots = sorted(q for q in incumbent if int(q) >= T_NOW_ORACLE)
        _need(bool(self.slots), "oracle found no future slot", absent=absent)
        self.tasks = {
            q: sorted(t for t in incumbent[q] if t in parser.tasks)
            for q in self.slots
        }
        workforce = set(parser.nurses)
        self.available = {
            q: sorted(i for i in workforce if i != self.absent
                      and _slot_allowed(q, parser.nurses[i].slots))
            for q in self.slots
        }
        self.candidates = {}
        for q in self.slots:
            avail = set(self.available[q])
            for task in self.tasks[q]:
                eligible = sorted(
                    i for i in parser.tasks[task].candidate_nurses if i in avail)
                incumbent_nurse = incumbent[q][task]
                _need(bool([i for i in parser.tasks[task].candidate_nurses
                            if i in workforce]),
                      "oracle pre-absence candidate set is empty",
                      slot=q, task=task, regime=regime)
                if incumbent_nurse != self.absent:
                    _need(incumbent_nurse in set(
                        parser.tasks[task].candidate_nurses),
                        "oracle independently rejects an ineligible incumbent",
                        slot=q, task=task, nurse=incumbent_nurse)
                    _need(_slot_allowed(
                        q, parser.nurses[incumbent_nurse].slots),
                        "oracle independently rejects an off-duty incumbent",
                        slot=q, task=task, nurse=incumbent_nurse)
                self.candidates[(q, task)] = (
                    [incumbent_nurse]
                    if regime == "A1" and incumbent_nurse != self.absent
                    else eligible)

        self.duration = {
            (q, task): (float(parser.tasks[task].duration)
                        + TRAVEL_PER_TASK_ORACLE)
            for q in self.slots for task in self.tasks[q]
        }
        self.xbar = {
            (q, incumbent[q][task], task)
            for q in self.slots for task in self.tasks[q]
            if incumbent[q][task] != self.absent
            and incumbent[q][task] in set(self.available[q])
        }
        self.xkeys = [
            (q, nurse, task)
            for q in self.slots for task in self.tasks[q]
            for nurse in self.candidates[(q, task)]
        ]
        self.xindex = {key: pos for pos, key in enumerate(self.xkeys)}
        offset = len(self.xkeys)
        self.oindex = {}
        for q in self.slots:
            for nurse in self.available[q]:
                self.oindex[(q, nurse)] = offset + len(self.oindex)
        offset += len(self.oindex)
        self.uindex = {}
        for q in self.slots:
            for task in self.tasks[q]:
                self.uindex[(q, task)] = offset + len(self.uindex)
        self.nvar = offset + len(self.uindex)

        self.dev_linear = np.zeros(self.nvar)
        for key, pos in self.xindex.items():
            self.dev_linear[pos] = 1.0 - 2.0 * float(key in self.xbar)
        self.dev_constant = float(len(self.xbar))
        self.rows, self.lower, self.upper = self._structural_rows()

    def _structural_rows(self):
        rows, lower, upper = [], [], []
        for q in self.slots:
            for task in self.tasks[q]:
                row = np.zeros(self.nvar)
                for nurse in self.candidates[(q, task)]:
                    row[self.xindex[(q, nurse, task)]] = 1.0
                row[self.uindex[(q, task)]] = 1.0
                rows.append(row); lower.append(1.0); upper.append(1.0)
        for q in self.slots:
            for nurse in self.available[q]:
                load = np.zeros(self.nvar)
                for task in self.tasks[q]:
                    pos = self.xindex.get((q, nurse, task))
                    if pos is not None:
                        load[pos] = self.duration[(q, task)]
                overtime = load.copy()
                overtime[self.oindex[(q, nurse)]] = -1.0
                rows.append(overtime)
                lower.append(-np.inf); upper.append(H_STD_ORACLE)
                rows.append(load)
                lower.append(-np.inf); upper.append(H_MAX_ORACLE)
        return rows, lower, upper

    def _urow(self):
        row = np.zeros(self.nvar)
        for pos in self.uindex.values():
            row[pos] = 1.0
        return row

    def _wrow(self):
        row = np.zeros(self.nvar)
        for pos in self.oindex.values():
            row[pos] = 1.0
        return row

    def epsilon_budget(self, u_eq, requested_epsilon):
        """Independently derive the exact fixed-D-parity budget tightening."""
        _need(u_eq is not None and float(u_eq).is_integer(),
              "oracle epsilon parity tightening needs integer U",
              u_eq=u_eq, requested_epsilon=requested_epsilon)
        _need(float(requested_epsilon).is_integer(),
              "oracle epsilon must be an integer",
              epsilon=requested_epsilon)
        requested = int(requested_epsilon)
        n_tasks = sum(len(values) for values in self.tasks.values())
        selected_pairs = n_tasks - int(u_eq)
        _need(0 <= selected_pairs <= n_tasks,
              "oracle U implies invalid selected-pair count",
              u_eq=u_eq, n_tasks=n_tasks,
              selected_pairs=selected_pairs)
        parity = int((len(self.xbar) + selected_pairs) % 2)
        effective = requested - ((requested - parity) % 2)
        return {
            "requested": requested,
            "effective": int(effective),
            "parity": parity,
            "tightened_by": int(requested - effective),
            "rule": EPSILON_PARITY_RULE_ORACLE,
        }

    def _solve(self, objective, *, u_eq=None, epsilon=None, w_le=None):
        rows = list(self.rows)
        lower = list(self.lower)
        upper = list(self.upper)
        if u_eq is not None:
            rows.append(self._urow())
            lower.append(float(u_eq)); upper.append(float(u_eq))
        epsilon_meta = None
        if epsilon is not None:
            epsilon_meta = self.epsilon_budget(u_eq, epsilon)
            rows.append(self.dev_linear.copy())
            lower.append(-np.inf)
            upper.append(float(epsilon_meta["effective"]) - self.dev_constant)
        if w_le is not None:
            rows.append(self._wrow())
            lower.append(-np.inf)
            upper.append(float(w_le) + LEX_W_ROW_SLACK_ORACLE)

        c = np.zeros(self.nvar)
        if objective == "U":
            c = self._urow()
        elif objective == "W":
            c = self._wrow()
        elif objective == "D":
            c = self.dev_linear.copy()
        else:
            raise OracleFailure("unknown oracle objective {!r}".format(objective))

        lo = np.zeros(self.nvar)
        hi = np.ones(self.nvar)
        for pos in self.oindex.values():
            # Independent exact bound: the oracle's own hard-capacity row gives
            # load <= H_MAX_ORACLE, so an overtime witness never needs to exceed
            # H_MAX_ORACLE-H_STD_ORACLE.  This preserves every assignment/value.
            hi[pos] = OVERTIME_VARIABLE_UPPER_BOUND_ORACLE
        integer = np.zeros(self.nvar)
        for pos in self.xindex.values():
            integer[pos] = 1
        for pos in self.uindex.values():
            integer[pos] = 1
        result = milp(
            c=c,
            constraints=LinearConstraint(
                np.asarray(rows), np.asarray(lower), np.asarray(upper)),
            integrality=integer, bounds=Bounds(lo, hi),
            options={"presolve": self.presolve,
                     "time_limit": self.time_limit_s})
        status = int(getattr(result, "status", -1))
        if status != 0 or result.x is None:
            if objective == "U":
                lex_stage = "lex1_minU"
            elif objective == "W":
                lex_stage = "lex2_minW"
            elif w_le is None:
                lex_stage = "grid_minD"
            else:
                lex_stage = "lex3_minD"
            failure_record = {
                "event": "R1_ORACLE_SOLVER_STATUS_FAILURE",
                "stage": "R1",
                "component": "independent_oracle",
                "rho": self.rho,
                "seed": self.seed,
                "absent": self.absent,
                "regime": self.regime,
                "epsilon": None if epsilon is None else int(epsilon),
                "effective_epsilon": (
                    None if epsilon_meta is None
                    else int(epsilon_meta["effective"])),
                "deviation_parity": (
                    None if epsilon_meta is None
                    else int(epsilon_meta["parity"])),
                "epsilon_tightened_by": (
                    None if epsilon_meta is None
                    else int(epsilon_meta["tightened_by"])),
                "lex_stage": lex_stage,
                "objective_stage": objective,
                "solver_status_code": status,
                "solver_message": str(getattr(result, "message", "")),
                "time_limit_s": self.time_limit_s,
                "presolve": self.presolve,
            }
            print(json.dumps(failure_record, sort_keys=True,
                             separators=(",", ":"), allow_nan=False),
                  file=sys.stderr, flush=True)
        _need(status == 0 and result.x is not None,
              "independent oracle did not return code-0 Optimal",
              status=status, solver_message=str(getattr(result, "message", "")),
              absent=self.absent, regime=self.regime,
              objective=objective, epsilon=epsilon)
        point = self._readback(result.x)
        point.update({
            "effective_epsilon": (
                None if epsilon_meta is None
                else int(epsilon_meta["effective"])),
            "deviation_parity": (
                None if epsilon_meta is None
                else int(epsilon_meta["parity"])),
            "epsilon_tightened_by": (
                None if epsilon_meta is None
                else int(epsilon_meta["tightened_by"])),
        })
        return point, status

    def _readback(self, vector):
        vector = np.asarray(vector, dtype=float)
        _need(np.all(np.isfinite(vector)),
              "oracle solver vector contains a non-finite value")
        chosen = set()
        for key, pos in self.xindex.items():
            value = vector[pos]
            _need(abs(value - round(value)) <= INT_TOL_ORACLE,
                  "oracle integral variable is fractional",
                  key=key, value=value)
            if round(value) == 1:
                chosen.add(key)
        assignment = {}
        for q in self.slots:
            for task in self.tasks[q]:
                nurses = [i for i in self.candidates[(q, task)]
                          if (q, i, task) in chosen]
                _need(len(nurses) <= 1,
                      "oracle assigned a task more than once",
                      slot=q, task=task, nurses=nurses)
                assignment[(q, task)] = nurses[0] if nurses else None
                u_value = vector[self.uindex[(q, task)]]
                _need(abs(u_value - round(u_value)) <= INT_TOL_ORACLE,
                      "oracle unserved indicator is fractional",
                      slot=q, task=task, value=u_value)
                _need(bool(round(u_value)) == (not nurses),
                      "oracle unserved indicator disagrees with assignment",
                      slot=q, task=task, value=u_value, nurses=nurses)
        overtime = 0.0
        for q in self.slots:
            for nurse in self.available[q]:
                load = sum(
                    self.duration[(q, task)] for task in self.tasks[q]
                    if assignment[(q, task)] == nurse)
                _need(load <= H_MAX_ORACLE
                      + 8.0 * math.ulp(H_MAX_ORACLE),
                      "oracle hard capacity violation",
                      slot=q, nurse=nurse, load=load)
                overtime += max(0.0, load - H_STD_ORACLE)
        unserved = sum(value is None for value in assignment.values())
        deviation = len(chosen - self.xbar) + len(self.xbar - chosen)
        return {"U": int(unserved), "W": float(overtime),
                "D": int(deviation)}

    def min_u(self):
        return self._solve("U")[0]["U"]

    def min_d_at_common_service(self, u_common):
        """Independent deviation floor at the registered common service level."""
        point, status = self._solve("D", u_eq=u_common)
        return dict(point, solver_status_chain=str(status))

    def point(self, u_common, epsilon=None):
        """Lexicographic min-W/min-D point, optionally under an epsilon budget."""
        w_point, status_w = self._solve(
            "W", u_eq=u_common, epsilon=epsilon)
        d_point, status_d = self._solve(
            "D", u_eq=u_common, epsilon=epsilon, w_le=w_point["W"])
        _need(d_point["U"] == int(u_common),
              "oracle U equality failed", got=d_point["U"],
              expected=u_common)
        _need(abs(d_point["W"] - w_point["W"]) <= 1e-9,
              "oracle D refinement changed optimal W",
              w_opt=w_point["W"], w_final=d_point["W"])
        if epsilon is not None:
            _need(d_point["D"] <= int(epsilon),
                  "oracle point violates epsilon", epsilon=epsilon,
                  deviation=d_point["D"])
        return dict(d_point, solver_status_chain=(
            "{}|{}".format(status_w, status_d)))

    def frontier_point(self, u_common, requested_epsilon, *,
                       w_floor_point, effective_cache):
        """Independently select or solve the exact proof anchor for one grid row."""
        meta = self.epsilon_budget(u_common, requested_epsilon)
        requested = meta["requested"]
        effective = meta["effective"]
        if int(w_floor_point["D"]) <= effective:
            anchor = w_floor_point
            source = EPSILON_W_FLOOR_REUSE_SOURCE_ORACLE
            proof = EPSILON_W_FLOOR_REUSE_PROOF_ORACLE
            anchor_epsilon = -1
            solver_executed = False
        elif effective in effective_cache:
            anchor = effective_cache[effective]
            source = EPSILON_PARITY_REUSE_SOURCE_ORACLE
            proof = EPSILON_PARITY_REUSE_PROOF_ORACLE
            anchor_epsilon = effective
            solver_executed = False
        else:
            _need(requested == effective,
                  "oracle parity-equivalent epsilon lacks its prior anchor",
                  requested_epsilon=requested,
                  effective_epsilon=effective,
                  cached_effective_budgets=sorted(effective_cache))
            anchor = self.point(u_common, effective)
            effective_cache[effective] = anchor
            source = EPSILON_DIRECT_SOURCE_ORACLE
            proof = EPSILON_DIRECT_PROOF_ORACLE
            anchor_epsilon = effective
            solver_executed = True
        point = dict(anchor)
        point.update({
            "requested_epsilon": requested,
            "effective_epsilon": effective,
            "deviation_parity": meta["parity"],
            "epsilon_tightened_by": meta["tightened_by"],
            "solve_anchor_epsilon": int(anchor_epsilon),
            "solver_executed_for_requested_epsilon": bool(solver_executed),
            "epsilon_solution_source": source,
            "epsilon_reuse_proof": proof,
            "epsilon_equivalence_rule": EPSILON_PARITY_RULE_ORACLE,
        })
        _need(int(point["D"]) <= effective,
              "oracle proof anchor violates effective epsilon",
              requested_epsilon=requested,
              effective_epsilon=effective, anchor_D=point["D"], source=source)
        _need(int(point["D"]) % 2 == int(meta["parity"]),
              "oracle proof anchor violates derived D parity",
              requested_epsilon=requested,
              effective_epsilon=effective, anchor_D=point["D"],
              deviation_parity=meta["parity"], source=source)
        return point


def oracle_disruption(hhcp_path, incumbent_path, absent, *,
                      time_limit_s, presolve=False, rho=None, seed=None):
    """Derive and solve the oracle's own shared epsilon grid.

    The production frontier does not supply ``eps_values``.  For each regime the
    oracle independently computes (i) the minimum attainable D at ``U_common``
    and (ii) D at that regime's lexicographic minimum-W point.  It then applies
    the registered shared-grid rule

        lo = max_r D_floor[r],  hi = max_r D_at_Wfloor[r],

    and solves every integer epsilon in ``[lo, hi]`` for both regimes.
    """
    parser = HHCPFullParser(hhcp_path)
    incumbent = read_incumbent_csv(incumbent_path)
    models = {
        regime: OracleHorizonModel(
            parser, incumbent, absent, regime,
            time_limit_s=time_limit_s, presolve=presolve,
            rho=rho, seed=seed)
        for regime in ("A1", "A2")
    }
    u_min = {regime: models[regime].min_u() for regime in ("A1", "A2")}
    _need(u_min["A2"] <= u_min["A1"],
          "oracle contradicts A1-subset-A2 service ordering",
          u_min=u_min, absent=absent)
    u_common = int(u_min["A1"])
    d_floor = {
        regime: models[regime].min_d_at_common_service(u_common)
        for regime in ("A1", "A2")
    }
    w_floor = {
        regime: models[regime].point(u_common, epsilon=None)
        for regime in ("A1", "A2")
    }
    grid_lo = max(int(d_floor[r]["D"]) for r in ("A1", "A2"))
    grid_hi = max(int(w_floor[r]["D"]) for r in ("A1", "A2"))
    _need(grid_hi >= grid_lo,
          "oracle grid derivation produced hi < lo",
          absent=absent, grid_lo=grid_lo, grid_hi=grid_hi,
          d_floor={r: d_floor[r]["D"] for r in d_floor},
          d_at_w_floor={r: w_floor[r]["D"] for r in w_floor})
    epsilon_grid = list(range(grid_lo, grid_hi + 1))
    points = {}
    for regime in ("A1", "A2"):
        effective_cache = {}
        for epsilon in epsilon_grid:
            points[(regime, epsilon)] = models[regime].frontier_point(
                u_common, epsilon,
                w_floor_point=w_floor[regime],
                effective_cache=effective_cache)
    return {
        "u_min": u_min,
        "u_common": u_common,
        "d_floor": {r: int(d_floor[r]["D"]) for r in d_floor},
        "d_at_w_floor": {r: int(w_floor[r]["D"]) for r in w_floor},
        "grid_lo": grid_lo,
        "grid_hi": grid_hi,
        "epsilon_grid": epsilon_grid,
        "points": points,
        "independence_scope": (
            "independent_model_and_incumbent_parser;"
            "shared_HHCPFullParser_data_layer"
        ),
    }
