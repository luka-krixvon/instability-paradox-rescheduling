"""R5 scenario-clustered descriptive analysis for the formal R1b sweep.

The R1b rows are *not* independent observations.  R5 first checks the complete
all-eligible disruption axis against ``rho_manifest.csv`` and every committed
``S_*.csv``.  It then averages paired disruption effects within each
``(rho, seed)`` scenario.  Per-rho summaries are descriptive (median, IQR and
range over five scenarios); no per-rho p-value is produced by default.

The pooled sensitivity interval uses one shared seed-index vector across all
rho strata in each of 10,000 bootstrap replicates.  It gives every rho stratum
equal weight and is explicitly descriptive, not population inference.

Formal example::

    python journal_p0/r5_stats.py \
      --e4-input /results/R1b/e4_raw.csv \
      --rho-manifest journal_p0/rho_manifest.csv \
      --output-dir /results/R5.partial \
      --snap-id <content-addressed-SNAP-sha256> --formal

The optional exact sign-flip/Holm supplement is disabled unless a separate
pre-registration JSON is supplied.  It never creates a pooled cross-rho
p-value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _path in (str(HERE), str(EXPERIMENTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from displacement_milp import SLOTS_PER_DAY, T_NOW, load_static  # noqa: E402
from s2_core import (PipelineFailure, assert_no_absolute_runtime_paths,  # noqa: E402
                     assert_no_staging_import, require, require_snap_binding,
                     sha256_file)


STAGE = "R5"
DESIGN_MANIFEST = HERE / "r5_design_manifest.json"
OUTPUT_SCENARIO = "r5_scenario_effects.csv"
OUTPUT_BY_RHO = "r5_by_rho.csv"
OUTPUT_BOOTSTRAP = "r5_bootstrap.csv"
OUTPUT_HOLM = "r5_holm.csv"
OUTPUT_RUN_MANIFEST = "r5_run_manifest.json"
OUTPUT_HASHES = "r5_output_hashes.json"
SCIENTIFIC_OUTPUTS = (OUTPUT_SCENARIO, OUTPUT_BY_RHO, OUTPUT_BOOTSTRAP)
KEY_COLUMNS = ("rho", "seed", "absent", "method", "w4")
COMMON_SERVICE_COLUMNS = ("U_min_A1", "U_min_A2", "U_common", "U")
FLOAT_ATOL = 1e-9


def _fail(message: str, **context: Any) -> None:
    raise PipelineFailure(message, **context)


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    require(path.is_file(), f"{label} does not exist", path=str(path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        _fail(f"{label} is not valid UTF-8 JSON", path=str(path), error=str(exc))
    require(isinstance(value, dict), f"{label} root must be an object", path=str(path))
    return value


def _load_contract() -> Dict[str, Any]:
    contract = _load_json(DESIGN_MANIFEST, "R5 design manifest")
    require(contract.get("stage") == STAGE, "R5 manifest stage mismatch")
    grid = contract.get("design_grid")
    bootstrap = contract.get("bootstrap")
    acceptance = contract.get("r1b_acceptance")
    require(isinstance(grid, dict), "R5 manifest lacks design_grid")
    require(isinstance(bootstrap, dict), "R5 manifest lacks bootstrap")
    require(isinstance(acceptance, dict), "R5 manifest lacks r1b_acceptance")
    require(grid.get("absence_rule") == "all_active_future_task_holders",
            "R5 manifest altered the all-eligible absence rule")
    require(int(grid.get("t_now")) == int(T_NOW),
            "R5 t_now differs from executable model", manifest=grid.get("t_now"),
            executable=T_NOW)
    require(int(grid.get("slots_per_day")) == int(SLOTS_PER_DAY),
            "R5 slot convention differs from executable model",
            manifest=grid.get("slots_per_day"), executable=SLOTS_PER_DAY)
    require(int(bootstrap.get("n_resamples")) == 10000,
            "R5 bootstrap replicate count must remain pre-registered at 10000")
    require(int(bootstrap.get("rng_seed")) == 20260730,
            "R5 bootstrap seed must remain pre-registered at 20260730")
    require(bootstrap.get("shared_seed_indices_across_rho") is True,
            "R5 must reuse one seed-index vector across rho strata")
    require(bootstrap.get("pooled_p_value") is False,
            "R5 pooled cross-rho p-values are forbidden")
    require(list(acceptance.get("common_service_columns", []))
            == list(COMMON_SERVICE_COLUMNS),
            "R5 common-service schema differs from the executable contract")
    return contract


def _read_csv_strict(path: Path, label: str) -> pd.DataFrame:
    require(path.is_file(), f"{label} does not exist", path=str(path))
    require(path.stat().st_size > 0, f"{label} is empty", path=str(path))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
        _fail(f"{label} header cannot be read", path=str(path), error=str(exc))
    duplicates = sorted({name for name in header if header.count(name) > 1})
    require(not duplicates, f"{label} has duplicate column names", columns=duplicates)
    require(all(str(name).strip() for name in header),
            f"{label} has a blank column name", header=header)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        _fail(f"{label} cannot be parsed as CSV", path=str(path), error=str(exc))
    require(not frame.empty, f"{label} has no data rows", path=str(path))
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{label} is missing required columns", missing=missing,
            available=list(frame.columns))


def _numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    bad: List[str] = []
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(converted.to_numpy(dtype=float))
        if not bool(np.all(finite)):
            lines = (np.flatnonzero(~finite) + 2).tolist()
            bad.append(f"{column}:lines={lines[:8]}")
        frame[column] = converted
    require(not bad, f"{label} has nonnumeric or nonfinite cells", offending=bad)


def _as_bool(value: Any, *, column: str, line: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    _fail("R1b solver Boolean is malformed", column=column, line=line, value=value)


def _rho_value(value: Any) -> float:
    number = float(value)
    require(math.isfinite(number), "rho is nonfinite", value=value)
    return float(round(number, 12))


def _integer_value(value: Any, name: str, **context: Any) -> int:
    number = float(value)
    require(math.isfinite(number) and number.is_integer(),
            f"{name} must be an integer", value=value, **context)
    return int(number)


def _task_tokens(value: Any) -> List[int]:
    if pd.isna(value):
        return []
    out = []
    for token in str(value).replace(" ", "").strip(";").split(";"):
        if token:
            out.append(_integer_value(token, "TaskIDs token"))
    return out


def _active_absentees(solution_path: Path, contract: Mapping[str, Any]) -> List[int]:
    """Recompute the eligible axis using the same load_static semantics as R1b."""
    frame = _read_csv_strict(solution_path, "committed S file")
    _require_columns(frame, ["NurseID", "TaskIDs"], "committed S file")
    if "GlobalSlot" not in frame.columns:
        _require_columns(frame, ["Day", "Shift"], "committed S file")
    # Calling the production parser is intentional: the validator's independent
    # axis comes from S, but its interpretation cannot silently drift from R1b.
    assignment = load_static(str(solution_path))
    t_now = int(contract["design_grid"]["t_now"])
    active = sorted({int(nurse) for slot, tasks in assignment.items()
                     if int(slot) >= t_now for nurse in tasks.values()})
    require(active, "committed S file has no active future-task holder",
            file=solution_path.name, t_now=t_now)

    # Independent row-level cross-check against the same documented slot formula.
    row_active = set()
    for line_index, row in frame.iterrows():
        if "GlobalSlot" in frame.columns:
            slot = _integer_value(row["GlobalSlot"], "GlobalSlot", line=int(line_index) + 2)
        else:
            day = _integer_value(row["Day"], "Day", line=int(line_index) + 2)
            shift = _integer_value(row["Shift"], "Shift", line=int(line_index) + 2)
            slot = (day - 1) * int(contract["design_grid"]["slots_per_day"]) + shift
        if slot >= t_now and _task_tokens(row["TaskIDs"]):
            row_active.add(_integer_value(row["NurseID"], "NurseID",
                                          line=int(line_index) + 2))
    require(row_active == set(active),
            "row-level and production-parser eligible axes disagree",
            file=solution_path.name, row_axis=sorted(row_active),
            parser_axis=active)
    return active


def _manifest_axis(manifest_path: Path, contract: Mapping[str, Any]) \
        -> Tuple[Dict[Tuple[float, int], List[int]], List[Path], pd.DataFrame]:
    frame = _read_csv_strict(manifest_path, "rho_manifest.csv")
    _require_columns(frame, ["rho_target", "seed", "sol"], "rho_manifest.csv")
    _numeric(frame, ["rho_target", "seed"], "rho_manifest.csv")
    frame["_rho"] = frame["rho_target"].map(_rho_value)
    frame["_seed"] = frame["seed"].map(lambda value: _integer_value(value, "seed"))
    require(not frame.duplicated(["_rho", "_seed"]).any(),
            "rho_manifest.csv has duplicate scenario keys")

    expected_rhos = {_rho_value(value) for value in
                     contract["design_grid"]["rho_levels"]}
    expected_seeds = {int(value) for value in contract["design_grid"]["seed_labels"]}
    expected_scenarios = set(itertools.product(expected_rhos, expected_seeds))
    observed_scenarios = set(zip(frame["_rho"], frame["_seed"]))
    require(observed_scenarios == expected_scenarios,
            "rho_manifest scenario grid differs from the pre-registered grid",
            missing=sorted(expected_scenarios - observed_scenarios),
            extra=sorted(observed_scenarios - expected_scenarios))

    root = manifest_path.resolve().parent
    axis: Dict[Tuple[float, int], List[int]] = {}
    inputs: List[Path] = [manifest_path.resolve(), DESIGN_MANIFEST.resolve()]
    for _, row in frame.sort_values(["_rho", "_seed"]).iterrows():
        relative = Path(str(row["sol"]))
        require(not relative.is_absolute(),
                "rho_manifest sol paths must be snapshot-relative", sol=str(relative))
        solution = (root / relative).resolve()
        require(solution.parent == root or root in solution.parents,
                "rho_manifest sol path escapes its directory", sol=str(relative))
        require(solution.is_file(), "rho_manifest points at a missing S file",
                sol=str(relative))
        if "s_sha256" in frame.columns and not pd.isna(row["s_sha256"]):
            declared = str(row["s_sha256"]).strip().lower()
            require(len(declared) == 64 and all(ch in "0123456789abcdef" for ch in declared),
                    "rho_manifest contains malformed s_sha256", sol=str(relative),
                    value=declared)
            require(sha256_file(str(solution)) == declared,
                    "committed S bytes disagree with rho_manifest s_sha256",
                    sol=str(relative), declared=declared,
                    landed=sha256_file(str(solution)))
        key = (_rho_value(row["_rho"]), int(row["_seed"]))
        axis[key] = _active_absentees(solution, contract)
        inputs.append(solution)
    return axis, inputs, frame


def _solver_acceptance(raw: pd.DataFrame, contract: Mapping[str, Any]) -> None:
    required = contract["r1b_acceptance"]["required_solver_evidence"]
    _require_columns(raw, required.keys(), "R1b e4_raw.csv")
    violations: List[str] = []
    for column, wanted in required.items():
        if isinstance(wanted, bool):
            for index, value in enumerate(raw[column], start=2):
                if _as_bool(value, column=column, line=index) is not wanted:
                    violations.append(f"line={index} {column}={value!r}")
        else:
            converted = pd.to_numeric(raw[column], errors="coerce")
            for offset, value in enumerate(converted, start=2):
                if not math.isfinite(float(value)) or float(value) != float(wanted):
                    violations.append(f"line={offset} {column}={raw[column].iloc[offset-2]!r}")
    require(not violations,
            "R1b input contains a non-optimal or malformed solver certificate",
            offending=violations[:16], n=len(violations))
    chain_columns = list(contract["r1b_acceptance"]["required_zero_status_chains"])
    complete_column = str(contract["r1b_acceptance"]["complete_chain_json"])
    _require_columns(raw, chain_columns + [complete_column], "R1b e4_raw.csv")
    chain_violations = []
    for index, row in raw.iterrows():
        line = int(index) + 2
        for column in chain_columns:
            text = str(row[column]).strip()
            tokens = text.split("|")
            parsed = []
            for token in tokens:
                match = re.fullmatch(r"(?:[A-Za-z0-9_]+:)?(-?\d+)", token)
                if match is None:
                    chain_violations.append(
                        f"line={line} {column} malformed={text!r}")
                    parsed = []
                    break
                parsed.append(int(match.group(1)))
            if not parsed or set(parsed) != {0}:
                chain_violations.append(
                    f"line={line} {column} nonoptimal={text!r}")
        try:
            complete = json.loads(str(row[complete_column]))
        except ValueError:
            complete = None
        if not isinstance(complete, list) or not complete:
            chain_violations.append(
                f"line={line} {complete_column} malformed")
        else:
            for group in complete:
                names = group.get("stage_names") if isinstance(group, dict) else None
                codes = group.get("status_codes") if isinstance(group, dict) else None
                if (not isinstance(names, list) or not isinstance(codes, list)
                        or len(names) != len(codes) or not codes
                        or {int(value) for value in codes} != {0}):
                    chain_violations.append(
                        f"line={line} {complete_column} incomplete_or_nonoptimal")
                    break
    require(not chain_violations,
            "R1b solver stage/group chain is incomplete or non-optimal",
            offending=chain_violations[:16], n=len(chain_violations))


def validate_r1b(raw_path: Path, manifest_path: Path, contract: Mapping[str, Any],
                 *, formal: bool, snap_id: str) \
        -> Tuple[pd.DataFrame, Dict[Tuple[float, int], List[int]], List[Path], Dict[str, int]]:
    axis, inputs, _ = _manifest_axis(manifest_path, contract)
    raw = _read_csv_strict(raw_path, "R1b e4_raw.csv")
    required = (list(KEY_COLUMNS) + ["deviation", "overtime", "unserved"]
                + list(COMMON_SERVICE_COLUMNS)
                + list(contract["r1b_acceptance"]["coverage_evidence_columns"]))
    _require_columns(raw, required, "R1b e4_raw.csv")
    _numeric(raw, ["rho", "seed", "absent", "w4", "deviation", "overtime",
                   "unserved"] + list(COMMON_SERVICE_COLUMNS), "R1b e4_raw.csv")
    raw["rho"] = raw["rho"].map(_rho_value)
    for column in ("seed", "absent", "w4"):
        raw[column] = raw[column].map(lambda value, name=column:
                                      _integer_value(value, name))
    raw["method"] = raw["method"].astype(str).str.strip()
    require(not raw.duplicated(list(KEY_COLUMNS)).any(),
            "R1b e4_raw.csv has duplicate primary keys",
            sample=raw.loc[raw.duplicated(list(KEY_COLUMNS), keep=False),
                           list(KEY_COLUMNS)].head(10).to_dict("records"))

    methods = [str(value) for value in contract["design_grid"]["methods"]]
    weights = [int(value) for value in contract["design_grid"]["w4_sweep"]]
    expected = {(rho, seed, absent, method, weight)
                for (rho, seed), absent_ids in axis.items()
                for absent in absent_ids for method in methods for weight in weights}
    actual = {tuple(row) for row in raw[list(KEY_COLUMNS)].itertuples(index=False,
                                                                      name=None)}
    require(actual == expected,
            "R1b coverage is not the dynamic all-eligible product",
            dynamic_disruptions=sum(len(values) for values in axis.values()),
            expected_rows=len(expected), actual_rows=len(actual),
            missing=sorted(expected - actual, key=repr)[:12],
            extra=sorted(actual - expected, key=repr)[:12])

    _solver_acceptance(raw, contract)
    coverage_violations = []
    for index, row in raw.iterrows():
        key = (_rho_value(row["rho"]), int(row["seed"]))
        expected_count = len(axis[key])
        if not _as_bool(row["eligible_disruption"],
                        column="eligible_disruption", line=int(index) + 2):
            coverage_violations.append(
                f"line={int(index)+2} eligible_disruption={row['eligible_disruption']!r}")
        try:
            persisted_count = _integer_value(
                row["eligible_disruptions_in_scenario"],
                "eligible_disruptions_in_scenario", line=int(index) + 2)
        except PipelineFailure as exc:
            coverage_violations.append(str(exc))
            persisted_count = -1
        if persisted_count != expected_count:
            coverage_violations.append(
                f"line={int(index)+2} eligible_disruptions_in_scenario="
                f"{persisted_count} dynamic={expected_count}")
        if str(row["absence_rule"]).strip() != "all_active_future_task_holders":
            coverage_violations.append(
                f"line={int(index)+2} absence_rule={row['absence_rule']!r}")
        if formal and str(row["absence_cap_mode"]).strip() != "all_active_no_cap":
            coverage_violations.append(
                f"line={int(index)+2} absence_cap_mode={row['absence_cap_mode']!r}")
    require(not coverage_violations,
            "R1b persisted coverage evidence disagrees with the dynamic S-file axis",
            offending=coverage_violations[:16], n=len(coverage_violations))
    if formal:
        _require_columns(raw, ["snap_id"], "formal R1b e4_raw.csv")
        observed_snap = {str(value).strip().lower() for value in raw["snap_id"]}
        require(observed_snap == {snap_id},
                "formal R1b rows do not all carry the CLI SNAP identifier",
                expected=snap_id, observed=sorted(observed_snap))
    elif "snap_id" in raw.columns:
        observed_snap = {str(value).strip() for value in raw["snap_id"].fillna("")}
        require(observed_snap in ({""}, {"nan"}),
                "rehearsal R1b input may not attach or invent a SNAP identifier",
                observed=sorted(observed_snap))

    for column in ("deviation", "overtime", "unserved") + COMMON_SERVICE_COLUMNS:
        require(bool((raw[column] >= -FLOAT_ATOL).all()),
                "R1b metric is negative", column=column)
    for column in ("unserved",) + COMMON_SERVICE_COLUMNS:
        values = raw[column].to_numpy(dtype=float)
        require(bool(np.allclose(values, np.rint(values), rtol=0, atol=FLOAT_ATOL)),
                "R1b service count is nonintegral", column=column)
    require(bool(np.allclose(raw["unserved"], raw["U"], rtol=0, atol=FLOAT_ATOL)),
            "legacy unserved alias differs from formal achieved-service column U")
    require(bool(np.allclose(raw["U"], raw["U_common"], rtol=0,
                             atol=FLOAT_ATOL)),
            "R1b did not impose common service on every reported row")
    require(bool(np.allclose(raw["U_common"],
                             np.maximum(raw["U_min_A1"], raw["U_min_A2"]),
                             rtol=0, atol=FLOAT_ATOL)),
            "U_common is not max(U_min_A1,U_min_A2)")
    require(bool((raw["U_min_A2"] <= raw["U_min_A1"] + FLOAT_ATOL).all()),
            "U_min_A2 exceeds U_min_A1 despite checked A1 subset A2")
    require(bool(np.allclose(raw["U_common"], raw["U_min_A1"],
                             rtol=0, atol=FLOAT_ATOL)),
            "U_common differs from U_min_A1 under the checked nesting relation")

    metadata_violations = []
    for key, group in raw.groupby(["rho", "seed", "absent"], sort=True):
        for column in ("U_min_A1", "U_min_A2", "U_common"):
            if group[column].nunique(dropna=False) != 1:
                metadata_violations.append({"key": key, "column": column,
                                            "values": sorted(group[column].unique().tolist())})
        for weight, pair in group.groupby("w4", sort=True):
            if set(pair["method"]) != set(methods) or len(pair) != len(methods):
                metadata_violations.append({"key": key, "w4": int(weight),
                                            "methods": pair["method"].tolist()})
    require(not metadata_violations,
            "common-service metadata is not shared by both regimes and all weights",
            sample=metadata_violations[:12], n=len(metadata_violations))

    inputs.insert(0, raw_path.resolve())
    counts = {
        "scenarios": len(axis),
        "dynamic_disruptions": sum(len(values) for values in axis.values()),
        "expected_raw_rows": len(expected),
        "actual_raw_rows": len(raw),
    }
    return raw, axis, inputs, counts


def _estimand_specs(contract: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    specs = {str(item["id"]): item for item in contract["estimands"]}
    require(set(specs) == {"delta_overtime_A2_minus_A1",
                           "delta_deviation_A2_minus_A1",
                           "A2_cost_of_stability"},
            "R5 estimand family differs from the pre-registered family",
            observed=sorted(specs))
    return specs


def build_scenario_effects(raw: pd.DataFrame, axis: Mapping[Tuple[float, int], List[int]],
                           contract: Mapping[str, Any], snap_id: str) -> pd.DataFrame:
    specs = _estimand_specs(contract)
    rows: List[Dict[str, Any]] = []
    base_counts = (raw[["rho", "seed", "absent", "U_common"]]
                   .drop_duplicates(["rho", "seed", "absent"]))
    count_lookup = {}
    for key, group in base_counts.groupby(["rho", "seed"], sort=True):
        count_lookup[(_rho_value(key[0]), int(key[1]))] = (
            len(group), int(np.isclose(group["U_common"], 0.0,
                                       rtol=0, atol=FLOAT_ATOL).sum()))

    for estimand_id in ("delta_overtime_A2_minus_A1",
                         "delta_deviation_A2_minus_A1"):
        spec = specs[estimand_id]
        metric = str(spec["column"])
        paired = raw.pivot(index=["rho", "seed", "absent", "w4"],
                           columns="method", values=metric)
        require(set(paired.columns) == {"A1", "A2"},
                "paired method pivot does not contain exactly A1 and A2",
                estimand=estimand_id, columns=list(paired.columns))
        paired["effect"] = paired["A2"] - paired["A1"]
        for (rho, seed, weight), group in paired.reset_index().groupby(
                ["rho", "seed", "w4"], sort=True):
            key = (_rho_value(rho), int(seed))
            n_disruptions, n_zero = count_lookup[key]
            require(len(group) == len(axis[key]),
                    "scenario effect omitted an eligible disruption",
                    estimand=estimand_id, rho=rho, seed=seed, w4=weight,
                    expected=len(axis[key]), actual=len(group))
            rows.append({
                "stage": STAGE,
                "estimand": estimand_id,
                "rho": _rho_value(rho),
                "seed": int(seed),
                "w4": int(weight),
                "w4_lo": "",
                "w4_hi": "",
                "n_disruptions": n_disruptions,
                "n_zero_drop_disruptions": n_zero,
                "estimate": float(group["effect"].mean()),
                "unit": spec["unit"],
                "effect_direction": spec["effect_direction"],
                "snap_id": snap_id,
            })

    estimand_id = "A2_cost_of_stability"
    spec = specs[estimand_id]
    low, high = int(spec["w4_lo"]), int(spec["w4_hi"])
    a2 = raw[raw["method"] == "A2"]
    endpoint = a2[a2["w4"].isin([low, high])].pivot(
        index=["rho", "seed", "absent"], columns="w4", values=str(spec["column"]))
    require(set(endpoint.columns) == {low, high},
            "A2 endpoint pivot lacks a pre-registered weight", columns=list(endpoint.columns))
    endpoint["effect"] = endpoint[high] - endpoint[low]
    for (rho, seed), group in endpoint.reset_index().groupby(["rho", "seed"], sort=True):
        key = (_rho_value(rho), int(seed))
        n_disruptions, n_zero = count_lookup[key]
        require(len(group) == len(axis[key]),
                "cost-of-stability effect omitted an eligible disruption",
                rho=rho, seed=seed, expected=len(axis[key]), actual=len(group))
        rows.append({
            "stage": STAGE,
            "estimand": estimand_id,
            "rho": _rho_value(rho),
            "seed": int(seed),
            "w4": "",
            "w4_lo": low,
            "w4_hi": high,
            "n_disruptions": n_disruptions,
            "n_zero_drop_disruptions": n_zero,
            "estimate": float(group["effect"].mean()),
            "unit": spec["unit"],
            "effect_direction": spec["effect_direction"],
            "snap_id": snap_id,
        })
    result = pd.DataFrame(rows)
    result["_w4_sort"] = pd.to_numeric(result["w4"], errors="coerce").fillna(-1)
    result = result.sort_values(["estimand", "rho", "seed", "_w4_sort"],
                                kind="mergesort").drop(columns="_w4_sort").reset_index(drop=True)
    return result


def _analysis_cells(frame: pd.DataFrame) -> Iterable[Tuple[Tuple[Any, ...], pd.DataFrame]]:
    # Empty string endpoint fields survive CSV round-trips as NaN; normalize them
    # only for grouping/sorting, never for scientific computation.
    work = frame.copy()
    for column in ("w4", "w4_lo", "w4_hi"):
        work[column] = work[column].map(lambda value: "" if pd.isna(value) else value)
    return work.groupby(["estimand", "w4", "w4_lo", "w4_hi"],
                        sort=True, dropna=False)


def build_by_rho(scenario: pd.DataFrame, contract: Mapping[str, Any],
                 snap_id: str) -> pd.DataFrame:
    required_n = int(contract["per_rho_summary"]["required_scenarios"])
    rows: List[Dict[str, Any]] = []
    for cell, cell_frame in _analysis_cells(scenario):
        estimand, weight, low, high = cell
        for rho, group in cell_frame.groupby("rho", sort=True):
            require(len(group) == required_n and group["seed"].nunique() == required_n,
                    "per-rho summary does not contain the five registered scenarios",
                    estimand=estimand, w4=weight, rho=rho, rows=len(group),
                    unique_seeds=int(group["seed"].nunique()))
            values = group["estimate"].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
            rows.append({
                "stage": STAGE,
                "estimand": estimand,
                "rho": _rho_value(rho),
                "w4": weight,
                "w4_lo": low,
                "w4_hi": high,
                "n_scenarios": len(group),
                "n_disruptions_total": int(group["n_disruptions"].sum()),
                "n_zero_drop_disruptions_total":
                    int(group["n_zero_drop_disruptions"].sum()),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "per_rho_p_value": "NOT_COMPUTED",
                "unit": str(group["unit"].iloc[0]),
                "effect_direction": str(group["effect_direction"].iloc[0]),
                "snap_id": snap_id,
            })
    result = pd.DataFrame(rows)
    result["_w4_sort"] = pd.to_numeric(result["w4"], errors="coerce").fillna(-1)
    return (result.sort_values(["estimand", "rho", "_w4_sort"], kind="mergesort")
            .drop(columns="_w4_sort").reset_index(drop=True))


def _bootstrap_indices(contract: Mapping[str, Any]) -> np.ndarray:
    config = contract["bootstrap"]
    rng = np.random.default_rng(int(config["rng_seed"]))
    n_scenarios = int(contract["per_rho_summary"]["required_scenarios"])
    return rng.integers(0, n_scenarios,
                        size=(int(config["n_resamples"]), n_scenarios),
                        dtype=np.int64)


def _index_sha256(indices: np.ndarray) -> str:
    payload = json.dumps(indices.tolist(), separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_bootstrap(scenario: pd.DataFrame, contract: Mapping[str, Any],
                    snap_id: str) -> Tuple[pd.DataFrame, str]:
    rhos = [_rho_value(value) for value in contract["design_grid"]["rho_levels"]]
    seeds = [int(value) for value in contract["design_grid"]["seed_labels"]]
    indices = _bootstrap_indices(contract)
    index_hash = _index_sha256(indices)
    alpha = 1.0 - float(contract["bootstrap"]["ci_level"])
    rows: List[Dict[str, Any]] = []
    for cell, group in _analysis_cells(scenario):
        estimand, weight, low, high = cell
        pivot = group.pivot(index="rho", columns="seed", values="estimate")
        require(list(pivot.index.map(_rho_value)) == rhos,
                "bootstrap cell does not contain the registered rho strata",
                estimand=estimand, w4=weight, observed=list(pivot.index))
        require([int(value) for value in pivot.columns] == seeds,
                "bootstrap cell does not contain the shared five seed labels",
                estimand=estimand, w4=weight, observed=list(pivot.columns))
        matrix = pivot.to_numpy(dtype=float)  # rho x shared seed label
        # One row of `indices` is one seed-label vector.  The identical row is
        # applied to every rho stratum, preserving the crossed seed family.
        draws = np.empty(len(indices), dtype=float)
        for replicate, selected in enumerate(indices):
            rho_means = matrix[:, selected].mean(axis=1)
            draws[replicate] = float(rho_means.mean())
        lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0],
                             method="linear")
        rows.append({
            "stage": STAGE,
            "estimand": estimand,
            "w4": weight,
            "w4_lo": low,
            "w4_hi": high,
            "point_estimate": float(matrix.mean(axis=1).mean()),
            "ci_level": float(contract["bootstrap"]["ci_level"]),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_rho_strata": len(rhos),
            "n_shared_seed_clusters": len(seeds),
            "n_resamples": len(indices),
            "rng_seed": int(contract["bootstrap"]["rng_seed"]),
            "seed_index_matrix_sha256": index_hash,
            "rho_weighting": "equal_weight",
            "shared_seed_indices_across_rho": True,
            "interpretation": contract["bootstrap"]["interpretation"],
            "pooled_p_value": "NOT_COMPUTED",
            "unit": str(group["unit"].iloc[0]),
            "effect_direction": str(group["effect_direction"].iloc[0]),
            "snap_id": snap_id,
        })
    result = pd.DataFrame(rows)
    result["_w4_sort"] = pd.to_numeric(result["w4"], errors="coerce").fillna(-1)
    result = (result.sort_values(["estimand", "_w4_sort"], kind="mergesort")
              .drop(columns="_w4_sort").reset_index(drop=True))
    return result, index_hash


def _exact_two_sided_sign_flip(values: Sequence[float]) -> Tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    require(len(vector) > 0 and bool(np.isfinite(vector).all()),
            "exact sign-flip requires a nonempty finite cluster vector")
    observed = abs(float(vector.mean()))
    extreme = 0
    total = 2 ** len(vector)
    for signs in itertools.product((-1.0, 1.0), repeat=len(vector)):
        statistic = abs(float(np.mean(vector * np.asarray(signs))))
        if statistic >= observed - 1e-12:
            extreme += 1
    return float(extreme / total), float(2.0 / total)


def _holm_adjust(raw_p: Sequence[float]) -> np.ndarray:
    values = np.asarray(raw_p, dtype=float)
    require(bool(np.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()),
            "Holm adjustment received an invalid p-value")
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (total - rank))
        running = max(running, candidate)  # required cumulative maximum
        adjusted[index] = min(1.0, running)
    return adjusted


def build_optional_holm(scenario: pd.DataFrame, registration_path: Optional[Path],
                        contract: Mapping[str, Any], snap_id: str) \
        -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    if registration_path is None:
        return None, None
    registration = _load_json(registration_path, "R5 Holm pre-registration")
    required = {"stage", "family_id", "estimand", "w4", "rho_levels", "procedure",
                "registered_before_r5", "two_sided"}
    require(required <= set(registration),
            "Holm pre-registration is missing fields", missing=sorted(required-set(registration)))
    require(registration["stage"] == STAGE and registration["registered_before_r5"] is True,
            "Holm registration must explicitly predate R5")
    require(registration["procedure"] == "exact_cluster_sign_flip_then_Holm"
            and registration["two_sided"] is True,
            "Holm registration selected an unregistered procedure")
    expected_rhos = [_rho_value(value) for value in contract["design_grid"]["rho_levels"]]
    require([_rho_value(value) for value in registration["rho_levels"]] == expected_rhos,
            "Holm family must contain exactly the six registered rho levels in order")
    estimand = str(registration["estimand"])
    require(estimand in _estimand_specs(contract),
            "Holm registration names an unknown estimand", estimand=estimand)
    weight = registration["w4"]
    selected = scenario[scenario["estimand"] == estimand].copy()
    if estimand == "A2_cost_of_stability":
        require(weight is None, "cost-of-stability Holm family must use w4=null")
    else:
        weight = _integer_value(weight, "Holm w4")
        require(weight in [int(value) for value in contract["design_grid"]["w4_sweep"]],
                "Holm registration names an unregistered w4", w4=weight)
        selected = selected[pd.to_numeric(selected["w4"], errors="coerce") == weight]
    rows = []
    for rho in expected_rhos:
        group = selected[np.isclose(selected["rho"].astype(float), rho,
                                    rtol=0, atol=FLOAT_ATOL)].sort_values("seed")
        require(len(group) == 5 and group["seed"].nunique() == 5,
                "Holm cell lacks five scenario clusters", rho=rho, rows=len(group))
        values = group["estimate"].to_numpy(dtype=float)
        p_raw, p_floor = _exact_two_sided_sign_flip(values)
        sd = float(values.std(ddof=1))
        standardized: Any = "" if sd <= FLOAT_ATOL else float(values.mean() / sd)
        rows.append({
            "stage": STAGE,
            "family_id": registration["family_id"],
            "estimand": estimand,
            "rho": rho,
            "w4": "" if weight is None else weight,
            "n_clusters": len(values),
            "mean_effect": float(values.mean()),
            "median_effect": float(np.median(values)),
            "standardized_mean_effect_dz": standardized,
            "p_raw": p_raw,
            "minimum_attainable_two_sided_p": p_floor,
            "procedure": "two_sided_exact_cluster_sign_flip",
            "exchangeability_caveat": "five seed families derived from one base instance",
            "snap_id": snap_id,
        })
    result = pd.DataFrame(rows)
    result["p_holm"] = _holm_adjust(result["p_raw"].to_numpy(dtype=float))
    registration_meta = {
        "file": registration_path.name,
        "sha256": sha256_file(str(registration_path)),
        "family_id": registration["family_id"],
    }
    return result.sort_values("rho").reset_index(drop=True), registration_meta


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _relative_input_key(path: Path, manifest_path: Path, raw_path: Path,
                        registration_path: Optional[Path] = None) -> str:
    path = path.resolve()
    manifest_root = manifest_path.resolve().parent
    if path == raw_path.resolve():
        return "R1b/e4_raw.csv"
    if path == DESIGN_MANIFEST.resolve():
        return "experiments/journal_p0/r5_design_manifest.json"
    if path == manifest_path.resolve():
        return "experiments/journal_p0/rho_manifest.csv"
    if registration_path is not None and path == registration_path.resolve():
        return "R5-preregistration/" + registration_path.name
    try:
        return "experiments/journal_p0/" + path.relative_to(manifest_root).as_posix()
    except ValueError:
        _fail("R5 input path is outside the snapshot-relative manifest tree", path=str(path))


def _validate_output_frames(scenario: pd.DataFrame, by_rho: pd.DataFrame,
                            bootstrap: pd.DataFrame, contract: Mapping[str, Any],
                            snap_id: str) -> None:
    require(not scenario.empty and not by_rho.empty and not bootstrap.empty,
            "R5 produced an empty scientific artifact")
    require(not scenario.duplicated(["estimand", "rho", "seed", "w4", "w4_lo",
                                     "w4_hi"]).any(),
            "R5 scenario output has duplicate analysis keys")
    require(not by_rho.duplicated(["estimand", "rho", "w4", "w4_lo", "w4_hi"]).any(),
            "R5 per-rho output has duplicate analysis keys")
    require(not bootstrap.duplicated(["estimand", "w4", "w4_lo", "w4_hi"]).any(),
            "R5 bootstrap output has duplicate analysis keys")
    require(set(by_rho["per_rho_p_value"]) == {"NOT_COMPUTED"},
            "R5 default output contains a per-rho p-value")
    require(set(bootstrap["pooled_p_value"]) == {"NOT_COMPUTED"},
            "R5 default output contains a pooled cross-rho p-value")
    if snap_id:
        for label, frame in ((OUTPUT_SCENARIO, scenario), (OUTPUT_BY_RHO, by_rho),
                             (OUTPUT_BOOTSTRAP, bootstrap)):
            require(set(frame["snap_id"]) == {snap_id},
                    "R5 output SNAP mismatch", artifact=label)
    expected_scenario_rows = (
        2 * len(contract["design_grid"]["w4_sweep"])
        * len(contract["design_grid"]["rho_levels"])
        * len(contract["design_grid"]["seed_labels"])
        + len(contract["design_grid"]["rho_levels"])
        * len(contract["design_grid"]["seed_labels"]))
    require(len(scenario) == expected_scenario_rows,
            "R5 scenario-effect shape differs from the registered estimand grid",
            expected=expected_scenario_rows, actual=len(scenario))


def run(e4_input: Path, rho_manifest: Path, output_dir: Path, *, formal: bool = False,
        snap_id: Optional[str] = None, holm_registration: Optional[Path] = None) \
        -> Dict[str, Any]:
    assert_no_staging_import()
    assert_no_absolute_runtime_paths(__file__)
    contract = _load_contract()
    if formal:
        snap = require_snap_binding(str(DESIGN_MANIFEST), str(snap_id or ""), stage=STAGE)
    else:
        require(not snap_id,
                "a rehearsal may not attach or invent a SNAP identifier",
                stage=STAGE, snap_id=snap_id)
        snap = ""
    e4_input = e4_input.resolve()
    rho_manifest = rho_manifest.resolve()
    output_dir = output_dir.resolve()
    require(e4_input.name == "e4_raw.csv", "R5 primary input must be named e4_raw.csv",
            path=str(e4_input))
    require(rho_manifest.name == "rho_manifest.csv",
            "R5 scenario manifest must be named rho_manifest.csv",
            path=str(rho_manifest))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, axis, inputs, counts = validate_r1b(
        e4_input, rho_manifest, contract, formal=formal, snap_id=snap)
    scenario = build_scenario_effects(raw, axis, contract, snap)
    by_rho = build_by_rho(scenario, contract, snap)
    bootstrap, index_hash = build_bootstrap(scenario, contract, snap)
    holm, holm_meta = build_optional_holm(
        scenario, holm_registration.resolve() if holm_registration else None,
        contract, snap)
    _validate_output_frames(scenario, by_rho, bootstrap, contract, snap)
    if holm is None and (output_dir / OUTPUT_HOLM).exists():
        _fail("stale r5_holm.csv exists but this run had no registration",
              path=str(output_dir / OUTPUT_HOLM))

    if holm_registration:
        inputs.append(holm_registration.resolve())
    input_hashes = {
        _relative_input_key(path, rho_manifest, e4_input, holm_registration):
        sha256_file(str(path))
                    for path in sorted(set(inputs), key=lambda item: str(item))}
    require(len(input_hashes) == len(set(inputs)),
            "R5 input hash keys collide after snapshot-relative normalization")

    temp_dir = Path(tempfile.mkdtemp(prefix=".R5TMP.", dir=str(output_dir)))
    try:
        _write_csv(scenario, temp_dir / OUTPUT_SCENARIO)
        _write_csv(by_rho, temp_dir / OUTPUT_BY_RHO)
        _write_csv(bootstrap, temp_dir / OUTPUT_BOOTSTRAP)
        output_names = list(SCIENTIFIC_OUTPUTS)
        if holm is not None:
            _write_csv(holm, temp_dir / OUTPUT_HOLM)
            output_names.append(OUTPUT_HOLM)

        # Re-read the landed temporary bytes before they can be promoted.
        scenario_check = _read_csv_strict(temp_dir / OUTPUT_SCENARIO, OUTPUT_SCENARIO)
        by_rho_check = _read_csv_strict(temp_dir / OUTPUT_BY_RHO, OUTPUT_BY_RHO)
        bootstrap_check = _read_csv_strict(temp_dir / OUTPUT_BOOTSTRAP, OUTPUT_BOOTSTRAP)
        _validate_output_frames(scenario_check, by_rho_check, bootstrap_check,
                                contract, snap)
        scientific_hashes = {name: sha256_file(str(temp_dir / name))
                             for name in output_names}
        run_manifest: Dict[str, Any] = {
            "stage": STAGE,
            "owner_script": "experiments/journal_p0/r5_stats.py",
            "formal": formal,
            "snap_id": snap,
            "design_manifest_sha256": sha256_file(str(DESIGN_MANIFEST)),
            "input_hashes": dict(sorted(input_hashes.items())),
            "dynamic_coverage": counts,
            "dynamic_disruptions_by_scenario": [
                {"rho": rho, "seed": seed, "n_eligible_disruptions": len(axis[(rho, seed)])}
                for rho, seed in sorted(axis)
            ],
            "scenario_aggregation": contract["scenario_aggregation"],
            "bootstrap": {
                **contract["bootstrap"],
                "seed_index_matrix_sha256": index_hash,
            },
            "holm": {
                "enabled": holm is not None,
                "registration": holm_meta,
            },
            "scientific_output_hashes": dict(sorted(scientific_hashes.items())),
            "nonclaims": [
                "No disruption-level iid inference is performed.",
                "No pooled cross-rho p-value is computed.",
                "The pooled bootstrap interval is descriptive sensitivity, not population inference.",
                "Single-file os.replace promotions are not family-atomic."
            ],
        }
        _write_json(run_manifest, temp_dir / OUTPUT_RUN_MANIFEST)
        output_names.append(OUTPUT_RUN_MANIFEST)
        hashes_payload = {
            "stage": STAGE,
            "snap_id": snap,
            "inputs": dict(sorted(input_hashes.items())),
            "outputs": {name: sha256_file(str(temp_dir / name))
                        for name in sorted(output_names)},
            "excluded_from_coverage": [OUTPUT_HASHES, "R5.log"],
            "self_hash": "NOT_COMPUTED_TO_AVOID_SELF_REFERENCE",
        }
        _write_json(hashes_payload, temp_dir / OUTPUT_HASHES)
        output_names.append(OUTPUT_HASHES)

        for name in output_names:
            os.replace(temp_dir / name, output_dir / name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"[R5] PASS dynamic scenarios={counts['scenarios']} "
          f"disruptions={counts['dynamic_disruptions']} "
          f"raw_rows={counts['actual_raw_rows']}")
    print(f"[R5] PASS scenario aggregation precedes every across-scenario summary")
    print(f"[R5] PASS bootstrap n={contract['bootstrap']['n_resamples']} "
          f"seed={contract['bootstrap']['rng_seed']} shared-index-sha256={index_hash}")
    print("[R5] PASS no per-rho or pooled cross-rho p-value was computed by default")
    print("[R5] NOTE regression/traceability evidence is not proof or validation of the science")
    return {
        "scenario": scenario,
        "by_rho": by_rho,
        "bootstrap": bootstrap,
        "holm": holm,
        "counts": counts,
        "snap_id": snap,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4-input", type=Path, required=True,
                        help="validated R1b e4_raw.csv")
    parser.add_argument("--rho-manifest", type=Path, required=True,
                        help="rho_manifest.csv whose sol column names committed S files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal", action="store_true",
                        help="require the content-addressed SNAP id on CLI and every R1b row")
    parser.add_argument("--snap-id", default=None,
                        help="64-hex content-addressed SNAP archive SHA256; formal only")
    parser.add_argument("--holm-registration", type=Path, default=None,
                        help="optional pre-existing JSON that registers one six-rho Holm family")
    args = parser.parse_args(argv)
    run(args.e4_input, args.rho_manifest, args.output_dir,
        formal=args.formal, snap_id=args.snap_id,
        holm_registration=args.holm_registration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
