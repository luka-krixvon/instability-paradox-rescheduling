"""
negative_controls.py — S2 corelib negative-control driver.

Every control mutilates a COPY of a tiny synthetic fixture (or constructs a tiny synthetic input)
and asserts that the corelib **raises / exits non-zero**. A validator that has never been shown to
fail is not a validator (rerun_plan_v2_1 §4 ``[negative]``).

Honest scope: a passing negative control is *regression evidence about this code path*. It is not
proof, not validation, not correctness, and not evidence of an absence of fabrication.

Nothing here writes to any canonical artifact. All temp files live beside this file.

    cd <repo>/tests/fixtures/corelib && python3 negative_controls.py
    exit 0  == every control failed as required
    exit 1  == at least one control did NOT fail (that is the alarming outcome)
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # tests/fixtures/corelib -> repo
JP0 = os.path.join(REPO, "experiments", "journal_p0")
for _p in (JP0, os.path.join(REPO, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s2_core                                                    # noqa: E402
from s2_core import COL                                           # noqa: E402

sys.path.insert(0, JP0)
import compare_baselines as cb                                    # noqa: E402

RUNS = os.path.join(HERE, "fixture_runs.csv")
INSTANCES = os.path.join(HERE, "fixture_instances.csv")
MANIFEST = os.path.join(HERE, "selftest_manifest.json")
STAGE = "S2-selftest-corelib"
TMP = os.path.join(HERE, "_nc_tmp")

RESULTS = []


def record(control, what, outcome, detail):
    RESULTS.append({"control": control, "what_is_broken": what, "outcome": outcome,
                    "detail": detail})
    flag = "OK(failed as required)" if outcome == "raised" else "!!! DID NOT FAIL !!!"
    print("{:<7} {:<58} {}".format(control, what, flag))
    if outcome == "raised":
        print("        {}".format(detail.splitlines()[0][:200]))


def expect_raise(control, what, fn):
    try:
        fn()
    except BaseException as exc:                     # noqa: BLE001 - the point is to catch it
        record(control, what, "raised", "{}: {}".format(type(exc).__name__, exc))
        return True
    record(control, what, "NO_RAISE", "the call returned normally, which is the defect")
    return False


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def read_runs():
    with open(RUNS, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def runs_spec(path, *, snap_id=None, stage=STAGE):
    policy = s2_core.StageSolverPolicy.from_manifest_file(MANIFEST, stage=STAGE)
    grid = s2_core.DesignGrid(
        name="S2-selftest-runs",
        axes={COL.STAGE: [STAGE], "T_q": [6, 8], "seed": [1, 2],
              COL.ALGORITHM: ["EXACT", "VNS", "ALNS", "cold"], COL.REPEAT_INDEX: [1, 2]},
        declared_shortfalls=[(STAGE, t, s, "cold", 2) for t in (6, 8) for s in (1, 2)],
        provenance="negative-control design manifest")
    return s2_core.StageOutputSpec(
        stage=stage, artifact_name=os.path.basename(path),
        primary_key=list(s2_core.RUN_LEVEL_KEY), grid=grid,
        required_columns=[COL.STAGE, "T_q", "seed", COL.ALGORITHM, COL.REPEAT_INDEX],
        numeric_columns=[COL.ITERS_BUDGET, COL.ITERS_EXECUTED, COL.EVALUATIONS, COL.WALL_TIME],
        solver_policy=policy, solver_row_filter=(COL.ALGORITHM, ["EXACT"]), run_level=True,
        stochastic_algorithms=["VNS", "ALNS"], input_paths=[MANIFEST], snap_id=snap_id,
        nullable_columns=[COL.GAP_SIGNED])


def mutate(name, fn):
    """Copy the good fixture, apply ``fn(header, rows)``, return the mutilated path."""
    header, rows = read_runs()
    header, rows = fn(header, rows)
    path = os.path.join(TMP, name)
    write_csv(path, header, rows)
    return path


def col(header, name):
    return s2_core.field_index_by_name(header, name, where="negative control")


# --------------------------------------------------------------------------------------------
def main():
    os.makedirs(TMP, exist_ok=True)
    for stale in os.listdir(TMP):
        os.unlink(os.path.join(TMP, stale))
    for required in (RUNS, INSTANCES, MANIFEST):
        if not os.path.isfile(required):
            print("MISSING fixture {}; run `python3 journal_p0/scalability.py --selftest` "
                  "first".format(required))
            return 2

    print("=== S2 corelib negative controls (each MUST fail) ===\n")

    # ---- baseline: the unmutilated fixture must PASS, else the controls prove nothing ----
    try:
        s2_core.validate_stage_output(RUNS, runs_spec(RUNS))
        print("BASE    unmutilated fixture validates                            OK(passes)\n")
    except BaseException as exc:                                          # noqa: BLE001
        print("BASE    unmutilated fixture FAILED validation -> controls are meaningless")
        traceback.print_exc()
        return 2

    # ---- NC-1 deleted row -------------------------------------------------------------
    p = mutate("nc1_deleted_row.csv", lambda h, r: (h, r[:-1]))
    expect_raise("NC-1", "one row deleted from the output",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-2 duplicate key -----------------------------------------------------------
    def dup(h, r):
        return h, r + [list(r[0])]
    p = mutate("nc2_duplicate_key.csv", dup)
    expect_raise("NC-2", "a primary key duplicated",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-3 NaN in a numeric column -------------------------------------------------
    def nan(h, r):
        r[0][col(h, COL.WALL_TIME)] = "nan"
        return h, r
    p = mutate("nc3_nan.csv", nan)
    expect_raise("NC-3", "NaN in a numeric column",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-4 solver status outside the declared admissible set ------------------------
    def bad_status(h, r):
        for row in r:
            if row[col(h, COL.ALGORITHM)] == "EXACT":
                row[col(h, COL.STATUS_CODE)] = "99"
                break
        return h, r
    p = mutate("nc4_bad_status.csv", bad_status)
    expect_raise("NC-4", "solver status outside the stage's admissible set",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-5 hash mismatch: the recorded ledger hash no longer matches the bytes -----
    shutil.copyfile(RUNS, os.path.join(TMP, "nc5_hash.csv"))
    p = os.path.join(TMP, "nc5_hash.csv")
    report = s2_core.validate_stage_output(p, runs_spec(p))
    recorded = list(report.hashes["outputs"].values())[0]
    with open(p, "ab") as handle:
        handle.write(b"# tampered\n")

    def hash_check():
        landed = s2_core.sha256_file(p)
        s2_core.require(landed == recorded,
                        "recorded SHA256 does not match the bytes on disk; the artifact was "
                        "modified after validation", recorded=recorded, landed=landed)
    expect_raise("NC-5", "artifact bytes changed after the hash was recorded", hash_check)

    # ---- NC-6 stop_reason outside the declared enumeration ----------------------------
    def bad_stop(h, r):
        r[0][col(h, COL.STOP_REASON)] = "converged"
        return h, r
    p = mutate("nc6_stop_reason.csv", bad_stop)
    expect_raise("NC-6", "stop_reason outside the declared enumeration",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-7 gap_full_objective inconsistent with the persisted P, B, C ---------------
    def bad_gap(h, r):
        for row in r:
            if row[col(h, COL.ALGORITHM)] == "EXACT":
                row[col(h, COL.GAP_FULL)] = "0.5"
                break
        return h, r
    p = mutate("nc7_gap_recompute.csv", bad_gap)
    expect_raise("NC-7", "persisted gap disagrees with the gap recomputed from P, B, C",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-8 a time-limit incumbent labelled proven_optimal ---------------------------
    def bad_label(h, r):
        for row in r:
            if row[col(h, COL.ALGORITHM)] == "EXACT":
                row[col(h, COL.PROVEN_OPTIMAL)] = "False"
                break
        return h, r
    p = mutate("nc8_optimality_label.csv", bad_label)
    expect_raise("NC-8", "proven_optimal contradicts the status the manifest declares optimal",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-9 iterations_executed > iterations_budget ---------------------------------
    def over_budget(h, r):
        for row in r:
            if row[col(h, COL.ALGORITHM)] == "VNS":
                row[col(h, COL.ITERS_EXECUTED)] = "999999"
                break
        return h, r
    p = mutate("nc9_iters.csv", over_budget)
    expect_raise("NC-9", "iterations_executed exceeds the declared budget",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-10 negative gap against a proven-optimal reference ------------------------
    def negative_gap(h, r):
        for row in r:
            if row[col(h, COL.ALGORITHM)] == "ALNS":
                row[col(h, COL.GAP_SIGNED)] = "-3.75"
                break
        return h, r
    p = mutate("nc10_negative_gap.csv", negative_gap)
    expect_raise("NC-10", "negative gap against a proven-optimal exact reference",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))
    expect_raise("NC-10b", "require_gap_sign_admissible on a negative gap",
                 lambda: s2_core.require_gap_sign_admissible(
                     -3.75, reference_proven_optimal=True, where="NC-10b"))

    # ---- NC-11 an R-stage artifact without a SNAP identifier --------------------------
    shutil.copyfile(RUNS, os.path.join(TMP, "nc11_no_snap.csv"))
    p = os.path.join(TMP, "nc11_no_snap.csv")
    expect_raise("NC-11", "stage named R* emits with no SNAP identifier",
                 lambda: s2_core.validate_stage_output(
                     p, runs_spec(p, snap_id=None, stage="R4-size")))

    # ---- NC-12 two repeats of one algorithm sharing a seed ----------------------------
    def shared_seed(h, r):
        seeds = [row for row in r if row[col(h, COL.ALGORITHM)] == "VNS"]
        seeds[1][col(h, COL.RNG_SEED)] = seeds[0][col(h, COL.RNG_SEED)]
        return h, r
    p = mutate("nc12_shared_seed.csv", shared_seed)
    expect_raise("NC-12", "two repeats of one algorithm share an RNG seed",
                 lambda: s2_core.validate_stage_output(p, runs_spec(p)))

    # ---- NC-13 one RNG object handed to two stochastic procedures ---------------------
    shared = random.Random(1234)
    prov_a = s2_core.RngProvenance("VNS", 1, 1, 1234, shared)
    prov_b = s2_core.RngProvenance("ALNS", 1, 1, 1234, shared)
    expect_raise("NC-13", "one RNG object shared by VNS and ALNS",
                 lambda: s2_core.require_distinct_rngs(prov_a, prov_b))
    expect_raise("NC-13b", "a bare random.Random passed to vns() instead of an RngProvenance",
                 lambda: cb.vns([0], {0: [0]}, {0: 60.0}, {}, 50.0, random.Random(1)))

    # ---- NC-14 a design grid derived from the artifact it validates -------------------
    expect_raise("NC-14", "design grid derived from the output (tautological coverage check)",
                 lambda: s2_core.DesignGrid.from_dataframe(RUNS))

    # ---- NC-15 canonical_write: a failing validator must leave the canonical file alone
    canonical = os.path.join(TMP, "nc15_canonical.csv")
    with open(canonical, "w", encoding="utf-8") as handle:
        handle.write("good,bytes\n1,2\n")
    before = s2_core.sha256_file(canonical)

    def failing_write():
        def always_fail(_tmp):
            raise s2_core.ValidationFailure("injected validator failure", ["injected"])
        with s2_core.canonical_write(canonical, always_fail, stage="NC-15") as tmp:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write("partial,garbage\n9,9\n")
    raised = expect_raise("NC-15", "failing validator must not promote a partial file",
                          failing_write)
    after = s2_core.sha256_file(canonical)
    diag = [f for f in os.listdir(TMP) if f.startswith("nc15_canonical.csv.FAILED_")]
    leftovers = [f for f in os.listdir(TMP) if ".S2TMP." in f]
    unchanged = before == after
    record("NC-15b", "canonical bytes unchanged + diagnostics written + no temp left",
           "raised" if (raised and unchanged and diag and not leftovers) else "NO_RAISE",
           "canonical_unchanged={} diagnostics={} temp_leftovers={}".format(
               unchanged, diag, leftovers))

    # ---- NC-16 the cap allowance: 1e-6 would pass, math.ulp(cap)*8 must not -----------
    over = 360.0 + 1e-7
    expect_raise("NC-16", "load exceeding the cap by 1e-7 (a 1e-6 guard would pass it)",
                 lambda: s2_core.require_flat_feasible({"w0": over}, 360.0, where="NC-16"))
    record("NC-16b", "the same overload against the old 1e-6 allowance",
           "raised" if (over <= 360.0 + 1e-6) else "NO_RAISE",
           "over-cap by 1e-7 is INSIDE a 1e-6 allowance (=> 1e-6 hides it) and outside "
           "math.ulp(360.0)*8 = {:.6g}".format(s2_core.cap_fp_slack(360.0)))

    # ---- NC-17 the base instance path must have no fallback --------------------------
    def missing_base():
        saved = s2_core._BASE_INSTANCE_RELATIVE
        try:
            s2_core._BASE_INSTANCE_RELATIVE = os.path.join("data_problems_solutions",
                                                           "DOES-NOT-EXIST.csv")
            s2_core.paths()
        finally:
            s2_core._BASE_INSTANCE_RELATIVE = saved
    expect_raise("NC-17", "base instance CSV renamed away (no silent fallback)", missing_base)

    # ---- NC-18 an unplaced task where the design forbids one -------------------------
    expect_raise("NC-18", "unplaced task where the design assumed full service",
                 lambda: s2_core.require_no_unplaced({0: 1, 1: None}, [0, 1], where="NC-18"))
    expect_raise("NC-18b", "empty candidate set where the design assumed non-empty",
                 lambda: s2_core.require_nonempty_candidates({0: [1], 1: []}, where="NC-18b"))

    # ---- NC-19 a manifest whose optimal set is not a subset of the admissible set -----
    expect_raise("NC-19", "manifest declares an optimal status that is not admissible",
                 lambda: s2_core.StageSolverPolicy.from_manifest(
                     {"stage": "X", "solver": {"admissible_statuses": [{"code": 0,
                                                                        "name": "Optimal"}],
                                               "optimal_statuses": [{"code": 1,
                                                                     "name": "TimeLimit"}],
                                               "accepts_time_limit_incumbent": True,
                                               "time_limit_s": 10.0}},
                     stage="X"))

    # ---- NC-20 a non-proven row described as optimal ---------------------------------
    expect_raise("NC-20", "a time-limit incumbent column named 'obj_optimal'",
                 lambda: s2_core.assert_not_labeled_optimal(
                     ["obj_optimal"], proven_optimal=False, where="NC-20"))

    # ---- summary ---------------------------------------------------------------------
    failed_to_fail = [r for r in RESULTS if r["outcome"] != "raised"]
    out = {
        "what_this_is": ("S2 corelib negative-control record. Each entry asserts the corelib "
                         "RAISES on a specific mutilation. A passing control is regression "
                         "evidence about this code path only: it is not proof, not validation, "
                         "not correctness, and not evidence of an absence of fabrication."),
        "controls_total": len(RESULTS),
        "controls_that_failed_as_required": len(RESULTS) - len(failed_to_fail),
        "controls_that_did_not_fail": [r["control"] for r in failed_to_fail],
        "results": RESULTS,
    }
    with open(os.path.join(HERE, "negative_controls_result.json"), "w",
              encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    print("\n{}/{} controls failed as required".format(
        out["controls_that_failed_as_required"], out["controls_total"]))
    return 1 if failed_to_fail else 0


if __name__ == "__main__":
    sys.exit(main())
