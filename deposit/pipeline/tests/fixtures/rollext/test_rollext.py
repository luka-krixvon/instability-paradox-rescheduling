"""
tests/fixtures/rollext/test_rollext.py — S2 regression fixtures for agent `rollext`.

Covers:
  * REQ-7 / REQ-8 on `experiments/journal_p0/rolling_horizon.py` (tiny synthetic stream only);
  * the HHCRSP parser defect and its shape/fail-fast guards in
    `experiments/journal_p0/external/hhcrsp_adapter.py` (tiny local fixture files only);
  * the cap-guard allowance and fail-fast guards in `external/instance_spec.py`;
  * the UNSET-parameter and fail-fast guards in `external/run_family.py`.

EPISTEMIC SCOPE (`rerun_plan_v2_1.md` E1/E2). Everything below is a REGRESSION TEST and a
NEGATIVE CONTROL record. Nothing here is a proof, a validation, a correctness argument, or
evidence about the absence of fabrication. No canonical artifact is read or written, no external
solve is run, and no formal R-stage is entered. The only solves performed are on the tiny
synthetic stream in `synthetic_rolling.py`.

    python3 tests/fixtures/rollext/test_rollext.py
Exit code 0 == every positive check ran without a violation AND every negative control failed the
way it was supposed to. Exit code 1 == at least one check did not behave as specified.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
import traceback
from fractions import Fraction

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
EXPERIMENTS = os.path.join(REPO, "experiments")
JOURNAL = os.path.join(EXPERIMENTS, "journal_p0")
EXTERNAL = os.path.join(JOURNAL, "external")
for _p in (EXPERIMENTS, JOURNAL, EXTERNAL, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rolling_horizon as RH          # noqa: E402
import instance_spec as IS            # noqa: E402
import hhcrsp_adapter as HA           # noqa: E402
import run_family as RF               # noqa: E402
import synthetic_rolling as SYN       # noqa: E402

RESULTS = []          # (id, kind, name, ok, detail)


def check(cid, name, fn, kind="positive"):
    try:
        detail = fn()
        RESULTS.append((cid, kind, name, True, detail if isinstance(detail, str) else ""))
        print(f"  PASS {cid:<6} {name}" + (f"  [{detail}]" if isinstance(detail, str) and detail else ""))
    except Exception as exc:                                  # noqa: BLE001
        RESULTS.append((cid, kind, name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL {cid:<6} {name}\n        {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)


def expect_raises(exc_types, fn, must_mention=()):
    """Return a describing string, or raise AssertionError if the guard did not fire."""
    try:
        fn()
    except exc_types as exc:
        msg = str(exc)
        for frag in must_mention:
            if frag not in msg:
                raise AssertionError(f"raised {type(exc).__name__} but message lacks {frag!r}: {msg}")
        return f"{type(exc).__name__} raised as required"
    except Exception as exc:                                   # noqa: BLE001
        raise AssertionError(f"raised {type(exc).__name__} ({exc}), expected {exc_types}")
    raise AssertionError(f"NO exception raised; expected {exc_types}")


# =============================================================================================== #
#  Build the synthetic stream once (this is the only thing in this file that solves MILPs)
# =============================================================================================== #
SEEDS = SYN.manifest_like()
RULE = "busiest_committed"
PARSER, COMMITTED, S_ALL = SYN.build()


def build_frames():
    raw, slots, sched, epochs = [], [], [], []
    for seed in SEEDS:
        r, sl, sc, ep, _seq = RH.frames_for_seed(PARSER, COMMITTED, S_ALL, seed, RULE)
        raw.extend(r); slots.extend(sl); sched.extend(sc); epochs.extend(ep)
    return (pd.DataFrame(raw), pd.DataFrame(slots), pd.DataFrame(sched), pd.DataFrame(epochs))


print("=== building the tiny synthetic stream (2 seeds x 4 configurations) ===")
RAW, SLOT, SCHED, EPOCH = build_frames()
SLOTS_BY_SEED = {s: sorted(COMMITTED) for s in SEEDS}
print(RAW[["seed", "regime", "w4", "absent_seq", "ot_full_plan", "ot_recoverable_horizon",
           "ot_frozen_prefix", "ot_cumulative", "D", "dev_terminal_tasks", "churn_total",
           "U"]].to_string(index=False))


# =============================================================================================== #
#  REQ-7 — the absence sequence
# =============================================================================================== #
print("\n=== REQ-7: exogenous absence sequence ===")


def t_r7_unset_rule():
    return expect_raises(RH.RollingDesignUnregistered,
                         lambda: RH.absence_sequence(COMMITTED, 1),
                         must_mention=("UNSET", "REQ-7"))


def t_r7_deterministic():
    seqs = {tuple(RH.absence_sequence(COMMITTED, 1, rule=RULE)) for _ in range(5)}
    assert len(seqs) == 1, f"5 identical calls produced {len(seqs)} different sequences"
    other = tuple(RH.absence_sequence(COMMITTED, 2, rule=RULE))
    return f"seed1={seqs.pop()} seed2={other}"


def t_r7_generator_signature():
    """REQ-7 acceptance test 2: the generator's parameters and body may not reference the policy."""
    src = inspect.getsource(RH.absence_sequence)
    for token in ("regime", "w4", "A1", "A2", "cur["):
        assert token not in src, f"absence_sequence source references {token!r}"
    params = list(inspect.signature(RH.absence_sequence).parameters)
    assert params == ["committed", "seed", "epochs", "rule"], params
    return f"params={params}, no policy token in source"


def t_r7_identical_across_configurations():
    per_seed = RAW.groupby("seed").absent_seq.nunique()
    assert int(per_seed.max()) == 1, f"absent_seq differs within a seed: {per_seed.to_dict()}"
    assert RAW.groupby("seed").ngroups == len(SEEDS)
    counts = RAW.groupby("seed").size().to_dict()
    assert all(v == RH.N_CONFIGURATIONS for v in counts.values()), counts
    return "; ".join(f"seed {s}: {g.absent_seq.iloc[0]}" for s, g in RAW.groupby("seed"))


def t_r7_four_configurations_preserved():
    got = {(r.regime, float(r.w4)) for r in RAW.itertuples()}
    assert got == set(RH.CONFIGURATIONS), got
    assert RH.N_CONFIGURATIONS == 4
    return f"configurations={sorted(got)}"


def t_r7_epoch_table():
    for seed, g in EPOCH.groupby("seed"):
        assert list(g.epoch) == RH.EPOCHS, (seed, list(g.epoch))
        assert g.absent_seq.nunique() == 1
    return f"{len(EPOCH)} (seed, epoch) rows persisted"


def t_r7_configurations_actually_differ():
    """Guards against a vacuous version of the identity test: if all four configurations produced
    byte-identical results the shared sequence would be trivially shared."""
    sig = RAW[["seed", "ot_recoverable_horizon", "D", "churn_total"]].drop_duplicates()
    assert len(sig) > len(SEEDS), (
        "the four configurations are indistinguishable on this fixture, so the cross-policy "
        "identity check is vacuous")
    return f"{len(sig)} distinct (seed, ot_recov, D, churn) tuples over {len(RAW)} rows"


check("R7-1", "absence rule is UNSET by default and raises", t_r7_unset_rule)
check("R7-2", "sequence is deterministic in (committed, seed, rule)", t_r7_deterministic)
check("R7-3", "generator sees neither policy nor weight (source + signature)", t_r7_generator_signature)
check("R7-4", "the four configurations of a seed share ONE absence sequence", t_r7_identical_across_configurations)
check("R7-5", "the four pre-existing configurations are preserved", t_r7_four_configurations_preserved)
check("R7-6", "per-(seed, epoch) absence table is persisted", t_r7_epoch_table)
check("R7-7", "the four configurations are not indistinguishable", t_r7_configurations_actually_differ)


# =============================================================================================== #
#  REQ-8 — overtime estimands and the three deviation quantities
# =============================================================================================== #
print("\n=== REQ-8: four overtime estimands, three deviation quantities ===")


def t_r8_identity():
    for r in RAW.itertuples():
        lhs = r.ot_full_plan - r.ot_recoverable_horizon
        assert lhs == r.ot_frozen_prefix, (
            f"seed={r.seed} {r.regime}/{r.w4}: {r.ot_full_plan} - {r.ot_recoverable_horizon} "
            f"= {lhs} != {r.ot_frozen_prefix}")
    return f"exact on all {len(RAW)} rows"


def t_r8_prefix_independent_and_invariant():
    independent = float(RH.overtime_total(PARSER, COMMITTED, q_max_excl=RH.E0))
    assert independent > 0, "the fixture has a zero frozen prefix; the check would be vacuous"
    for seed, g in RAW.groupby("seed"):
        assert g.ot_frozen_prefix.nunique() == 1, g.ot_frozen_prefix.tolist()
        assert float(g.ot_frozen_prefix.iloc[0]) == independent, (
            g.ot_frozen_prefix.iloc[0], independent)
    return f"ot_frozen_prefix={independent} recomputed from the committed plan, equal on all rows"


def t_r8_estimands_present_and_distinct():
    for k in RH.OT_ESTIMANDS:
        assert k in RAW.columns, k
    # Distinctness is DEFINITIONAL, so the test is that no pair is the same column — not that they
    # differ on every row. On this fixture A2 happens to drive the post-epoch-2 overtime to zero, so
    # ot_cumulative coincides with ot_recoverable_horizon on some A2 rows; that is a numeric
    # coincidence of the fixture and asserting otherwise would be asserting the fixture, not the code.
    pairs = (("ot_cumulative", "ot_full_plan"), ("ot_cumulative", "ot_recoverable_horizon"),
             ("ot_full_plan", "ot_recoverable_horizon"), ("ot_full_plan", "ot_frozen_prefix"))
    for a, b in pairs:
        assert (RAW[a] != RAW[b]).any(), f"{a} and {b} are indistinguishable on every row"
    # ot_cumulative depends on the intermediate plans, so its per-epoch terms must be persisted
    for r in RAW.itertuples():
        terms = [float(x) for x in str(r.ot_cumulative_terms).split("|") if x]
        assert len(terms) == len(RH.EPOCHS), (r.seed, r.regime, terms)
        assert abs(sum(terms) - r.ot_cumulative) <= max(abs(r.ot_cumulative), 1.0) * 1e-6
    doc = " ".join(RH.__doc__.split())
    for k, v in RH.OT_ESTIMANDS.items():
        head = v.split(",")[0][:48]
        assert head in doc, f"definition of {k} not verbatim in the docstring"
    coincide = int((RAW.ot_cumulative == RAW.ot_recoverable_horizon).sum())
    return (f"4 estimands, every pair distinguishable somewhere; per-epoch terms persisted and "
            f"re-sum; {coincide}/{len(RAW)} rows happen to have "
            f"ot_cumulative == ot_recoverable_horizon on this fixture")


def t_r8_no_privileged_estimand():
    """Grep gate: no 'true value' / 'correct overtime' style label anywhere in the module."""
    src = open(os.path.join(JOURNAL, "rolling_horizon.py"), encoding="utf-8").read().lower()
    for bad in ("the true value", "the true overtime", "the correct overtime",
                "the actual overtime", "the real overtime"):
        # the module is allowed to FORBID the phrase; count only assertive uses
        for line in src.splitlines():
            if bad in line and "not" not in line and "never" not in line and "none of" not in line:
                raise AssertionError(f"privileging phrase {bad!r} in: {line.strip()[:100]}")
    assert RH.W_ESTIMAND in RH.OT_ESTIMANDS, RH.W_ESTIMAND
    for r in RAW.itertuples():
        assert r.W == getattr(r, RH.W_ESTIMAND)
        assert isinstance(r.W_scope, str) and r.W_scope
    return f"W names its estimand ({RH.W_ESTIMAND}) and its scope"


def t_r8_three_deviations_distinct():
    for r in RAW.itertuples():
        assert r.D >= r.dev_terminal_tasks, (r.D, r.dev_terminal_tasks)
        assert r.churn_per_step == r.churn_total / r.n_epochs_executed
    moved = RAW[RAW.dev_terminal_tasks > 0]
    assert len(moved), "no configuration deviated at all; the distinctness check would be vacuous"
    strict = moved[moved.D > moved.dev_terminal_tasks]
    assert len(strict), "D never exceeds the terminal deviation, so they may be conflated"
    # churn_total must not equal either terminal quantity as a column
    assert (RAW.churn_total != RAW.D).any() and (RAW.churn_total != RAW.dev_terminal_tasks).any()
    return (f"{len(strict)}/{len(moved)} deviating rows have D > terminal deviation; "
            f"churn_per_step = {RH.CHURN_PER_STEP_DEFINITION}")


def t_r8_churn_is_not_a_symmetric_difference():
    """The per-solve deviation is measured against the RETAINED incumbent `xbar`, which excludes
    pairs whose caregiver became absent. So churn counts a forced re-placement once where a full
    symmetric difference counts it twice. This micro-check pins that difference down, because it is
    the reason `churn_total` may legitimately be SMALLER than `D` and must never be reconciled
    with it."""
    q = 5
    holder = COMMITTED[q][q * 3]                        # caregiver holding the first task of slot q
    before = {q: dict(COMMITTED[q])}
    assign, dev, _ot, _st = RH.solve_assign(PARSER, q, COMMITTED[q], {holder}, S_ALL, 0.0, "A2")
    after = {q: dict(assign)}
    moved = [t for t, who in assign.items() if COMMITTED[q].get(t) != who]
    assert moved, "the absence did not force any re-placement; the check would be vacuous"
    full_sd = RH.symmetric_difference_D(after, before)
    assert full_sd > dev, (
        f"per-solve deviation {dev} is not smaller than the full symmetric difference {full_sd}; "
        f"the two notions have been conflated")
    return (f"slot {q}, caregiver {holder} absent: per-solve deviation={dev}, full symmetric "
            f"difference={full_sd}, {len(moved)} task(s) re-placed")


def t_r8_per_slot_persisted():
    agg = SLOT.groupby(["seed", "regime", "w4"]).ot_slot.sum()
    for r in RAW.itertuples():
        got = float(agg.loc[(r.seed, r.regime, float(r.w4))])
        assert abs(got - r.ot_full_plan) <= max(abs(r.ot_full_plan), 1.0) * 1e-9, (got, r.ot_full_plan)
    assert set(SLOT.columns) >= {"slot", "ot_slot", "cell_loads", "in_recoverable_horizon"}
    # schedule evidence: enough to recompute the overtime from scratch
    assert set(SCHED.columns) >= {"slot", "task", "caregiver", "committed_caregiver", "flat_duration"}
    rebuilt = {}
    for row in SCHED.itertuples():
        if row.caregiver == "" or (isinstance(row.caregiver, float) and math.isnan(row.caregiver)):
            continue
        key = (row.seed, row.regime, float(row.w4))
        rebuilt.setdefault(key, {}).setdefault((int(row.caregiver), row.slot), 0.0)
        rebuilt[key][(int(row.caregiver), row.slot)] += row.flat_duration
    for r in RAW.itertuples():
        cells = rebuilt[(r.seed, r.regime, float(r.w4))]
        ot = sum(max(0.0, v - RH.H_STD) for v in cells.values())
        assert abs(ot - r.ot_full_plan) <= max(abs(r.ot_full_plan), 1.0) * 1e-9, (ot, r.ot_full_plan)
    return f"{len(SLOT)} per-slot rows and {len(SCHED)} schedule rows recompute ot_full_plan"


def t_r8_solver_status():
    for r in RAW.itertuples():
        got = {int(x) for x in str(r.solver_status_distinct).split("|") if x != ""}
        assert got <= set(RH.ADMISSIBLE_SOLVER_STATUS_R3), got
    return (f"all statuses within {sorted(RH.ADMISSIBLE_SOLVER_STATUS_R3)}; "
            f"max |milp - recomputed| overtime = "
            f"{RAW.milp_vs_recomputed_ot_max_abs_diff.max():.3e} (reported, NOT gated)")


check("R8-1", "ot_full_plan - ot_recoverable_horizon == ot_frozen_prefix, per row", t_r8_identity)
check("R8-2", "ot_frozen_prefix is independent and policy-invariant", t_r8_prefix_independent_and_invariant)
check("R8-3", "four estimands present, ot_cumulative is not one of the others", t_r8_estimands_present_and_distinct)
check("R8-4", "no estimand is privileged; W names estimand and scope", t_r8_no_privileged_estimand)
check("R8-5", "D, terminal deviation and step churn stay distinct", t_r8_three_deviations_distinct)
check("R8-5b", "per-solve churn is not a full symmetric difference", t_r8_churn_is_not_a_symmetric_difference)
check("R8-6", "per-slot overtime and schedule evidence recompute the total", t_r8_per_slot_persisted)
check("R8-7", "solver status admissible for this stage (REQ-13)", t_r8_solver_status)


# =============================================================================================== #
#  Validator + NEGATIVE CONTROLS (the validator must FAIL on mutilated output)
# =============================================================================================== #
print("\n=== validator and negative controls ===")


def _validate(raw=None, slot=None, sched=None, seeds=None, slots_by_seed=None, status=None):
    return RH.validate_rolling(
        RAW.copy() if raw is None else raw,
        SLOT.copy() if slot is None else slot,
        SCHED.copy() if sched is None else sched,
        SEEDS if seeds is None else seeds,
        SLOTS_BY_SEED if slots_by_seed is None else slots_by_seed,
        **({} if status is None else {"admissible_status": status}))


def t_validator_clean():
    fails = _validate()
    assert fails == [], fails
    return "0 violations on the clean frames"


check("V-1", "validator reports no violation on clean output", t_validator_clean)

NEGATIVE = []


def negative(nid, name, mutate, expect_tag):
    def run():
        raw, slot, sched = RAW.copy(), SLOT.copy(), SCHED.copy()
        kwargs = mutate(raw, slot, sched) or {}
        fails = _validate(raw=raw, slot=slot, sched=sched, **kwargs)
        assert fails, "the validator PASSED on mutilated input — it is not a validator"
        tags = [f.split("]")[0] + "]" for f in fails]
        assert expect_tag in tags, f"expected {expect_tag} among {tags}: {fails[:3]}"
        NEGATIVE.append(dict(id=nid, name=name, expected_tag=expect_tag,
                             observed_tags=sorted(set(tags)),
                             first_message=fails[0][:300], validator_failed=True))
        return f"{expect_tag} fired ({len(fails)} violation(s))"
    check(nid, name, run, kind="negative")


negative("N-1", "delete one row from rolling_raw",
         lambda raw, slot, sched: raw.drop(raw.index[0], inplace=True), "[shape]")


def _dup_key(raw, slot, sched):
    raw.loc[len(raw)] = raw.iloc[0]
    raw.drop(raw.index[1], inplace=True)          # keep the row count right so [keys] is the finding
    return {}


negative("N-2", "duplicate a primary key (seed, regime, w4)", _dup_key, "[keys]")


def _break_identity(raw, slot, sched):
    raw.loc[raw.index[0], "ot_frozen_prefix"] = raw.loc[raw.index[0], "ot_frozen_prefix"] + 1.0
    return {}


negative("N-3", "perturb ot_frozen_prefix on one row", _break_identity, "[identity]")


def _break_absence(raw, slot, sched):
    raw.loc[raw.index[0], "absent_seq"] = "9|9|9|9|9"
    return {}


negative("N-4", "corrupt absent_seq on one configuration", _break_absence, "[absence]")


def _nan(raw, slot, sched):
    raw.loc[raw.index[0], "ot_cumulative"] = float("nan")
    return {}


negative("N-5", "NaN in a numeric column", _nan, "[numeric]")


def _drop_slot(raw, slot, sched):
    idx = slot.index[(slot.seed == SEEDS[0]) & (slot.regime == "A1") & (slot.w4 == 0.0)][0]
    slot.drop(idx, inplace=True)
    return {}


negative("N-6", "drop one slot from the per-slot table", _drop_slot, "[slots]")


def _conflate(raw, slot, sched):
    raw.loc[raw.index[0], "D"] = -1
    return {}


negative("N-7", "set D below the terminal deviation (conflation)", _conflate, "[deviation]")


def _bad_status(raw, slot, sched):
    raw.loc[raw.index[0], "solver_status_distinct"] = "0|1"
    return {}


negative("N-8", "non-admissible solver status on one row", _bad_status, "[solver]")


def _drop_col(raw, slot, sched):
    raw.drop(columns=["churn_per_step"], inplace=True)
    return {}


negative("N-9", "drop a REQ-8-required column", _drop_col, "[columns]")


def _bad_coverage(raw, slot, sched):
    raw.loc[raw.index[0], "w4"] = 55.0
    return {}


negative("N-10", "replace a designed configuration with an undesigned one", _bad_coverage, "[coverage]")


def _touch_prefix(raw, slot, sched):
    raw.loc[raw.index[0], "ot_frozen_prefix_final_plan"] = \
        raw.loc[raw.index[0], "ot_frozen_prefix"] + 5.0
    return {}


negative("N-11", "final plan differs from committed inside the frozen prefix", _touch_prefix, "[identity]")


def _break_slot_sum(raw, slot, sched):
    slot.loc[slot.index[0], "ot_slot"] = slot.loc[slot.index[0], "ot_slot"] + 7.0
    return {}


negative("N-12", "per-slot overtime no longer sums to ot_full_plan", _break_slot_sum, "[slots]")


def _erase_caregiver_without_updating_unserved(raw, slot, sched):
    sched.loc[sched.index[0], "caregiver"] = float("nan")
    return {}


negative("N-12b", "nullable caregiver must close to persisted unserved counts",
         _erase_caregiver_without_updating_unserved, "[slots]")


# fail-fast negative controls that raise instead of being validated
def t_neg_solve_fatal():
    """Every caregiver absent -> empty candidate structure -> FATAL, not a silent skip."""
    detail = expect_raises(RH.RollingSolveFailure,
                           lambda: RH.solve_assign(PARSER, 3, COMMITTED[3], set(S_ALL), S_ALL,
                                                   0.0, "A2"),
                           must_mention=("EMPTY candidate structure", "REQ-2"))
    NEGATIVE.append(dict(id="N-13", name="empty candidate structure is FATAL",
                         expected_tag="RollingSolveFailure", observed_tags=["RollingSolveFailure"],
                         first_message=detail, validator_failed=True))
    return detail


check("N-13", "empty candidate structure raises instead of silently keeping the old plan",
      t_neg_solve_fatal, kind="negative")


def t_neg_absent_unknown():
    seq = [999] + [None] * (len(RH.EPOCHS) - 1)
    detail = expect_raises(RH.RollingFatal,
                           lambda: RH.run_stream(PARSER, COMMITTED, S_ALL, "A2", 0.0, seq,
                                                 seed=1, absence_rule=RULE),
                           must_mention=("not a known caregiver",))
    NEGATIVE.append(dict(id="N-14", name="unknown absentee is FATAL", expected_tag="RollingFatal",
                         observed_tags=["RollingFatal"], first_message=detail,
                         validator_failed=True))
    return detail


check("N-14", "an absentee not in the caregiver set is FATAL", t_neg_absent_unknown, kind="negative")


# =============================================================================================== #
#  HHCRSP parser — the decimal-splitting defect and its guards
# =============================================================================================== #
print("\n=== external parser (local fixture files only; NO external solve) ===")

FIX_OK = os.path.join(HERE, "tiny_hhcrsp_ok.txt")
FIX_WIDTH = os.path.join(HERE, "tiny_hhcrsp_bad_rowwidth.txt")
FIX_DUP = os.path.join(HERE, "tiny_hhcrsp_dup_key.txt")
FIX_WINDOW = os.path.join(HERE, "tiny_hhcrsp_bad_window.txt")
FIX_UNB = os.path.join(HERE, "tiny_hhcrsp_unbuildable.txt")
FIX_VNS = os.path.join(HERE, "tiny_vns_other_layout.txt")

#: The pre-S2 implementation, reproduced verbatim so the defect is demonstrated rather than asserted.
import re                                                    # noqa: E402


def _legacy_rows(block):
    return [[int(x) for x in re.findall(r"-?\d+", g)] for g in re.findall(r"\[([^\[\]]*)\]", block)]


def t_parser_defect_reproduced():
    txt = open(FIX_OK, encoding="utf-8").read()
    block = HA._block(txt, "p")
    legacy = _legacy_rows(block)
    S = HA._scalar(txt, "nbServi")
    widths = {len(r) for r in legacy}
    assert widths == {S, 2 * S}, widths
    decimal_rows = [r for r in legacy if len(r) == 2 * S]
    assert decimal_rows[0] == [14, 0] * S, decimal_rows[0]
    # what the legacy indexing then returned for p[node][staff][service]
    p_legacy = [legacy[i * 2:(i + 1) * 2] for i in range(5)]
    got = {"node1 s0": p_legacy[1][0][0], "node2 s1": p_legacy[2][1][1],
           "node3 s0": p_legacy[3][0][0], "node3 s1": p_legacy[3][1][1]}
    assert got == {"node1 s0": 14, "node2 s1": 5, "node3 s0": 31, "node3 s1": 0}, got
    return (f"legacy integer regex: 14.0 -> [14, 0]; row widths {sorted(widths)} for nbServi={S}; "
            f"p[node2][staff1][s1] read as 5 instead of 20.5; p[node3][staff1][s1] read as 0 "
            f"(-> the task was silently DROPPED)")


def t_parser_fixed():
    hh = HA.parse_hhcrsp(FIX_OK)
    assert hh["N"] == 5 and hh["K"] == 2 and hh["S"] == 2, hh
    assert all(len(r) == hh["S"] for r in hh["r"]), hh["r"]
    assert all(len(row) == hh["S"] for node in hh["p"] for row in node)
    assert hh["p"][1][0][0] == 14.0 and hh["p"][2][1][1] == 20.5 and hh["p"][3][1][1] == 31.0, hh["p"]
    spec = HA.to_spec(hh, num_slots=3, on_unbuildable_task="error")
    durs = sorted(t.duration for t in spec.tasks)
    assert durs == [14.0, 20.5, 31.0, 31.0], durs
    assert spec.meta["tasks_required"] == 4 and spec.meta["tasks_built"] == 4, spec.meta
    assert spec.meta["dropped_no_qualified_staff"] == 0
    assert spec.meta["dropped_no_positive_time"] == 0
    assert sorted(t.slot for t in spec.tasks) == [0, 1, 2, 2], [t.slot for t in spec.tasks]
    return f"4/4 required services built; durations {durs}; slots {sorted(t.slot for t in spec.tasks)}"


check("P-1", "the decimal-splitting defect is reproduced on the fixture", t_parser_defect_reproduced)
check("P-2", "the fixed parser recovers all 4 tasks and the exact durations", t_parser_fixed)

check("P-3", "row-WIDTH violation raises and names the key and row",
      lambda: expect_raises(HA.HHCRSPFormatError, lambda: HA.parse_hhcrsp(FIX_WIDTH),
                            must_mention=("block 'r'", "row")), kind="negative")
check("P-4", "duplicated key raises instead of silently taking the first",
      lambda: expect_raises(HA.HHCRSPFormatError, lambda: HA.parse_hhcrsp(FIX_DUP),
                            must_mention=("nbVehi", "appears")), kind="negative")
check("P-5", "inverted time window raises",
      lambda: expect_raises(HA.HHCRSPFormatError, lambda: HA.parse_hhcrsp(FIX_WINDOW),
                            must_mention=("time window inverted",)), kind="negative")
check("P-6", "the other released layout (InstanzVNS_*) is rejected with a diagnostic",
      lambda: expect_raises(HA.HHCRSPFormatError, lambda: HA.parse_hhcrsp(FIX_VNS),
                            must_mention=("nbNodes", "InstanzVNS")), kind="negative")
check("P-7", "num_slots is UNSET by default and raises (R0-D)",
      lambda: expect_raises(HA.ExternalDesignUnregistered,
                            lambda: HA.load(FIX_OK, on_unbuildable_task="error"),
                            must_mention=("num_slots is UNSET", "REQ-9")), kind="negative")
check("P-8", "on_unbuildable_task is UNSET by default and raises (R0-D)",
      lambda: expect_raises(HA.ExternalDesignUnregistered,
                            lambda: HA.load(FIX_OK, num_slots=3),
                            must_mention=("on_unbuildable_task is UNSET",)), kind="negative")
check("P-9", "policy 'error' aborts on a required service with no qualified staff",
      lambda: expect_raises(HA.HHCRSPFormatError,
                            lambda: HA.load(FIX_UNB, num_slots=3, on_unbuildable_task="error"),
                            must_mention=("cannot be built", "REQ-2")), kind="negative")


def t_parser_drop_and_record():
    spec = HA.load(FIX_UNB, num_slots=3, on_unbuildable_task="drop_and_record")
    m = spec.meta
    assert m["dropped_no_qualified_staff"] == 1, m
    assert m["tasks_required"] == m["tasks_built"] + m["dropped_no_qualified_staff"] \
        + m["dropped_no_positive_time"], m
    assert m["dropped_task_keys"], m
    return (f"required={m['tasks_required']} built={m['tasks_built']} "
            f"dropped_no_qualified_staff={m['dropped_no_qualified_staff']} "
            f"keys={m['dropped_task_keys']}")


check("P-10", "policy 'drop_and_record' preserves the drop as evidence", t_parser_drop_and_record)


# =============================================================================================== #
#  instance_spec — cap-guard allowance and fail-fast
# =============================================================================================== #
print("\n=== instance_spec: cap guard and fail-fast ===")


def t_cap_allowance():
    for cap in (0.45 * IS.H_MAX, 0.85 * IS.H_MAX, IS.H_MAX, 1.0):
        assert IS.cap_allowance(cap) == math.ulp(float(cap)) * 8
        assert IS.cap_allowance(cap) < 1e-6
    src = open(os.path.join(EXTERNAL, "instance_spec.py"), encoding="utf-8").read()
    offenders = [ln.strip() for ln in src.splitlines()
                 if "1e-6" in ln and not ln.strip().startswith("#")
                 and "cap_allowance" not in ln and "1e-6`" not in ln]
    assert not offenders, f"1e-6 still used as a cap allowance: {offenders}"
    return (f"cap_allowance(0.85*H_MAX) = {IS.cap_allowance(0.85 * IS.H_MAX):.3e} "
            f"(the pre-S2 value was 1e-6, ~2.2e9x larger)")


def t_cap_guard_rejects():
    """A cell exactly one ULP over the cap must be rejected; the old 1e-6 would have admitted it."""
    cap = 0.85 * IS.H_MAX
    add = cap - IS.TRAVEL_PER_TASK                    # a single task that exactly fills the cell
    spec = IS.InstanceSpec("cap-probe",
                           [IS.TaskSpec(0, add, 0, frozenset({0})),
                            IS.TaskSpec(1, 1e-7, 0, frozenset({0}))],
                           [IS.WorkerSpec(0, frozenset({0}))], 1).validate()
    sa, rep = IS.build_flat_feasible_incumbent(spec, cap=cap, quiet=True)
    assert rep["unplaced"] == 1, rep
    assert rep["cap_allowance"] == IS.cap_allowance(cap), rep
    old_would_place = (IS.TRAVEL_PER_TASK + 1e-7) + add <= cap + 1e-6
    assert old_would_place, "the probe does not distinguish the two allowances"
    return (f"a {1e-7 + IS.TRAVEL_PER_TASK:.7f}-minute overflow is now unplaced; "
            f"under the pre-S2 1e-6 allowance it would have been placed")


check("I-1", "cap allowance is math.ulp(cap)*8 and 1e-6 is gone", t_cap_allowance)
check("I-2", "a cell over the cap by less than 1e-6 is now rejected", t_cap_guard_rejects)
check("I-3", "InstanceSpec.validate raises InstanceSpecError, not bare assert",
      lambda: expect_raises(IS.InstanceSpecError,
                            lambda: IS.InstanceSpec(
                                "bad", [IS.TaskSpec(0, -1.0, 0, frozenset({0}))],
                                [IS.WorkerSpec(0, frozenset({0}))], 1).validate(),
                            must_mention=("nonpositive",)), kind="negative")


def t_strict_incumbent():
    spec = IS.InstanceSpec("tight", [IS.TaskSpec(0, 300.0, 0, frozenset({0})),
                                     IS.TaskSpec(1, 300.0, 0, frozenset({0}))],
                           [IS.WorkerSpec(0, frozenset({0}))], 1).validate()
    return expect_raises(IS.InstanceSpecError,
                         lambda: IS.build_flat_feasible_incumbent(spec, cap=0.85 * IS.H_MAX,
                                                                  quiet=True, strict=True),
                         must_mention=("not flat-feasible", "REQ-2", "REQ-9"))


check("I-4", "strict=True makes an unplaced task FATAL", t_strict_incumbent, kind="negative")


def t_cover_records_drops():
    spec = IS.InstanceSpec("cover", [IS.TaskSpec(0, 60.0, 0, frozenset({0, 1})),
                                     IS.TaskSpec(1, 60.0, 0, frozenset({2}))],
                           [IS.WorkerSpec(i, frozenset({0})) for i in range(3)], 1).validate()
    red = IS.greedy_worker_cover(spec, min_coverage=1)
    for k in ("cover_workers_rejected", "cover_tasks_dropped", "cover_tasks_dropped_ids",
              "cover_tasks_under_covered", "cover_extra_requested"):
        assert k in red.meta, k
    return (f"cover_workers={red.meta['cover_workers']} "
            f"rejected={red.meta['cover_workers_rejected']} "
            f"tasks_dropped={red.meta['cover_tasks_dropped']}")


check("I-5", "greedy_worker_cover records rejected workers and dropped tasks", t_cover_records_drops)


# =============================================================================================== #
#  run_family — UNSET parameters and fail-fast
# =============================================================================================== #
print("\n=== run_family: UNSET parameters and fail-fast ===")


def _dummy_spec():
    return IS.InstanceSpec("dummy", [IS.TaskSpec(0, 60.0, 0, frozenset({0, 1}))],
                           [IS.WorkerSpec(0, frozenset({0})), IS.WorkerSpec(1, frozenset({0}))],
                           1).validate()


for _cid, _kw, _frag in (("F-1", {}, "rho is UNSET"),
                         ("F-2", {"rho": 0.85}, "min_coverage is UNSET"),
                         ("F-3", {"rho": 0.85, "min_coverage": 2}, "max_absent is UNSET"),
                         ("F-4", {"rho": 0.85, "min_coverage": 2, "max_absent": 6},
                          "absence_rule is UNSET")):
    check(_cid, f"run_instance refuses to run: {_frag}",
          (lambda kw=_kw, frag=_frag: expect_raises(
              HA.ExternalDesignUnregistered,
              lambda: RF.run_instance(_dummy_spec(), **kw),
              must_mention=(frag, "REQ-9"))), kind="negative")


def t_rf_safe_is_sensitivity_only():
    src = open(os.path.join(EXTERNAL, "run_family.py"), encoding="utf-8").read()
    assert "safe_filter_sensitivity" in RF.ABSENCE_RULES
    # Scan run_instance's EXECUTABLE lines only: the docstring and the comments quote the removed
    # fallback deliberately, so a whole-file grep would be fooled by the documentation of the fix.
    body = "\n".join(ln for ln in inspect.getsource(RF.run_instance).splitlines()
                     if ln.strip() and not ln.strip().startswith("#"))
    assert "or ranked[:max_absent]" not in body, "the silent unfiltered fallback is still present"
    assert "or ranked[:max_absent]" in src, "the removed fallback is no longer documented"
    assert "SENSITIVITY analysis only" in src
    # S2.1 moved every scientific parameter into the hashed R0-D manifest.
    # The CLI may expose only the formal-stage envelope, and all four fields are
    # mandatory.  Re-introducing a per-parameter CLI switch would create an
    # unregistered shadow override.
    main_src = inspect.getsource(RF.main)
    for opt in ("--stage", "--manifest", "--snap-id", "--out-dir"):
        i = main_src.index(f'"{opt}"')
        seg = main_src[
            i:main_src.index("ap.add_argument", i + 1)
            if "ap.add_argument" in main_src[i + 1:]
            else i + 400
        ]
        assert "required=True" in seg, f"{opt} is not required"
        assert "default=" not in seg, f"{opt} still carries a default"
    for opt in ("--absence-rule", "--max-absent", "--min-coverage", "--slots", "--rho",
                "--on-unbuildable-task"):
        assert opt not in main_src, f"{opt} is an unregistered CLI shadow override"
    return (f"rules={RF.ABSENCE_RULES}; scientific parameters are manifest-only; "
            "formal envelope fields are required with no default")


def t_rf_stage_guard():
    import argparse
    try:
        RF.main.__globals__  # noqa: B018  (just to keep the reference explicit)
        old = sys.argv
        sys.argv = ["run_family.py", "--benchmark", "hhcrsp", "--glob", "nope"]
        try:
            RF.main()
        finally:
            sys.argv = old
    except SystemExit as exc:
        assert exc.code != 0, exc.code
        return f"bare invocation exits {exc.code} instead of writing an artifact"
    raise AssertionError("run_family.main() did not refuse a bare invocation")


check("F-5", "safe() is sensitivity-only and no silent fallback remains", t_rf_safe_is_sensitivity_only)
check("F-6", "run_family.main() refuses a bare invocation", t_rf_stage_guard, kind="negative")


def t_rh_stage_guard():
    old = sys.argv
    sys.argv = ["rolling_horizon.py"]
    try:
        RH.main()
    except SystemExit as exc:
        assert exc.code != 0, exc.code
        return f"bare invocation exits {exc.code} instead of overwriting rolling_raw.csv"
    finally:
        sys.argv = old
    raise AssertionError("rolling_horizon.main() did not refuse a bare invocation")


check("F-7", "rolling_horizon.main() refuses a bare invocation", t_rh_stage_guard, kind="negative")


print("\n=== temp -> validate -> promote (scratch directory only) ===")


def t_promote_success():
    import tempfile
    with tempfile.TemporaryDirectory() as dest, tempfile.TemporaryDirectory() as stage:
        prior = os.path.join(dest, RH.RAW_BASENAME)
        with open(prior, "w", encoding="utf-8") as fh:
            fh.write("PRIOR\n")
        staged = {}
        for base, frame in ((RH.RAW_BASENAME, RAW), (RH.SLOT_BASENAME, SLOT)):
            p = os.path.join(stage, base)
            frame.to_csv(p, index=False)
            staged[base] = p
        moved = RH.promote(dict(staged), dest)
        assert len(moved) == 2, moved
        landed = pd.read_csv(os.path.join(dest, RH.RAW_BASENAME))
        assert len(landed) == len(RAW)
        assert not [f for f in os.listdir(dest) if f.endswith(".s2rollback")], os.listdir(dest)
        return f"{len(moved)} files promoted, no rollback residue left behind"


def t_promote_rollback():
    """A failure part-way through the promotion must restore the files already moved. This is a
    per-file os.replace loop: rollback is demonstrated, ATOMICITY IS NOT and is not claimed."""
    import tempfile
    with tempfile.TemporaryDirectory() as dest, tempfile.TemporaryDirectory() as stage:
        first = os.path.join(dest, RH.RAW_BASENAME)
        with open(first, "w", encoding="utf-8") as fh:
            fh.write("PRIOR-RAW\n")
        os.mkdir(os.path.join(dest, RH.SLOT_BASENAME))      # a directory: os.replace must fail here
        staged = {}
        for base in (RH.RAW_BASENAME, RH.SLOT_BASENAME):
            p = os.path.join(stage, base)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("NEW\n")
            staged[base] = p
        try:
            RH.promote(staged, dest)
        except OSError:
            restored = open(first, encoding="utf-8").read()
            assert restored == "PRIOR-RAW\n", f"rollback left {restored!r}"
            assert not [f for f in os.listdir(dest) if f.endswith(".s2rollback")]
            return "the already-moved file was restored to its prior bytes"
        raise AssertionError("promote() did not raise on an impossible destination")


def t_failed_validation_writes_nothing():
    """A CAUGHT validation failure must leave no new canonical file. Verified on a scratch dir."""
    import tempfile
    with tempfile.TemporaryDirectory() as dest, tempfile.TemporaryDirectory() as stage:
        broken = RAW.copy()
        broken.loc[broken.index[0], "ot_frozen_prefix"] += 1.0
        for base, frame in ((RH.RAW_BASENAME, broken), (RH.SLOT_BASENAME, SLOT)):
            frame.to_csv(os.path.join(stage, base), index=False)
        fails = RH.validate_rolling(pd.read_csv(os.path.join(stage, RH.RAW_BASENAME)),
                                    pd.read_csv(os.path.join(stage, RH.SLOT_BASENAME)),
                                    SCHED.copy(), SEEDS, SLOTS_BY_SEED)
        assert fails, "the staged bytes validated clean despite being mutilated"
        assert os.listdir(dest) == [], os.listdir(dest)
        return f"{len(fails)} violation(s) on the STAGED BYTES; destination still empty"


check("W-1", "promote() moves validated files and leaves no residue", t_promote_success)
check("W-2", "promote() rolls back files already moved when a later move fails",
      t_promote_rollback, kind="negative")
check("W-3", "a failed validation of the staged bytes promotes nothing",
      t_failed_validation_writes_nothing, kind="negative")


def t_shared_core_bridge():
    """The shared S2 helper module is owned by another agent. These files must run with or without
    it, and must not silently disagree with it where both define the same quantity."""
    present = RH.s2_core is not None and IS._s2_core is not None
    assert RH.sha256_file(FIX_OK) == RH._local_sha256_file(FIX_OK)
    cap = 0.85 * IS.H_MAX
    assert IS.cap_allowance(cap) == math.ulp(float(cap)) * 8
    if present:
        shared = getattr(IS._s2_core, "cap_fp_slack", None)
        assert callable(shared) and shared(cap) == IS.cap_allowance(cap), (
            "s2_core.cap_fp_slack and instance_spec.cap_allowance disagree")
    return (f"s2_core {'present' if present else 'ABSENT'}; sha256 and cap allowance agree "
            f"between the shared helper and the local fallback")


check("S-1", "shared s2_core bridge agrees with the local fallbacks", t_shared_core_bridge)


def t_no_absolute_paths():
    for rel in ("journal_p0/rolling_horizon.py", "journal_p0/external/hhcrsp_adapter.py",
                "journal_p0/external/instance_spec.py", "journal_p0/external/run_family.py"):
        src = open(os.path.join(EXPERIMENTS, rel), encoding="utf-8").read()
        assert "/Users/" not in src, f"{rel} contains an absolute runtime path"
        assert "gate_g2_staging" not in src, f"{rel} references gate_g2_staging"
        assert "__file__" in src, f"{rel} does not resolve its paths from __file__"
    return "4 files: no /Users/ path, no gate_g2_staging import, paths from __file__"


check("F-8", "owned runtime files carry no absolute path and no staging import", t_no_absolute_paths)


# =============================================================================================== #
#  Summary + negative-control record
# =============================================================================================== #
n_fail = sum(1 for _c, _k, _n, ok, _d in RESULTS if not ok)
n_pos = sum(1 for _c, k, _n, _ok, _d in RESULTS if k == "positive")
n_neg = sum(1 for _c, k, _n, _ok, _d in RESULTS if k == "negative")
print(f"\n=== {len(RESULTS)} checks ({n_pos} positive, {n_neg} negative controls), "
      f"{n_fail} did not behave as specified ===")
print("These are REGRESSION TESTS. They are not proof, not validation, not correctness, and not "
      "evidence about fabrication (rerun_plan_v2_1.md E1/E2).")

record = dict(
    slug="rollext",
    round="S2 (code + design fixes only)",
    generated_by="tests/fixtures/rollext/test_rollext.py",
    epistemic_status=("Negative controls and regression tests only. NOT a proof, NOT a validation, "
                      "NOT a correctness argument, and NOT evidence about the absence of "
                      "fabrication. No canonical artifact was read or written; no external solve "
                      "was run; no formal R-stage was entered."),
    fixture=dict(kind="tiny synthetic + tiny local format fixtures",
                 rolling_seeds=SEEDS, rolling_absence_rule=RULE,
                 rolling_slots=len(COMMITTED), rolling_tasks=len(PARSER.tasks),
                 rolling_caregivers=len(S_ALL), configurations=list(RH.CONFIGURATIONS),
                 hhcrsp_fixtures=[os.path.basename(p) for p in
                                  (FIX_OK, FIX_WIDTH, FIX_DUP, FIX_WINDOW, FIX_UNB, FIX_VNS)]),
    checks=[dict(id=c, kind=k, name=n, behaved_as_specified=ok, detail=d)
            for c, k, n, ok, d in RESULTS],
    negative_controls=NEGATIVE,
    counts=dict(total=len(RESULTS), positive=n_pos, negative=n_neg, unexpected=n_fail),
    all_as_specified=(n_fail == 0),
)
out = os.path.join(REPO, "audit_outputs", "s2_negative", "rollext.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=2, sort_keys=False)
    fh.write("\n")
print(f"negative-control record -> {os.path.relpath(out, REPO)}")

sys.exit(1 if n_fail else 0)
