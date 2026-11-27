"""
run_epse4_fixtures.py — the S2 fixture + negative-control driver for the two files
agent `epse4` owns: `experiments/journal_p0/eps_constraint.py` (stage R1) and
`experiments/journal_p0/run_e4.py` (stage R1b).

Everything here runs on TINY SYNTHETIC fixtures under this directory.  No canonical
artifact is read, written, or compared against; `eps_range.csv`, `eps_frontier.csv`,
`e4_raw.csv` and `e4_summary.csv` in `experiments/journal_p0/` are untouched.

EPISTEMIC STATUS, stated once and meant literally: a fixture pass and a hash
agreement here are REGRESSION AND TRACEABILITY EVIDENCE ONLY.  They are not proof,
not validation, not correctness, and not evidence of an absence of fabrication.
The formal runs are R1 / R1b and neither has been executed.

    python tests/fixtures/epse4/run_epse4_fixtures.py
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
_EXPERIMENTS = os.path.join(_REPO, "experiments")
_JP0 = os.path.join(_EXPERIMENTS, "journal_p0")
for _p in (_THIS, _JP0, _EXPERIMENTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eps_constraint as EPS                     # noqa: E402
import eps_oracle_independent as FORMAL_ORACLE   # noqa: E402
import run_e4 as E4                              # noqa: E402
from s2_core import (PipelineFailure, assert_no_absolute_runtime_paths,  # noqa: E402
                     assert_no_staging_import, cap_fp_slack, sha256_file,
                     validate_stage_output)
from SubgroupOptimizer import H_MAX, H_STD, HHCPFullParser   # noqa: E402
from displacement_milp import load_static                    # noqa: E402
import oracle_independent as ORACLE              # noqa: E402

OUT = os.path.join(_THIS, "out")
EVIDENCE = os.path.join(_THIS, "evidence")
RESULTS = {"checks": [], "negative_controls": []}


# --------------------------------------------------------------------------- #
def _rec(kind, name, status, detail, **extra):
    row = {"name": name, "status": status, "detail": detail}
    row.update(extra)
    RESULTS[kind].append(row)
    mark = {"PASS": "PASS", "FAIL": "FAIL", "INFO": "INFO"}[status]
    print(f"  [{mark}] {name}: {detail}")
    return row


def check(name, cond, detail, **extra):
    return _rec("checks", name, "PASS" if cond else "FAIL", detail, **extra)


def info(name, detail, **extra):
    return _rec("checks", name, "INFO", detail, **extra)


def expect_fatal(name, fn, must_contain=(), **extra):
    """A negative control: `fn` MUST abort.  If it returns, the control failed."""
    try:
        fn()
    except PipelineFailure as exc:
        msg = str(exc)
        missing = [t for t in must_contain if t not in msg]
        status = "PASS" if not missing else "FAIL"
        return _rec("negative_controls", name, status,
                    f"aborted as required: {type(exc).__name__}",
                    error_type=type(exc).__name__,
                    message=msg[:900], missing_tokens=missing, **extra)
    except Exception as exc:                      # wrong failure mode is a failure
        return _rec("negative_controls", name, "FAIL",
                    f"aborted with an UNEXPECTED exception type: "
                    f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                    traceback=traceback.format_exc()[-1200:], **extra)
    return _rec("negative_controls", name, "FAIL",
                "DID NOT ABORT — the guard is not a guard", **extra)


def _run_quiet(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


def _save(name, text):
    os.makedirs(EVIDENCE, exist_ok=True)
    path = os.path.join(EVIDENCE, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# =========================================================================== #
# part 1 — the epsilon stage (REQ-4 / REQ-5)
# =========================================================================== #
def part1_eps():
    print("\n=== part 1: epsilon stage on fixture A (REQ-4 / REQ-5) ===")
    man = os.path.join(_THIS, "manifest_epse4_A.csv")
    out_a = os.path.join(OUT, "eps_A")
    shutil.rmtree(out_a, ignore_errors=True)
    (rng, front, design, hashes, reports, _), log = _run_quiet(
        lambda: EPS.run(man, out_a, verbose=True))
    _save("eps_A_stdout.txt", log)
    check("eps.A.ran", len(front) == 12,
          f"frontier rows={len(front)} range rows={len(rng)}")

    # ---- REQ-4 test 4 (partial): byte determinism over two runs -------------
    out_a2 = os.path.join(OUT, "eps_A_rerun")
    shutil.rmtree(out_a2, ignore_errors=True)
    _run_quiet(lambda: EPS.run(man, out_a2, verbose=False))
    names = ("eps_range.csv", "eps_frontier.csv")
    whole = {n: (sha256_file(os.path.join(out_a, n))
                 == sha256_file(os.path.join(out_a2, n))) for n in names}
    # `wall_time_s` is a measured duration and CANNOT be byte-stable. It is kept in
    # the artifact because it is honest provenance; dropping it to manufacture a
    # byte-identical file would be worse. The determinism check is therefore run on
    # the artifact with that ONE column projected out, and the column is named.
    volatile = ["wall_time_s"]
    proj = {}
    for n in names:
        a = pd.read_csv(os.path.join(out_a, n))
        b = pd.read_csv(os.path.join(out_a2, n))
        a = a.drop(columns=[c for c in volatile if c in a.columns])
        b = b.drop(columns=[c for c in volatile if c in b.columns])
        proj[n] = a.equals(b)
    check("eps.A.determinism_modulo_measured_time", all(proj.values()),
          f"two consecutive runs identical on every column except "
          f"{volatile}: {proj}; whole-file byte equality (which the timing column "
          f"necessarily breaks): {whole}",
          volatile_columns=volatile, projected_equal=proj, whole_file_equal=whole,
          sha256={n: sha256_file(os.path.join(out_a, n)) for n in names})
    info("eps.A.solver_seed_variation_DEFERRED",
         "scipy.optimize.milp exposes no HiGHS random-seed option, so 'two runs "
         "with different solver seeds / variable orderings' (REQ-4 test 4) is only "
         "partially demonstrated here. Varying the variable order requires a "
         "permuted-index harness and belongs to R1. DEFERRED.")

    # ---- REQ-4 test 3: the INDEPENDENT oracle ------------------------------
    parser = HHCPFullParser(os.path.join(_THIS, "H_epse4_A.csv"))
    hp = os.path.join(_THIS, "H_epse4_A.csv")
    sp = os.path.join(_THIS, "S_epse4_A.csv")
    max_dW, n_triples, bad = 0.0, 0, []
    for absent in sorted(set(int(a) for a in front["absent"])):
        orc = ORACLE.oracle_disruption(hp, sp, absent, parser=parser)
        sub = front[front["absent"] == absent]
        rr = rng[rng["absent"] == absent].iloc[0]
        if (int(rr["U_min_A1"]), int(rr["U_min_A2"])) != (orc["u_min"]["A1"],
                                                          orc["u_min"]["A2"]):
            bad.append(("u_min", absent, dict(rr[["U_min_A1", "U_min_A2"]]),
                        orc["u_min"]))
        if (int(rr["eps_grid_lo"]), int(rr["eps_grid_hi"])) != (orc["grid_lo"],
                                                               orc["grid_hi"]):
            bad.append(("grid", absent, (int(rr["eps_grid_lo"]),
                                         int(rr["eps_grid_hi"])),
                        (orc["grid_lo"], orc["grid_hi"])))
        for _, r in sub.iterrows():
            key = (r["method"], int(r["eps"]))
            p = orc["points"].get(key)
            if p is None:
                bad.append(("missing", absent, key, None)); continue
            n_triples += 1
            dW = abs(float(r["W"]) - float(p["W"]))
            max_dW = max(max_dW, dW)
            if dW > EPS.ORACLE_W_ATOL or int(r["D"]) != p["D"] \
                    or int(r["U"]) != p["U"]:
                bad.append(("value", absent, key,
                            {"script": (r["W"], int(r["D"]), int(r["U"])),
                             "oracle": (p["W"], p["D"], p["U"])}))
    check("eps.A.independent_oracle", not bad,
          f"triples checked={n_triples}  max|W_script-W_oracle|={max_dW:.3e} "
          f"(tolerance {EPS.ORACLE_W_ATOL:.0e})  D and U exact  "
          f"disagreements={len(bad)}",
          triples=n_triples, max_abs_W_diff=max_dW, disagreements=bad[:6])

    # Exercise the exact oracle module used by the formal R1 path as well as
    # this fixture's deliberately separate enumerative oracle.  A prior bug in
    # its solver-error evidence used `message` both positionally and by keyword,
    # which only a real call through this module can catch.
    formal_absent = sorted(set(int(a) for a in front["absent"]))[0]
    formal_orc = FORMAL_ORACLE.oracle_disruption(
        hp, sp, formal_absent, time_limit_s=30.0, presolve=False)
    formal_sub = front[front["absent"] == formal_absent]
    formal_bad = []
    for _, r in formal_sub.iterrows():
        key = (r["method"], int(r["eps"]))
        point = formal_orc["points"].get(key)
        if point is None or abs(float(r["W"]) - float(point["W"])) > EPS.ORACLE_W_ATOL \
                or int(r["D"]) != int(point["D"]) or int(r["U"]) != int(point["U"]):
            formal_bad.append((key, point))
    check("eps.A.formal_oracle_runtime_path", not formal_bad,
          f"formal oracle executed its independently derived "
          f"{formal_orc['grid_lo']}..{formal_orc['grid_hi']} grid for "
          f"absent={formal_absent}; disagreements={len(formal_bad)}",
          absent=formal_absent, disagreements=formal_bad[:6])

    # ---- what the fixture actually exercises -------------------------------
    a2_better = int((rng["U_min_A2"] < rng["U_common"]).sum())
    info("eps.A.exercises",
         f"U_min_A2 < U_common on {a2_better} of {len(rng)} disruption(s) (the "
         f"structure the 35 real disruptions have, and none is excluded); "
         f"epsilon grids per disruption: "
         f"{sorted(zip(rng['absent'], rng['eps_grid_lo'], rng['eps_grid_hi']))}; "
         f"W_A2 strictly below W_A1 on "
         f"{int((front.pivot_table(index=['absent','eps'], columns='method', values='W')['A2'] < front.pivot_table(index=['absent','eps'], columns='method', values='W')['A1']).sum())}"
         f" paired cell(s)")
    return {"man": man, "out": out_a, "rng": rng, "front": front,
            "design": design, "hp": hp, "sp": sp, "parser": parser}


# =========================================================================== #
# part 2 — the weighted-sum stage (REQ-6)
# =========================================================================== #
def part2_e4():
    print("\n=== part 2: weighted-sum stage on fixture A (REQ-6) ===")
    man = os.path.join(_THIS, "manifest_epse4_A.csv")
    out_ps = os.path.join(OUT, "e4_A_per_slot")
    shutil.rmtree(out_ps, ignore_errors=True)
    (raw, summ, prov, design, hashes, reports, _), log = _run_quiet(
        lambda: E4.run(man, out_ps, mode="per-slot", verbose=True))
    _save("e4_A_stdout.txt", log)
    check("e4.A.ran", len(raw) == 24,
          f"raw rows={len(raw)} summary rows={len(summ)} "
          f"provenance rows={len(prov)}")

    # ---- REQ-6 item 7: separability tested, not assumed --------------------
    out_h = os.path.join(OUT, "e4_A_horizon")
    shutil.rmtree(out_h, ignore_errors=True)
    (raw_h, _s, _p, _d, _hh, _rp, _l), _ = _run_quiet(
        lambda: E4.run(man, out_h, mode="horizon", verbose=False))
    key = ["rho", "seed", "absent", "method", "w4"]
    a = raw.set_index(key)[["deviation", "overtime", "unserved"]].sort_index()
    b = raw_h.set_index(key)[["deviation", "overtime", "unserved"]].sort_index()
    same = a.equals(b)
    check("e4.A.separability_per_slot_equals_horizon", same,
          "per-slot decomposition and the monolithic horizon solve agree on "
          "(deviation, overtime, unserved) for all "
          f"{len(a)} cells" if same else
          f"MODES DISAGREE on {int((a != b).any(axis=1).sum())} cell(s)",
          disagreeing=None if same else a[(a != b).any(axis=1)].to_dict("index"))

    # ---- REQ-6 item 1: what the tie-break actually changes -----------------
    parser = HHCPFullParser(os.path.join(_THIS, "H_epse4_A.csv"))
    plan = load_static(os.path.join(_THIS, "S_epse4_A.csv"))
    S = set(parser.nurses)
    pol = E4.solver_policy()
    rows = []
    for absent in (0, 1):
        for regime in ("A1", "A2"):
            for q in E4.future_slots(plan):
                m = E4.WeightedModel(parser, plan, S, absent, regime, [q], pol)
                s_raw, _ = m._solve("Z", w4=0, stage_label="untiebroken")
                s_lex = m.solve_lex(0)
                rows.append({"absent": absent, "regime": regime, "slot": q,
                             "D_untiebroken": s_raw["D"], "D_lex": s_lex["D"],
                             "W_untiebroken": s_raw["W"], "W_lex": s_lex["W"],
                             "U_untiebroken": s_raw["U"], "U_lex": s_lex["U"]})
    df = pd.DataFrame(rows)
    changed = df[(df.D_untiebroken != df.D_lex) | (df.W_untiebroken != df.W_lex)]
    info("e4.A.tie_break_effect_at_w4_0",
         f"cells where the un-tie-broken single solve and the lexicographic solve "
         f"return a different (D, W): {len(changed)} of {len(df)} — "
         f"{changed[['absent', 'regime', 'slot', 'D_untiebroken', 'D_lex']].to_dict('records')}. "
         f"Where they agree, that is a fact about THIS solver on THIS instance and "
         f"NOT evidence that the tie-break is unnecessary: the tie exists whenever "
         f"a w4=0 optimum is non-unique, and which optimum a solver happens to "
         f"return is not a property of the model. The dominated-point control is "
         f"the decisive check.",
         n_changed=int(len(changed)), n_cells=int(len(df)),
         table=df.to_dict("records"))
    _save("e4_tiebreak_at_w4_0.json", json.dumps(df.to_dict("records"), indent=2))

    # ---- REQ-6 item 1: canonical assignment stage --------------------------
    m = E4.WeightedModel(parser, plan, S, 0, "A1", [4], pol)
    c1 = m.solve_lex(0, canonical_assignment=True)
    c2 = m.solve_lex(0, canonical_assignment=True)
    plain = m.solve_lex(0)
    check("e4.canonical_assignment.deterministic",
          c1["canonical_x_key"] == c2["canonical_x_key"]
          and (c1["D"], c1["W"], c1["U"]) == (plain["D"], plain["W"], plain["U"]),
          f"canonical key {c1['canonical_x_key']!r} reproduced across two runs "
          f"({c1['canonical_probes']} probe solves) and leaves (D, W, U) = "
          f"{(c1['D'], c1['W'], c1['U'])} unchanged")
    return {"man": man, "out": out_ps, "raw": raw, "summary": summ, "prov": prov,
            "design": design, "parser": parser, "plan": plan, "S": S, "pol": pol}


# =========================================================================== #
# part 3 — negative controls
# =========================================================================== #
def part3_negative(p1, p2):
    print("\n=== part 3: negative controls ===")

    # -- NC-01 eligibility guard (fixture B) --------------------------------
    expect_fatal(
        "NC-01.incumbent_not_candidate_eligible",
        lambda: EPS.run(os.path.join(_THIS, "manifest_epse4_B.csv"),
                        os.path.join(OUT, "eps_B_must_not_appear"), verbose=False),
        must_contain=("incumbent eligibility guard FAILED",),
        why="without this guard A1's candidate set need not be contained in A2's, "
            "so U_common := U_min_A1 would be unsupported")

    # -- NC-02 the same guard in the weighted-sum stage ---------------------
    expect_fatal(
        "NC-02.incumbent_not_candidate_eligible_e4",
        lambda: E4.run(os.path.join(_THIS, "manifest_epse4_B.csv"),
                       os.path.join(OUT, "e4_B_must_not_appear"), verbose=False),
        must_contain=("incumbent eligibility guard FAILED",))

    # -- NC-03 [shape]/[coverage]: delete one row ---------------------------
    mut = os.path.join(OUT, "mutants")
    os.makedirs(mut, exist_ok=True)
    src = os.path.join(p1["out"], "eps_frontier.csv")
    dst = os.path.join(mut, "eps_frontier_row_deleted.csv")
    df = pd.read_csv(src)
    df.drop(index=df.index[3]).to_csv(dst, index=False)
    spec = EPS.frontier_spec(p1["design"], EPS.solver_policy())
    expect_fatal("NC-03.validator_fails_on_deleted_row",
                 lambda: _run_quiet(lambda: validate_stage_output(dst, spec)),
                 must_contain=("[shape]", "[coverage]"),
                 mutant=os.path.relpath(dst, _REPO))

    # -- NC-04 a validator that passes a mutilated file is not a validator --
    dst2 = os.path.join(mut, "eps_frontier_U_broken.csv")
    df = pd.read_csv(src)
    df.loc[df.index[0], "U"] = int(df.loc[df.index[0], "U"]) + 1
    df.to_csv(dst2, index=False)
    expect_fatal("NC-04.U_not_equal_U_common_is_caught",
                 lambda: EPS.extra_checks(pd.read_csv(os.path.join(p1["out"],
                                                                  "eps_range.csv")),
                                          pd.read_csv(dst2), p1["design"],
                                          verbose=False),
                 must_contain=("U != U_common",),
                 mutant=os.path.relpath(dst2, _REPO))

    # -- NC-05 monotonicity ------------------------------------------------
    dst3 = os.path.join(mut, "eps_frontier_W_nonmonotone.csv")
    df = pd.read_csv(src)
    tgt = df[(df.method == "A2") & (df.absent == 0)].sort_values("eps")
    if len(tgt) >= 2:
        df.loc[tgt.index[-1], "W"] = float(df.loc[tgt.index[0], "W"]) + 1.0
    df.to_csv(dst3, index=False)
    expect_fatal("NC-05.W_increasing_in_epsilon_is_caught",
                 lambda: EPS.extra_checks(pd.read_csv(os.path.join(p1["out"],
                                                                  "eps_range.csv")),
                                          pd.read_csv(dst3), p1["design"],
                                          verbose=False),
                 must_contain=("monotone",),
                 mutant=os.path.relpath(dst3, _REPO))

    # -- NC-06 epsilon must be an absolute INTEGER budget -------------------
    mdl = EPS.HorizonModel(p1["parser"], load_static(p1["sp"]),
                           set(p1["parser"].nurses), 0, "A2", EPS.solver_policy())
    expect_fatal("NC-06.fractional_epsilon_refused",
                 lambda: mdl._solve("W", u_eq=1, eps=1.5),
                 must_contain=("epsilon must be an absolute INTEGER budget",))

    # -- NC-07 the cap allowance: 1e-6 is NOT admissible --------------------
    cap = float(H_MAX)
    slack = cap_fp_slack(cap)
    over = cap + 1e-7
    verdict = {"cap": cap, "allowance_ulp8": slack, "allowance_1e-6": 1e-6,
               "load": over,
               "passes_under_1e-6": bool(over <= cap + 1e-6),
               "passes_under_ulp8": bool(over <= cap + slack),
               "exact_cap_passes_under_ulp8": bool(cap <= cap + slack),
               "one_ulp_over_passes": bool(cap + slack / 2 <= cap + slack)}
    ok = (verdict["passes_under_1e-6"] and not verdict["passes_under_ulp8"]
          and verdict["exact_cap_passes_under_ulp8"])
    _rec("negative_controls", "NC-07.cap_allowance_1e-6_hides_a_violation",
         "PASS" if ok else "FAIL",
         f"a load of cap+1e-7 = {over!r} PASSES a 1e-6 allowance and FAILS the "
         f"math.ulp(cap)*8 allowance ({slack:.3e}); a load of exactly cap still "
         f"passes. 1e-6 is {1e-6 / slack:.3g}x too loose.", **verdict)

    # -- NC-08 the cap guard on the real code path (fault injection) --------
    def _inject_overload():
        m2 = EPS.HorizonModel(p1["parser"], load_static(p1["sp"]),
                              set(p1["parser"].nurses), 0, "A2",
                              EPS.solver_policy())
        s, _ = m2._solve("W", u_eq=1)
        # make one served task 1e-7 longer than the cap allows, then re-read back
        served = [(k, v) for k, v in s["assignment"].items() if v is not None]
        (q, t), i = served[0]
        m2.dur[(q, t)] = float(H_MAX) + 1e-7
        return m2.readback(_x_of(m2, s["assignment"]), "cap_fault_injection")

    def _x_of(model, assignment):
        x = np.zeros(model.n)
        for (q, t), i in assignment.items():
            if i is not None:
                x[model.xidx[(q, i, t)]] = 1.0
            else:
                x[model.uidx[(q, t)]] = 1.0
        for (q, i), k in model.oidx.items():
            load = sum(model.dur[(q, tt)] for tt in model.tasks[q]
                       if assignment.get((q, tt)) == i)
            x[k] = max(0.0, load - float(H_STD))
        return x

    expect_fatal("NC-08.cap_guard_fires_on_a_1e-7_overload",
                 _inject_overload,
                 must_contain=("hard capacity violated", "math.ulp(cap)*8"),
                 why="a 1e-6 allowance would have admitted this assignment")

    # -- NC-09 big weights are NOT a lexicographic order --------------------
    m3 = EPS.HorizonModel(p1["parser"], load_static(p1["sp"]),
                          set(p1["parser"].nurses), 0, "A2", EPS.solver_policy())
    grid = sorted(set(int(e) for e in
                      p1["front"][(p1["front"].absent == 0)
                                  & (p1["front"].method == "A2")]["eps"]))
    eps_top = grid[-1]
    lex = m3.solve_lex(1, eps=eps_top)
    scal = _big_weight_scalarization(m3, u_eq=1, eps=eps_top, big=1000.0)
    differs = (abs(lex["W"] - scal["W"]) > 1e-9) or (lex["D"] != scal["D"])
    verdict = ("different, so a big weight does NOT implement min W then min D"
               if differs else
               "IDENTICAL, so this fixture does not separate them")
    _rec("negative_controls", "NC-09.big_weight_is_not_lexicographic",
         "PASS" if differs else "FAIL",
         "at eps={} the genuine sequential lexicographic solve returns "
         "(W={}, D={}) while minimising the scalarized W + 1000*D returns "
         "(W={}, D={}) — {}".format(eps_top, lex["W"], lex["D"], scal["W"],
                                    scal["D"], verdict),
         lex={"W": lex["W"], "D": lex["D"]}, scalarized=scal, big=1000.0)

    # -- NC-10 the U equality is load-bearing ------------------------------
    m4 = EPS.HorizonModel(p1["parser"], load_static(p1["sp"]),
                          set(p1["parser"].nurses), 0, "A2", EPS.solver_policy())
    eps_lo = grid[0]
    s_no_u, _ = m4._solve("W", eps=eps_lo, stage_label="no_U_equality")
    pinned = p1["front"][(p1["front"].absent == 0) & (p1["front"].method == "A2")
                         & (p1["front"].eps == eps_lo)].iloc[0]
    uncontrolled = s_no_u["U"] != int(pinned["U_common"])
    _rec("negative_controls", "NC-10.dropping_the_U_equality_changes_the_answer",
         "PASS" if uncontrolled else "FAIL",
         f"with U pinned to U_common={int(pinned['U_common'])} the A2 point at "
         f"eps={eps_lo} is (U={int(pinned['U'])}, W={float(pinned['W'])}); with the "
         f"equality removed the SAME program reaches (U={s_no_u['U']}, "
         f"W={s_no_u['W']}) — it buys overtime with unserved care. The two points "
         f"sit at different service levels, which is exactly the comparison REQ-5 "
         f"declares invalid, and the artifact could not even be checked for it "
         f"before REQ-4 added the U columns.",
         eps=eps_lo,
         with_equality={"U": int(pinned["U"]), "W": float(pinned["W"])},
         without_equality={"U": s_no_u["U"], "W": s_no_u["W"]})

    # -- NC-11 a per-slot cap is not a horizon epsilon ---------------------
    st, dt, ex = _per_slot_vs_global(p1)
    _rec("negative_controls", "NC-11.per_slot_cap_is_not_a_horizon_epsilon",
         st, dt, **ex)

    # -- NC-12 dominated point (REQ-6 test 4) ------------------------------
    raw = p2["raw"].copy()
    tgt = raw[(raw.absent == 0) & (raw.method == "A2")].sort_values("w4")
    # make the w4=0 row strictly worse than the w4=5 row in deviation
    i0 = tgt.index[0]
    i5 = tgt.index[1]
    raw.loc[i0, "deviation"] = float(raw.loc[i5, "deviation"]) + 7
    raw.loc[i0, "overtime"] = float(raw.loc[i5, "overtime"])
    raw.loc[i0, "unserved"] = float(raw.loc[i5, "unserved"])
    dpath = os.path.join(mut, "e4_raw_dominated_w4_0.csv")
    raw.to_csv(dpath, index=False)
    expect_fatal("NC-12.dominated_w4_point_is_caught",
                 lambda: E4.assert_no_dominated_w4_point(pd.read_csv(dpath)),
                 must_contain=("dominated by another w4 point",),
                 mutant=os.path.relpath(dpath, _REPO),
                 note="a synthetic dominated pair is injected; no published "
                      "number is used as a target")

    # -- NC-13 provenance: mutate one source_row_key -----------------------
    prov = p2["prov"].copy()
    prov.loc[prov.index[0], "source_row_key"] = "rho=0.0000|seed=901|absent=99" \
                                                "|method=A2|w4=0"
    ppath = os.path.join(mut, "e4_provenance_bad_key.csv")
    prov.to_csv(ppath, index=False)
    expect_fatal("NC-13.provenance_bad_source_row_key",
                 lambda: E4.verify_provenance(p2["summary"], p2["raw"],
                                              pd.read_csv(ppath)),
                 must_contain=("does not exist in the raw artifact",),
                 mutant=os.path.relpath(ppath, _REPO))

    # -- NC-14 provenance: drop a citation ---------------------------------
    prov2 = p2["prov"].copy()
    prov2 = prov2.drop(index=prov2.index[0])
    ppath2 = os.path.join(mut, "e4_provenance_missing_citation.csv")
    prov2.to_csv(ppath2, index=False)
    expect_fatal("NC-14.provenance_incomplete_citation_set",
                 lambda: E4.verify_provenance(p2["summary"], p2["raw"],
                                              pd.read_csv(ppath2)),
                 must_contain=("citation set does not match",),
                 mutant=os.path.relpath(ppath2, _REPO))

    # -- NC-15 the superseded-CI label may not be quietly removed ----------
    summ = p2["summary"].copy()
    summ.loc[summ.index[0], "ci95_method"] = "clustered bootstrap"
    expect_fatal("NC-15.ci95_supersession_label_cannot_be_altered",
                 lambda: E4.verify_provenance(summ, p2["raw"], p2["prov"]),
                 must_contain=("superseded by R5",))

    # -- NC-16 cross-stage transcription gate ------------------------------
    expect_fatal("NC-16.cross_stage_read_refused",
                 lambda: E4._read_csv_audited(
                     os.path.join(p1["out"], "eps_frontier.csv"),
                     why="negative control"),
                 must_contain=("cross-stage transcription refused",))

    # -- NC-17 canonical path protection ----------------------------------
    for label, mod in (("eps", EPS), ("e4", E4)):
        expect_fatal(f"NC-17.{label}.canonical_write_refused_without_flag",
                     lambda mod=mod: mod.main(["--manifest",
                                               os.path.join(_THIS,
                                                            "manifest_epse4_A.csv")]),
                     must_contain=("--promote-canonical",),
                     why="without this, a fixture run could overwrite a canonical "
                         "artifact; regeneration is R1 / R1b")

    # -- NC-18 lint gates -------------------------------------------------
    for label, path in (("eps_constraint.py", EPS.__file__),
                        ("run_e4.py", E4.__file__)):
        try:
            assert_no_absolute_runtime_paths(path)
            assert_no_staging_import()
            _rec("negative_controls", f"NC-18.lint_clean.{label}", "PASS",
                 "no absolute home-directory runtime path literal; no module "
                 "loaded from audit_outputs/")
        except PipelineFailure as exc:
            _rec("negative_controls", f"NC-18.lint_clean.{label}", "FAIL",
                 str(exc)[:400])
    bad = os.path.join(mut, "lint_offender.py")
    with open(bad, "w", encoding="utf-8") as fh:
        fh.write("PATH = '/" + "Users" + "/someone/repo/x.csv'\n")
    expect_fatal("NC-18b.lint_gate_fails_on_a_mutated_copy",
                 lambda: assert_no_absolute_runtime_paths(bad),
                 must_contain=("absolute runtime path literal",))

    # -- NC-19 prohibited-value grep (REQ-4 test 6) ------------------------
    st, dt, ex = _prohibited_value_grep()
    _rec("negative_controls", "NC-19.no_old_endpoint_acceptance_target",
         st, dt, **ex)

    # -- NC-20 the precision fixture --------------------------------------
    st, dt, ex = _precision_fixture()
    _rec("negative_controls", "NC-20.rounding_to_one_decimal_hides_the_optimum",
         st, dt, **ex)

    # -- NC-21 the HiGHS presolve finding ---------------------------------
    st, dt, ex = _presolve_finding()
    _rec("negative_controls", "NC-21.highs_presolve_infeasible_finding",
         st, dt, **ex)


def _big_weight_scalarization(model, u_eq, eps, big):
    """min sum(O) + big*D in ONE solve — the thing REQ-4 item 3 forbids."""
    c = model._w_row() + float(big) * model.dev_coef
    rows = list(model._base_rows) + [model._u_row(), model.dev_coef.copy()]
    lb = list(model._base_lb) + [float(u_eq), -np.inf]
    ub = list(model._base_ub) + [float(u_eq), float(int(eps)) - model.dev_const]
    from scipy.optimize import Bounds, LinearConstraint, milp
    lo = np.zeros(model.n)
    hi = np.ones(model.n)
    for k in model.oidx.values():
        hi[k] = np.inf
    integ = np.zeros(model.n)
    for k in range(len(model.xpairs)):
        integ[k] = 1
    for k in model.uidx.values():
        integ[k] = 1
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), np.array(lb),
                                                 np.array(ub)),
               integrality=integ, bounds=Bounds(lo, hi),
               options={"time_limit": 60})
    sol = model.readback(res.x, "big_weight_scalarization")
    return {"W": sol["W"], "D": sol["D"], "U": sol["U"], "status": int(res.status)}


def _per_slot_vs_global(p1):
    """Show that a per-slot deviation cap is split-dependent while a horizon
    budget is not, on the same instance and the same total budget."""
    plan = ORACLE.load_plan(p1["sp"])
    per_slot = {}
    for spec in ORACLE._structures(p1["parser"], plan, 0, "A2"):
        per_slot[spec["q"]] = ORACLE.slot_achievable(spec)
    ach = ORACLE.horizon_achievable(p1["parser"], plan, 0, "A2")
    u_common = ORACLE.min_unserved(
        ORACLE.horizon_achievable(p1["parser"], plan, 0, "A1"))
    total = 3
    glob = ORACLE.lex_point(ach, u_common, eps=total)
    slots = sorted(per_slot)
    splits = {}
    for c0 in range(0, total + 1):
        caps = {slots[0]: c0, slots[1]: total - c0}
        best = None
        for (u0, d0), w0 in per_slot[slots[0]].items():
            if d0 > caps[slots[0]]:
                continue
            for (u1, d1), w1 in per_slot[slots[1]].items():
                if d1 > caps[slots[1]] or u0 + u1 != u_common:
                    continue
                if best is None or w0 + w1 < best:
                    best = w0 + w1
        splits[f"{caps[slots[0]]}+{caps[slots[1]]}"] = best
    attainable = [v for v in splits.values() if v is not None]
    worse = [k for k, v in splits.items()
             if v is None or v > glob["W"] + 1e-9]
    ok = bool(worse) and glob is not None
    return ("PASS" if ok else "FAIL",
            f"one horizon budget sum_q D_q <= {total} attains W={glob['W']} at "
            f"D={glob['D']}. The SAME total budget split per slot attains "
            f"{splits} — {len(worse)} of {len(splits)} splits are strictly worse "
            f"or infeasible, so a per-slot cap is split-dependent and is not the "
            f"same constraint. This is why `frac` in the published scheme was not "
            f"an epsilon.",
            {"horizon": {"eps": total, "W": glob["W"], "D": glob["D"]},
             "per_slot_splits": splits, "worse_or_infeasible_splits": worse,
             "u_common": u_common,
             "n_attainable_splits": len(attainable)})


def _prohibited_value_grep():
    """REQ-4 test 6: no old endpoint may appear as an acceptance target."""
    needles = ["126,53", "(126, 53)", "207,49", "(207, 49)", "25,1811", "1811",
               "1810", "1799", "(240, 49)", "240,49", "(94, 49)", "94,49"]
    files = [EPS.__file__, E4.__file__, os.path.abspath(__file__),
             os.path.join(_THIS, "oracle_independent.py"),
             os.path.join(_JP0, "eps_constraint_design_manifest.json"),
             os.path.join(_JP0, "run_e4_design_manifest.json")]
    hits, self_file = [], os.path.abspath(__file__)
    for f in files:
        # THIS driver necessarily contains the needle literals (they are the search
        # list). For it, a hit only counts when the line is an assertion or a
        # comparison, i.e. when the number is being used as a target.
        target_only = os.path.abspath(f) == self_file
        with open(f, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                low = line.lower()
                is_target = any(tok in low for tok in
                                ("assert", "require(", "expected", "== "))
                if target_only and not is_target:
                    continue
                for needle in needles:
                    if needle in line and "forbidden" not in low \
                            and "prohibit" not in low:
                        hits.append(f"{os.path.basename(f)}:{n} {needle!r}")
                if re.search(r"(assert|require|==)\s*[^\n]*\b213\b", line) \
                        and "forbidden" not in low and "prohibit" not in low:
                    hits.append(f"{os.path.basename(f)}:{n} bare 213 target")
    return ("PASS" if not hits else "FAIL",
            f"grep over {len(files)} file(s) for the published endpoints "
            f"{needles} and for a bare 213 acceptance target: {len(hits)} hit(s). "
            f"The epsilon grid and every acceptance value are derived from the "
            f"instance, never from a published number.",
            {"files": [os.path.relpath(f, _REPO) for f in files], "hits": hits})


def orc_parser(hp):
    return HHCPFullParser(hp)


def _precision_fixture():
    """Fixture C: two achievable overtime levels collide under round(W, 1)."""
    man = os.path.join(_THIS, "manifest_epse4_C.csv")
    out_c = os.path.join(OUT, "eps_C")
    shutil.rmtree(out_c, ignore_errors=True)
    (rng, front, design, hashes, reports, _), log = _run_quiet(
        lambda: EPS.run(man, out_c, max_absent=1, allow_fractional=True,
                        verbose=True))
    _save("eps_C_stdout.txt", log)
    hp = os.path.join(_THIS, "H_epse4_C_precision.csv")
    sp = os.path.join(_THIS, "S_epse4_C_precision.csv")
    orc = ORACLE.oracle_disruption(hp, sp, 0)
    plan_c = ORACLE.load_plan(sp)
    row = front[front.method == "A2"].iloc[0]
    script_W = float(row["W"])
    oracle_W = orc["points"][("A2", int(row["eps"]))]["W"]
    exact = abs(script_W - oracle_W) <= EPS.ORACLE_W_ATOL
    # the FULL (unreduced) achievable set: two configurations can share
    # (unserved, deviation) and still differ in overtime
    u_common = int(row["U_common"])
    allcfg = []
    for spec in ORACLE._structures(orc_parser(hp), plan_c, 0, "A2"):
        allcfg += ORACLE.slot_achievable_all(spec)
    levels = sorted({round(w, 12) for (u, d, w) in allcfg if u == u_common})
    collide = [(a, b) for i, a in enumerate(levels) for b in levels[i + 1:]
               if a != b and round(a, 1) == round(b, 1)]
    # a strictly suboptimal level that a one-decimal comparator cannot tell from
    # the true optimum
    wrong = [b for (a, b) in collide if abs(a - script_W) < 1e-9]
    hidden = bool(wrong) and round(float(wrong[0]), 1) == round(script_W, 1) \
        and abs(float(wrong[0]) - script_W) > EPS.ORACLE_W_ATOL
    ok = exact and bool(collide) and hidden \
        and str(rng["w_level_separation"].iloc[0]) == "unknown"
    return ("PASS" if ok else "FAIL",
            f"achievable total-overtime levels on fixture C: {levels}. The pair "
            f"{collide} is distinct at full precision but IDENTICAL under "
            f"round(W, 1). The stage returns W={script_W!r} and the independent "
            f"oracle returns W={oracle_W!r}, agreeing to "
            f"{abs(script_W - oracle_W):.3e} <= {EPS.ORACLE_W_ATOL:.0e}. A "
            f"comparator that rounded to one decimal would have accepted a "
            f"strictly suboptimal answer "
            "as a match (wrong_level={!r}). Non-integral durations are flagged in "
            "the artifact as w_level_separation={!r}.".format(
                (float(wrong[0]) if wrong else None),
                str(rng["w_level_separation"].iloc[0])),
            {"levels": levels, "colliding_pairs": collide,
             "script_W": script_W, "oracle_W": oracle_W,
             "w_level_separation": str(rng["w_level_separation"].iloc[0]),
             "out_dir": os.path.relpath(out_c, _REPO)})


def _presolve_finding():
    import subprocess
    script = os.path.join(_THIS, "negative", "highs_presolve_infeasible.py")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True,
                          cwd=_REPO)
    _save("highs_presolve_infeasible_stdout.txt", proc.stdout + proc.stderr)
    ok = proc.returncode == 0 and "REPRODUCED" in proc.stdout
    return ("PASS" if ok else "INFO",
            "scipy/HiGHS returns status=2 (infeasible) with presolve ON for a "
            "model with an explicit feasible point, and status=0 with presolve "
            "OFF. In this R1b/Z-row reproducer the tolerance was NOT relaxed to make it pass: "
            "the provably "
            "redundant Z row was removed from the final lexicographic stage (the "
            "constraint is kept as a post-hoc assertion), and every stage now "
            "passes the previous stage's solution as known_feasible_x so a "
            "refuted infeasibility ABORTS instead of being retried. "
            "REPORTABLE FINDING.",
            {"script": os.path.relpath(script, _REPO),
             "returncode": proc.returncode,
             "stdout_tail": proc.stdout[-700:]})


# =========================================================================== #
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(EVIDENCE, exist_ok=True)
    p1 = part1_eps()
    p2 = part2_e4()
    part3_negative(p1, p2)

    n_fail = sum(1 for r in RESULTS["checks"] + RESULTS["negative_controls"]
                 if r["status"] == "FAIL")
    RESULTS["summary"] = {
        "checks": len(RESULTS["checks"]),
        "negative_controls": len(RESULTS["negative_controls"]),
        "failures": n_fail,
        "epistemic_status": ("regression and traceability evidence only; NOT "
                             "proof, NOT validation, NOT correctness, NOT "
                             "evidence of an absence of fabrication"),
        "stages_executed": "NONE — R1 and R1b were not run; only tiny synthetic "
                           "fixtures under tests/fixtures/epse4/ were solved",
        "snap": "no SNAP identifier exists and none was invented",
    }
    path = _save("epse4_fixture_results.json", json.dumps(RESULTS, indent=2,
                                                          default=str))
    print(f"\n=== {len(RESULTS['checks'])} check(s), "
          f"{len(RESULTS['negative_controls'])} negative control(s), "
          f"{n_fail} failure(s) ===")
    print(f"results: {os.path.relpath(path, _REPO)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
