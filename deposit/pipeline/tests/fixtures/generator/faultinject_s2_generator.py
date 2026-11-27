"""
faultinject_s2_generator.py — NEGATIVE CONTROLS for the official generator
`experiments/journal_p0/make_rho_instance.py` (S2 round, generator agent).

WHAT THIS IS AND IS NOT
-----------------------
Each case deliberately BREAKS one thing and requires the generator to (a) trip a
named FATAL guard, (b) exit non-zero, and (c) leave NO canonical `H_<tag>.csv` /
`S_<tag>.csv` pair behind, with an empty `_tmp/` and a machine-readable reason
file in `_quarantine/`.

**A guard that fires is evidence the guard is wired, not evidence the generator
is correct.**  These are negative controls, never a proof, never a validation of
any result.  The `baseline` case is a positive control on a TINY SYNTHETIC
fixture only; it says nothing about any real instance.

The generator is NOT edited to make these cases possible.  Injection uses the two
emitters that `generate()` already reaches through the module globals
(`emit_hhcp_with_correct_metadata`, `emit_schedule_flat`), plus mutated COPIES of
the tiny fixture for the input-side cases.  No test seam exists in the generator.

    cd <repo root> && python tests/fixtures/generator/faultinject_s2_generator.py --all
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))   # repo root
_JP0 = os.path.join(_ROOT, "experiments", "journal_p0")
if _JP0 not in sys.path:
    sys.path.insert(0, _JP0)

FIXTURE = os.path.join(_HERE, "tiny_hhcp_fixture.csv")
OUTROOT = os.path.join(_HERE, "faultinject_out")
RHO, NURSES, SEED = 0.85, 2, 1
TAG = f"rho{int(round(RHO * 100)):03d}_n{NURSES}_s{SEED}"

# case -> (expected primary guard, expected exit code)
CASES = {
    "baseline":                     (None, 0),
    # ---- input-side mutations (a mutated COPY of the fixture) ----------------
    "srcmeta":                      ("source_metadata_inconsistent", 3),
    "candslot":                     ("cand_slot_violation", 3),
    "srcfield_patient":             ("source_persisted_field_mismatch", 3),
    "srcfield_nurse_skills":        ("source_persisted_field_mismatch", 3),
    "srcfield_nurse_slots":         ("source_persisted_field_mismatch", 3),
    # ---- landed-H mutations (H emitter intercepted) --------------------------
    "landed_meta":                  ("landed_metadata_mismatch", 3),
    "landedh_reqslot":              ("landed_h_not_equivalent_to_intended", 3),
    "landedh_verbatim_x":           ("landed_h_not_equivalent_to_intended", 3),
    "landedh_persisted_task_count": ("landed_persisted_field_mismatch", 3),
    # ---- landed-S mutations (schedule emitter intercepted) -------------------
    "s_multiset_swap":              ("served_task_multiset_mismatch", 3),
    "s_drop_task":                  ("served_task_multiset_mismatch", 3),
    "s_ownership":                  ("row_patient_task_mismatch", 3),
    "s_slot":                       ("requested_slot_violation", 3),
    "s_nurse_noncandidate":         ("availability_candidate_violation", 3),
    "s_capover":                    ("landed_max_load_gt_cap", 3),
    "s_schema":                     ("landed_s_schema_invalid", 3),
    "s_flatload":                   ("landed_flatload_mismatch", 3),
}
INPUT_CASES = ("srcmeta", "candslot", "srcfield_patient", "srcfield_nurse_skills",
               "srcfield_nurse_slots")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ───────────────────────── tiny text-level helpers ───────────────────────────
def _hdr(lines, prefix):
    hits = [i for i, l in enumerate(lines) if l.lower().startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"expected one {prefix!r} header, found {hits}")
    return hits[0]


def _sections(lines):
    pi = _hdr(lines, "patient,")
    ni = _hdr(lines, "nurse,")
    ti = _hdr(lines, "task,")
    ci = _hdr(lines, "candidate solution")
    return pi, ni, ti, ci


def _row_line(lines, lo, hi, rid):
    for i in range(lo, hi):
        p = lines[i].split(",")
        if p and p[0].strip().isdigit() and int(p[0]) == rid:
            return i
    raise SystemExit(f"row id {rid} not found in [{lo},{hi})")


def _field_index(header_line, name):
    names = [f.strip().lower() for f in header_line.split(",")]
    hits = [i for i, nm in enumerate(names) if nm == name]
    if len(hits) != 1:
        raise SystemExit(f"field {name!r} appears {len(hits)} times in {names}")
    return hits[0]


def _set_field(lines, i, k, value):
    p = lines[i].split(",")
    old = p[k]
    p[k] = str(value)
    lines[i] = ",".join(p)
    return old


def _histogram_lines(lines):
    for i, l in enumerate(lines):
        if l.strip().lower().startswith("#(patients) in each slot"):
            return i + 2
    raise SystemExit("per-slot histogram block not found")


def _set_histogram(lines, i, counts):
    trailing = lines[i].endswith(",")
    lines[i] = ",".join(str(c) for c in counts) + ("," if trailing else "")


# ───────────────────── input-side mutated fixture builders ───────────────────
def build_mutated_input(case, dest):
    with open(FIXTURE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    pi, ni, ti, ci = _sections(lines)
    note = {}
    if case == "srcmeta":
        hi = _histogram_lines(lines)
        before = lines[hi]
        _set_histogram(lines, hi, [1, 1, 1, 0, 1, 0])
        note = dict(what="declared per-slot patient histogram only; patient rows "
                         "untouched", before=before, after=lines[hi])
    elif case == "candslot":
        k = _field_index(lines[pi], "requested slot")
        i = _row_line(lines, pi + 1, ni, 0)
        old = _set_field(lines, i, k, 4)
        hi = _histogram_lines(lines)
        _set_histogram(lines, hi, [1, 1, 1, 0, 1, 0])
        note = dict(what=f"patient 0 requested slot {old}->4 (no caregiver works slot 4); "
                         "the declared histogram is kept CONSISTENT so this case isolates "
                         "the candidate/allowed-slot probe")
    elif case == "srcfield_patient":
        k = _field_index(lines[pi], "#(tasks)")
        i = _row_line(lines, pi + 1, ni, 0)
        old = _set_field(lines, i, k, 3)
        note = dict(what=f"patient 0 declared #(tasks) {old}->3 while the task list still "
                         "holds 2 ids")
    elif case == "srcfield_nurse_skills":
        i = _row_line(lines, ni + 1, ti, 0)
        p = lines[i].split(",")
        clean_pos = [j for j, v in enumerate(p) if v.strip() != ""]
        # cleaned layout: 0 id,1 x,2 y,3 class,4..7 prefs,8 #slot,9.. slots,
        # then #skills and the skill list
        n_slots = int(p[clean_pos[8]])
        k_sk = clean_pos[9 + n_slots]
        n_sk = int(p[k_sk])
        j = clean_pos[9 + n_slots + 1 + (n_sk - 1)]        # last skill token
        old = _set_field(lines, i, j, p[clean_pos[9 + n_slots + 1]].strip())
        note = dict(what=f"nurse 0 last skill token {old}->duplicate of the first, "
                         f"#(skills)={n_sk} left unchanged, so the declared count exceeds "
                         f"the parsed skill SET size")
    elif case == "srcfield_nurse_slots":
        i = _row_line(lines, ni + 1, ti, 0)
        p = lines[i].split(",")
        clean_pos = [j for j, v in enumerate(p) if v.strip() != ""]
        n_slots = int(p[clean_pos[8]])
        j = clean_pos[9 + n_slots - 1]                     # last slot token
        old = _set_field(lines, i, j, p[clean_pos[9]].strip())
        note = dict(what=f"nurse 0 last slot token {old}->duplicate of the first, "
                         f"#(slot)={n_slots} left unchanged (this ALSO removes a real "
                         f"availability, so a candidate/slot collateral trip is expected)")
    else:
        raise SystemExit(f"{case} is not an input-side case")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return note


# ─────────────────────── in-process injection wrappers ───────────────────────
def _wrap_hhcp(case, orig, injection):
    def wrapped(src_lines, S, parser):
        lines, npairs, report = orig(src_lines, S, parser)
        lines = list(lines)
        pi, ni, ti, ci = _sections(lines)
        if case == "landed_meta":
            mh = next(j for j, l in enumerate(lines[:pi])
                      if "#(nurse)" in {f.strip().lower() for f in l.split(",")})
            k = _field_index(lines[mh], "#(nurse)")
            old = _set_field(lines, mh + 1, k, 99)
            injection.update(what=f"landed H declared #(nurse) {old}->99 while the emitted "
                                  f"nurse rows are unchanged")
        elif case == "landedh_reqslot":
            k = _field_index(lines[pi], "requested slot")
            i = _row_line(lines, pi + 1, ni, 0)
            old = _set_field(lines, i, k, 2)
            injection.update(what=f"ONE field of ONE landed-H patient line: patient 0 "
                                  f"requested slot {old}->2")
        elif case == "landedh_verbatim_x":
            k = _field_index(lines[pi], "x")
            i = _row_line(lines, pi + 1, ni, 1)
            old = _set_field(lines, i, k, 301)
            injection.update(what=f"ONE field of ONE landed-H patient line: patient 1 x "
                                  f"{old}->301. The patient block is a VERBATIM copy of "
                                  f"the source, and patient x is not compared field-wise "
                                  f"against the source anywhere else, so only the "
                                  f"byte-level verbatim check can see this")
        elif case == "landedh_persisted_task_count":
            k = _field_index(lines[ti], "#(nurses)")
            i = _row_line(lines, ti + 1, ci, 0)
            old = _set_field(lines, i, k, int(_field_value(lines, i, k)) + 1)
            injection.update(what=f"landed H task 0 declared #(nurses) {old}->"
                                  f"{int(old) + 1} while the candidate list is unchanged")
        else:
            raise SystemExit(f"{case} is not an H-side case")
        return lines, npairs, report
    return wrapped


def _field_value(lines, i, k):
    return lines[i].split(",")[k].strip()


def _wrap_schedule(case, orig, injection):
    def wrapped(parser, cell_pt, cell_load):
        df = orig(parser, cell_pt, cell_load)
        df = df.copy()
        if case == "s_multiset_swap":
            before = df.at[0, "TaskIDs"]
            tids = [t for t in str(before).split(";") if t != ""]
            df.at[0, "TaskIDs"] = f"{tids[0]};{tids[0]};"
            injection.update(what=f"row 0 TaskIDs {before!r} -> {df.at[0, 'TaskIDs']!r}: "
                                  f"task {tids[0]} appears TWICE and task {tids[1]} "
                                  f"disappears. The row COUNT and the flat load are "
                                  f"unchanged, which is exactly why a count can never "
                                  f"substitute for task-set equality")
        elif case == "s_drop_task":
            before = df.at[0, "TaskIDs"]
            tids = [t for t in str(before).split(";") if t != ""]
            df.at[0, "TaskIDs"] = f"{tids[0]};"
            from displacement_milp import TRAVEL_PER_TASK
            keep = parser.tasks[int(tids[0])].duration + TRAVEL_PER_TASK
            df.at[0, "FlatLoad"] = round(keep, 1)
            injection.update(what=f"row 0 TaskIDs {before!r} -> {df.at[0,'TaskIDs']!r}: "
                                  f"task {tids[1]} is never scheduled; FlatLoad is "
                                  f"corrected with it so the case isolates the task set")
        elif case == "s_ownership":
            before = int(df.at[0, "PatientID"])
            df.at[0, "PatientID"] = int(df.at[1, "PatientID"])
            injection.update(what=f"row 0 PatientID {before}->{int(df.at[0,'PatientID'])} "
                                  f"while its TaskIDs still belong to patient {before}")
        elif case == "s_slot":
            i = len(df) - 1
            before = int(df.at[i, "Shift"])
            df.at[i, "Shift"] = before - 1
            injection.update(what=f"last row Shift {before}->{before - 1}: the caregiver "
                                  f"still works that slot, but the slot is OUTSIDE the "
                                  f"patient's allowed-slot SET")
        elif case == "s_nurse_noncandidate":
            i = len(df) - 1
            before = int(df.at[i, "NurseID"])
            df.at[i, "NurseID"] = int(df.at[i - 1, "NurseID"])
            injection.update(what=f"last row NurseID {before}->"
                                  f"{int(df.at[i,'NurseID'])}: present in the landed H and "
                                  f"available in that slot, but NOT a candidate for the "
                                  f"row's tasks")
        elif case == "s_capover":
            before = int(df.at[1, "NurseID"])
            df.at[1, "NurseID"] = int(df.at[0, "NurseID"])
            injection.update(what=f"row 1 NurseID {before}->{int(df.at[1,'NurseID'])}: a "
                                  f"legal candidate, available in that slot, and the slot "
                                  f"is allowed -- but the cell load now exceeds the cap")
        elif case == "s_schema":
            df = df.drop(columns=["NurseID"])
            injection.update(what="the NurseID column is removed from the landed S")
        elif case == "s_flatload":
            before = float(df.at[0, "FlatLoad"])
            df.at[0, "FlatLoad"] = before + 789.0
            injection.update(what=f"row 0 FlatLoad {before}->{before + 789.0} only; the "
                                  f"task set and every assignment are untouched")
        else:
            raise SystemExit(f"{case} is not an S-side case")
        return df
    return wrapped


# ────────────────────────────── child process ────────────────────────────────
def child(case, outdir, src):
    import make_rho_instance as M
    injection = dict(case=case)
    if case in ("landed_meta", "landedh_reqslot", "landedh_verbatim_x",
                "landedh_persisted_task_count"):
        M.emit_hhcp_with_correct_metadata = _wrap_hhcp(
            case, M.emit_hhcp_with_correct_metadata, injection)
    elif case.startswith("s_"):
        M.emit_schedule_flat = _wrap_schedule(case, M.emit_schedule_flat, injection)
    elif case not in ("baseline",) + INPUT_CASES:
        raise SystemExit(f"unknown case {case}")
    code = M.run_one(RHO, nurses=NURSES, seed=SEED, src=src, outdir=outdir, quiet=False)
    print("INJECTION_JSON " + json.dumps(injection, default=str))
    return code


# ────────────────────────────── parent process ───────────────────────────────
def canon(outdir):
    return (os.path.join(outdir, f"H_{TAG}.csv"), os.path.join(outdir, f"S_{TAG}.csv"))


def listing(d):
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d))


def guards_from_stdout(text):
    for line in text.splitlines():
        if line.startswith("MANIFEST_JSON "):
            try:
                return json.loads(line[len("MANIFEST_JSON "):]).get("guards_tripped", [])
            except Exception:
                return ["<unparseable MANIFEST_JSON>"]
    return []


def injection_from_stdout(text):
    for line in text.splitlines():
        if line.startswith("INJECTION_JSON "):
            try:
                return json.loads(line[len("INJECTION_JSON "):])
            except Exception:
                return {}
    return {}


def run_case(case):
    want_guard, want_exit = CASES[case]
    outdir = os.path.join(OUTROOT, case)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    src = FIXTURE
    input_note = {}
    if case in INPUT_CASES:
        src = os.path.join(outdir, f"MUTATED_INPUT_{case}.csv")
        input_note = build_mutated_input(case, src)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", case,
         "--outdir", outdir, "--src", src],
        capture_output=True, text=True, cwd=_ROOT)
    wall = round(time.time() - t0, 3)
    with open(os.path.join(outdir, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(os.path.join(outdir, "stderr.txt"), "w", encoding="utf-8") as f:
        f.write(proc.stderr)

    h, s = canon(outdir)
    tmpd, qd = os.path.join(outdir, "_tmp"), os.path.join(outdir, "_quarantine")
    guards = guards_from_stdout(proc.stdout)
    qfiles = listing(qd)
    rec = dict(
        case=case, expected_primary_guard=want_guard, expected_exit=want_exit,
        exit_code=proc.returncode, wall_seconds=wall,
        guards_tripped=guards,
        primary_guard_present=(want_guard in guards) if want_guard else None,
        collateral_guards=[g for g in guards if g != want_guard],
        canonical_H_exists=os.path.exists(h), canonical_S_exists=os.path.exists(s),
        tmp_listing=listing(tmpd), quarantine_listing=qfiles,
        reason_json_present=any(x.startswith("FAILED_REASONS_") for x in qfiles),
        input_mutation=input_note or None,
        injection=injection_from_stdout(proc.stdout) or None,
        stdout_sha256=hashlib.sha256(proc.stdout.encode()).hexdigest(),
        stderr_sha256=hashlib.sha256(proc.stderr.encode()).hexdigest(),
    )
    checks = []
    checks.append(("exit_code", rec["exit_code"] == want_exit))
    if want_guard is None:                    # positive control
        checks.append(("both_canonical_present",
                       rec["canonical_H_exists"] and rec["canonical_S_exists"]))
        checks.append(("no_guard_tripped", guards == []))
        checks.append(("tmp_clean", rec["tmp_listing"] == []))
        checks.append(("no_quarantine", rec["quarantine_listing"] == []))
        if os.path.exists(h):
            rec["h_sha256"] = sha256_of(h)
        if os.path.exists(s):
            rec["s_sha256"] = sha256_of(s)
    else:
        checks.append(("primary_guard_present", bool(rec["primary_guard_present"])))
        checks.append(("canonical_H_absent", not rec["canonical_H_exists"]))
        checks.append(("canonical_S_absent", not rec["canonical_S_exists"]))
        checks.append(("tmp_clean", rec["tmp_listing"] == []))
        checks.append(("reason_json_present", rec["reason_json_present"]))
    rec["failed_checks"] = [k for k, ok in checks if not ok]
    rec["verdict"] = "PASS" if not rec["failed_checks"] else "FAIL"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--src", type=str, default=FIXTURE)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case", type=str, default=None)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    if args.child:
        sys.exit(child(args.child, args.outdir, args.src))

    cases = list(CASES) if args.all else ([args.case] if args.case else [])
    if not cases:
        ap.error("pass --all or --case NAME")
    os.makedirs(OUTROOT, exist_ok=True)
    recs = [run_case(c) for c in cases]
    for r in recs:
        print(f"{r['verdict']:4}  {r['case']:30} exit={r['exit_code']} "
              f"guards={r['guards_tripped']} H={r['canonical_H_exists']} "
              f"S={r['canonical_S_exists']} tmp={r['tmp_listing']} "
              f"failed_checks={r['failed_checks']}")
    n_pass = sum(1 for r in recs if r["verdict"] == "PASS")
    summary = dict(
        harness=os.path.basename(__file__),
        harness_sha256=sha256_of(os.path.abspath(__file__)),
        generator=os.path.join("experiments", "journal_p0", "make_rho_instance.py"),
        generator_sha256=sha256_of(os.path.join(_JP0, "make_rho_instance.py")),
        fixture=os.path.relpath(FIXTURE, _ROOT), fixture_sha256=sha256_of(FIXTURE),
        fixture_is_tiny_and_synthetic=True,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python=sys.version.split()[0],
        rho=RHO, nurses_start=NURSES, seed=SEED, tag=TAG,
        cases=len(recs), passed=n_pass, failed=len(recs) - n_pass,
        epistemic_status="NEGATIVE CONTROLS. A guard that fires shows the guard is "
                         "wired. This is NOT a proof, NOT a validation, and NOT "
                         "evidence that any published instance or result is correct.",
        results=recs)
    print(json.dumps(dict(summary, results="<see json-out>"), indent=1))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=str)
        print(f"wrote {args.json_out}")
    sys.exit(0 if n_pass == len(recs) else 1)


if __name__ == "__main__":
    main()
