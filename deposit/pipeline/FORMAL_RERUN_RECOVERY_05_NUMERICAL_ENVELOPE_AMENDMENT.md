# Formal rerun recovery-05: conditional numerical-envelope amendment

Status: **pre-run design amendment**. No recovery-05 scientific result exists at the time this file is written.

Date: 2026-08-08

## Triggering evidence

Recovery-04 used SNAP `026a4f146ad6c639f7ce6f4e967e2b17f74bc1eef8255daa4a85bb1b30c6b0dd`. Its R1 family completed, but R1b aborted for rho 0.65, 0.75, 0.85, and 0.95. Every failing solve had SciPy/HiGHS status 0, an incumbent, no timeout, and no OOM. The raw linearized `(stage,P,B,C,step)` contexts were:

| rho | stage | P | B | C | step |
|---:|:---:|---:|---:|---:|---:|
| 0.65 | Z | -390 | -389.9999999999995 | 815 | 5 |
| 0.75 | W | 0 | 9.769962616701367e-15 | 0 | 1 |
| 0.85 | W | 0 | 1.8010456216615096e-14 | 0 | 1 |
| 0.95 | Z | 1125 | 1125.0000000000125 | 0 | 5 |

The recovery-04 certificate subtracted eight ULPs. In these four contexts the reported dual bound remained microscopically above the exact feasible readback after that subtraction, so upward lattice closure jumped to the next level and failed closed. Recovery-04 remains a failed attempt; none of its partial R1b or downstream outputs may be promoted.

## Registered recovery-05 rule

Raw solver evidence is immutable:

\[
F=\texttt{res.fun},\quad P=\text{exact assignment readback},\quad
B=\texttt{mip\_dual\_bound},\quad P_f=P+C,\quad B_f=B+C.
\]

Structural identities, coefficient integrality, and persisted numeric replay retain their original eight-ULP allowance. A separate dual-bound numerical envelope is registered only for the lattice certificate:

\[
S=\max(1,|P|,|B|,|C|,|P_f|,|B_f|,s,|o|),
\qquad e=128\,\operatorname{ulp}(S),
\]

where `s` and `o` are the source-authorized lattice step and offset. Before closure, every role must satisfy all of the following:

1. status code is exactly 0, an incumbent exists, and all operands are finite;
2. the directed-up raw discrepancy `up(|F-P|)` is at most `2e-6` in absolute objective units;
3. the directed-up raw signed overshoot `up(max(0,B-P))` is at most `e`;
4. `e + 2e-6 < s/2`.

The two differences are computed from raw linearized values. They are not computed after adding `C`, because large-constant cancellation could hide their sign or magnitude.

The conservative closure is

\[
B_{f,\mathrm{safe}}=\operatorname{down}(B_f-e),
\]

\[
q_{\mathrm{safe}}=\operatorname{down}
\left(\frac{B_{f,\mathrm{safe}}-o}{s}\right),\qquad
L=o+s\lceil q_{\mathrm{safe}}\rceil.
\]

The certificate passes only if the exact feasible readback is on the independently established lattice and `L` equals its lattice level. No clamp is permitted. Neither `P`, `B`, `C`, `F`, HiGHS' native gap, nor the recomputed full-objective gap is replaced by `L`.

## Scope of the claim

This is a **conditional numerical-solver certificate**, not an unconditional exact-arithmetic proof of HiGHS. It excludes the next lower attainable objective level under all registered premises: the source-derived lattice, exact feasible assignment readback, code-0 solver result, and the premise that the reported dual lower bound's upward numerical error is no larger than `e`.

The 128-ULP envelope is a pre-recovery-05 registered power-of-two bound chosen after forensic observation of recovery-04, whose largest failing overshoot was about 81 ULP at the registered scale. It is not represented as a HiGHS theorem. Every recovery-05 row must remain inside it; a larger overshoot aborts rather than widening the envelope. The `2e-6` `F/P` condition is only a solver-fun/readback consistency gate and never moves the bound.

## Mandatory hostile and replay controls

- replay all four recovery-04 contexts and all five recovery-03 lattice regressions;
- reject a true `P-step` bound, the nextafter-above boundary case, an adjacent/common-mode solver shift, and any overshoot outside the 128-ULP envelope;
- reject equality at either directed-up guard boundary, nonfinite evidence, a nonpositive step, and any envelope spanning half a step;
- reject large positive or negative `C` when cancellation or ULP scale makes the certificate unsafe;
- recompute both solver-native `|F-B|/|F|` and full-objective `(P_f-B_f)/|P_f|` gap evidence;
- independently replay method, step, offset, 128-ULP envelope, safe bound, scaled closure, incumbent level, raw evidence, and strict scalar types;
- reject any status-1 or timeout incumbent regardless of lattice closure.

## New SNAP and rerun scope

Recovery-05 must use a new deterministic content-addressed SNAP and a new isolated Docker results root. Because executable and verifier semantics changed, the complete registered chain `environment -> preflight -> R0_generator -> R1 -> R1b -> R5` must run under that one SNAP. Recovery-04's completed R1 may be retained only as forensic comparison; it cannot enter the recovery-05 owner merge, stage hashes, report, or ledger.

Changed SNAP surfaces are:

- `FORMAL_RERUN_RECOVERY_05_NUMERICAL_ENVELOPE_AMENDMENT.md`;
- `experiments/journal_p0/run_e4.py`;
- `experiments/journal_p0/run_e4_design_manifest.json`;
- `experiments/journal_p0/repro/build_snapshot.py`;
- `tests/fixtures/epse4/test_r1b_lattice_certificate.py`.

The external recovery-05 verifier is intentionally stored outside the SNAP. Its SNAP ID, member count, member hashes, report names, and ledger names are pinned only after deterministic snapshot construction.
