# Formal rerun recovery-04: exact objective-lattice amendment

Status: **pre-run design amendment**. No recovery-04 scientific result exists at the time this file is written.

Date: 2026-08-08

## Triggering evidence

Recovery-03 used SNAP `665829efa452bf7c8123ac5fb56efb7e18d28fbfa809e69f7b4518a11449452b`. R1 completed, but five R1b rho shards exited after their five seeds because the validator required the raw HiGHS dual bound to equal the incumbent within `1e-6`. The failing full-objective `(P,B)` pairs were `(700,698.6842105263167)`, `(275,274.999998)`, `(5100,5099.373076980055)`, `(1630,1629.9257322311669)`, and `(4265,4264.560351718618)`.

All five solver results were status 0; none was a timeout incumbent or OOM. Recovery-03 therefore remains FAIL and its `.R1b.partial` family is not reusable.

## Registered mathematical correction

Raw solver evidence remains immutable:

\[
P_f=P+C,\qquad B_f=B+C,\qquad
g_f=(P_f-B_f)/|P_f|.
\]

For a minimisation objective with attainable levels `offset + k*step`, the validator computes a dual bound rounded downward by eight floating-point ULPs and then

\[
L(B_f)=offset+step\left\lceil\frac{B_{f,safe}-offset}{step}\right\rceil.
\]

The solve is accepted as exactly optimal only when the incumbent lies on the registered lattice and `L(B_f)=P_f`. `INT_TOL=1e-6` is not used to round a dual bound.

Registered full-objective lattices:

| objective | step | offset | basis |
|---|---:|---:|---|
| U | 1 | 0 | sum of binary unserved indicators |
| W | 1 | 0 | integral effective durations and integral H_STD imply integral canonical minimum overtime |
| D | 1 | 0 | binary-assignment Hamming distance |
| Z | 5 | 0 | `gcd(W2=5,w4,W5=1000)=5` for every registered w4 |

Every U/Z/W/D solve must pass this certificate immediately after a code-0 result and before its value can constrain a later lexicographic stage. Lex3-W evidence is now persisted explicitly rather than appearing only in the stage chain.

The W/Z premise is not authorized by an output label. Before **any** MILP is
called, the runner re-hashes and parses the complete 30-scenario H/S family,
including rho levels that a shard will not solve. It derives the premise from
`H_STD` and every future task's `duration + TRAVEL_PER_TASK`, records H/S
receipts, and checks those receipts again immediately before the first solve
and after the last solve before publication. `H_STD` must be the manifest-bound
integer value 240. A fractional scenario later in the manifest therefore
aborts the whole shard before its first MILP; `--rho` cannot hide it.

HiGHS' native `mip_gap` is recomputed on its own objective scale. Every solve
now preserves `solver_fun_raw = res.fun` in both its role certificate and the
diagnostics sidecar, verifies it against `c @ res.x` at solve time, and later
replays

\[
g_{solver}=|F-B|/|F|
\]

from persisted `(F,B)` (with the exact zero-objective case handled
separately). This is intentionally distinct from the full-objective gap based
on read-back `(P,B,C)`: read-back assignment arithmetic can differ slightly
from `res.fun`, so `P/B` alone cannot reconstruct HiGHS' reported gap.

## Non-negotiable guards

- Status 1 and every timeout incumbent remain fatal even if a lattice happens to close.
- Formal mode rejects `--allow-fractional-durations`.
- Non-integral effective durations or a non-integral `H_STD` anywhere in the
  full registered family are rejected before the first solver call.
- Raw P, B, C, solver mip gap, and recomputed full-objective gap are never replaced by the closed lattice bound.
- Post-hoc validation independently replays the certificate from raw P/B/C,
  replays the solver-native gap from raw F/B, and rejects altered proof fields.
- Integer evidence is parsed exactly (no truncating `8.9` to `8`); boolean
  evidence accepts only canonical true/false or exact 0/1 values. Formal
  diagnostics must cover exactly the independently derived
  disruption/method/w4/objective-stage grid.
- `w_level_separation` is checked against the source-derived premise but can
  never create or upgrade that premise.
- The five recovery-03 partial shards are evidence only and cannot be merged or promoted under recovery-04.

## Changed SNAP surfaces

- `experiments/journal_p0/run_e4.py`
- `experiments/journal_p0/run_e4_design_manifest.json`
- `experiments/journal_p0/repro/run_all.py`
- `experiments/journal_p0/repro/build_snapshot.py`
- `tests/fixtures/epse4/run_e4_common_service.py`
- `tests/fixtures/epse4/test_r1b_lattice_certificate.py`
- this amendment

`experiments/journal_p0/s2_core.py` is intentionally unchanged because its raw bound and gap semantics are correct.

Recovery-04 must use a new content-addressed SNAP, a new results root, and the same isolated Docker controls. Only a completely promoted and hash-verified stage family may be used by the paper.
