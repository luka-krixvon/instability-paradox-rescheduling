# Formal rerun recovery-06: same-lattice-level and binary64 round-trip amendment

Status: **pre-run design amendment**. No recovery-06 scientific result exists at the time this file is written. Fixture and preflight success are regression evidence only; they are not a formal result, proof of scientific truth, or permission to consume a failed attempt.

## 1. Recovery-05 is a failed, non-consumable attempt

Recovery-05 used SNAP `43dd4664c534808c2a5aefa64137c269f9948a2345221edb8a03672880df1fd1`. Its isolated container exited with code 2 and `OOMKilled=false`. R1 completed, but R1b did not complete: rho 0.45 and 0.55 finished only inside the failed `.R1b.partial` stage, rho 0.65, 0.75, 0.85, and 0.95 failed, the R1b owner merge never started, and R5 never started. No recovery-05 R1b shard, partial aggregate, downstream output, PASS report, or ledger is admissible input to recovery-06 or to the paper.

The preserved failures identify two distinct defects:

1. At rho 0.65, post-write certificate replay row 306 compared a persisted full-objective gap of `5.128202928972314e-09` with `5.128197244630428e-09` obtained after pandas' default decimal-to-binary64 conversion. The stored decimal token had not been replayed to the same binary64 value. This is a parser/replay defect; it is not permission to alter the raw gap.
2. At rho 0.75, 0.85, and 0.95, the lex3 W certificates respectively had `(P,F,B,C)` equal to `(112,111.99999619999844,111.99999619999844,0)`, `(173,172.99999619999838,172.99999619999838,0)`, and `(132,131.99999619999818,131.99999619999818,0)`. Their directed-up `abs(F-P)` values were about `3.8e-06`, so the recovery-05 fixed `2e-6` readback test aborted even though every value remained far inside the same unit-width W lattice cell.

These facts do not establish that a larger fixed absolute tolerance would be valid. The rejected response is to replace `2e-6` by a larger number or to retune HiGHS until these three rows happen to pass.

## 2. Registered recovery-06 certificate

The formal certificate method is `objective_lattice_ceiling_v3`. For a code-0 minimization role, let:

- `P` be the objective recomputed from the exact feasible assignment readback;
- `F` be the raw solver incumbent objective;
- `B` be the raw solver dual bound;
- `C` be the immutable additive objective constant;
- `step > 0` and `offset` define attainable full-objective levels `offset + k*step`.

Before any certificate is constructed, the canonical readback must be an exact
feasible point for the current lexicographic stage. In addition to assignment,
integrality, service-indicator, and hard-capacity checks, the rebuilt values
must satisfy every active refinement row: exact `U == u_eq`, exact
`Z <= z_le + LEX_TOL`, and exact `W <= w_le + LEX_TOL`. Here `LEX_TOL` is the
already registered row slack; this amendment does not add another tolerance.

Define `P_full=P+C`, `B_full=B+C`, and the registered dual envelope

```text
S = max(1, |P|, |B|, |C|, |P+C|, |B+C|, step, |offset|)
e = 128 * ulp(S).
```

All inputs must be finite. The existing eight-ULP structural identity checks remain unchanged, and `P_full` must lie on the declared objective lattice. The executable then computes from the raw linearized values, before adding `C`,

```text
delta = nextafter(abs(F-P), +infinity)
dual_above_primal = nextafter(max(0, B-P), +infinity).
```

It must establish both

```text
delta + e < step/2
dual_above_primal <= e.
```

The first inequality is a strict same-lattice-level guard: the solver incumbent and exact feasible readback must identify the same attainable objective level. It is not an absolute readback tolerance and does not authorize an adjacent lattice cell. Equality at `step/2` fails closed. The second inequality retains the recovery-05 dual-bound policy unchanged at 128 ULP.

Only after those gates pass is the dual bound closed on the lattice:

```text
safe        = nextafter(B_full - e, -infinity)
scaled_safe = nextafter((safe-offset)/step, -infinity)
closed      = offset + step*ceil(scaled_safe).
```

`closed` must equal the exact primal lattice level. No clamp, replacement, rounding-to-incumbent, or rewriting of `P`, `B`, `C`, `F`, solver gap, or full-objective gap is permitted. The result remains a conditional numerical-solver certificate, not an unconditional theorem about floating-point HiGHS.

## 3. Exact binary64 replay

Every audited CSV read used to validate or rebuild certificate-bearing artifacts must use pandas `float_precision="round_trip"`. The authority is the persisted decimal token interpreted back to the same binary64 value that produced it. A default parser that selects a neighbouring binary64 value is not admissible evidence for an exact proof-field comparison.

Round-trip parsing does not relax equality and does not repair a changed token. Post-write validation and owner merge must independently recompute every certificate from the landed raw bytes and still require exact agreement with every immutable proof field.

## 4. Solver policy is unchanged

Recovery-06 does not tune the solver to fit the failed rows. Formal R1b retains exactly:

```text
presolve = false
time_limit = 1800.0 seconds
admissible status = 0 (Optimal) only
retry policy = none
```

No `mip_feasibility_tolerance`, `primal_feasibility_tolerance`, or larger fixed `F/P` tolerance is introduced. Status 1, timeout incumbents, missing incumbents, nonfinite evidence, and any envelope or lattice violation continue to abort.

## 5. Required negative controls

Before a formal launch, preflight must exercise at least the following cases:

1. The exact recovery-05 rho-0.65 persisted decimal row replays to identical binary64 proof values under `float_precision="round_trip"`, while a default-parser mutation is detected rather than silently accepted.
2. The exact rho-0.75, rho-0.85, and rho-0.95 W contexts pass `objective_lattice_ceiling_v3` without changing any raw evidence or the 128-ULP envelope.
3. `delta+e == step/2`, `delta+e > step/2`, and an adjacent solver lattice cell all fail closed even with status 0.
4. Disagreement hidden by adding a large positive or negative `C` fails because `F-P` and `B-P` are checked before the constant is added.
5. A dual overshoot beyond 128 ULP, a lower attainable dual level, a non-lattice primal, nonfinite input, or a nonpositive step fails closed.
6. Tampering a persisted certificate field or replaying a certificate through a non-round-trip parser fails closed.
7. A mocked status-0 incumbent whose canonical exact readback violates any of
   `u_eq`, `z_le`, or `w_le` fails before `SolverResult` construction and before
   lattice closure.

Passing these controls establishes only that the implementation enforces the registered rule on the tested cases.

## 6. New SNAP and exact stage replay

Recovery-06 must use a new deterministic content-addressed SNAP, a new isolated Docker container, and a new empty results root. The entire registered chain must execute under that one SNAP:

```text
environment -> preflight -> R0_generator -> R1 -> R1b -> R5
```

This is an exact stage replay, not a resume. Recovery-05's completed R1 and its two completed R1b shards remain forensic comparison only. All six R1 shards, all six R1b shards, both owner merges, R5, and every stage hash must be regenerated. R5 may start only after the completed R1b owner merge has passed.

The external recovery-06 verifier remains outside the SNAP and is pinned only after deterministic snapshot construction. Only container exit 0, no OOM, exact SNAP and image binding, complete stage inventories, owner merges, independent certificate replay, external verifier PASS, a published PASS report, and the final single-exclusion ledger authorize consumption. Until then, no recovery-06 value may enter the manuscript.

## 7. SNAP-bound design inputs

The recovery-06 SNAP must include and hash at least:

- this amendment;
- `FORMAL_RERUN_RECOVERY_05_NUMERICAL_ENVELOPE_AMENDMENT.md` as superseded lineage;
- `experiments/journal_p0/run_e4.py`;
- `experiments/journal_p0/run_e4_design_manifest.json`;
- the recovery-06 lattice and replay negative-test sources;
- the unchanged formal orchestration and snapshot builders.

The in-archive SNAP identifier remains null to avoid circular self-hashing. The closed archive SHA-256 becomes the runtime `--snap-id` and the sole authority for the new attempt.
