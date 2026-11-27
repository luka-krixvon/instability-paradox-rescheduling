# Formal Rerun Recovery Amendment

## 1. Status and scope

This amendment records the disposition of formal rerun attempt 1 and the only
authorised recovery changes. It does not validate, accept, or publish any result
from the failed attempt. It does not change the mathematical estimands, designed
axes, epsilon grid, solver presolve policy, or admissible solver-status set.

## 2. Attempt 1 failure record

- **Attempt:** formal rerun attempt 1
- **SNAP ID:** `5502de57fd1b6f2d3c414734b8908582d1e7bc1a958c2e9c1777d1aabb47931f`
- **Stage:** R1
- **Failure key:** `rho=0.95`, `seed=1`, `absent=8`
- **Observed failing solve:** frontier `lex2_minW`
- **Observed solver status:** `1`
- **Registered per-solve time limit:** `300 s`
- **Regime:** pending / not recoverable from the presently confirmed failure message
- **Epsilon:** pending / not recoverable from the presently confirmed failure message
- **Disposition:** **REJECTED / NOT CONSUMED**

Status 1 is outside R1's pre-registered optimal-only admissible set. No
status-1 incumbent, partial scientific artifact, completed sibling shard, or
other output from attempt 1 may be reused in an accepted formal result family.
Attempt 1 must remain preserved as failure evidence and must not be overwritten
or relabelled as a successful run.

Final execution facts from the preserved container and result root:

- **Container final state:** `exited`; the root operator deliberately stopped
  this already-rejected attempt to release its two remaining CPU workers.
- **Container exit code:** `143` (operator termination after the independent
  R1 status-1 failure had already been preserved).
- **Container OOMKilled state:** `false`.
- **Started / stopped (UTC):** `2026-08-01T05:38:36.086282245Z` /
  `2026-08-01T06:35:42.455159598Z`.
- **Sibling-shard final statuses:** rho 0.45, 0.55 and 0.65 atomically landed;
  rho 0.75 and 0.85 remained partial when the rejected attempt was stopped;
  rho 0.95 failed with solver status 1. None is consumable.
- **rho 0.95 failure-log SHA-256:**
  `1de896e9253d93f78280c45c5a0ecd8657eccafa9f9aac5451f0c4724f6904e6`.
- **Preserved result tree:** 102 files; aggregate
  `40501678ecdc6e6ea5edfe47bc55971b9ce94280ffeeeb8ca936b5718c0f89e5`,
  computed as SHA-256 of the byte stream produced by sorting the relative
  NUL-delimited file paths and applying `sha256sum` to every file.

The operator termination does not alter the rejection decision: the decisive
status-1 failure occurred and was logged before the stop.

## 3. Authorised recovery changes

The recovery revision is limited to all of the following:

1. Change the uniform R1 and R1b per-solve time ceiling from `300 s` to
   `1800 s`. The same `1800 s` ceiling must be declared in the applicable
   design manifests, executable runtime contracts, command receipts, solver
   diagnostics, and independent-oracle evidence wherever required.
2. Set the overtime slack-variable upper bound exactly to
   `H_MAX - H_STD`. This is a bounding correction only; reported overtime must
   continue to be recomputed from the integral assignment under the existing
   definitions and tolerances.
3. Improve failure logging so that every failed solve identifies, at minimum,
   the stage, `rho`, seed, absent caregiver, regime, epsilon when applicable,
   lexicographic substage, solver status code and message, registered time
   limit, and presolve setting. The enclosing immutable stage log and execution
   receipt bind that event to the SNAP ID, container, and attempt identifier.

No other implementation, configuration, model, or reporting change was
authorised by the attempt-1 amendment.  The later exact epsilon-equivalence
recovery is separately recorded in Section 7 and supersedes this sentence only
to that explicitly bounded extent.

## 4. Scientific invariants that remain unchanged

The recovery must preserve all of the following exactly:

- the R1 and R1b mathematical estimands and objective definitions;
- the common-service construction and disruption eligibility rule;
- the complete registered rho/scenario/disruption axes;
- the shared, contiguous integer epsilon-grid construction and coverage rules;
- the lexicographic objective order and existing numerical tolerances, except
  for the explicitly stated exact overtime slack upper bound;
- `presolve=false`;
- optimal-only acceptance, with solver status code `0` as the only admissible
  and proven-optimal status for R1 and R1b;
- rejection of every time-limit incumbent and every incomplete or skipped row;
- independent-oracle coverage and agreement requirements;
- canonical-assignment policy and all existing provenance/hash requirements.

The increased ceiling is not permission to accept status 1, weaken a validator,
skip a designed cell, retry with unregistered options, or select successful
outputs from attempt 1.

## 5. Mandatory new formal run

Because this amendment changes SNAP-covered code and design-contract bytes, the
recovery requires a **new content-addressed SNAP ID**. The old SNAP ID must not
be attached to any amended execution or artifact.

The accepted recovery must use a fresh result root and run the complete formal
chain from **R0 through R5**, including all six R1 shards and all six R1b shards,
under the new SNAP. In particular:

1. rebuild and independently verify the new deterministic SNAP archive;
2. record the image and container identities and effective isolation settings;
3. rerun environment and preflight gates;
4. rerun R0 and verify the complete generated family;
5. rerun **all six** R1 shards, perform the owner merge, and atomically promote
   only after every validation and hash gate passes;
6. rerun **all six** R1b shards, perform the owner merge, and atomically promote
   only after every validation and hash gate passes;
7. rerun R5 only after the completed R1 and R1b gates pass;
8. build the final execution receipt and checksum ledger last, with attempt 1
   explicitly marked `REJECTED / NOT CONSUMED` and no cross-attempt artifact
   lineage.

No shard or scientific output from attempt 1 may be copied, merged, relabelled,
or otherwise carried into the new-SNAP result family.

## 6. Remaining human blocker

Dataset provenance, lawful acquisition, privacy/synthetic status, licence, and
redistribution permission remain **HUMAN-BLOCKED**. A technically successful
R0-to-R5 rerun does not resolve this blocker and does not by itself authorise
submission or public release of the underlying dataset.

## 7. Exact epsilon-equivalence recovery after SNAP `fb0c3fa1...`

The next isolated attempt used SNAP
`fb0c3fa1aaa796420201699e0832ad41ce8cc1a7169cf6023e321233d2588cfb`.
Its preflight and R0 passed, and R1 rho shards 0.45, 0.55, and 0.65 landed.
The rho 0.85 shard then correctly failed fast at
`seed=1, absent=25, regime=A2, epsilon=69, lex2_minW`: SciPy/HiGHS returned
status 1 after the registered 1800-second limit.  The operator preserved the
attempt and stopped its two remaining workers; no output from that SNAP is
consumable by a later SNAP.

The failure exposed a mathematically redundant solve, not permission to accept
a time-limit incumbent.  At fixed common service level, every integral
assignment satisfies

    D = |xbar| + (n_tasks - U_common) - 2 |x intersect xbar|.

Consequently all attainable deviations have parity
`(|xbar| + n_tasks - U_common) mod 2`.  For the failing coordinate,
`n_tasks=186`, `|xbar|=178`, and `U_common=0`, so D is even and the integer
feasible sets under `D <= 68` and `D <= 69` are identical.  The code-0 epsilon
68 solve had already succeeded; re-solving the looser but integer-equivalent
epsilon 69 formulation only weakened its LP relaxation under the required
`presolve=false` policy.

The authorised exact recovery therefore:

1. independently derives this parity in production and oracle code;
2. strengthens each MILP RHS to the largest parity-compatible integer not
   exceeding the requested epsilon;
3. solves each new effective budget only to code-0 Optimal;
4. reuses an anchor only when fixed-D parity proves identical integer feasible
   sets, or when the already code-0 unconstrained lexicographic W-floor point is
   feasible under the requested epsilon;
5. retains every requested integer epsilon row and explicitly persists the
   effective budget, anchor, execution/reuse flag, and proof rule; and
6. keeps synthetic proof-reuse diagnostics distinct from actual solver calls.

This transformation changes neither the requested epsilon grid nor any
integer-feasible assignment, objective, lexicographic order, time limit,
presolve setting, or admissible status.  A non-code-0 anchor still aborts the
stage.  Production and the independently implemented oracle must derive the
same effective budgets and agree on every requested `(U,W,D)` point.  The
change requires another new content-addressed SNAP, a fresh result root, and a
complete rerun; no shard from `fb0c3fa1...` may be mixed into it.
