# Recovery-06 pre-run addendum: attained structural lower bounds

Status: **superseding pre-run design addendum**.  This addendum changes no
scientific estimand, input instance, objective weight, solver option, seed, or
published result.  The earlier recovery-06 SNAP
`3db6d7de65f3320afbe4422d79e498bc73ddbba417ba7cc4a4b4ee2423b926d8`
was used only for a targeted smoke test and is rejected.  It produced no
admissible result artifact and must never be consumed by the manuscript.

## 1. Observed smoke failure

The isolated smoke container exited `1` with `OOMKilled=false` at
`rho=0.75, seed=1, absent=4, method=A1, w4=0, role=lex3_minW`.  The exact
assignment readback gave `P=0`, while HiGHS returned
`F=B=4.884981308350699e-14` and `C=0`.  This is approximately 220 times
`ulp(1)` and correctly violated the pre-registered 128-ULP dual-overshoot
premise.  The failure is numerical certificate evidence, not a scientific
result or a model-infeasibility result.

## 2. Analytic certificate

The executable now permits a second exact certificate method,
`attained_structural_lower_bound_v1`.  Its authority is an integer recomputation
from the exact feasible assignment, never the floating expression `P+C`:

- `U` is a count of nonnegative unserved indicators;
- `W` is the sum of canonical `max(0, integer load - integer H_STD)` overtime;
- `D` is the nonnegative assignment-deviation count; and
- `Z = 5 W + w4 D + 1000 U`, with every registered `w4` nonnegative.  At the
  fixed common-service equality, `Z >= 1000 U_common`.

If the exact integer objective equals the registered integer lower bound, then

`lower_bound <= OPT <= exact_feasible_objective = lower_bound`,

so optimality follows without the solver dual.  Raw `P`, `B`, `C`, `F`, solver
gap, status, and stop reason remain unchanged.  `F` and `B` must each remain
strictly within half of one declared objective-lattice step from `P`; this is a
non-tunable corruption check, not a lower-bound premise.  Status 1, timeout
incumbents, non-finite evidence, failed exact feasibility, adjacent/half-cell
diagnostics, negative weights, or non-exact attainment still fail closed.

When the exact objective does not attain the analytic lower bound, the existing
`objective_lattice_ceiling_v3` path remains unchanged, including its 128-ULP
dual premise.  The constant is not enlarged to 256 or 512.

## 3. Required evidence and acceptance

Every certificate persists the exact integer objective, the registered
structural lower bound, the lower-bound source, and immutable raw solver
evidence.  Post-write replay uses the round-trip decimal parser, independently
recomputes the exact objective and floor from the scientific row, and derives
the deviation constant again from the re-parsed H/S inputs; persisted proof
fields are comparison evidence, never authority.  Negative controls must show
that the exact smoke tuple passes only through structural attainment, while a
nonzero U/W/D objective claiming the zero floor, a Z objective not equal to its
fixed-U floor, near-floor values, coordinated P/C cancellation, field
tampering, adjacent levels, and non-admissible statuses fail.

`e4_solver_diagnostics.csv` is a non-authoritative operational mirror.  The
final external seal must nevertheless require its complete coordinate/stage
axis and exact cross-link every mirrored proof field to the authoritative raw
role certificate; deletion or coordinated relabeling therefore fails the
formal acceptance chain.

No formal recovery-06 result exists until a new deterministic SNAP is built,
the targeted real cell and complete rho-0.75 shard pass in isolated Docker,
the six-shard formal run exits zero without OOM, both owner merges and R5
complete, and the external verifier, PASS report, and final single-exclusion
ledger all pass.
