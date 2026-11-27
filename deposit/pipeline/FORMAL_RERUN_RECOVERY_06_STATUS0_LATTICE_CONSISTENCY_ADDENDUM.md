# Formal rerun recovery-06: status-0 lattice-consistency addendum

Date: 2026-08-09  
Status: **pre-run executable contract; no scientific result is authorized by this document**

Precedence: for formal integral-lattice R1b roles that do not attain a
registered analytic structural lower bound, this addendum replaces
`objective_lattice_ceiling_v3` as the certificate authority with
`solver_status0_lattice_consistency_v1`.  It does not enlarge the historical
128-ULP dual envelope, does not weaken any exact-readback requirement, and does
not change the legacy non-lattice fallback.  The independently sufficient
`attained_structural_lower_bound_v1` method remains authoritative when its
exact-attainment premises hold.

## 1. Rejected smoke-02 evidence

The content-addressed SNAP
`49811ff8ab15cd8c23a9974037997e0c9d25b52c80e3175b77963a3636549a5b`
was exercised in the isolated container
`dchhr-49811ff8ab15-r1b075-smoke-02`.  The container exited 1, was not
OOM-killed, emitted no result files, and is rejected for all scientific and
manuscript use.  Seeds 1--4 completed inside that failed smoke, but remain
non-consumable; seed 5 aborted at:

- rho=0.75, seed=5, absent=4, method=A2, w4=0;
- role=`lex4_minD`, objective stage D;
- raw linearized values `P=-54.0`,
  `B=-53.999999999994884`, `F=-53.999999999994884`;
- immutable additive constant `C=155.0`, hence `P_full=101.0`;
- exact assignment-derived full-objective integer `101`;
- the previously supplied structural floor was 0 with source
  `nonnegative_assignment_deviation`;
- directed-up `max(0,B-P)=5.115907697472722e-12`, exceeding the unchanged
  128-ULP envelope `e=3.637978807091713e-12`.

The failed context did **not** log or persist X=`c @ res.x`, and its failure
JSON did not contain the raw solver mip gap.  Neither value may be invented or
attributed to the remote failure evidence.  A subsequent read-only diagnostic
recomputation found 186 future tasks and `U_common=0`; these are diagnostic
recomputations, not fields captured in the failed remote stderr and not proof
that the failed row passes the replacement contract.

The failure shows that the v3 registered envelope cannot authorize this row.
The envelope remains 128 ULP and is not enlarged.  Because v3 cannot close the
observed diagnostic B to P under its own premise, it is no longer a formal
authority for non-attainment integral-lattice roles.

## 2. New formal method and prerequisites

The new method name is exactly
`solver_status0_lattice_consistency_v1`.  It is admissible only when every one
of the following conditions holds for the current role:

1. SciPy/HiGHS returns status code 0 named `Optimal`, `proven_optimal=true`,
   no timeout incumbent is accepted, and the registered stop reason and raw
   status evidence are internally consistent.
2. The canonical assignment readback is exactly feasible for every active
   structural and lexicographic row, and its objective P+C is an exact integer
   on the independently registered objective lattice.
3. Before adding C, the live returned vector satisfies, with
   `up(d)=nextafter(abs(d), +infinity)`, both
   `up(X-P)<step/2` and `up(F-X)<step/2`.
4. The persisted and independently round-trip-replayed raw values satisfy all
   three pairwise conditions `up(F-P)<step/2`, `up(B-P)<step/2`, and
   `up(F-B)<step/2`.
5. The persisted raw solver gap is finite and exactly consistent, under the
   registered gap arithmetic, with the immutable raw F and B.  The recomputed
   full-objective gap (or its registered zero-denominator reason) is likewise
   exactly consistent with immutable P, B, and C.
6. Raw P, B, C, F, the raw solver gap, and the full-objective gap evidence are
   never clamped, rounded to one another, replaced, or rewritten.

All inequalities are strict.  Equality at half a lattice step, an adjacent
cell, non-finite evidence, status 1, a timeout incumbent, a failed exact
readback, inconsistent gap evidence, or a mutation of any raw field fails
closed.  The checks before C prevent cancellation from hiding disagreement.

This method establishes that the code-0 solver evidence and an exactly
feasible assignment readback identify one unique attainable objective-lattice
cell.  Its claim is precisely:

> Under the registered status-0 Optimal solver contract and exact-feasible
> lattice readback, the returned incumbent, live vector reconstruction, and
> persisted primal/dual diagnostics are mutually consistent with the same
> unique attainable objective level.

It is a conditional solver-status consistency certificate.  It does **not**
derive an independent exact lower bound, does not prove the floating-point
solver correct, and therefore records `proves_exact_optimality=false`.

## 3. Legacy evidence-field semantics

The scientific schema retains historical field names so a method change cannot
silently reorder or delete evidence columns.  For
`solver_status0_lattice_consistency_v1` their meanings are exactly:

- `lattice_bound_ulps=0` and `lattice_boundary_margin=0`;
- `lattice_safe_dual_bound_full_objective` stores immutable raw `B_full=B+C`
  only as a diagnostic.  Despite the legacy word `safe`, it is not a safe
  bound, a rounded bound, or proof authority;
- `lattice_closed_dual_bound_full_objective` stores the exact P lattice center
  `P_full=P+C`.  Despite the legacy word `closed`, no dual closure was
  performed;
- `lattice_primal_level_full_objective` also stores that exact P lattice
  center;
- `exact_certificate_lower_bound_source` is exactly
  `solver_status0_unique_lattice_cell_no_exact_bound`;
- the raw B_full diagnostic and P lattice center must be recomputed from landed
  raw bytes and compared exactly during post-write validation, owner merge,
  and external verification.

The unchanged 128-ULP value may remain in lineage documentation and v3 hostile
tests, but it is not used, enlarged, or emitted as the bound of the new method.

## 4. Structural attainment and non-lattice scope

`attained_structural_lower_bound_v1` remains an independently sufficient exact
optimality certificate.  It requires an exact feasible integer readback equal
to its correctly registered analytic lower bound and continues to record
`proves_exact_optimality=true`; raw solver evidence remains immutable
diagnostic evidence and must remain in the same strict lattice cell.

Historical non-lattice fractional fixtures retain their prior exact-raw-bound
fallback and eight-ULP F/X identity rule.  Formal R1b forbids fractional
effective durations and cannot use that fallback.  The new status-0 method
does not relax either branch.

## 5. Replay, negative controls, and release

Preflight must cover the observed smoke-02 tuple without changing any recorded
value; strict half-step equality and adjacent-cell mutations for X, F, or B;
large-C cancellation; status/time-limit mutations; infeasible exact readback;
raw and full-gap mutations; and tampering with the legacy safe/closed field
semantics.  The manifest and loader must exact-pin the method, claim wording,
required conditions, legacy-field meanings, `proves_exact_optimality=false`,
and the separate structural-attainment truth value.

Fixture success is regression evidence only.  Release still requires a new
content-addressed SNAP, fresh isolated smoke, fresh formal R0/R1/R1b/R5 chain,
exit 0, no OOM, both owner merges, complete artifacts and stage hashes,
external verification, final report, and single-exclusion ledger.  No output
from recovery-05 or either rejected recovery-06 smoke may be reused.
