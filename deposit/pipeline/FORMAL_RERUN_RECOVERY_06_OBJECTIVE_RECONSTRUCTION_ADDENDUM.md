# Formal rerun recovery-06: live objective reconstruction addendum

Date: 2026-08-09  
Status: **pre-run executable contract; no scientific result is authorized by this document**

Precedence: this addendum supersedes only the former fixed eight-ULP F/X
identity gate for lattice objectives.  It is additive to, and does not replace,
the recovery-06 same-level/round-trip and structural-lower-bound amendments.

## 1. Rejected smoke evidence

The content-addressed SNAP
`221aef66bee99904d6c9cfba882a2f96ca9956c96e62ac263656335636aca445`
was exercised only as an isolated R1b rho-0.75 smoke.  Its container exited 1,
was not OOM-killed, emitted no result artifact, and is rejected for all
scientific or manuscript use.

The failure context recorded:

- coordinate: rho=0.75, seed=1, absent=28, method=A2,
  role=`lex2_minZ_at_Ucommon`, objective stage Z;
- solver-reported raw linearized objective F:
  `-100.0000000000013`;
- live NumPy reconstruction X=`c @ res.x`:
  `-100.00000000000114`;
- absolute F/X difference: `1.5631940186722204e-13`, equal to 11 ULP at
  scale 100;
- former fixed eight-ULP identity margin:
  `1.1368683772161603e-13`.

The failed exception did **not** record w4, exact readback P, or raw dual B.
Those fields must therefore not be attributed to the remote failure log.  The
hostile test anchors the two logged F/X values to an explicitly synthetic,
compatible Z-lattice center; it is a guard regression, not reconstruction
evidence for the failed solve.  Because B was not captured, no claim is made
that the next certificate will pass: a new smoke must execute the complete
F/P, B-envelope, and lattice-closure gates.

## 2. Cause and rejected alternatives

The old F/X check used a dimension-insensitive eight-ULP bound.  It is suitable
for small structural identities but is not a registered bound for two
different objective summation paths under cancellation.  The failure therefore
does not establish a different attainable objective level.

The shared `LATTICE_STRUCTURAL_ULPS=8` constant is **not changed**.  Increasing
it globally would also weaken unrelated coefficient, lattice, and replay
identities.

A Higham-style gamma-n dot-product envelope is also not adopted as a formal F
bound.  The SNAP does not establish that HiGHS constructs `res.fun` from the
returned vector with a specified number, order, scaling, precision, or rounding
mode of IEEE-754 operations.  Applying a local NumPy error theorem to that
unknown internal path would add an unverified premise.  Such a calculation may
be diagnostic, but it is not proof authority.

## 3. Replacement live gate

Let all quantities below be raw linearized values before adding the deviation
constant C:

- F = the immutable solver-reported `res.fun`;
- X = the live reconstruction `c @ res.x` from the returned vector;
- P = the assignment-derived exact readback used by the objective certificate;
- s = the positive objective-lattice step declared by executable source and
  design manifest;
- `up(d) = nextafter(abs(d), +infinity)`.

For a lattice objective, execution must satisfy both new inequalities:

1. `up(X-P) < s/2`;
2. `up(F-X) < s/2`.

The existing certificate independently retains its F/P requirement:

- attained structural lower bound: `up(F-P) < s/2`;
- lattice-v3: `up(F-P) + registered_dual_envelope < s/2`.

Thus F, X, and P are pairwise bound to one unique objective lattice cell.  This
is a scalar objective-level consistency statement, not a coordinate-wise proof
about every component of `res.x`.  Exact equality
to half a step is rejected.  The checks occur before C is added so cancellation
by a large positive or negative constant cannot hide disagreement.  NaN,
infinity, a non-positive step, an adjacent cell, and opposite-side separation
of at least half a step all fail closed.

For historical non-lattice fractional fixtures only, the original eight-ULP
F/X identity check remains.  Formal R1b forbids fractional effective durations
and therefore cannot enter that fallback.

## 4. Evidence and persistence boundary

X and the raw solver vector are live execution evidence and are not added to
the scientific CSV schema.  Persisting X without the full raw vector would not
make it independently recomputable and would enlarge five role schemas without
adding verification authority.  The final external verifier may accept this
gate only by pinning and validating the executable SNAP member; it must not
describe X as post-hoc replayed evidence.

Raw P, B, C, F, solver gaps, exact assignment readback, structural floors, and
lattice certificates remain unchanged.  The replacement gate neither clamps a
dual bound nor rewrites an incumbent.

## 5. Mandatory negative controls and release condition

Before a new formal attempt, the SNAP must demonstrate that:

- the logged 11-ULP F/X regression passes while remaining far inside the
  strict half-step cell;
- small F and X rounding on opposite sides of P passes only when every
  pairwise separation is strictly below half a step;
- exact half-step, adjacent-cell, coordinated wrong-cell, non-finite, and
  large-C cancellation cases fail;
- the non-lattice branch still enforces its unchanged eight-ULP identity;
- all prior objective-certificate, exact-readback, round-trip parsing, shard
  merge, and external-verifier hostile tests still pass.

Success of these fixtures is regression evidence only.  Scientific release
still requires a new content-addressed SNAP, a fresh isolated smoke, a fresh
formal R0/R1/R1b/R5 chain, exit 0, no OOM, owner merges, complete artifacts,
stage hashes, external verification, final report, and a single-exclusion
ledger.  No recovery-05 or rejected recovery-06 artifact may be reused.
