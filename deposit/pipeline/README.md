# Runnable pipeline — re-solve the MILPs, do not just recompute from them

The deposit one directory up lets you **recompute** every reported number from sealed stage
outputs. This directory is the other half: the **model, the scenario family and the solver
pipeline**, so you can re-solve the MILPs yourself and produce those stage outputs from
scratch.

> *Auditing Stability Controls in Dynamic Workforce Reassignment: Assignment-Only versus
> Displacement Recovery in Home Healthcare* — Teng-Ruei Chen, Chun-Cheng Lin.
> XSci preprint, DOI [10.66977/xsci.2611.0042](https://doi.org/10.66977/xsci.2611.0042)

It exists because two of the three reviews of v2 made the same criticism, and it was right:
the deposit shipped no model, no scenario generator and no base instance, so a reader could not
re-solve a single MILP. Two of those three are now here. The third — the base instance — is
not, and most of this file is about what that costs and what we changed to work around it.

---

## What runs here, and what does not

| Stage | What it does | Runs from this release? |
|---|---|:--:|
| R0 generator | derives the 30 scenarios from the base instance | **no** — the base instance is withheld |
| R1 | ε-constraint frontier, 6 ρ strata, with independent oracle recheck | yes |
| R1b | the w₄ × method × disruption cost family | yes |
| R5 | clustered bootstrap over scenario effects | yes |
| R4 | scalability | **no** — `scalability.py` reads the base instance and stops |
| plots | figures | **no** — it requires R3 and R4 to have completed |
| R3 | rolling horizon | untested here; its code does not read the base instance, but we did not run it |

R1, R1b and R5 are the stages behind every number in Section VI, and they are the ones we
re-ran and checked. Use `--stage primary-sharded`, which runs exactly those.

The 30 scenarios ship **pre-generated**, as the 60 `H_rho*.csv` / `S_rho*.csv` files beside
`experiments/journal_p0/rho_manifest.csv`. Before R1 reads anything, all 60 are digest-verified
against that manifest and the run aborts on a mismatch. What shipping them pre-generated costs
is the ability to re-derive them, and with it the byte-for-byte comparison of a regenerated
family against the committed one that the sealed run performed. The digest check is the weaker
substitute, and the relaxations section below says so again.

### The withheld input

`experiments/data_problems_solutions/HHCP-patient-65-task-256-nurse-59.csv` is not here. It is
a crowdsourced home-healthcare benchmark extending a co-author's earlier work; authorisation
for its use in this study is not in dispute, but no data-specific public-redistribution licence
has been established for it.

It is worth saying plainly what that file is, because "patient instance" invites the wrong
assumption. An internal provenance audit records that it is synthetic — produced by a C++
generator seeded from the wall clock, with the seed recovered and a replay reproducing all 65
patients, 59 caregivers and 256 tasks with zero differences. We report that as the audit's
finding rather than as evidence we can hand you: the replay script and its output were never
archived. The reason the file is withheld is licensing, not privacy.

**The released scenarios nonetheless contain most of it.** They reproduce the base instance's
patient table and task attributes completely and four fifths of its caregiver rows. The exact
measurements, and the terms that follow from them, are in
[`LICENSE-DATA.md`](../../LICENSE-DATA.md). Read that before reusing the data: it is released
for checking this paper, not as a redistributable dataset.

### What `BASE_INSTANCE_WITHHELD` licenses

That file at the root of this tree names, by basename, every input withheld from this release.
The marker is consulted in five places across four files, and this is all of them:

1. **R0 does not run.** `run_all.py` writes `R0_GENERATOR_WITHHELD.json` into the stage
   directory instead of generating, so the result tree states the skip on its face. In its
   place, `_require_committed_family()` digest-verifies all 60 shipped scenarios against
   `rho_manifest.csv`. What is lost is regeneration, not verification — but a regenerated
   family would also have been compared against the committed one byte for byte, and that
   independent check is gone.
2. **The base instance's provenance digest is accepted as recorded rather than recomputed.**
   `eps_constraint.py` and `run_e4.py` resolve three provenance roles per scenario — source
   instance, generator, family runner. The generator and family runner are in this tree and
   are still hashed and checked. The source instance cannot be, so its `src_sha256` in
   `rho_manifest.csv` stands undisputed:
   `5b59f162dd4736d3cb26e80a4e011eac0cf4127745c5b6bc8e967a7e6ae3b41e`.
3. **The archive builder tolerates the marker-named absences, and ships the marker.**
   `build_snapshot.py` consults the marker twice: once to accept the base instance's absence
   from the mandatory member set, and once for the `AUDIT_INPUTS` list, from which three
   internal project-management and audit documents are withheld. Those three are withheld
   whole rather than edited, so that no provenance record appears here in a form its author
   did not write. In the first branch it
   also adds `BASE_INSTANCE_WITHHELD` itself to the archive, because the relaxations it
   licenses are evaluated against the extracted tree rather than the host.
4. **`scalability.py` may be imported without the base instance**, so that preflight's
   `--selftest` gate and the 25 negative controls after it can run. The same check also drops
   the base instance from that fixture's declared input list, since a file that is not here
   cannot be hashed. Every other mode of the script still stops with the original message,
   checked against the mode actually dispatched rather than the flag.

Everything else is unchanged and still fails closed: every H/S digest, every generator and
family-runner digest, every solver status code, every oracle flag, every row count. One
boundary on that, so it is not read as more than it is: the builder's fail-closed check covers
inputs it enumerates explicitly, not the source files it gathers by glob. Deleting a globbed
`.py` would shrink the archive and change its digest rather than abort the build. That is the
sealed builder's behaviour too, unchanged here, but the marker does not extend it.

Two caveats we would rather state than have found. First, **the shipped test suite does not
exercise these relaxations.** Its negative control for a missing base instance goes through
`s2_core.paths()`, which never consults the marker, so it fails closed for named and unnamed
paths alike and does not demonstrate the scoping. The scoping is visible by reading
`_source_is_withheld` in the two solver scripts, and the authors tested the branch logic
separately, but that test is not in this tree. Second, **restoring strictness is not simply
deleting the marker.** The three withheld audit documents are otherwise-mandatory archive
members, so a tree with the marker deleted will not build an archive at all. To get the sealed
behaviour you need the base instance back, those three documents back, and no marker.

---

## Running it

The pipeline runs only inside the pinned container, from a content-addressed archive it
verifies on extraction. It refuses to run from a source tree. This is deliberate: it is what
makes "it reproduced" mean something.

**1 — build the archive and the image.**

```sh
python3 experiments/journal_p0/repro/build_snapshot.py --root . --output-dir ./snap
```

That prints a `snap_id`: the SHA-256 of the archive, which is also its filename. Verify it
yourself rather than trusting the name:

```sh
shasum -a 256 snap/SNAP_*.tar
```

Built from this tree as published, the archive is

```
ec1668727a4407f0903554550b78ad021e9d8b88092a39bca11bf728742f49e1
```

and that is the archive we publish. If yours differs, you have changed something in the tree —
the builder is deterministic, and this README is not one of the archive's members, so editing
it does not move the digest.

The sealed results in `../run/results/` were **not** produced from this archive. They came from
`d6c9fa4de2c8ca88bd2fa260f8975398ac42ed9ef14c0527213d0865f2980c60`, built before the base
instance and the three audit documents were withheld, and differing from this one only by
those four absences and the five patched files. That earlier digest is the `snap_id` stamped into
every row of the deposited CSVs, which is why your own output will not carry the same one.

```sh
docker build --platform linux/amd64 -t dchhr-repro -f experiments/journal_p0/repro/Dockerfile .
```

The base image is pinned by digest (`python:3.9.19-slim-bookworm`), and the four scientific
dependencies are pinned exactly in `requirements.lock`. The MILP solver is the HiGHS embedded
in SciPy 1.13.1, reached through `scipy.optimize.milp`. Presolve is disabled by
pre-registration, enforced in the run code, not merely stated — see the recorded reason and its
reproducer at `tests/fixtures/epse4/negative/highs_presolve_infeasible.py`.

**2 — run the pipeline.** Substitute the `snap_id` you just verified.

```sh
mkdir -p results
```

```sh
docker run --rm --platform linux/amd64 --user $(id -u):$(id -g) --tmpfs /work:rw,size=1g -v "$PWD/snap/SNAP_<snap_id>.tar:/snap/archive.tar:ro" -v "$PWD/results:/results" dchhr-repro --archive /snap/archive.tar --snap-id <snap_id> --result-root /results --stage primary-sharded
```

The `mkdir` is not optional: Docker creates a missing bind-mount target as root, and the
container runs as you, so it could not then write its results.

Use `--stage primary-sharded`. Do not use `--stage all`, which is argparse's default and would
go on to attempt R3, R4 and plots; R4 stops on the withheld base instance and plots cannot
complete without it.

`primary-sharded` runs environment, preflight, the withheld-R0 receipt, then R1 and R1b as six
concurrent ρ strata, then R5. Set `DCHHR_SHARD_JOBS` to run fewer than six at once; six is the
maximum and the registered default. Each shard is single-threaded by design
(`OMP_NUM_THREADS=1` and friends), so six shards want six cores.

**3 — compare.** Each stage lands atomically and prints `FORMAL_STAGE_COMPLETE`. A stage that
fails leaves a `.<stage>.partial` directory and never a finished one, so a partial result can
never be mistaken for a complete family.

Compare your `results/R1/`, `results/R1b/` and `results/R5/` CSVs against the sealed copies in
`../run/results/`. **Most of them will not match byte for byte, and should not.** Seven of the
eight deposited CSVs carry a `snap_id` column written from the running archive's own digest, so
every row of those differs from the sealed copy in that column. Normalise it before comparing,
or compare the columns you care about. Two files behave differently and are worth knowing
about: `e4_summary.csv` carries no `snap_id` at all and should match byte for byte, and
`e4_raw.csv` additionally differs in three shard-digest columns — `shard_raw_sha256`,
`shard_provenance_sha256` and `shard_diagnostics_sha256` — each of which hashes a shard file
that itself carries `snap_id`, and the last of which also carries elapsed times.

Across the whole result tree, exactly six columns can legitimately differ: `snap_id`,
`wall_time_s`, and the four digests that hash files containing them (`source_sha256`,
`shard_raw_sha256`, `shard_provenance_sha256`, `shard_diagnostics_sha256`). **Any other column
differing is a real difference and we want to hear about it.** The headline counts are 30
scenarios, 518 disruptions, 488 zero-drop, 253/518 A2 multi-cost and 224/488 after
qualification.

---

## What this does and does not establish

**It does** let a third party re-solve every MILP behind Section VI on their own hardware, with
the same model, the same scenarios, the same solver and the same pre-registered design
manifests, and compare the result against ours.

**The pipeline is deterministic, and we checked.** Re-running the sealed archive in a fresh
container from a clean result directory reproduced the sealed stage outputs exactly, with one
exception. Of the ten result CSVs, seven are byte-identical — including `eps_frontier.csv`
(32 372 rows), `eps_oracle_validation.csv` (32 372 rows) and all three R5 files. The remaining
three differ in exactly two columns between them: `wall_time_s`, and the
`shard_diagnostics_sha256` that hashes the file containing it. Every cost, every solver status
code, every branch-and-bound node count and every oracle flag is identical. The only quantity
that does not reproduce is measured elapsed time — which is also the only quantity the paper
states it draws no conclusion from.

**We also ran this released tree itself, not just the sealed archive.** Built into its own
content-addressed archive and executed from a clean result directory, it completed every stage
it claims to run — the R0 withheld-input receipt, environment, all seventeen preflight gates,
25 of 25 negative controls, R1, R1b and R5 — with zero aborts, and verified all 60 committed
scenario files against their recorded digests before any stage read them. Comparing all 60
result CSVs it produced against the sealed run's: seven byte-identical, 53 differing only in
the six columns listed above, and **none with a substantive difference**. `e4_raw.csv` agrees
in 232 of its 236 columns. `e4_summary.csv` — which the pipeline recomputes inside the
container from the landed `e4_raw.csv` — comes back byte-identical to the sealed one, which is
an independent check on the raw file's scientific content rather than on its bytes.

**It does not** establish machine-independence. That re-run was executed on the same host that
produced the sealed run. A bounded cross-platform check was done on a second host of a
different CPU architecture — arm64 macOS running this pinned `linux/amd64` image under
emulation — where the environment stage, all seventeen preflight gates and all 25 negative
controls behave identically, and the recorded environment differs only in the kernel version
string, the container hostname and the platform string. R1, R1b and R5 were not re-solved
there. Whether a reader on different hardware obtains identical frontier values is untested.
If you run this and a numeric column differs from `../run/results/`, that is worth reporting to
us — it is a result, not a failure on your part.

**It does not** make the solutions optimal by fiat. The gates are consistency controls on a
recorded run — row counts, digests, solver status codes, oracle flags — not optimality proofs.
80 of the 32 372 R1 frontier rows carry a material solver gap; all 80 terminated with solver
status 0 against an 1 800-second limit never reached, and on each the incumbent exceeds the
dual bound by exactly one unit of an integer-valued objective. Section VI-D reports what
happens when all 80 are deleted: the headline counts do not move.

**It does not** re-derive the 30 scenarios, for the reason given above. A reader who wants to
check the scenario family itself can check it against the recorded digests, which is weaker
than regeneration.

---

## Cost

Measured on the authors' re-run: an AMD EPYC 7J13, 24 cores, 46 GB, six concurrent shards,
each single-threaded.

| Stage | Wall clock |
|---|---|
| environment + preflight + withheld-R0 receipt | about 1 minute |
| R1 (6 ρ strata, concurrent) | 3 h 10 min |
| R1b (6 ρ strata, concurrent) | 1 h 31 min |
| R5 | seconds |
| **total** | **4 h 43 min** |

The cost is concentrated almost entirely in the three tight capacity strata. Within R1, the
ρ = 0.45, 0.55 and 0.65 shards each finished in about two minutes; ρ = 0.75 took 81 minutes,
ρ = 0.85 took 88, and ρ = 0.95 took just over three hours. That distribution matches the one
the paper reports for its solver gaps: 74 of the 80 material-gap rows lie at ρ = 0.95.

Two qualifications. Another job shared the machine during part of the R1 window, so the
tight-stratum wall times are somewhat inflated. And the sealed run's own records put the total
at about 5.38 CPU-hours, which is the figure to plan against rather than the wall clock above.
With fewer than six cores, set `DCHHR_SHARD_JOBS` lower and expect the total to approach the
CPU-hour figure.
