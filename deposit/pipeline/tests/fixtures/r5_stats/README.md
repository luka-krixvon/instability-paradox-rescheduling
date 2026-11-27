# R5 clustered-statistics fixture

`test_r5_stats.py` builds a complete six-rho by five-seed fixture in a fresh
temporary directory.  The committed `S_*.csv` files intentionally contain a
varying number of future task holders, so the expected disruption count must be
derived from the files; it cannot be inferred from `e4_raw.csv` or a fixed
literal.

The checks cover:

- the positive scenario/per-rho/bootstrap path;
- byte-identical scientific outputs after independent row reorderings of the
  manifest, every S file, and R1b raw input;
- the 10,000-replicate shared-index bootstrap and its fixed seed;
- the exact two-sided five-cluster sign-flip floor and cumulative-maximum Holm
  implementation;
- missing/duplicate/extra design cells, a newly eligible caregiver absent from
  raw, broken common service, `unserved != U`, non-optimal solver evidence, and
  missing formal SNAP provenance.

Run from the snapshot root:

```bash
python3 tests/fixtures/r5_stats/test_r5_stats.py
```

A PASS is regression and traceability evidence only.  It is not proof of the
method, validation of formal R1b values, or evidence that fabricated data are
absent.

