# moduli_scan

Scan harnesses, aggregated data and figure scripts for the two models of the paper:

- **`model1/`** — the complex-structure scan: couplings vs `psi1` (real axis and complex
  plane), at the fixed symmetric Kähler point `t=(1,1,1,1)`.
- **`model2/`** — the Kähler-moduli scan: couplings vs the D-flat shape modulus `t3`
  (with `t2=1`, `t1=(t3-2)(t3+1)`, `t4=t1/t3`), at fixed complex structure
  `psi0=2, psi1=1`.

Each directory is self-contained: pipeline scripts, an example SLURM submission, the
aggregated scan data under `data/`, and figure scripts under `plots/`. See the README in
each. Run all commands from the repository root.
