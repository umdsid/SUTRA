# Getting started

`part1_production/run_sutra.sh` is the canonical public entry point. Historical stage-numbered scripts are retained for provenance; they are not separate SUTRA versions.

```bash
git clone https://github.com/umdsid/SUTRA.git
cd SUTRA/part1_production
./run_sutra.sh --help
```

Populate the project-level `data/` and `resources/` required by the released analysis. The five specimen keys are `healthy_reference`, `alzheimers`, `gbm_reference_addon`, `nondiseased_kidney`, and `prcc`.

Preflight before the full sequential run:

```bash
./run_sutra.sh --preflight
./run_sutra.sh --run
```

The frozen paper-reproduction record is distinct from a fresh production run:

```bash
cd ..
python part2_paper/validate_frozen_products.py
./test_release.sh
```
