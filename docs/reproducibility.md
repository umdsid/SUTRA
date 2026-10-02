# Reproducibility and provenance

SUTRA separates computation, frozen scientific products and presentation.

## Production computation
`part1_production/run_sutra.sh` is the canonical public entry point. Stage-numbered scripts retained beside it preserve the validated execution history; they are not competing public releases.

## Frozen scientific products
`part2_paper/` retains approved panel products, numerical audits/manifests, supplementary analyses and hierarchy movies.

```bash
python part2_paper/validate_frozen_products.py
./test_release.sh
```

File checksums are recorded in `SHA256SUMS.tsv`.

## Presentation is downstream
Figure composition, manuscript layout and documentation should not silently redefine a numerical quantity, rerun a scientific analysis, or replace a frozen result.

## Movies
`part2_paper/movies/MOVIE_MANIFEST.json` records the released five-movie set. Frozen movies and movie-generation capability are distinct provenance claims.

## Interpretation
The SUTRA hierarchy is indexed by organizational scale `u`, not time, temperature, or a thermodynamic control parameter. Transition-like structure across the hierarchy is interpreted in finite, spatially embedded biological systems rather than assumed to establish an equilibrium critical point.
