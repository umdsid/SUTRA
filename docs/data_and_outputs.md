# Data and outputs

SUTRA reconstructs each specimen locally. Raw spatial-transcriptomic data and large resources are not stored in the public Git repository; production expects project-level `data/` and `resources/`.

Released specimen keys:

```text
healthy_reference
alzheimers
gbm_reference_addon
nondiseased_kidney
prcc
```

The hierarchy retains traceable tissue objects and, across the released analyses, spatial, molecular, geometric, topological/adjacency and relational information. The organizational coordinate `u` describes scale, not time.

A fresh production run and the frozen paper record serve different purposes. `part1_production/` contains the public production workflow; `part2_paper/` contains frozen scientific products for paper reproduction and audit. Do not overwrite frozen products with an unvalidated fresh run.

The release contains five hierarchy movies under `part2_paper/movies/`, with `MOVIE_MANIFEST.json` providing the release-level record. Presence of a frozen MP4 establishes the released product; regeneration should only be claimed when its producer and provenance are present and validated.
