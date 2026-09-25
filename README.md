# SUTRA — Spatial Unified Transcriptomic Reconstruction and Analysis

Release-candidate repository organized as two reproducibility layers.

## Part 1 — production
`part1_production/` contains the scientific engine, v0911 configuration family, tests, and production entry points. Raw 10x datasets and large generated results are intentionally not committed.

## Part 2 — paper reproduction
`part2_paper/` contains only validated standalone panel products, frozen source tables, panel-producing code snapshots, supplementary products, and the scientific movie.

**No assembled main figures are shipped.** Main Figure 2 contains only A–H standalone panels. Main Figure 3 contains only A–D plus tissue-specific E/F standalone panels.

### Supplementary renumbering
The manuscript currently has no previous Supplementary Figure 1. Therefore:
old SuppFig02 → new SuppFig1  
old SuppFig03 → new SuppFig2  
old SuppFig04 → new SuppFig3  
old SuppFig05 → new SuppFig4  
old SuppFig06 → new SuppFig5

The uploaded old SuppFig05 archive contains no panel PNG/PDF, so new SuppFig4 is explicitly marked incomplete rather than being replaced by a wrong panel.

## Before GitHub upload
```bash
./test_release.sh
```
Do not upload if it does not finish with `ALL RELEASE TESTS PASSED`.

See `docs/GITHUB_UPLOAD.md` and `docs/RELEASE_AUDIT.md`.
