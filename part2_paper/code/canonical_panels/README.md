# SUTRA canonical panel-only source

This directory is a presentation-only staging layer. Original scientific renderers are
not overwritten. The staged copies suppress Matplotlib `set_title`/`suptitle` calls and
strict standalone A–H panel-letter text calls.

Do not use composite/compositor scripts for canonical paper production.

Important:
- Main Fig. 3 and Fig. 4 are sourced from the latest frozen Part-2/Fig4 production
  branches found locally.
- Main Fig. 5 and Fig. 6 use the installed FINE renderers.
- Supp. Figs. 3–5 use the common disease-collective renderer.
- Supp. Fig. 6 prefers v0.5.9 standalone; falls back only if it is absent.
- Supp. Fig. 1 remains to be materialized as a construction/mechanics audit family;
  no scientifically authoritative renderer was present in the audited source set, so
  this installer does not invent one.
