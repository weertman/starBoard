# Interaction Widget Map (Retired)

This file is intentionally kept as a retirement notice rather than an active
widget inventory.

The previous widget-by-widget map fell out of sync with the live UI and the
current interaction logging implementation, so it should not be used as a
source of truth.

Use these references instead:

- `docs/starboard-interface-guide.md` for the current user-facing UI layout
- `docs/INTERACTION_LOGGING_GAPS.md` for the current engineering audit of known
  logging gaps
- `_ilog.log(...)` call sites under `src/ui/` and
  `src/utils/interaction_logger.py` for implementation truth

This map was retired on 2026-03-06 after a repo-wide review found:

- no inbound references to this document in the repository
- missing coverage for newer UI surfaces such as `Morphometric` and
  `Gallery Review`
- stale "not logged" status for widgets that are now instrumented in
  `TabFirstOrder`, `TabSetup`, `TabSecondOrder`, `TabPastMatches`, and
  `TabDeepLearning`

The original snapshot remains available in git history if it ever needs to be
reconstructed for historical comparison.
