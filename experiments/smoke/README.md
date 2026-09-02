# Manual smoke checks

These scripts are exploratory, environment-dependent checks rather than
automated tests. Run them individually from the repository root. They may
require network access, cached Hugging Face models, local datasets, checkpoints,
GPU resources, or writable output directories.

They intentionally live outside `tests/` so normal pytest collection remains
fast, deterministic, and self-contained.
