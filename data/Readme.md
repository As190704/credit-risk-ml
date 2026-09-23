# Data directory

- `raw/` — place the original, immutable CSV dataset here. Never edit in place.
- `processed/` — reserved for Phase 2+ generated artifacts (e.g. cached splits).

## Dataset assumptions (documented, not hardcoded)

The pipeline expects:
- A single flat CSV file, one row per applicant/loan.
- A target column (name configurable via `DataConfig.target_column`)
  representing credit risk outcome (e.g. default / good-bad), ideally binary.
- A mix of numerical and categorical columns; exact names are NOT assumed
  anywhere in `src/` — they are inferred at runtime or provided via config.

If a specific dataset requires columns to be excluded (IDs, leakage-prone
post-outcome fields, etc.), add them to `DataConfig.excluded_columns` with a
matching reason string in `DataConfig.exclusion_reasons`.s