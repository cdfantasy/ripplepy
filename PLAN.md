# Plan

## Status

`ripplepy` provides a Python/Fortran implementation of the vacuum-field effective
ripple, magnetic-axis search and radial surface scans. The core evaluation chain is
complete and documented in `README.md`; the repository is kept code-only.

- Core library: `python/ripplepy/` + `fortran/`.
- Build: scikit-build-core + f90wrap + f2py (see `README.md`).
- Example drivers and development benchmarks: `tests/`.
- Run data, figures, logs, presentations and build artefacts are not committed.

## Near term

1. Keep numerical regression tests green with the local `mgrid`/`wout` fixtures.
2. Maintain the public API and README together with any signature changes.
3. Keep the H1 scan and surrogate scripts as examples; do not add generated data.
4. Add a small synthetic/CI test that does not require external NetCDF fixtures.

## Longer term

- Optional: document the current-space surrogate and boundary-classification workflow
  as a separate example, with generated data kept outside the repository.
- Packaging: keep the wheel limited to the Fortran extension, the generated wrapper and
  the Python package; no runtime data in the wheel.
- Maintenance: track numerical conventions and gate definitions in code/tests rather
  than in local notes.

## Repository policy

- Commit source, build definitions, README, PLAN and `CLAUDE.md`.
- Do not commit: `*.h5`, `*.csv`, `*.log`, `*.png`, `*.pptx`, `figs/`, build trees and
  local run data (see `.gitignore`).
- Internal hand-off notes and campaign-specific documents are kept outside this
  repository.
