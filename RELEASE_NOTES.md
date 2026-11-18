# v1.0.0-rc2

Release candidate date: 2026-11-18

## Purpose

This release candidate packages the reorganized implicit DG RANS-SA solver for
independent testing and reproducible research. It provides a portable,
Q2/OpenMP implementation with CGNS mesh input, SA-neg,
assembled automatic-differentiation Jacobians, Newton-GMRES, block ILU(0),
checkpoint/restart, isolated run directories, and regression tests.

## Reproducibility defaults

- Steady normalized Navier-Stokes tolerance: `1e-6`
- Steady normalized SA tolerance: `1e-6`
- SA relative reduction is diagnostic only and cannot independently stop a run
- Physical-output stability is additionally required where the case defines a
  production convergence monitor
- SA limiter default: `finite-only` (SA-neg); the legacy floor is diagnostic
  only

## Supported case families

- NASA/TMR zero-pressure-gradient flat plate
- NASA/TMR smooth bump in channel
- NASA/TMR NACA 0012

## Release-candidate scope

This repository is not a complete reproduction package for work requiring the
following components, which are outside this RC's scope:

- MPI domain decomposition;
- a backward-facing-step case;
- the complete Q1-Q3 manufactured-solution study;
- the separate FV-JFNK comparison solver.

The current implementation is Q2 and uses serial or OpenMP execution. These
boundaries should be stated whenever results from this release are reported.

## Validation status

Strict `1e-6` formal runs are complete for the two coarsest flat-plate levels
and the coarse bump level. All 47 standalone regression tests passed in the
release build. The formal NACA campaign is pending; therefore this snapshot is
an RC rather than the final `v1.0.0` release.

## RC2 packaging cleanup

- Consolidated user instructions into `README.md` and removed internal design
  and development reports.
- Removed disabled CGNS examples, tests, GUI tools, sample scripts, CI files,
  and release documents while retaining the ADF library sources and license.
- Retained solver regression tests, input meshes, launch scripts, and compact
  NASA/TMR reference data.
