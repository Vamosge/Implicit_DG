# GSX implicit DG RANS-SA solver

GSX is a research implementation of a two-dimensional, nodal discontinuous
Galerkin solver for compressible RANS flow with the Spalart--Allmaras
turbulence model. The implicit algorithm uses assembled forward-AD Jacobians,
Newton iteration, restarted GMRES, and block ILU(0). Serial and OpenMP builds
are supported.

This is release candidate `v1.0.0-rc2`. It is intended for testing and
reproducible research. It is not yet the final validation release.

## Scope

The release contains one Q2 solver implementation and the following case
families:

| Case | Grid | Cells |
| --- | --- | ---: |
| `bump89` | smooth bump, 89 x 41 points | 88 x 40 |
| `bump177` | smooth bump, 177 x 81 points | 176 x 80 |
| `bump353` | smooth bump, 353 x 161 points | 352 x 160 |
| `plate35` | flat plate, 35 x 25 points | 34 x 24 |
| `plate69` | flat plate, 69 x 49 points | 68 x 48 |
| `plate137` | flat plate, 137 x 97 points | 136 x 96 |
| `plate273` | flat plate, 273 x 193 points | 272 x 192 |
| `naca113` | NACA 0012 with nested geometry | 112 x 32 |
| `naca113-familyII` | downsampled Family-II mesh | 112 x 32 |
| `naca225` | NACA 0012 Family-II mesh | 224 x 64 |

The NACA qualification campaign is incomplete. This repository does not
contain MPI domain decomposition, an FV-JFNK solver, a backward-facing-step
case, or a complete Q1--Q3 manufactured-solution campaign.

## Requirements

- CMake 3.24 or newer;
- a C++17 compiler;
- OpenMP for threaded execution (optional);
- PowerShell 5 or newer for the Windows convenience scripts.

The required CGNS ADF library source is included under `third_party/cgns`.
HDF5 and Fortran support are disabled. An installed CGNS package can instead
be selected with `-DGSX_USE_SYSTEM_CGNS=ON`.

## Windows quick start

Open an x64 Visual Studio Developer PowerShell in the repository root. The
bootstrap command downloads portable CMake and Ninja into the ignored
`.tools` directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\bootstrap-build-tools.ps1
```

Check the coarse bump case, build it, run the regression tests, and perform a
100-step smoke run:

```powershell
.\tools\run-case.cmd -Case bump89 -CheckOnly
.\tools\run-case.cmd -Case bump89 -Steps 100 -Threads 8 `
  -RunName bump89-smoke100 -CflPolicy residual-aware `
  -SaLimiter finite-only -CflMax 500
```

Results are written to `runs/<RunName>`. After the first build, add
`-SkipBuild` to subsequent commands for the same case.

Continue from a checkpoint:

```powershell
.\tools\run-case.cmd -Case bump89 -Steps 1000 -Threads 8 `
  -RunName bump89-continuation -Restart .\runs\bump89-smoke100\checkpoint_latest.bin `
  -CflPolicy residual-aware -SaLimiter finite-only -CflMax 500 -SkipBuild
```

The wrapper also accepts `plate35`, `plate69`, `naca113`,
`naca113-familyII`, and `naca225`. The finer bump and flat-plate grids can be
built directly by setting `GSX_CASE` with CMake.

## Portable CMake build

The following commands configure the coarse bump case. Set `GSX_CASE` to any
case in the table above.

```text
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DGSX_CASE=bump89 -DGSX_ENABLE_OPENMP=ON
cmake --build build/release --parallel
ctest --test-dir build/release --output-on-failure
```

Run the executable with:

```text
build/release/gsx_dg --max-steps 100 \
  --cfl-policy residual-aware --sa-limiter finite-only --cfl-max 500
```

Use `gsx_dg --help` for the complete command-line interface.

## Convergence and checkpoints

The default normalized Navier--Stokes and SA convergence tolerances are both
`1e-6`. The recommended SA-neg setting is `--sa-limiter finite-only`, which
rejects non-finite states without clipping finite negative values. The legacy
floor limiter remains available only for controlled comparisons.

Normal runs write:

- `residual_history.dat` and `run_summary.dat`;
- solution snapshots and aerodynamic histories where applicable;
- `checkpoint_latest.bin` at the requested checkpoint interval and on exit;
- diagnostic histories for line search, CFL control, GMRES, and SA limiting.

Restart compatibility is checked against the case, mesh, flow conditions,
and discretization signature.

## Repository contents

- `src/`, `include/dg/`: solver and reusable numerical components;
- `tests/`: regression, mesh, AD, boundary, nonlinear, and linear-solver tests;
- `tools/`: Windows build and isolated-run launchers;
- `reference/`: compact NASA/TMR comparison data;
- `third_party/cgns/`: pruned CGNS 4.5.2 ADF source and its license;
- root-level `.cgns` and `.msh` files: meshes used by the named cases.

## Citation and license

The solver is distributed under the BSD-3-Clause license in `LICENSE`.
Software citation metadata are provided in `CITATION.cff`; contributors are
listed in `AUTHORS.md`. Third-party software and reference-data provenance are
recorded in `THIRD_PARTY_NOTICES.md`.
