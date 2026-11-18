# Third-party notices and data provenance

## CGNS 4.5.2

The release vendors CGNS 4.5.2 under `third_party/cgns`. CGNS is distributed
under its upstream zlib-style license. The authoritative notice is retained at
`third_party/cgns/license.txt` and must not be removed.

Upstream project: <https://github.com/CGNS/CGNS>

Vendored upstream commit recorded by the development repository:
`3a972d92559787a49c5f9a54974ea1946ecb9d21`.

## NASA Turbulence Modeling Resource data

The verification meshes, validation meshes, and compact comparison data in
this package originate from or are derived from the NASA Turbulence Modeling
Resource case families:

- Flat plate: <https://tmbwg.github.io/turbmodels/flatplate.html>
- Bump in channel: <https://tmbwg.github.io/turbmodels/bump.html>
- NACA 0012: <https://tmbwg.github.io/turbmodels/naca0012_val.html>

NASA and the Turbulence Modeling Resource do not endorse this solver or its
results. Users should consult the authoritative case pages for definitions,
reference conditions, and the latest reference data.

## Bundled CGNS subset

The bundled CGNS tree is pruned to the files needed to configure and build the
static ADF library used by this solver. CGNS examples, tests, GUI utilities,
sample scripts, CI configuration, and release documents are not redistributed.
