# CGNS 4.5.2 source subset

This directory contains a pruned copy of CGNS 4.5.2 used to build the static
ADF reader required by the GSX solver. It is not a complete CGNS source
distribution: upstream examples, tests, GUI utilities, sample scripts, CI
configuration, and release documents have been omitted.

The source corresponds to upstream commit
`3a972d92559787a49c5f9a54974ea1946ecb9d21` from
<https://github.com/CGNS/CGNS>. The upstream license is retained in
`license.txt`.

The parent GSX build disables CGNS HDF5, Fortran, shared-library, and test
options. Use an installed full CGNS package with `GSX_USE_SYSTEM_CGNS=ON` when
those capabilities are required independently of GSX.
