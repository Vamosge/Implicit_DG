$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cmake = Join-Path $root '.tools\cmake\bin\cmake.exe'
if (-not (Test-Path -LiteralPath $cmake)) {
    throw 'Portable build tools are missing. Run tools/bootstrap-build-tools.ps1 first.'
}
& $cmake --preset release-bump89
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmake --build --preset release-bump89
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $root '.tools\cmake\bin\ctest.exe') --preset release-bump89
exit $LASTEXITCODE
