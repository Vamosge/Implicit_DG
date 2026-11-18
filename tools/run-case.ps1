[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('bump89', 'plate35', 'plate69', 'naca113', 'naca113-original',
        'naca113-familyII', 'naca225')]
    [string]$Case,

    [ValidateRange(1, 2147483647)]
    [int]$Steps = 1000,

    [ValidateRange(1, 1024)]
    [int]$Threads = 8,

    [string]$RunName = '',
    [string]$Restart = '',

    [Nullable[double]]$Mach = $null,
    [Nullable[double]]$Reynolds = $null,
    [Nullable[double]]$AlphaDegrees = $null,

    [double]$NsTolerance = 1e-6,
    [double]$SaTolerance = 1e-6,

    [ValidateSet('legacy', 'residual-aware')]
    [string]$CflPolicy = 'legacy',

    [ValidateSet('floor', 'finite-only')]
    [string]$SaLimiter = 'finite-only',

    [ValidateRange(1, 2147483647)]
    [int]$CheckpointEvery = 100,

    [double]$CflInitial = 0.0,
    [double]$CflMax = 0.0,
    [double]$SaCflScale = 1.0,
    [double]$SaBr2Factor = 1.0,
    [switch]$SaQ2Reset,
    [switch]$FreezeNs,
    [switch]$CheckOnly,
    [switch]$ResidualCheck,
    [switch]$JacobianCheck,
    [switch]$Fresh,
    [switch]$SkipBuild,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cmake = Join-Path $root '.tools\cmake\bin\cmake.exe'
$ctest = Join-Path $root '.tools\cmake\bin\ctest.exe'

$cases = @{
    'bump89' = @{
        Preset = 'release-bump89-openmp-windows'
        Mach = '0.2'; Reynolds = '3000000'; Alpha = '0'
    }
    'plate35' = @{
        Preset = 'release-plate35-openmp-windows'
        Mach = '0.2'; Reynolds = '5000000'; Alpha = '0'
    }
    'plate69' = @{
        Preset = 'release-plate69-openmp-windows'
        Mach = '0.2'; Reynolds = '5000000'; Alpha = '0'
    }
    'naca113-original' = @{
        Preset = 'release-naca113-openmp-windows'
        Mach = '0.15'; Reynolds = '6000000'; Alpha = '10'
    }
    'naca113' = @{
        Preset = 'release-naca113-openmp-windows'
        Mach = '0.15'; Reynolds = '6000000'; Alpha = '10'
    }
    'naca113-familyII' = @{
        Preset = 'release-naca113-familyII-openmp-windows'
        Mach = '0.15'; Reynolds = '6000000'; Alpha = '10'
    }
    'naca225' = @{
        Preset = 'release-naca225-openmp-windows'
        Mach = '0.15'; Reynolds = '6000000'; Alpha = '10'
    }
}

$configuration = $cases[$Case]
$preset = $configuration.Preset
$exe = Join-Path $root "build\$preset\gsx_dg.exe"

if ($Case -eq 'naca113-familyII') {
    Write-Warning ('naca113-familyII is an experimental downsampled linear-mesh route. ' +
        'Use -Case naca113 for the release validation baseline with nested Q4 geometry.')
}

$invariantCulture = [Globalization.CultureInfo]::InvariantCulture
if ($NsTolerance -ne $NsTolerance -or $NsTolerance -le 0.0 `
    -or $NsTolerance -gt [double]::MaxValue) {
    throw 'NsTolerance must be a finite positive number.'
}
if ($SaTolerance -ne $SaTolerance -or $SaTolerance -le 0.0 `
    -or $SaTolerance -gt [double]::MaxValue) {
    throw 'SaTolerance must be a finite positive number.'
}
$machValue = $configuration.Mach
$reynoldsValue = $configuration.Reynolds
$alphaValue = $configuration.Alpha
if ([double]::IsNaN($SaCflScale) -or [double]::IsInfinity($SaCflScale) `
    -or $SaCflScale -le 0.0) {
    throw 'SaCflScale must be a finite positive number.'
}
if ([double]::IsNaN($SaBr2Factor) -or [double]::IsInfinity($SaBr2Factor) `
    -or $SaBr2Factor -le 0.0) {
    throw 'SaBr2Factor must be a finite positive number.'
}
if ($null -ne $Mach) {
    $machOverride = [double]$Mach
    if ([double]::IsNaN($machOverride) -or [double]::IsInfinity($machOverride) `
        -or $machOverride -le 0.0) {
        throw 'Mach must be a finite positive number.'
    }
    $machValue = $machOverride.ToString('R', $invariantCulture)
}
if ($null -ne $Reynolds) {
    $reynoldsOverride = [double]$Reynolds
    if ([double]::IsNaN($reynoldsOverride) `
        -or [double]::IsInfinity($reynoldsOverride) `
        -or $reynoldsOverride -le 0.0) {
        throw 'Reynolds must be a finite positive number.'
    }
    $reynoldsValue = $reynoldsOverride.ToString('R', $invariantCulture)
}
if ($null -ne $AlphaDegrees) {
    $alphaOverride = [double]$AlphaDegrees
    if ([double]::IsNaN($alphaOverride) `
        -or [double]::IsInfinity($alphaOverride)) {
        throw 'AlphaDegrees must be finite.'
    }
    $alphaValue = $alphaOverride.ToString('R', $invariantCulture)
}

if (-not $SkipBuild) {
    if (-not (Test-Path -LiteralPath $cmake)) {
        throw 'Portable CMake is missing. Run tools/bootstrap-build-tools.ps1 first.'
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw 'MSVC is not active. Run this script from a Developer PowerShell for Visual Studio.'
    }
    $configureArguments = @('--preset', $preset)
    if ($Fresh) { $configureArguments = @('--fresh') + $configureArguments }
    & $cmake @configureArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $cmake --build --preset $preset -j $Threads
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not $SkipTests) {
        & $ctest --preset $preset --output-on-failure
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Solver executable is missing: $exe"
}

$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NUM_THREADS = [string]$Threads

if (([int]$CheckOnly.IsPresent + [int]$ResidualCheck.IsPresent + [int]$JacobianCheck.IsPresent) -gt 1) {
    throw 'CheckOnly, ResidualCheck, and JacobianCheck are mutually exclusive.'
}
if ($CheckOnly) {
    & $exe --check-case `
        --mach $machValue `
        --reynolds $reynoldsValue `
        --alpha-deg $alphaValue
    exit $LASTEXITCODE
}
if ($ResidualCheck) {
    & $exe --residual-check `
        --mach $machValue `
        --reynolds $reynoldsValue `
        --alpha-deg $alphaValue
    exit $LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = '{0}-{1}' -f $Case, (Get-Date -Format 'yyyyMMdd-HHmmss')
}
if ([IO.Path]::GetFileName($RunName) -ne $RunName `
    -or $RunName.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) {
    throw 'RunName must be one valid directory name, not a path.'
}

$runsRoot = Join-Path $root 'runs'
if (-not (Test-Path -LiteralPath $runsRoot)) {
    New-Item -ItemType Directory -Path $runsRoot | Out-Null
}
$runDirectory = Join-Path $runsRoot $RunName
if (Test-Path -LiteralPath $runDirectory) {
    $existing = Get-ChildItem -LiteralPath $runDirectory -Force
    if ($existing.Count -gt 0) {
        throw "Run directory already exists and is not empty: $runDirectory"
    }
}
else {
    New-Item -ItemType Directory -Path $runDirectory | Out-Null
}

$restartArgument = ''
if (-not [string]::IsNullOrWhiteSpace($Restart)) {
    $restartSource = (Resolve-Path -LiteralPath $Restart).Path
    $restartCopy = Join-Path $runDirectory 'checkpoint_start.bin'
    Copy-Item -LiteralPath $restartSource -Destination $restartCopy
    $restartArgument = 'checkpoint_start.bin'
}

$solverArguments = @(
    $(if ($JacobianCheck) { '--jacobian-check' } else { '--max-steps' }),
    $(if ($JacobianCheck) { $null } else { [string]$Steps }),
    '--mach', $machValue,
    '--reynolds', $reynoldsValue,
    '--alpha-deg', $alphaValue,
    '--ns-tolerance', $NsTolerance.ToString('R', $invariantCulture),
    '--sa-tolerance', $SaTolerance.ToString('R', $invariantCulture),
    '--cfl-policy', $CflPolicy,
    '--sa-limiter', $SaLimiter,
    '--sa-cfl-scale', $SaCflScale.ToString('R', $invariantCulture),
    '--sa-br2-factor', $SaBr2Factor.ToString('R', $invariantCulture),
    '--checkpoint-every', [string]$CheckpointEvery
) | Where-Object { $null -ne $_ }
if ($CflInitial -gt 0.0) {
    $solverArguments += @('--cfl-initial', [string]$CflInitial)
}
elseif ($CflInitial -lt 0.0) {
    throw 'CflInitial must be positive when specified.'
}
if ($CflMax -gt 0.0) {
    $solverArguments += @('--cfl-max', [string]$CflMax)
}
elseif ($CflMax -lt 0.0) {
    throw 'CflMax must be positive when specified.'
}
if ($restartArgument) {
    $solverArguments += @('--restart', $restartArgument)
}
if ($SaQ2Reset) {
    if (-not $restartArgument) {
        throw 'SaQ2Reset requires -Restart.'
    }
    $solverArguments += '--sa-q2-reset-on-restart'
}
if ($FreezeNs) {
    if (-not $restartArgument) {
        throw 'FreezeNs requires -Restart.'
    }
    $solverArguments += '--freeze-ns'
}

$gitRevision = 'unavailable'
$gitStatus = 'unavailable'
if (Get-Command git.exe -ErrorAction SilentlyContinue) {
    $gitRevision = (& git.exe -C $root rev-parse HEAD 2>$null)
    $dirtyLines = @(& git.exe -C $root status --short 2>$null)
    $gitStatus = if ($dirtyLines.Count -eq 0) {
        'clean'
    }
    else {
        'dirty: ' + ($dirtyLines -join '; ')
    }
}

$displayArguments = $solverArguments | ForEach-Object {
    if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}
$commandText = '"{0}" {1}' -f $exe, ($displayArguments -join ' ')
$provenance = @(
    'started_local ' + (Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK'),
    'case ' + $Case,
    'preset ' + $preset,
    'run_directory ' + $runDirectory,
    'threads ' + $Threads,
    'git_revision ' + $gitRevision,
    'git_status ' + $gitStatus,
    'command ' + $commandText
)
Set-Content -LiteralPath (Join-Path $runDirectory 'run_provenance.txt') `
    -Value $provenance -Encoding UTF8

Write-Host "Run directory: $runDirectory"
Write-Host "Command: $commandText"
Push-Location $runDirectory
try {
    # Merge the native streams in cmd.exe before PowerShell sees them. Windows
    # PowerShell 5.1 otherwise converts normal solver warnings on stderr into
    # noisy NativeCommandError records (and ErrorAction=Stop aborts the run).
    $nativeCommand = $commandText + ' 2>&1'
    & $env:ComSpec /d /s /c $nativeCommand |
        Tee-Object -FilePath 'solver.log'
    $solverExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $solverExitCode
