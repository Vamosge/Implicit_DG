$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$toolRoot = Join-Path $root '.tools'
New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null

function Install-ZipTool([string]$Url, [string]$Destination, [string]$NestedDirectory) {
    if (Test-Path -LiteralPath $Destination) { return }
    $archive = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName() + '.zip')
    $extract = Join-Path $env:TEMP ([IO.Path]::GetRandomFileName())
    try {
        Invoke-WebRequest -Uri $Url -OutFile $archive
        Expand-Archive -LiteralPath $archive -DestinationPath $extract
        $source = if ($NestedDirectory) { Join-Path $extract $NestedDirectory } else { $extract }
        Move-Item -LiteralPath $source -Destination $Destination
    }
    finally {
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Install-ZipTool 'https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-windows-x86_64.zip' `
    (Join-Path $toolRoot 'cmake') 'cmake-4.4.4-windows-x86_64'
Install-ZipTool 'https://github.com/ninja-build/ninja/releases/download/v1.13.1/ninja-win.zip' `
    (Join-Path $toolRoot 'ninja') ''
Install-ZipTool 'https://ziglang.org/download/0.17.0/zig-x86_64-windows-0.17.0.zip' `
    (Join-Path $toolRoot 'zig') 'zig-x86_64-windows-0.17.0'

Write-Host 'Portable CMake, Ninja, and Zig toolchains are ready.'
