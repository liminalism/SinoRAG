# Builds the Document Intake release zip for a GitHub release:
#
#   .\make-release.ps1                # rebuild the payload, build, zip
#   .\make-release.ps1 -SkipPayload   # reuse the current sinorag.7z
#
# Output: dist\SinoRAG-Document-Intake-<sinorag version>-windows-x64.zip
# holding sinorag-intake.exe, sinorag-intake-uninstaller.exe and README.md.
[CmdletBinding()]
param([switch]$SkipPayload)

$here = $PSScriptRoot
Push-Location $here
try {
    if (-not $SkipPayload) {
        & (Join-Path $here 'make-payload.ps1')
        if (-not $?) { throw 'make-payload.ps1 failed' }
    }

    # cargo reports progress on stderr; judge it by exit code.
    $ErrorActionPreference = 'Continue'
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
    $ErrorActionPreference = 'Stop'

    $targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $here 'target' }
    $release = Join-Path $targetDir 'release'
    $files = @(
        (Join-Path $release 'sinorag-intake.exe'),
        (Join-Path $release 'sinorag-intake-uninstaller.exe'),
        (Join-Path $here 'README.md')
    )
    foreach ($f in $files) { if (-not (Test-Path $f)) { throw "missing $f" } }

    # The version of the SinoRAG the intake carries (build.rs checked the payload).
    $cargoToml = Get-Content (Join-Path $here '..\..\Cargo.toml') -Raw
    if ($cargoToml -notmatch '(?ms)^\[package\].*?^version\s*=\s*"([^"]+)"') { throw 'no SinoRAG version found' }
    $version = $Matches[1]

    $dist = Join-Path $here 'dist'
    New-Item -ItemType Directory -Force -Path $dist | Out-Null
    $zip = Join-Path $dist "SinoRAG-Document-Intake-$version-windows-x64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $files -DestinationPath $zip -CompressionLevel Optimal

    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    $size = '{0:N1} MB' -f ((Get-Item $zip).Length / 1MB)
    Set-Content -Path "$zip.sha256" -Value "$hash  $(Split-Path $zip -Leaf)" -Encoding ascii
    Write-Host "Release package: $zip ($size)"
    Write-Host "SHA-256: $hash"
} finally {
    Pop-Location
}
