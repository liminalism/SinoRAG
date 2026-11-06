# Builds sinorag.exe from this checkout and packs it as the intake payload
# (sinorag.7z next to this script, which build.rs prefers over the payload
# the CBETA installers share).
#
#   .\make-payload.ps1                      # default features
#   .\make-payload.ps1 --features tensorrt  # extra arguments go to cargo build
#
# Run it whenever the SinoRAG sources change. The intake writes the indexes
# with the sinorag library it links and the installed sinorag.exe reads them,
# so both have to be built from the same source; build.rs refuses a payload
# of another version.

$here = $PSScriptRoot
$root = (Resolve-Path (Join-Path $here '..\..')).Path

# cargo and tar report progress on stderr. Windows PowerShell turns that into
# errors when the output is redirected, so they run under 'Continue' and are
# judged by exit code; everything else stops on the first error.
$ErrorActionPreference = 'Continue'
Push-Location $root
try {
    cargo build --release --bin sinorag @args
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
$ErrorActionPreference = 'Stop'

$targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root 'target' }
$exeDir = Join-Path $targetDir 'release'
$exe = Join-Path $exeDir 'sinorag.exe'
if (-not (Test-Path $exe)) { throw "cargo did not produce $exe" }

# Pack with the bsdtar that ships with Windows 10 and later; the intake
# unpacks it with sevenz-rust. Write beside the target and rename, so an
# interrupted run never leaves a truncated payload for build.rs to embed.
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
$out = Join-Path $here 'sinorag.7z'
$partial = "$out.partial"
$ErrorActionPreference = 'Continue'
& $tar --format=7zip --options '7zip:compression=lzma2,7zip:compression-level=9' -cf $partial -C $exeDir sinorag.exe
if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
$ErrorActionPreference = 'Stop'
Move-Item -Force $partial $out

$version = (& $exe --version)
$size = '{0:N1} MB' -f ((Get-Item $out).Length / 1MB)
Write-Host "Packed $version into $out ($size)"
