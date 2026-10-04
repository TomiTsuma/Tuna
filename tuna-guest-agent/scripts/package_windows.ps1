param(
    [Parameter(Mandatory = $true)]
    [string]$BuildDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$QtBinDirectory
)

$ErrorActionPreference = "Stop"

$buildRoot = (Resolve-Path -LiteralPath $BuildDirectory).Path
$binaryDirectory = Join-Path $buildRoot "bin"
$guiExecutable = Join-Path $binaryDirectory "tuna_gui.exe"
$sampleExecutable = Join-Path $binaryDirectory "tuna_sample_app.exe"
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)

if (!(Test-Path -LiteralPath $guiExecutable -PathType Leaf)) {
    throw "GUI executable was not found: $guiExecutable"
}
if (!(Test-Path -LiteralPath $sampleExecutable -PathType Leaf)) {
    throw "Sample client executable was not found: $sampleExecutable"
}

$windeployqt = $null
if ($QtBinDirectory) {
    $candidate = Join-Path (Resolve-Path -LiteralPath $QtBinDirectory).Path "windeployqt.exe"
    if (!(Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "windeployqt.exe was not found in QtBinDirectory: $candidate"
    }
    $windeployqt = $candidate
} else {
    $command = Get-Command "windeployqt.exe" -ErrorAction SilentlyContinue
    if ($command) {
        $windeployqt = $command.Source
    }
}
if (!$windeployqt) {
    throw "Set QtBinDirectory or add the matching Qt kit's windeployqt.exe to PATH."
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
Copy-Item -LiteralPath $guiExecutable -Destination $outputRoot -Force
Copy-Item -LiteralPath $sampleExecutable -Destination $outputRoot -Force

Get-ChildItem -LiteralPath $binaryDirectory -Filter "*.dll" -File |
    Copy-Item -Destination $outputRoot -Force

$packagedGui = Join-Path $outputRoot "tuna_gui.exe"
& $windeployqt --no-translations --dir $outputRoot $packagedGui
if ($LASTEXITCODE -ne 0) {
    throw "windeployqt failed with exit code $LASTEXITCODE."
}

Write-Output "Packaged unsigned development build at $outputRoot"
