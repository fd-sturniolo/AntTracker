param (
    [Alias("h")]
    [switch]
    $help=$false,

    [switch]
    $Labeler=$false,
    [switch]
    $Tracker=$false,
    [switch]
    $All=$false
)

$invalid_call = $(!$($Labeler -xor $Tracker -xor $All))

if ($help -or $invalid_call) {
    Write-Host "
Compila AntTracker, AntLabeler, o ambos y empaqueta AntLabeler dentro de AntTracker.

Usar una de las siguientes flags:
    -Labeler
    -Tracker
    -All

Ejemplo:
./build.ps1 -All
    "
    exit
}

if !(Test-Path .\.env_info) {
    Write-Host "
Cree los environments primero: ver create-env.ps1
    "
    exit
}

foreach ($line in $(Get-Content -Path .\.env_info)) {
    if ($line.StartsWith("tracker")) {
        $tracker_env = $line.Split(":")[1]
    }
    if ($line.StartsWith("labeler")) {
        $labeler_env = $line.Split(":")[1]
    }
}

if ($Labeler -or $All) {
    conda activate $labeler_env
    pyinstaller AntLabeler.spec --onedir -y
}
if ($Tracker -or $All) {
    conda activate $tracker_env
    pyinstaller AntTracker.spec --onedir -y
}
if ($All) {
    python copy_labeler_files.py
}
conda activate base
