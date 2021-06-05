param (
    [Alias("h")]
    [switch]
    $help=$false
)

if ($help) {
    Write-Host "
Compila AntTracker y AntLabeler.
    "
    exit
}

if (!(Test-Path .\.env_info)) {
    Write-Host "
Cree el environment primero: ver create-env.ps1
    "
    exit
}

foreach ($line in $(Get-Content -Path .\.env_info)) {
    if ($line.StartsWith("tracker")) {
        $env = $line.Split(":")[1]
    }
    else {
        Write-Host "
Hay un error con el environment. Chequee $(Get-Item env.log)
"
        exit
    }
}

conda activate $env
pyinstaller build.spec -y
conda deactivate
