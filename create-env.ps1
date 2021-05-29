param (
    [Parameter(Position=0,
               HelpMessage="Nombre del environment para AntTracker")]
    [ValidateNotNullOrEmpty()][string]
    $tracker_env_name,
    [Parameter(Position=1,
               HelpMessage="Nombre del environment para AntLabeler")]
    [ValidateNotNullOrEmpty()][string]
    $labeler_env_name,
    [Parameter(Mandatory=$false)]
    [Alias("h")]
    [switch]
    $help=$false
)

$invalid_call = $($($tracker_env_name -eq "") -or $($labeler_env_name -eq "") -or $($ffmpeg_dir -eq ""))

if ($help -or $invalid_call) {
    Write-Host "
Instala los conda-environments necesarios para el proyecto.

Se necesitan:
    conda
    git

Parámetros:
    tracker_env_name: Nombre del environment para AntTracker
    labeler_env_name: Nombre del environment para AntLabeler

Ejemplo:
./create-env.ps1 AntTracker AntLabeler
    "
    exit
}

try {
    $CWD=$(get-item .)
    #############################################################################
    ## AntTracker:
    Write-Host "Creando env: " $tracker_env_name -ForegroundColor Green
    conda create --name $tracker_env_name python=3.8 -y
    conda activate $tracker_env_name

    pip install -r requirements-tracker.txt

    #############################################################################
    ## AntLabeler:
    Write-Host "Creando env: " $labeler_env_name -ForegroundColor Green
    conda create --name $labeler_env_name python=3.8 -y
    conda activate $labeler_env_name

    pip install -r requirements-labeler.txt
    pip install PyForms-GUI==4.904.152 --no-deps

    ## patch pyforms
    $to = (get-item $(get-command python).Source ).DirectoryName+'\Lib\site-packages'
    Copy-Item ant_tracker\labeler\pyforms_patch\pyforms_gui -Destination $to -Recurse -Force

    #############################################################################
    # poner los nombres de los env en .env_info, después check_env.py se fija ahí
    # si el env correcto está activado
    Write-Output tracker:$tracker_env_name, labeler:$labeler_env_name > .env_info

    Write-Host "Finalizado. Sus environments son:" -ForegroundColor Green
    Get-Content .env_info
    conda activate base
}
catch {
    Write-Host "Ocurrió un error al configurar:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    Write-Host "Removiendo environments..."
    conda activate base
    Set-Location $CWD
    Remove-Item -Force -ErrorAction Ignore .env_info
    conda remove --name $tracker_env_name --all -y *>$null
    conda remove --name $labeler_env_name --all -y *>$null
    exit
}
