param (
    [Parameter(Position=0,
               HelpMessage="Nombre del environment a crear")]
    [ValidateNotNullOrEmpty()][string]
    $env_name,
    [Parameter(Mandatory=$false)]
    [Alias("h")]
    [switch]
    $help=$false,
    [Parameter(Mandatory=$false)]
    [switch]
    $force=$false
)

$invalid_call = $($env_name -eq "")

if ($help -or $invalid_call) {
    Write-Host "
Instala el conda environment necesario para el proyecto.

Deberá tener instalado conda, y tener la dirección del ejecutable en PATH.

Parámetros:
    env_name: Nombre del environment creado

Ejemplo:
./create-env.ps1 AntTracker
    "
    exit
}

if ($(Test-Path .env_info) -and $(!$force)) {
    Write-Host $force
    $env = $(Get-Content .env_info).Split(':')[1]
    Write-Host "
Ya existe un conda environment creado para este proyecto: $env
Puede ver un log de la creación del environment en:
    $(Get-Item env.log)
Si desea crear un nuevo environment para este proyecto, elimine el archivo .env_info y vuelva a ejecutar este script.
Alternativamente, ejecute el script con la opción -force
    "
    exit
}

try {
    $CWD=$(get-item .)

    Start-Transcript -Path env.log -IncludeInvocationHeader

    Write-Host "Creando env: $env_name" -ForegroundColor Green
    conda create --name $env_name python=3.8 -y                                                 2>&1 | Write-Output
    conda activate $env_name                                                                    2>&1 | Write-Output

    pip install -r .\requirements.txt                                                           2>&1 | Write-Output

    pip install PyForms-GUI==4.904.152 --no-deps                                                2>&1 | Write-Output

    ## patch pyforms
    $to = (get-item $(get-command python).Source ).DirectoryName+'\Lib\site-packages'
    Copy-Item ant_tracker\labeler\pyforms_patch\pyforms_gui -Destination $to -Recurse -Force    2>&1 | Write-Output

    # poner el nombre del env en .env_info, después check_env.py se fija ahí
    Write-Output tracker:$env_name | Out-File ".env_info" -Encoding utf8                        2>&1 | Write-Output

    Write-Host "Finalizado. Su environment es: $env_name" -ForegroundColor Green
    Write-Host "Ejecute ``conda activate $env_name`` para activarlo."
    conda activate base

    Stop-Transcript
}
catch {
    Write-Host "Ocurrió un error al configurar:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    Write-Host "Removiendo environment..."
    conda activate base
    Set-Location $CWD
    Remove-Item -Force -ErrorAction Ignore .env_info
    conda remove --name $env_name --all -y *>$null
    Stop-Transcript
    exit
}
