param (
    [Parameter(Position=0,
               HelpMessage="Nombre del environment para AntTracker")]
    [ValidateNotNullOrEmpty()][string]
    $tracker_env_name,
    [Parameter(Position=1,
               HelpMessage="Nombre del environment para AntLabeler")]
    [ValidateNotNullOrEmpty()][string]
    $labeler_env_name,
    [Parameter(Position=2,
               HelpMessage="Directorio que contiene ffmpeg")]
    [ValidateNotNullOrEmpty()][string]
    $ffmpeg_dir,
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
    ffmpeg (dev/shared)
Puede conseguir binarios compilados de ffmpeg en http://acyun.org/ o https://ottverse.com/ffmpeg-builds/
De otra manera, deberá compilar de la fuente: https://ffmpeg.org

Parámetros:
    tracker_env_name: Nombre del environment para AntTracker
    labeler_env_name: Nombre del environment para AntLabeler
    ffmpeg_dir: Directorio que contiene ffmpeg

Ejemplo:
./create-env.ps1 AntTracker AntLabeler C:\ffmpeg-dev
    "
    exit
}

## TODO: chequear de alguna forma más robusta si con la carpeta provista se puede buildear PyAV
if (-not $(Test-Path $ffmpeg_dir"\lib\avcodec.lib")) {
    Write-Host "El directorio $ffmpeg_dir no contiene los archivos necesarios."
    Write-Host "
Puede conseguir binarios compilados de ffmpeg en http://acyun.org/ o https://ottverse.com/ffmpeg-builds/
De otra manera, deberá compilar de la fuente: https://ffmpeg.org

Asegúrese de obtener la versión dev o shared.
    "
    exit
}

try {
    $CWD=$(get-item .)

    ## AntTracker:
    Write-Host "Creando env: " $tracker_env_name -ForegroundColor Green
    conda create --name $tracker_env_name python=3.8 -y
    conda activate $tracker_env_name

    pip install -r requirements-tracker.txt

    ## you're also gonna need to build PyAV from source, using cython=0.28 and bundling ffmpeg (shared or dev)
    ## I've managed to get it from http://acyun.org/ or https://ottverse.com/ffmpeg-builds/
    ## https://pyav.org/docs/6.1.2/installation.html::build-on-windows:~:text=On%20Windows%20you%20must%20indicate%20the%20location%20of%20your%20FFmpeg%2C%20e.g.%3A
    conda install cython=0.28 -y
    git clone https://github.com/PyAV-Org/PyAV temp/PyAV
    Set-Location temp/PyAV
    python setup.py build --ffmpeg-dir=$ffmpeg_dir
    python setup.py install
    conda remove cython -y
    Set-Location ../..
    Remove-Item -Recurse -Force temp

    # mandar las libs a la carpeta lib del environment
    Write-Output (get-item lib).ToString() > ($env:CONDA_PREFIX+'\lib\site-packages\ants.pth')


    ## AntLabeler:
    Write-Host "Creando env: " $labeler_env_name -ForegroundColor Green
    conda create --name $labeler_env_name python=3.8 -y
    conda activate $labeler_env_name

    pip install -r requirements-labeler.txt
    pip install PyForms-GUI==4.904.152 --no-deps

    ## patch pyforms
    $to = (get-item $(get-command python).Source ).DirectoryName+'\Lib\site-packages'
    Copy-Item ant_tracker\labeler\pyforms_patch\pyforms_gui -Destination $to -Recurse -Force

    # mandar las libs a la carpeta lib del environment
    Write-Output (Get-Item lib).ToString() > ($env:CONDA_PREFIX+'\lib\site-packages\ants.pth')
    #|| goto error

    # poner los nombres de los env en .env_info, después check_env.py se fija ahí
    # si el env correcto está activado
    Write-Output tracker:$tracker_env_name, labeler:$labeler_env_name > .env_info

    Write-Host "Finalizado. Sus environments son:" -ForegroundColor Green
    Get-Content .env_info
    conda activate base
}
catch {
    Write-Host "Ocurrió un error al instalar:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    Write-Host "Removiendo environments..."
    conda activate base
    Set-Location $CWD
    Remove-Item -Recurse -Force -ErrorAction Ignore temp
    Remove-Item -Force -ErrorAction Ignore .env_info
    conda remove --name $tracker_env_name --all -y *>$null
    conda remove --name $labeler_env_name --all -y *>$null
    exit
}
