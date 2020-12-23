# AntTracker

## Instrucciones

El proyecto hace uso de dos environments de `conda` con algunas librerías en común, 
y se compone de un módulo `ant_tracker` con tres submódulos:

| Módulo | Environment|
|--------|------------|
|`labeler` |`NOMBRE_ENV_LABELER`|
|`tracker` |`NOMBRE_ENV_TRACKER`|
|`tracker_gui`|`NOMBRE_ENV_TRACKER`|

El script `create-env.ps1` le ayudará a crear los environments.

#### Requerimientos
- `git`
- `conda` (Miniconda o Anaconda)
- Binarios de `ffmpeg-dev` o `ffmpeg-shared`.
    - Puede conseguir binarios compilados para Windows en http://acyun.org/ o https://ottverse.com/ffmpeg-builds/
    - De otra manera, deberá compilar [desde la fuente](https://ffmpeg.org). Buena suerte.

#### Setup & Compilación
```powershell
git clone "https://github.com/fd-sturniolo/AntTracker.git"
cd AntTracker
.\create-env NOMBRE_ENV_TRACKER NOMBRE_ENV_LABELER DIR_FFMPEG
.\build -All
```

#### Distribución

Actualmente la carpeta generada `dist/AntTracker` se empaqueta en un instalador con 
[InstallSimple](http://installsimple.com/). El ejecutable requiere instalar el 
[paquete Visual C++ Redistributable](https://www.microsoft.com/es-es/download/details.aspx?id=48145).
A futuro, se empaquetará mediante [NSIS](https://nsis.sourceforge.io/Main_Page) 
y la instalación del Redist será automática.

## Información

Desarrollado durante 2019-2020 por Francisco Daniel Sturniolo,
en el marco de su Proyecto Final de Carrera para el título de Ingeniero en Informática
de la Facultad de Ingeniería y Ciencias Hídricas de la Universidad Nacional del Litoral,
bajo la dirección de Leandro Bugnon y la co-dirección de Julián Sabattini, 
titulado "Desarrollo de una herramienta para identificación automática del ritmo de forrajeo
de hormigas cortadoras de hojas a partir de registros de video".


El mismo pretende analizar el comportamiento de forrajeo de las HCH a partir de videos tomados de la salida de un
hormiguero (tales como los obtenidos a partir del dispositivo AntVRecord), detectando las trayectorias tomadas por las
hormigas y su posible carga de hojas, para luego extraer estadísticas temporales de su comportamiento
y volumen vegetal recolectado.


También incluido con este programa se encuentra AntLabeler, una utilidad de etiquetado para videos de la misma índole, 
que fue utilizada para validar los resultados obtenidos por AntTracker sobre videos de prueba. El uso de esta
herramienta actualmente se encuentra supercedido por AntTracker, pero se provee como una forma de revisar con precisión
las trayectorias y cargas detectadas.


## Legales

This software uses libraries from the FFmpeg project under the LGPLv2.1.
