# AntTracker

## Instalación
Busque el instalador para la última versión en la [sección Releases](https://github.com/fd-sturniolo/AntTracker/releases).

## Desarrollo

Para configurar el entorno de desarrollo necesitará ejecutar el script `create-env.ps1`.
Es imperativo que use este script en vez de instalar las dependencias manualmente ya que
hay post-procesado necesario luego de instalarlas.

El proyecto se compone de un módulo `ant_tracker` con tres submódulos:

- `labeler`
- `tracker`
- `tracker_gui`

### Requerimientos
- `git`
- `conda` (Miniconda o Anaconda)
- [`MakeNSIS`](https://nsis.sourceforge.io/Download) (para construir el instalador)

### Setup & Compilación
```powershell
git clone "https://github.com/fd-sturniolo/AntTracker.git"
cd AntTracker
.\create-env NOMBRE_ENV
conda activate NOMBRE_ENV
.\build                     # Compila los ejecutables a dist/AntTracker
MakeNSIS make_installer.nsi # Crea el instalador
```

### Nuevas versiones

Para incrementar el número de versión del software deberá usar [`bump2version`](https://github.com/c4urself/bump2version) 
(el cual se instala automáticamente en el environment de desarrollo).
Antes conviene agregar los cambios al [changelog](https://github.com/fd-sturniolo/AntTracker/blob/main/CHANGELOG.md).
El siguiente comando aumenta la versión `major`/`minor`/`patch` del software e instalador:

```powershell
bump2version [major/minor/patch]
```

Al pushear el commit y tag generado por el comando anterior a este repositorio (`git push --follow-tags`), una Release
es generada automáticamente con el instalador adjunto mediante
[una GitHub Action](https://github.com/fd-sturniolo/AntTracker/blob/main/.github/workflows/release.yml).

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
