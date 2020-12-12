# AntLabeler

AntLabeler es un programa que permite etiquetar manualmente videos de hormigas en movimiento, guardando progresivamente el etiquetado en un archivo .json. Cuenta con la posibilidad de continuar el etiquetado de un mismo video a lo largo de múltiples sesiones.

## Uso

### Inicio

Al ejecutar el programa se abren dos selectores de archivos en secuencia. El primero permite seleccionar el video a etiquetar. El segundo permite seleccionar un archivo de etiquetas existente. Si no se selecciona ninguno, el archivo se generará automáticamente usando el mismo nombre de archivo que el video, con una extensión `.tag`. El proceso de generación puede tardar unos minutos.

### Reproductor

El componente principal es un reproductor de video, donde se visualiza el video a etiquetar, y superpuesto sobre él, el etiquetado vigente. En un inicio, las regiones que el proceso de generación reconoció como hormigas se encuentran marcadas en color rojo y un marcador cian. Estas áreas son las _regiones **inetiquetadas**_. El usuario deberá buscar  regiones inetiquetadas y asignarles una etiqueta, manteniendo correspondencia temporal cuadro a cuadro.

El video puede reproducirse normalmente con el botón _play_, o avanzar cuadro por cuadro con las teclas `A/D` o `⬅/➡`. Además, se puede hacer _zoom_ en el video con la rueda del mouse, y mover la imagen manteniendo apretado el botón de la rueda.

### Listado de objetos y selección

Del lado derecho se encuentra una lista de objetos con botones `+` y `-` para agregar y eliminar hormigas. Una vez agregadas, se podrá seleccionar una para proceder a etiquetarla. Cualquier acción de etiquetado que se realice con una hormiga seleccionada aplicará su identificador.

Además de su número y color identificador, en la lista aparece el intervalo de cuadros en el cual la hormiga se encuentra involucrada. Es esperable que haya sólo un intervalo, ya que la hormiga se mantiene en cámara desde que ingresa hasta que se retira, y no vuelve a aparecer; pero si así fuese etiquetada, aquí aparecerían más intervalos.

En esta lista se puede asignar el estado de cargada/no cargada a la hormiga. Este se mantiene durante toda la vida de la hormiga, siguiendo los supuestos del punto anterior.

Cuando se elimina una hormiga, todas las regiones en las que estaba etiquetada se convierten en inetiquetadas.

### Acciones de etiquetado

Hay tres modos de etiquetado, nombrados por su equivalente en un programa de dibujo: _Dibujar_, _Borrar_ y _Rellenar_.

En el modo Dibujar, hacer click sobre el cuadro etiqueta un círculo con el identificador de la hormiga seleccionada. El radio del círculo está dado por el selector que se encuentra al tope de la pantalla. El modo Borrar elimina cualquier etiqueta que se encuentre debajo de un círculo del mismo radio. Estos dos modos son útiles para refinar la segmentación producida a la creación del archivo de etiquetas, o bien para corregir errores.

El modo Rellenar detecta una región debajo del mouse y la etiqueta con el identificador seleccionado. Esta región puede ser inetiquetada o bien ya haberse etiquetado con otro identificador.

Cualquier acción de etiquetado con estos modos puede deshacerse presionando (y posiblemente manteniendo pulsado) `Ctrl+Z`.

#### Rellenado a futuro

La opción rellenado a futuro simplifica el etiquetado de una sola hormiga, al hacer una predicción de cuales regiones en los cuadros siguientes son probables a corresponder a la hormiga que acaba de etiquetarse. Al etiquetar en el modo Rellenar, se etiquetará esa misma hormiga en tantos cuadros como AntLabeler tenga seguridad de que no se producirán errores. Sin embargo, cualquier etiquetado que se produzca en cuadros futuros no podrá deshacerse con `Ctrl+Z`.

### Atajos de teclado

`(A/D)` o `(⬅/➡)`
~ Retroceder/avanzar un cuadro

`(W/S)` o `(⬆/⬇)`
~ Mover la selección de hormiga

`R`, `T`, `Y`
~ Dibujar, Borrar, Rellenar

`Ctrl+Z`
~ Deshacer último cambio (en este cuadro)

Tecla más/menos (`+`/`-`)
~ Aumentar/disminuir radio de dibujo

`Espacio`
~ Reproduce/detiene el video

`U`
~ Activar/desactivar rellenado a futuro

`M`
~ Mostrar/Ocultar máscara de etiquetado
