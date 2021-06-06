from collections import deque  # for edit history
from os.path import splitext, exists
from pathlib import Path
from typing import List, Tuple

from PyQt5.QtGui import QMouseEvent
from packaging.version import Version

import numpy as np
from AnyQt import QtCore, QtGui
from AnyQt.QtWidgets import QToolTip, QTreeWidgetItem, QMessageBox, QProgressDialog, QCheckBox, QApplication

# import pyforms_gui.allcontrols
from pyforms.basewidget import BaseWidget
from pyforms.controls import (ControlButton, ControlCheckBox,
                              ControlLabel, ControlList,
                              ControlSlider, ControlText)

import cv2 as cv
from .classes import *
from .gui_classes import *
from .PreLabeler import labelVideo

def clip(x, y, shape):
    if y >= shape[0]: y = shape[0] - 1
    if x >= shape[1]: x = shape[1] - 1
    if y < 0: y = 0
    if x < 0: x = 0
    return int(x), int(y)

MAX_EDIT_HIST = 1000
DEBUG = False
SEL_UP = 1
SEL_DOWN = -1
VID_FORWARD = 1
VID_BACKWARD = -1
ANT_ID_COLUMN = 0
ICON_COLUMN = 1
LOADED_COLUMN = 2
INVOLVED_FRAMES_COLUMN = 3
SPACER_LABEL = "========================================="
EDITING_LABEL = "🚫 Edición deshabilitada"
WARNING_UNLABEL = "⚠️ Hay regiones sin etiquetar en este cuadro!"
WARNING_UNLABELS = "⚠️ Hay regiones sin etiquetar en los siguientes cuadros: "
WARNING_REPEATED = "❌ Hay hormigas con más de una región.\nSólo se guardará la región más grande."
AUTOFILL_QUESTION = "Desea activar el rellenado a futuro?\n\n"
AUTOFILL_HELP = "Cuando se rellena un área sin etiquetar, " + \
                "las áreas en los cuadros siguientes intentan etiquetarse " + \
                "automáticamente. \n\n️️️️️⚠️ ¡Cuidado! Luego de rellenar, " + \
                "los cambios en frames siguientes no pueden deshacerse."
AUTOFILL_WARNING = "¡Ya hay una hormiga etiquetada en este cuadro!\n\n" + \
                   "Normalmente solo debería ser necesario rellenar una " + \
                   "vez por hormiga, por cuadro. Si necesita rellenar más de " + \
                   "una vez, desactive Rellenado a futuro."
LEAF_IMAGE = cv.imread("./leaf.png", cv.IMREAD_UNCHANGED)

class AntLabeler(BaseWidget):
    def __init__(self):
        super().__init__('AntLabeler')

        self._videofile = ControlFileAnts('Video a etiquetar', filter="Video (*.mp4 *.avi *.wmv *.webm *.h264)")
        self._tagfile = ControlFileAnts('Archivo de etiquetas (si no hay uno disponible, cancelar)',
                                        filter="Archivo de etiqueta (*.tag)")
        self._player = ControlPlayerAnts('Player')
        self._radslider = ControlSlider('Radio (+,-)', default=5, minimum=1, maximum=25)
        self._drawbutton = ControlButton('Dibujar (R)')
        self._erasebutton = ControlButton('Borrar (T)')
        self._fillbutton = ControlButton('Rellenar (Y)')
        self._unlerasebutton = ControlButton('Borrar todas las regiones sin etiquetar del cuadro')
        self._autofillcheck = ControlCheckBox('Rellenado a futuro (U)', helptext=AUTOFILL_HELP,
                                              default=True if DEBUG else False)
        self._labelscheck = ControlCheckBox('Esconder n°s y hojas (N)', default=True)
        self._maskcheck = ControlCheckBox('Esconder máscara (M)', default=False)
        self._editinglabel = ControlLabelFont(EDITING_LABEL, font=QtGui.QFont("Times", 12))
        self._unlabelwarn = ControlLabelFont(WARNING_UNLABEL, font=QtGui.QFont("Times", 11))
        self._unlabelswarn = ControlLabelFont(WARNING_UNLABEL, font=QtGui.QFont("Times", 9))
        self._repeatedwarn = ControlLabelFont(WARNING_REPEATED, font=QtGui.QFont("Times", 12))
        self._spacerlabel = ControlLabelFont(SPACER_LABEL, font=QtGui.QFont("Times", 10))

        headers = [""] * 4
        headers[ANT_ID_COLUMN] = "ID"
        headers[ICON_COLUMN] = "Color"
        headers[LOADED_COLUMN] = "Cargada"
        headers[INVOLVED_FRAMES_COLUMN] = "Cuadros"
        self._objectlist = ControlListAnts('Hormigas',
                                           add_function=self.__add_new_ant,
                                           remove_function=self.__remove_selected_ant,
                                           horizontal_headers=headers,
                                           select_entire_row=True,
                                           resizecolumns=True,
                                           item_selection_changed_event=self.__select_ant)
        self._objectlist.readonly = True
        self._objectlist.single_select = True

        self._formset = [
            (
                '_player',
                '||', '|',
                [
                    '_radslider',
                    ('_drawbutton', '_erasebutton', '_fillbutton',),
                    '_unlerasebutton',
                    '_autofillcheck',
                    ('_maskcheck', '_labelscheck'),
                    '_objectlist',
                    '=',
                    '_spacerlabel',
                    '_unlabelwarn',
                    '_unlabelswarn',
                    '_repeatedwarn',
                    '_editinglabel',
                ],
            )
        ]

        # self.antCollection = AntCollection()
        if DEBUG:
            self._videofile.value = "C:/f/pfc/ants/labeled1/labeler/HD13.mp4"
            self._player.value = self._videofile.value
            self._tagfile.value = "C:/f/pfc/ants/labeled1/labeler/HD13.tag"
            self.antCollection = AntCollection.deserialize(video_path=self._videofile.value,
                                                           filename=self._tagfile.value)
        else:
            self.__openFiles()

        self.colored_mask = self.antCollection.getMask(0)
        self.hist = deque(maxlen=MAX_EDIT_HIST)

        # Define the event that will be called when the run button is processed
        self._drawbutton.value = self.__drawEvent
        self._erasebutton.value = self.__eraseEvent
        self._fillbutton.value = self.__fillEvent
        self._unlerasebutton.value = self.__eraseUnlabeled
        self._radslider.changed_event = self.__radiusChange
        self._player.process_frame_event = self.__process_frame
        self._player.before_frame_change = self.__before_frame_change
        self._player.after_frame_change = self.__after_frame_change

        self._player.drag_event = self.__drag
        self._player.click_event = self.__click
        self._player.move_event = self.__mouse_move
        self._player.key_release_event = self.__keyhandler
        self._objectlist.key_release_event = self.__keyhandler

        self._player.when_play_clicked = self.__play_clicked
        self._autofillcheck.changed_event = self.__toggle_autofill
        self._maskcheck.changed_event = self.__toggle_mask
        self._labelscheck.changed_event = self.__toggle_labels

        self.currentFrame = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.draw_radius = 5
        self.clickAction = "draw"
        self.autofillEnabled = True if DEBUG else False
        self.__first_fill = True
        self.should_save = False
        self._drawbutton.enabled = False
        self._erasebutton.enabled = True
        self._fillbutton.enabled = True

        self.selected_ant_id: Optional[int] = None
        self.showMask = not self._maskcheck.value
        self.showLabels = not self._labelscheck.value
        self.unlabeledframes = ""
        self.__update_warning()
        self.__updateEditing()
        self.__fill_list()

        # Sino no se ven los bordes del video
        self.video_widget = self._player._video_widget  # noqa
        self.video_widget.zoom = 0.2
        self.video_widget.update()

        self.unlabeledframes = self.__get_unlabeled_frames()
        self._objectlist.setFocus()
        self._player.call_next_frame()

    def __openFiles(self):
        # Abre un diálogo de elección de archivo
        self._videofile.click()

        __vid = cv.VideoCapture(self._videofile.value, cv.CAP_FFMPEG)
        if __vid.isOpened():
            print(__vid.getBackendName())
            vshape = (int(__vid.get(cv.CAP_PROP_FRAME_HEIGHT)), int(__vid.get(cv.CAP_PROP_FRAME_WIDTH)))
            vlen = int(__vid.get(cv.CAP_PROP_FRAME_COUNT))
            self._fps = __vid.get(cv.CAP_PROP_FPS)
            __vid.release()
        else:
            raise CancelingException("%s no es un video válido" % self._videofile.value)

        # Asignar video al reproductor
        self._player.value = self._videofile.value

        # Nombre default de archivo de etiquetas: nombre_video.tag
        default_tagfile = splitext(self._videofile.value)[0] + ".tag"
        # self._tagfile.value = default_tagfile
        self._tagfile.click()
        #TODO: #9
        self._bkp = self._tagfile.value[:-4] + "-backup.tag"
        try:
            self.antCollection = AntCollection.deserialize(video_path=self._videofile.value,
                                                           filename=self._tagfile.value)
            if self.antCollection._old_version < Version("2.0"):
                from shutil import copy2
                copy2(self._tagfile.value, self._bkp)
        except Exception as e:  # noqa
            print(e)
            print("%s no es un archivo de etiquetas válido, generando uno" % self._tagfile.value)
            questionDialog = ResolutionDialog("Pre-etiquetado")
            if not questionDialog.exec_(): raise CancelingException("Creación de archivo de etiquetas cancelada.")
            print(questionDialog.get_selection())
            minimum_ant_radius = {"low": 4, "med": 8, "high": 10}[questionDialog.get_selection()]
            info = LabelingInfo(video_path=Path(self._videofile.value), ants=[], unlabeled_frames=[])
            self.antCollection = AntCollection(np.empty(vshape), info=info)
            progress = QProgressDialog("Creando un archivo de etiquetas...", "Cancelar", 0, vlen, self)
            progress.setWindowTitle("AntLabeler")
            progress.setMinimumWidth(400)
            progress.setWindowModality(QtCore.Qt.ApplicationModal)
            progress.setValue(0)
            for frame, mask in enumerate(labelVideo(self._videofile.value, minimum_ant_radius=minimum_ant_radius)):
                self.antCollection.addUnlabeledFrame(frame, mask)
                progress.setValue(frame)
                if progress.wasCanceled():
                    raise CancelingException("Creación de archivo de etiquetas cancelada.")
            self.antCollection.videoSize = mask.size  # noqa
            self.antCollection.videoShape = tuple(mask.shape)
            self.antCollection.videoLength = frame + 1  # noqa

            if exists(default_tagfile):
                file_i = 2
                tagfile = splitext(self._videofile.value)[0] + str(file_i) + ".tag"
                while exists(tagfile):
                    file_i += 1
                    tagfile = splitext(self._videofile.value)[0] + str(file_i) + ".tag"
            else:
                tagfile = default_tagfile
            self.antCollection.info.save(tagfile, pretty=True)
            self._tagfile.value = tagfile
            progress.setValue(vlen)

        if not (self.antCollection.videoShape == vshape):
            raise ValueError("Video %s con tamaño %s no coincide con archivo de etiquetas %s con tamaño %s"
                             % (self._videofile.value, str(vshape), self._tagfile.value,
                                str(self.antCollection.videoShape)))
        self.number_of_frames = vlen
        self.update_collection_version()

    def update_collection_version(self):
        #TODO: #9
        vlen = self.number_of_frames
        if self.antCollection._old_version < Version("1.1"):
            print("Cleaning v1 errors")
            self.antCollection.cleanErrors(vlen)
        if self.antCollection._old_version < Version("2.0"):
            self.antCollection.videoLength = self._bkp
            last_frame_w_ant = self.antCollection.getLastLabeledFrame()
            if last_frame_w_ant < self.antCollection.videoLength - 1:
                questionDialog = ResolutionDialog(f"Reprocesar frames faltantes (desde el {last_frame_w_ant})")
                if questionDialog.exec_():
                    minimum_ant_radius = {"low": 4, "med": 6, "high": 8}[questionDialog.get_selection()]

                    def msg(f):
                        return f"Reprocesando frame {f}/{vlen - 1}...\n" \
                               f"Un backup del archivo original se encuentra en\n" \
                               f"{self._bkp}"

                    progress = QProgressDialog(msg(last_frame_w_ant + 1), "Cancelar", 0, vlen - last_frame_w_ant + 1,
                                               self)
                    progress.setWindowTitle("AntLabeler")
                    progress.setMinimumWidth(400)
                    progress.setWindowModality(QtCore.Qt.ApplicationModal)
                    progress.setValue(0)
                    for frame, mask in enumerate(
                            labelVideo(self._videofile.value, minimum_ant_radius=minimum_ant_radius,
                                       start_frame=last_frame_w_ant + 1), start=last_frame_w_ant + 1
                    ):
                        self.antCollection.overwriteUnlabeledFrame(frame, mask)
                        progress.setValue(frame - last_frame_w_ant + 1)
                        progress.setLabelText(msg(frame))
                        if progress.wasCanceled():
                            raise CancelingException("Actualización de archivo de etiquetas cancelada.")
                    progress.setValue(vlen - last_frame_w_ant + 1)
        if self.antCollection._old_version < Version("2.1"):
            self.antCollection.info.video_fps_average = self._fps
        self.antCollection.version = CollectionVersion
        self.antCollection.info.version = CollectionVersion

    def closeEvent(self, event):
        print("saving...")
        self.should_save = True
        self.__saveFrame(self.currentFrame, pretty=True)

        print("closing...")
        event.accept()

    def resizeEvent(self, event):
        # print("wWidth: ",str(self.geometry().width()))
        if len(self._splitters) < 2:
            raise ValueError("Tiene que haber dos splitters!")
        windowWidth = self.geometry().width()
        windowHeight = self.geometry().height()
        # tools to warnings
        pRatio = 0.75
        self._splitters[0].setSizes([windowHeight * pRatio, windowHeight * (1 - pRatio)])
        # window to tools
        pRatio = 0.80
        self._splitters[1].setSizes([windowWidth * pRatio, windowWidth * (1 - pRatio)])
        self._player.setFocus()
        # return super().resizeEvent(event)

    def __mouse_move(self, x_, y_):
        x, y = clip(x_, y_, self.colored_mask.shape)
        self.mouse_x = x
        self.mouse_y = y
        tooltip = ""
        if not self.showLabels:
            ant_id = self.colored_mask[y, x]
            if ant_id not in [0, -1]:
                tooltip = f'<span style="color: white">ID: {ant_id}</span>'
                if self.antCollection.getAnt(ant_id).loaded:
                    tooltip += '<span style="color: white"> - Cargada</span>'
        self.video_widget.setToolTip(tooltip)

    def __click(self, e: QMouseEvent, x_, y_):
        x, y = clip(x_, y_, self.colored_mask.shape)
        if e.button == QtCore.Qt.RightButton:
            ant_id = self.colored_mask[y, x]
            if ant_id not in [0, -1]:
                self.__set_selected_ant(ant_id)
        if self.clickAction == "fill" and self.editingEnabled and self.selected_ant_id is not None:
            # print("About to fill")
            print(str((x, y)))
            print(self.colored_mask[y, x])
            if self.colored_mask[y, x] == 0:
                # print("Background, no fill")
                # NO rellenar fondo
                return e  # findClosestRegion()?
            doAutofill = self.autofillEnabled and self.colored_mask[y, x] == -1

            if self.__first_fill and not self.autofillEnabled and self.colored_mask[y, x] == -1 and not DEBUG:
                self.__first_fill = False
                if QMessageBox().question(self, 'Rellenado a futuro',
                                          AUTOFILL_QUESTION + AUTOFILL_HELP,
                                          QMessageBox.Yes, QMessageBox.No) == QMessageBox.Yes:
                    self._autofillcheck.value = True
            if doAutofill and np.any(self.colored_mask == self.selected_ant_id):
                # No hagamos autofill si esa hormiga ya está etiquetada en este frame
                QMessageBox().critical(self, 'Error', AUTOFILL_WARNING)  # noqa
                return e
            self.hist.append(self.colored_mask.copy())
            upcasted_mask = self.colored_mask.astype('int32')
            cv.floodFill(image=upcasted_mask,
                         mask=None,
                         seedPoint=(x, y),
                         newVal=self.selected_ant_id,
                         loDiff=0,
                         upDiff=0)
            self.colored_mask = upcasted_mask.astype('int16').copy()
            if doAutofill:
                # print("autofilling")
                self.antCollection.updateAreas(self.currentFrame, self.colored_mask)
                self.antCollection.labelFollowingFrames(self.currentFrame, self.selected_ant_id)
                self.__update_list(True)
            self._player.refresh()
            self.should_save = True
        return e

    def __drag(self, _, end):
        def inbounds(_x, _y):
            ix, iy = (_x, _y)
            my, mx = self.antCollection.videoShape
            if _x < 0:
                ix = 0
            elif _x >= mx:
                ix = mx - 1
            if _y < 0:
                iy = 0
            elif _y >= my:
                iy = my - 1
            return ix, iy

        # Si estamos moviendo la pantalla
        if self.video_widget._move_img: return  # noqa
        # Le corresponde a __click()
        if self.clickAction == "fill": return

        if self.editingEnabled:
            (x, y) = inbounds(np.int(end[0]), np.int(end[1]))
            if self.clickAction == "draw" and self.selected_ant_id is not None:
                self.hist.append(self.colored_mask.copy())
                cv.circle(self.colored_mask, (x, y), self.draw_radius, self.selected_ant_id, -1)
            elif self.clickAction == "erase":
                self.hist.append(self.colored_mask.copy())
                cv.circle(self.colored_mask, (x, y), self.draw_radius, 0, -1)
            self._player.refresh()
        self.should_save = True

    def __color_frame(self, frame) -> (np.ndarray, List[int]):
        """Retorna el frame coloreado con las áreas etiquetadas y no etiquetadas,
        junto con una lista de ids y su cantidad de componentes conectadas
        """

        # De https://stackoverflow.com/a/52742571
        def put4ChannelImageOn3ChannelImage(back, fore, x, y):
            back4 = cv.cvtColor(back, cv.COLOR_BGR2BGRA)
            rows, cols, channels = fore.shape
            trans_indices = fore[..., 3] != 0  # Where not transparent
            overlay_copy = back4[y:y + rows, x:x + cols]
            overlay_copy[trans_indices] = fore[trans_indices]
            back4[y:y + rows, x:x + cols] = overlay_copy
            back = cv.cvtColor(back4, cv.COLOR_BGRA2BGR)
            return back

        (height, width) = self.antCollection.videoShape
        middle = (width // 2, height // 2)
        ids_and_colors = {}
        for ant_id in np.unique(self.colored_mask):
            if ant_id == 0:  # Fondo
                continue
            elif ant_id == -1:  # Regiones sin etiquetas
                coloring = np.zeros_like(self.colored_mask, dtype='uint8')
                coloring[self.colored_mask == -1] = 255
                _, _, _, centroids = cv.connectedComponentsWithStats(coloring)
                coloring = cv.cvtColor(coloring, cv.COLOR_GRAY2BGR)
                coloring[:, :, 0:2] = 0
                frame = cv.addWeighted(coloring, 0.5, frame, 1, 0, dtype=cv.CV_8U)
                for c in centroids[1:]:
                    size = max(self.antCollection.videoShape) // 100
                    cv.drawMarker(frame, tuple(c.astype(int)), (255, 255, 0), cv.MARKER_DIAMOND, -1, size)
            else:  # Región con hormiga
                ant = self.antCollection.getAnt(ant_id)
                if ant is None:
                    print("None ant: ", ant_id)
                    color = (255, 255, 0)
                else:
                    color = ant.color[::-1]  # gets bgr instead of rgb
                ids_and_colors[ant_id] = color
        coloring = np.zeros_like(frame, dtype='uint8')
        for ant_id, color in ids_and_colors.items():
            coloring[self.colored_mask == ant_id] = color
        frame = cv.addWeighted(coloring, 0.8, frame, 1, 0, dtype=cv.CV_8U)
        dups = []
        if self.showLabels:
            coloring = np.zeros_like(self.colored_mask, dtype='uint8')
            coloring[self.colored_mask != -1] = self.colored_mask[self.colored_mask != -1]
            nlabels, labels, _, centroids = cv.connectedComponentsWithStats(coloring.astype('uint8'))
            found = []
            for label in np.unique(labels):
                if label == 0: continue
                c = centroids[label]
                w = np.argwhere(labels == label)[0, :]
                ant_id = self.colored_mask[w[0], w[1]]
                vector_to_middle = np.array(middle) - c
                vector_to_middle = vector_to_middle / np.linalg.norm(vector_to_middle)
                pos = tuple((c + vector_to_middle * 2).astype(int))
                ant = self.antCollection.getAnt(ant_id)
                found.append(ant.id)
                if ant is not None and ant.loaded:  # Dibujar la hojita
                    try:
                        frame = put4ChannelImageOn3ChannelImage(frame, LEAF_IMAGE, pos[0], pos[1] + 10)
                    except:  # noqa
                        pass
                cv.putText(frame, str(ant.id), pos, cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
            u, c = np.unique(found, return_counts=True)
            dups = u[c > 1]
        return frame, dups

    def __process_frame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        dups = []
        if self.showMask:
            frame, dups = self.__color_frame(frame)
        self.__update_warning(dups)
        return frame

    def __drawEvent(self):
        self.clickAction = "draw"
        self._drawbutton.enabled = False
        self._erasebutton.enabled = True
        self._fillbutton.enabled = True

    def __eraseEvent(self):
        self.clickAction = "erase"
        self._drawbutton.enabled = True
        self._erasebutton.enabled = False
        self._fillbutton.enabled = True

    def __fillEvent(self):
        self.clickAction = "fill"
        self._drawbutton.enabled = True
        self._erasebutton.enabled = True
        self._fillbutton.enabled = False

    def __eraseUnlabeled(self):
        if QMessageBox().question(self,
                                  "Borrar regiones sin etiquetar",
                                  "¿Está seguro de que quiere borrar las regiones "
                                  f"sin etiquetar del cuadro {self.currentFrame}?",
                                  QMessageBox.Yes, QMessageBox.No) == QMessageBox.Yes:
            self.antCollection.deleteUnlabeledFrame(self.currentFrame)
            self.colored_mask[self.colored_mask == -1] = 0
            self.should_save = True
            self._player.refresh()

    def __radiusChange(self):
        self.draw_radius = self._radslider.value

    def __update_loaded(self):
        for row in range(self._objectlist.tableWidget.rowCount()):
            ant_id = int(self._objectlist.value[row][ANT_ID_COLUMN])
            loaded = self._objectlist.value[row][LOADED_COLUMN].isChecked()
            self.antCollection.update_load(ant_id, loaded)

    def __saveFrame(self, frame=None, pretty=False):
        """
        Si frame==None, sólo se guarda el estado de carga de las hormigas
        """
        if self.editingEnabled:
            if frame is not None and self.should_save:
                print("saving frame %d" % self.currentFrame)
                self.antCollection.updateAreas(frame, self.colored_mask)
                self.unlabeledframes = self.__get_unlabeled_frames()
            self.__update_loaded()
            if self.should_save:
                self.antCollection.info.save(f"{self._tagfile.value[:-4]}.tag", pretty=pretty)

        self.should_save = False

    def __before_frame_change(self):
        self.__saveFrame(self.currentFrame)
        pass

    def __after_frame_change(self):
        self.currentFrame = self._player.video_index
        self.colored_mask = self.antCollection.getMask(self.currentFrame)
        self.__update_list(dry=True)
        self._player.refresh()

    def __update_warning(self, dups=None):
        repeated = dups and len(dups) > 0
        unlabeled = self.colored_mask.min() == -1
        unlabeleds = self.unlabeledframes != ""
        if unlabeled:
            self._unlabelwarn.value = WARNING_UNLABEL
        else:
            self._unlabelwarn.value = " "
        if repeated:
            self._repeatedwarn.value = WARNING_REPEATED
        else:
            self._repeatedwarn.value = " "
        if unlabeleds:
            self._unlabelswarn.value = WARNING_UNLABELS + '\n' + self.unlabeledframes
        else:
            self._unlabelswarn.value = " "

    def __moveByFrame(self, direction):
        if direction == VID_FORWARD:
            if self.currentFrame != self._player.max:
                self.hist.clear()
                self._player.forward_one_frame()
        elif direction == VID_BACKWARD:
            if self.currentFrame != 0:
                self.hist.clear()
                self._player.back_one_frame()

    def __add_new_ant(self):
        new_ant = self.antCollection.newAnt()
        self._objectlist + self.__get_list_item(new_ant)
        return

    def __on_loaded_check(self, _):
        self.should_save = True
        self.__saveFrame()
        self._player.refresh()

    def __get_unlabeled_frames(self) -> str:
        def formatter(groups):
            strings = [[]]
            for n, group in enumerate(groups):
                if group[0] == group[1]:
                    strings[-1].append(f"{group[0]}")
                else:
                    strings[-1].append(f"({group[0]}→{group[1]})")
                if (n + 1) % 5 == 0:
                    strings.append([])
            return "\n".join([", ".join(s) for s in strings])

        grps, nframes = self.antCollection.getUnlabeledFrameGroups()
        if nframes < 20:
            return formatter(grps)
        else:
            return ""

    @staticmethod
    def __get_involved_frames(ant: Ant) -> str:
        def formatter(groups):
            strings = []
            for group in groups:
                strings.append("(%d→%d)" % (group[0], group[1]))
            return " ∪ ".join(strings)

        return formatter(ant.getGroupsOfFrames())

    def __get_list_item(self, ant: Ant):
        list_item: List[Any] = [""] * 4
        list_item[ANT_ID_COLUMN] = ant.id
        list_item[ICON_COLUMN] = ColorIcon(*ant.color)
        list_item[LOADED_COLUMN] = QCheckBox("", self)
        list_item[INVOLVED_FRAMES_COLUMN] = self.__get_involved_frames(ant)

        if ant.loaded: list_item[LOADED_COLUMN].toggle()
        list_item[LOADED_COLUMN].stateChanged.connect(self.__on_loaded_check)
        return list_item

    def __fill_list(self):
        ant: Ant
        for ant in self.antCollection.ants:
            self._objectlist + self.__get_list_item(ant)

    def __update_list(self, dry=False):
        """Un update dry solamente actualiza la lista de cuadros de cada hormiga"""
        if dry:
            for row in range(self._objectlist.tableWidget.rowCount()):
                _id = int(self._objectlist.value[row][ANT_ID_COLUMN])
                newIF = self.__get_involved_frames(self.antCollection.getAnt(_id))
                self._objectlist.tableWidget.item(row, INVOLVED_FRAMES_COLUMN).setText(newIF)
        else:
            row = self._objectlist.tableWidget.currentRow()
            self._objectlist.value = []
            self.__fill_list()
            self.__set_selection(row)

    def __remove_selected_ant(self):
        # Mucho de esto es innecesario, probablemente. TODO: Revisar qué es necesario realmente
        if QMessageBox().question(self,
                                  f"Eliminar hormiga {self.selected_ant_id}",
                                  f"¿Está seguro de que desea eliminar la hormiga con ID: {self.selected_ant_id}? \n"
                                  "Esta acción no puede revertirse.",
                                  QMessageBox.Yes, QMessageBox.No) == QMessageBox.No:
            return
        print("id of removed ant:", str(self.selected_ant_id))
        self.antCollection.updateAreas(self.currentFrame, self.colored_mask)
        # afterUpdate = self.antCollection.getMask(self.currentFrame)

        self.antCollection.deleteAnt(self.selected_ant_id)
        self.colored_mask = self.antCollection.getMask(self.currentFrame)

        self.antCollection.updateAreas(self.currentFrame, self.colored_mask)
        self.colored_mask = self.antCollection.getMask(self.currentFrame)

        self._objectlist - (-1)  # Remove current row
        self.__select_ant()
        self.hist.clear()
        self._player.refresh()

    def __select_ant(self):
        try:
            _id = int(self._objectlist.get_currentrow_value()[ANT_ID_COLUMN])
            self.selected_ant_id = _id
        except:  # noqa
            self.selected_ant_id = None

    def __get_selected_ant(self) -> Ant:
        ant = list(filter(lambda a: a.id == self.selected_ant_id, self.antCollection.ants))[0]
        return ant

    def __undo(self):
        if self.editingEnabled and (len(self.hist) != 0):
            self.colored_mask = self.hist.pop()
            self._player.refresh()

    def __play_clicked(self):
        # print("playclicked")
        if self._player.is_playing:
            self.__updateEditing()
        else:
            self.__updateEditing()

    def __setEditing(self, true_or_false):
        if true_or_false:
            self._editinglabel.value = " "
            self.colored_mask = self.antCollection.getMask(self.currentFrame)
            self._player.refresh()
        else:
            self._editinglabel.value = EDITING_LABEL
        self.editingEnabled = true_or_false

    def __updateEditing(self):
        if self.showMask and not self._player.is_playing:
            self.__setEditing(True)
        else:
            self.__setEditing(False)

    def __change_radius(self, amt):
        v = self._radslider.value + amt
        if v > self._radslider.max:
            v = self._radslider.max
        elif v < 1:
            v = 1
        self._radslider.value = v

    def __set_selected_ant(self, ant_id):
        for row_idx in range(self._objectlist.tableWidget.rowCount()):
            i = self._objectlist.get_value(column=ANT_ID_COLUMN, row=row_idx)
            if i == str(ant_id):
                self.__set_selection(row_idx)
                break

    def __set_selection(self, row):
        col = 1
        if row >= self._objectlist.tableWidget.rowCount():
            row = self._objectlist.tableWidget.rowCount() - 1
        elif row < 0:
            row = 0
        self._objectlist.tableWidget.setCurrentCell(row, col)
        self.__select_ant()

    def __change_selection(self, direction):
        if self.selected_ant_id is None:
            row = 0 if direction == SEL_DOWN else self._objectlist.tableWidget.rowCount() - 1
            col = 0
        else:
            row = self._objectlist.tableWidget.currentRow()
            col = self._objectlist.tableWidget.currentColumn()
            if direction == SEL_UP:
                row = (row - 1) % self._objectlist.tableWidget.rowCount()
            elif direction == SEL_DOWN:
                row = (row + 1) % self._objectlist.tableWidget.rowCount()
        self._objectlist.tableWidget.setCurrentCell(row, col)
        self.__select_ant()

    def __toggle_autofill(self):
        self.autofillEnabled = self._autofillcheck.value

    def __toggle_mask(self):
        self.showMask = not self._maskcheck.value
        self._labelscheck.enabled = self.showMask
        self.__updateEditing()
        self._player.refresh()

    def __toggle_labels(self):
        self.showLabels = not self._labelscheck.value
        self._player.refresh()

    def __keyhandler(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key_Plus:
            self.__change_radius(1)
        if key == QtCore.Qt.Key_Minus:
            self.__change_radius(-1)
        if key == QtCore.Qt.Key_Space:
            self._player.videoPlay_clicked()
        if key == QtCore.Qt.Key_R:
            self.__drawEvent()
        if key == QtCore.Qt.Key_T:
            self.__eraseEvent()
        if key == QtCore.Qt.Key_Y:
            self.__fillEvent()
        if key == QtCore.Qt.Key_U:
            self._autofillcheck.value = not self._autofillcheck.value
        if key == QtCore.Qt.Key_M:
            self._maskcheck.value = not self._maskcheck.value
        if key == QtCore.Qt.Key_N:
            self._labelscheck.value = not self._labelscheck.value
        elif key in (QtCore.Qt.Key_A, QtCore.Qt.Key_Left):
            self.__moveByFrame(VID_BACKWARD)
        elif key in (QtCore.Qt.Key_D, QtCore.Qt.Key_Right):
            self.__moveByFrame(VID_FORWARD)
        elif key in (QtCore.Qt.Key_S, QtCore.Qt.Key_Down):
            self.__change_selection(SEL_DOWN)
        elif key in (QtCore.Qt.Key_W, QtCore.Qt.Key_Up):
            self.__change_selection(SEL_UP)
        elif (modifiers & QtCore.Qt.ControlModifier) != 0 and key == QtCore.Qt.Key_Z:
            self.__undo()
        elif (modifiers & QtCore.Qt.ControlModifier) != 0 and key == QtCore.Qt.Key_Delete:
            raise Exception("Force crash")

class CancelingException(Exception):
    pass

def excepthook(exc_type, exc_value, exc_tb):
    import traceback
    import datetime

    if exc_type is not CancelingException:
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(tb)
        filename = f"error-{datetime.datetime.now(tz=None).strftime('%Y-%m-%dT%H_%M_%S')}.log"
        with open(filename, "w") as f:
            f.write(tb)
        QMessageBox.critical(None, 'Error',
                             f"Se produjo un error. El archivo {filename} contiene los detalles.\n" +
                             "Por favor envíe el mismo y el archivo .tag con el que estaba trabajando "
                             "a la persona de quien recibió este programa.")  # noqa
    QApplication.exit()  # noqa

def main():
    from pyforms import start_app
    import sys

    # from shutil import copy
    # try:
    # import cProfile
    # cProfile.run('start_app(AntLabeler)','profile')
    sys.excepthook = excepthook
    app = start_app(AntLabeler)
    sys.exit(0)
    # except:
    # copy("Video16cr.tag_original","Video16cr.tag")
    # raise Exception()
    # copy("Video16cr.tag_original","Video16cr.tag")

if __name__ == '__main__':
    main()
