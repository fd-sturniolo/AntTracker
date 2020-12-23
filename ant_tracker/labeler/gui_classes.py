import pyforms_gui.utils.tools as tools
from AnyQt import QtCore, _api, uic  # noqa
from AnyQt.QtGui import QIcon, QPixmap, QFont, QColor
from AnyQt.QtWidgets import (QAbstractItemView, QComboBox, QDialogButtonBox, QGridLayout, QLabel,
                             QTableWidgetItem,
                             QWidget, QDialog, QFileDialog)
from confapp import conf
from pyforms.controls import ControlBase, ControlList, ControlPlayer, ControlFile, ControlLabel

from .classes import *
from confapp import conf

class ColorIcon:
    def __init__(self, r=None, g=None, b=None):
        icon = QPixmap(15, 15)  # 15px
        if r is None:
            icon.fill(QColor(*getNextColor()))  # * = tuple expansion
        else:
            icon.fill(QColor(r, g, b))
        self.icon = QIcon(icon)
        for size in self.icon.availableSizes():
            self.icon.addPixmap(self.icon.pixmap(size, QIcon.Normal, QIcon.Off), QIcon.Selected, QIcon.Off)
            self.icon.addPixmap(self.icon.pixmap(size, QIcon.Normal, QIcon.On), QIcon.Selected, QIcon.On)

class ControlListAnts(ControlList):
    def set_value(self, column, row, value):
        if isinstance(value, QWidget):
            self.tableWidget.setCellWidget(row, column, value)
            value.show()
            self.tableWidget.setRowHeight(row, value.height())
        elif isinstance(value, ControlBase):
            self.tableWidget.setCellWidget(row, column, value.form)
            value.show()
            self.tableWidget.setRowHeight(row, value.form.height())
        elif isinstance(value, ColorIcon):
            item = QTableWidgetItem()
            item.setIcon(value.icon)
            # item.setData(QtCore.Qt.EditRole, *args)
            self.tableWidget.setItem(row, column, item)
        else:
            args = [value]
            item = QTableWidgetItem()
            item.setData(QtCore.Qt.EditRole, *args)
            self.tableWidget.setItem(row, column, item)

    @property
    def single_select(self):
        return self.tableWidget.selectionBehavior()

    @single_select.setter
    def single_select(self, value):
        if value:
            self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        else:
            pass
            # self.tableWidget.setSelectionMode(QAbstractItemView.DragSelectingState)

class ControlPlayerAnts(ControlPlayer):
    def __init__(self, *args, **kwargs):
        super(ControlPlayerAnts, self).__init__(args, kwargs)
        self.frame_cache_len = 10
        self.frame_cache = {}
        self._current_frame_index = 0

    @property
    def video_index(self):
        return self._current_frame_index

    @video_index.setter
    def video_index(self, value):
        if value < 0: value = -1
        if value >= self.max: value = self.max - 1
        self._current_frame_index = value

    def reload_queue(self, index, display_index):
        # print(f"VC index: {int(self._value.get(1))}, index: {index}, display index: {display_index}")
        n_frames_to_load = self.frame_cache_len
        if not int(self._value.get(1)) == display_index:
            self._value.set(1, index)
            n_frames_to_load *= 2

        self.frame_cache.clear()
        for i in range(n_frames_to_load):
            success, frame = self._value.read()
            if not success:
                break
            self.frame_cache[index + i] = frame

    def get_frame(self, index, get_previous=False):
        # print(f"{list(self.frame_cache.keys())}")
        f = self.frame_cache.get(index, None)
        if f is not None:
            return f
        else:
            first_frame_to_get = max(0, index - self.frame_cache_len // 2 + 1) if get_previous else index
            self.reload_queue(first_frame_to_get, index)
            return self.frame_cache.get(index, None)

    def call_next_frame(self, update_slider=True, update_number=True, increment_frame=True, get_previous=False):
        # move the player to the next frame
        self.before_frame_change()
        self.form.setUpdatesEnabled(False)
        self._current_frame_index = self.video_index

        # if the player is not visible, stop
        if not self.visible:
            self.stop()
            self.form.setUpdatesEnabled(True)
            return

        # if no video is selected
        if self.value is None:
            self._current_frame = None
            self._current_frame_index = None
            return

        if len(self.frame_cache) == 0:  # first time drawing
            self._current_frame_index = 0
        else:
            self._current_frame_index += 1
        frame = self.get_frame(self._current_frame_index, get_previous=get_previous)

        # # no frame available. leave the function
        if frame is None:
            self.stop()
            self.form.setUpdatesEnabled(True)
            return
        self._current_frame = frame
        frame = self.process_frame_event(
            self._current_frame.copy()
        )

        # draw the frame
        if isinstance(frame, list) or isinstance(frame, tuple):
            self._video_widget.paint(frame)
        else:
            self._video_widget.paint([frame])

        if not self.videoProgress.isSliderDown():

            if update_slider and self._update_video_slider:
                self._update_video_slider = False
                self.videoProgress.setValue(self._current_frame_index)
                self._update_video_slider = True

            if update_number:
                self._update_video_frame = False
                self.videoFrames.setValue(self._current_frame_index)
                self._update_video_frame = True

        self.form.setUpdatesEnabled(True)
        self.after_frame_change()

    def videoPlay_clicked(self):
        """Slot for Play/Pause functionality."""
        # self.before_frame_change()
        if self.is_playing:
            self.stop()
        else:
            self.play()
        self.when_play_clicked()
        # self.after_frame_change()

    def videoProgress_sliderReleased(self):
        # self.before_frame_change()
        if self._update_video_slider:
            new_index = self.videoProgress.value()
            self.video_index = new_index
            # self._value.set(1, new_index)
            self.call_next_frame(update_slider=False, increment_frame=False, get_previous=True)
        # self.after_frame_change()

    def video_frames_value_changed(self, pos):
        # self.before_frame_change()
        if self._update_video_frame:
            self.video_index = pos - 1
            # self._value.set(1, pos) # set the video position
            self.call_next_frame(update_number=False, increment_frame=False, get_previous=True)
        # self.after_frame_change()

    def back_one_frame(self):
        """
        Back one frame.
        :return:
        """
        self.video_index -= 2
        self.call_next_frame(get_previous=True)

    def before_frame_change(self):
        pass

    def after_frame_change(self):
        pass

    def when_play_clicked(self):
        pass

    @property
    def move_event(self):
        return self._video_widget.onMove

    @move_event.setter
    def move_event(self, value):
        self._video_widget.onMove = value

class ControlFileAnts(ControlFile):
    def __init__(self, *args, **kwargs):
        self.__exec_changed_event = True
        super(ControlFile, self).__init__(*args, **kwargs)
        self.use_save_dialog = kwargs.get('use_save_dialog', False)
        self.filter = kwargs.get('filter', None)

    def click(self):
        if self.use_save_dialog:
            value, _ = QFileDialog.getSaveFileName(self.parent, self._label, self.value) # noqa
        else:
            if conf.PYFORMS_DIALOGS_OPTIONS:
                value = QFileDialog.getOpenFileName(self.parent, self._label, self.value,
                                                    filter=self.filter,
                                                    options=conf.PYFORMS_DIALOGS_OPTIONS)
            else:
                value = QFileDialog.getOpenFileName(self.parent, self._label, self.value,
                                                    filter=self.filter) # noqa

        if _api.USED_API == _api.QT_API_PYQT5:
            value = value[0]
        elif _api.USED_API == _api.QT_API_PYQT4:
            value = str(value)

        if value and len(value) > 0: self.value = value

class ControlLabelFont(ControlLabel):
    def __init__(self, *args, **kwargs):
        self._font = kwargs.get('font', QFont())
        super(ControlLabel, self).__init__(*args, **kwargs)

    def init_form(self):
        import inspect
        path = inspect.getfile(ControlLabel)

        control_path = tools.getFileInSameDirectory(path, "label.ui")
        self._form = uic.loadUi(control_path)
        self._form.label.setText(self._label)
        self._form.label.setFont(self._font)
        self._selectable = False # noqa
        super(ControlLabel, self).init_form()

# from pyforms_gui.controls.control_player.AbstractGLWidget import AbstractGLWidget, MouseEvent

# def mouseReleaseEvent(self, event):
#     super(AbstractGLWidget, self).mousePressEvent(event)
#     self.setFocus(QtCore.Qt.MouseFocusReason)

#     self._mouse_pressed = True
#     self._mouseX = event.x()
#     self._mouseY = event.y()
#     self._mouse_clicked_event = MouseEvent(event)

#     self.repaint()

# AbstractGLWidget.mouseReleaseEvent = mouseReleaseEvent

class ResolutionDialog(QDialog):
    def __init__(self, window_title, parent=None):
        super(ResolutionDialog, self).__init__(parent) # noqa
        self.setWindowTitle(window_title)
        label = QLabel("Resolución del video?")
        self.combo = QComboBox()
        self.combo.addItems(["Baja (640x480)", "Media (720x510)", "Alta (1080x720)"])
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)

        lay = QGridLayout(self)
        lay.addWidget(label)
        lay.addWidget(self.combo)
        lay.addWidget(box)

        self.setMinimumWidth(len(window_title) * 10)

    def get_selection(self):
        return ['low', 'med', 'high'][self.combo.currentIndex()]
