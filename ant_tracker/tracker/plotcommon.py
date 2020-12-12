import matplotlib
import matplotlib.colors as colors
import mpl_toolkits.axes_grid1
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.pyplot import Axes
from matplotlib.widgets import Button
from typing import Dict, Optional, Union

from .common import BinaryMask, Image_T

class Animate:
    __hash: Dict[Axes, AxesImage] = dict()

    @classmethod
    def draw(cls, axis: Axes, img: Union[Image_T, BinaryMask], autoscale=False, override_hash=False, cmap='viridis'):
        def get_clim():
            if img.dtype in [bool, float, 'float32', 'float64']:
                _vmin, _vmax = 0, 1
            else:
                _vmin, _vmax = 0, 255
            if autoscale:
                _vmin, _vmax = None, None
            return _vmin, _vmax

        vmin, vmax = get_clim()
        axes_image: Optional[AxesImage] = cls.__hash.get(axis, None)
        if axes_image is None or override_hash:
            axis.set_yticks([])
            axis.set_xticks([])
            axes_image = axis.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
            cls.__hash[axis] = axes_image
        else:
            if autoscale:
                axes_image.set_norm(None)
            else:
                vmin, vmax = get_clim()
                axes_image.set_clim(vmin, vmax)
            axes_image.set_data(img)
            axes_image.set_cmap(cmap)

# Visto en https://stackoverflow.com/a/41152160
class PageSlider(matplotlib.widgets.Slider):
    def __init__(self, ax, label, numpages=10, valinit=0, valfmt='%1d',
                 closedmin=True, closedmax=True,
                 dragging=True, **kwargs):

        self.facecolor = kwargs.get('facecolor', "w")
        self.activecolor = kwargs.pop('activecolor', "b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = numpages

        super(PageSlider, self).__init__(ax, label, 0, numpages,
                                         valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i == valinit else self.facecolor
            r = matplotlib.patches.Rectangle((float(i) / numpages, 0), 1. / numpages, 1,
                                             transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i) / numpages + 0.5 / numpages, 0.5, str(i + 1),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax, label=u'$\u25C0$',
                                                     color=self.facecolor, hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax, label=u'$\u25B6$',
                                                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >= self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i + 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i - 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

class Log1pInvNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        return np.ma.masked_array((1 - (np.log1p(value) / np.log1p(self.vmax))) ** 0.85)
