import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_contours(f, xlim, ylim,
                     n_steps=200, n_contour=25,
                     colorbar=True,
                     ax=None, **kwargs):
    x = np.linspace(xlim[0], xlim[1], n_steps)
    y = np.linspace(ylim[0], ylim[1], n_steps)
    grid = np.dstack(np.meshgrid(x, y))
    if ax is None:
        ax = plt.gca()
    contourset = ax.contourf(x, y, f(grid), n_contour)
    if colorbar:
        plt.colorbar(contourset)
    return ax
