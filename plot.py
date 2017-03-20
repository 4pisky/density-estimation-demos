import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_contours(f, xlim, ylim,
                     n_steps=200, n_contour=25,
                     ax=None, **kwargs):
    x = np.linspace(xlim[0], xlim[1], n_steps)
    y = np.linspace(ylim[0], ylim[1], n_steps)
    grid = np.dstack(np.meshgrid(x, y))

    # Flatten out the grid to a simple list of co-ordinate points, this is
    # sometimes acceptable when a 3-d array input is not.
    coords_vec = grid.reshape(grid.shape[0]*grid.shape[1],-1)
    values_vec = f(coords_vec)
    # Reshape the output back into a grid:
    values_grid = values_vec.reshape(grid.shape[0],grid.shape[1])
    print("SHAPE",values_grid.shape)

    if ax is None:
        ax = plt.gca()
    contourset = ax.contourf(x, y, values_grid, n_contour, **kwargs)
    return contourset, ax

def plot_crosshairs():
    pass