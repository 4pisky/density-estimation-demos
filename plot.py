import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.patches import Ellipse


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

    if ax is None:
        ax = plt.gca()
    contourset = ax.contourf(x, y, values_grid, n_contour, **kwargs)
    return contourset, ax

def get_2d_confidence_ellipse(mvgauss, confidence=0.95):
    """
    Calculate the ellipse containing given proportion of probability mass. 
    
    Args:
        mvgauss (MvGauss): (2-d) Multivariate Gaussian to use.
        confidence (float): Value from 0 to 1. If confidence=0.95, 95% of 
            samples will fall within this ellipse. 
    
    Returns:
        matplotlib.patches.Ellipse: The ellipse bounding the confidence region.
    """
    # A useful refererence:
    # http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    if len(mvgauss.mu)!=2:
        raise ValueError("Can only get confidence ellipse for 2-d Gaussian.")
        # Might implement ellipsoids / projections at some point in the future
    # Invert the CDF at ``confidence`` to get the chisq value.
    chisq = scipy.stats.chi2(df=2).ppf(confidence)
    # print("PPF Chisq:", chisq)
    # chisq = scipy.stats.chi2(df=2).isf(1. - confidence)
    # print("ISF Chisq:", chisq)
    chi = chisq**0.5
    eigval, eigvec = mvgauss.eig
    rotated_std_dev = eigval**0.5
    rotation_angle = np.rad2deg(np.arctan2(eigvec[0][1],eigvec[0][0]))

    ell = Ellipse(mvgauss.mu,
                  2*chi*rotated_std_dev[0],
                  2*chi*rotated_std_dev[1],
                  rotation_angle,
                  fill=False)
    return ell

def plot_principal_axes(mvgauss, ax=None,
                 **plot_kwargs):
    """
    Plot lines along the principal components, connecting the +/- 1-sigma points.
    
    """
    #Calculate the start and end co-ordinates for each principal vector
    start = -1*mvgauss.pcv.add(-mvgauss.mu, axis='index')
    end = mvgauss.pcv.add(mvgauss.mu, axis='index')

    if ax is None:
        ax = plt.gca()

    # Iterate over the component indices, drawing a line for each
    for c_idx in start:
        line_data = pd.concat((start[c_idx], end[c_idx]), axis=1)
        ax.plot(line_data.iloc[0], line_data.iloc[1], **plot_kwargs)
    return ax
