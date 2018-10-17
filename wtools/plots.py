"""``plots``: This module provides various plotting routines that ensure we display
our spatially referenced data in logical, consistant ways across projects.
"""

__all__ = [
    'display',
    'plotStructGrid',
]

__displayname__ = 'Plotting Routines'

import numpy as np

from .geostats.grids import GridSpec

def display(plt, arr, x=None, y=None, **kwargs):
    """This provides a convienant class for plotting 2D arrays that avoids
    treating our data like images. Since most datasets we work with are defined
    on Cartesian coordinates, <i,j,k> == <x,y,z>, we need to transpose our arrays
    before plotting in image plotting libraries like ``matplotlib``.

    Args:
        plt (handle): your active plotting handle
        arr (np.ndarray): A 2D array to plot
        kwargs (dict): Any kwargs to pass to the ``pcolor`` plotting routine

    Return:
        plt.pcolor

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> arr = np.arange(1000).reshape((10,100))
        >>> wtools.display(plt, arr)
        >>> plt.title('What we actually want')
        >>> plt.colorbar()
        >>> plt.show()

    """
    if x is None or y is None:
        return plt.pcolor(arr.T, **kwargs)
    return plt.pcolor(x, y, arr.T, **kwargs)



def plotStructGrid(plt, outStruct, gridspecs, imeas=None):
    """Plot a semivariogram or covariogram produced from raster2structgrid

    Args:
        plt (handle): An active plotting handle. This allows us to use the
            plotted result after the routine.
        outStruct (np.ndarray): the data to plot
        gridspecs (list(GridSpec)): the spatial reference of your gdata
        imeas (str): key indicating which structural measure to label:
            ``'var'`` for semi-variogram or ``'covar'`` for covariogram.
            This simply adds a few labels to the active figure.
            If semi-variance use ``True``. If covariance, use ``False``.

    Return:
        plt.plot or plt.pcolor

    """
    # Check that gridspecs is a list of ``GridSpec`` objects
    if not isinstance(gridspecs, list):
        if not isinstance(gridspecs, GridSpec):
            raise RuntimeError('gridspecs arguments ({}) improperly defined.'. format(gridspecs))
        gridspecs = [gridspecs] # Make sure we have a list to index if only 1D

    # Check the `GridSpec` objects and ensure they have an ``nnodes``
    nDim = outStruct.ndim
    if nDim != len(gridspecs):
        raise RuntimeError('Number of data dimensions does not match given gridspecs.')
    for i, gs in enumerate(gridspecs):
        gs.validate()
        if gs.nnodes is None:
            raise RuntimeError('GridSpec object at index %d does not have an nnodes property.' % i)
        if gs.nnodes > gs.n*0.5:
            raise RuntimeError('For GridSpec at index %d nnodesOff > # of gridnodes/2.' % i);

    if nDim > 3:
        raise RuntimeError('Plotting routine can only handle 1D or 2D grids. Please extract a 2D slice.')

    # Check imeas
    variogram = None
    if imeas is not None:
        itypes = ['covar', 'var']
        if isinstance(imeas, int) and imeas < 2 and imeas > -1:
            imeas = itypes[imeas]
        if imeas not in itypes:
            raise RuntimeError("imeas argument must be one of 'covar' for covariogram or 'var' for semi-variance. Not {}".format(imeas))
        if imeas == 'var':
            variogram = True
        else:
            variogram = False

    if nDim == 1:  ### 1D case
        gs = gridspecs[0]

        if gs.n != outStruct.shape[0]:
            raise RuntimeError('gridspecs do not match input data')

        xax = gs.sz * np.arange(-gs.n / 2, gs.n / 2)

        if variogram is not None:
            plt.xlabel(r'lag distance $h$')
            if variogram:
                plt.title('Sample semivariogram')
                plt.ylabel(r'Semivariance  $\gamma (h)$')
            else:
                plt.title('Covariogram')
                plt.ylabel(r'Covariance  $\sigma (h)$')

        return plt.plot(xax, outStruct,'.-')

    # Otherwise it is 2D
    gsx = gridspecs[0]
    gsy = gridspecs[1]

    if gsx.n != outStruct.shape[0] or gsy.n != outStruct.shape[1]:
        raise RuntimeError('gridspecs do not match input data')

    xax = gsx.sz * np.arange(-gsx.n / 2, gsx.n / 2)
    yax = gsy.sz * np.arange(-gsy.n / 2, gsy.n / 2)

    Y_show, X_show = np.meshgrid(xax, yax)

    if variogram is not None:
        if variogram:
            plt.title('Sample Semivariogram Map' )
        else:
            plt.title('Sample Covariogram Map')

        plt.xlabel('h_x')
        plt.ylabel('h_y')
    return display(plt, outStruct, x=X_show, y=Y_show, cmap='nipy_spectral')
