"""``plots``: This module provides various plotting routines that ensure we display
our spatially referenced data in logical, consistant ways across projects.
"""

__all__ = [
    'display',
]

def display(plt, arr, **kwargs):
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
    return plt.pcolor(arr.T, **kwargs)
