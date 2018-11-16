"""``transform``: This module provides several conveinance methods for
transforming NumPy arrays in a Cartesian coordinate system.
"""

__all__ = [
    'meshgrid',
    'transpose',
    'emptyArray',
]


__displayname__ = 'Array Transforms'

import numpy as np

def meshgrid(x, y, z=None):
    """Use this convienance method for your meshgrid needs. This ensures that
    we always use <ij> indexing to stay consistant with Cartesian grids.

    This simply provides a wrapper for ``np.meshgrid`` ensuring we always use
    ``indexing='ij'`` which makes sense for typical Cartesian coordinate
    systems (<x,y,z>).

    Note:
        This method handles 2D or 3D grids.

    Example:
        >>> import wtools
        >>> import numpy as np
        >>> x = np.arange(20, 200, 10)
        >>> y = np.arange(20, 500, 20)
        >>> z = np.arange(0, 1000, 50)
        >>> xx, yy, zz = wtools.meshgrid(x, y, z)
        >>> # Now check that axii are ordered correctly
        >>> assert(xx.shape[0] == len(x))
        >>> assert(xx.shape[1] == len(y))
        >>> assert(xx.shape[2] == len(z))
    """
    if z is not None:
        return np.meshgrid(x, y, z, indexing='ij')
    return np.meshgrid(x, y, indexing='ij')


def transpose(arr):
    """Transpose matrix from Cartesian to Earth Science coordinate system.
    This is useful for UBC Meshgrids where +Z is down.

    Note:
        Works forward and backward.

    Args:
        arr (ndarray): 3D NumPy array to transpose with ordering: <i,j,k>

    Return:
        ndarray: same array transposed from <i,j,k> to <j,i,-k>

    Example:
        >>> import wtools
        >>> import numpy as np
        >>> model = np.random.random(1000).reshape((10, 20, 5))
        >>> wtools.transpose(model).shape
        (20, 10, 5)
    """
    if (len(arr.shape) != 3):
        raise RuntimeError('argument must have 3 dimensions.')
    return np.flip(np.swapaxes(arr, 0, 1), 2)


def emptyArray(shp):
    """Creates a NumPy ndarray of the given shape that is gaurnteed to be
    initialized to all NaN values

    Args:
        shp (tuple(int)): A tuple of integers specifiy the shape of the array
    """
    arr = np.empty(shp)
    arr[:] = np.nan
    return arr
