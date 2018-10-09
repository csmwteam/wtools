"""``mesh``: This module provides numerous methods and classes for discretizing data
in a convienant way that makes sense for our spatially refeerenced data/models.
"""

__all__ = [
    'meshgrid',
    'transpose',
    'saveUBC',
]

import numpy as np

def meshgrid(x, y, z=None):
    """Use this convienance method for your meshgrid needs. This ensures that
    we always use <ij> indexing to stay consistant with Cartesian grids.

    Note:
        This method handles 2D or 3D grids.
    """
    if z is not None:
        return np.meshgrid(x, y, z, indexing='ij')
    return np.meshgrid(x, y, indexing='ij')


def transpose(arr):
    """Transpose matrix from Cartesian to Earth Science coordinate system.
    This is useful for UBC Meshgrids where +Z is downself.
    <i,j,k> to <j,i,-k>
    """
    if (len(arr.shape) != 3):
        raise RuntimeError('argument must have 3 dimensions.')
    return np.flip(np.swapaxes(arr, 0, 1), 2)


def saveUBC(fname, x, y, z, models, header='Data', sz=False, origin=(0.0, 0.0, 0.0)):
    """Saves a 3D gridded array with spacing reference to the UBC mesh/model format.
    Use `PVGeo`_ to visualize this data. For more information on the UBC mesh
    format, reference the `GIFtoolsCookbook`_ website. This method assumes your
    mesh and data are defined on a normal cartesian system: <x,y,z>

    .. _PVGeo: http://pvgeo.org
    .. _GIFtoolsCookbook: https://giftoolscookbook.readthedocs.io/en/latest/content/fileFormats/mesh3Dfile.html

    Args:
        fname (str): the string file name of the mesh file. Model files will be
            saved next to this file.
        x (ndarray or float): a 1D array of unique coordinates along the X axis
        y (ndarray or float): a 1D array of unique coordinates along the Y axis
        z (ndarray or float): a 1D array of unique coordinates along the Z axis
        models (dict): a dictionary of models. Key is model name and value is a
            3D array with dimensions <x,y,z> containing cell data.
        header (str): a string header for your mesh/model files
        sx (bool): flag for whether to treat the (``x``, ``y``, ``z``) args as
            cell sizes
        origin (tuple(float)): optional origin value used only if ``sz==True``

    Example:
        >>> import numpy as np
        >>> # Create the unique coordinates along each axis
        >>> x = np.linspace(0, 100, 11)
        >>> y = np.linspace(220, 500, 11)
        >>> z = np.linspace(0, 50, 11)
        >>> # Create some model data
        >>> arr = np.array([i*j*k for i in range(10) for j in range(10) for k in range(10)]).reshape(10, 10, 10)
        >>> models = dict( foo=arr )
        >>> # Define the name of the file
        >>> fname = 'test'
        >>> # Perfrom the write out
        >>> saveUBC(fname, x, y, z, models, header='A simple model')


    """
    def arr2str(arr):
            return ' '.join(map(str, arr))

    shp = models.items()[0][1].shape
    for n, m in models.items():
        if m.shape != shp:
            raise RuntimeError('dimension mismatch in models.')

    nx, ny, nz = shp

    # Convert coordinates to cell sizes

    if sz:
        if isinstance(x, (float, int)):
            dx = np.full(nx, x)
        else:
            dx = x
        if isinstance(y, (float, int)):
            dy = np.full(ny, y)
        else:
            dy = y
        if isinstance(x, (float, int)):
            dz = np.full(nz, z)
        else:
            dz = z
        ox, oy, oz = origin
    else:
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        ox, oy, oz = np.min(x), np.min(y), np.max(z)




    if '.msh' not in fname:
        fname += '.msh'

    # Save out the data to UBC Tensor Mesh format
    with open(fname, 'w') as f:
        f.write('! %s\n' % header)
        f.write('%d %d %d\n' % (nx, ny, nz))
        f.write('%f %f %f\n' % (ox, oy, oz))
        f.write('%s\n' % arr2str(dx))
        f.write('%s\n' % arr2str(dy))
        f.write('%s\n' % arr2str(dz))

    # Save the model data
    for name, data in models.items():
        name = name.replace(' ', '-')
        mfnm = fname.split('.msh')[0] + '.%s' % name
        with open(mfnm, 'w') as f:
            f.write('! %s\n' % header)
            f.write('! %s\n' % ('Data name: %s' % name))
            f.write('%s\n' % '\n'.join(map(str, transpose(data).flatten())))

    return None
