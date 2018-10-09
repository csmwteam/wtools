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
    This is useful for UBC Meshgrids where +Z is down.

    Note:
        Works forward and backward.

    Args:
        arr (ndarray): 3D NumPy array to transpose with ordering: <i,j,k>

    Return:
        ndarray: same array transposed from <i,j,k> to <j,i,-k>
    """
    if (len(arr.shape) != 3):
        raise RuntimeError('argument must have 3 dimensions.')
    return np.flip(np.swapaxes(arr, 0, 1), 2)


def saveUBC(fname, x, y, z, models, header='Data', widths=False, origin=(0.0, 0.0, 0.0)):
    """Saves a 3D gridded array with spatail reference to the UBC mesh/model format.
    Use `PVGeo`_ to visualize this data. For more information on the UBC mesh
    format, reference the `GIFtoolsCookbook`_ website.

    Note:
        This method assumes your mesh and data are defined on a normal cartesian
        system: <x,y,z>

    .. _PVGeo: http://pvgeo.org
    .. _GIFtoolsCookbook: https://giftoolscookbook.readthedocs.io/en/latest/content/fileFormats/mesh3Dfile.html

    Args:
        fname (str): the string file name of the mesh file. Model files will be
            saved next to this file.
        x (ndarray or float): a 1D array of unique coordinates along the X axis,
            float for uniform cell widths, or an  array with ``widths==True``
            to treat as cell spacing on X axis
        y (ndarray or float): a 1D array of unique coordinates along the Y axis,
            float for uniform cell widths, or an  array with ``widths==True``
            to treat as cell spacing on Y axis
        z (ndarray or float): a 1D array of unique coordinates along the Z axis,
            float for uniform cell widths, or an  array with ``widths==True``
            to treat as cell spacing on Z axis
        models (dict): a dictionary of models. Key is model name and value is a
            3D array with dimensions <x,y,z> containing cell data.
        header (str): a string header for your mesh/model files
        widths (bool): flag for whether to treat the (``x``, ``y``, ``z``) args as
            cell sizes/widths
        origin (tuple(float)): optional origin value used if ``widths==True``,
            or used on a component basis if any of the ``x``, ``y``, or ``z``
            args are scalars.

    Return:
        None: saves out a mesh file named {``fname``}.msh and a model file for
            every key/value pair in the ``models`` argument (key is file
            extension for model file and value is the data.

    Examples:
        >>> import numpy as np
        >>> # Create the unique coordinates along each axis : 11 nodes on each axis
        >>> x = np.linspace(0, 100, 11)
        >>> y = np.linspace(220, 500, 11)
        >>> z = np.linspace(0, 50, 11)
        >>> # Create some model data: 10 cells on each axis
        >>> arr = np.array([i*j*k for i in range(10) for j in range(10) for k in range(10)]).reshape(10, 10, 10)
        >>> models = dict( foo=arr )
        >>> # Define the name of the file
        >>> fname = 'test'
        >>> # Perfrom the write out
        >>> saveUBC(fname, x, y, z, models, header='A simple model')
        >>> # Two files saved: 'test.msh' and 'test.foo'

        >>> # Uniform cell sizes
        >>> d = np.random.random(1000).reshape((10, 10, 10))
        >>> v = np.random.random(1000).reshape((10, 10, 10))
        >>> models = dict(den=d, vel=v)
        >>> saveUBC('volume', 25, 25, 2, models, widths=True, origin=(200.0, 100.0, 500.0))
        >>> # Three files saved: 'volume.msh', 'volume.den', and 'volume.vel'


    """
    def arr2str(arr):
            return ' '.join(map(str, arr))

    shp = list(models.items())[0][1].shape
    for n, m in models.items():
        if m.shape != shp:
            raise RuntimeError('dimension mismatch in models.')


    nx, ny, nz = shp

    def _getWidths(nw, w, widths=False):
        # Convert scalars if necessary
        if isinstance(w, (float, int)):
            dw = np.full(nw, w)
            return dw, None
        # Now get cell widths
        if widths:
            dw = w
            return dw, None
        dw = np.diff(w)
        o = np.min(w)
        return dw, o

    # Get proper cell widths
    dx, ox = _getWidths(nx, x, widths=widths)
    dy, oy = _getWidths(ny, y, widths=widths)
    dz, oz = _getWidths(nz, z, widths=widths)


    # Check lengths of cell widths against model space shape
    if len(dx) != nx:
        raise RuntimeError('X cells size does not match data.')
    if len(dy) != ny:
        raise RuntimeError('Y cells size does not match data.')
    if len(dz) != nz:
        raise RuntimeError('Z cells size does not match data.')


    # Check the origin
    if widths:
        ox, oy, oz = origin
    else:
        # Now check set ox, oy, oz
        if ox is None: ox = origin[0]
        if oy is None: oy = origin[1]
        if oz is None: oz = origin[2]



    # Perfrom the write out:
    if '.msh' not in fname:
        fname += '.msh'

    # Save the mesh to UBC Tensor Mesh format
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
