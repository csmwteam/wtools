"""``mesh``: This module provides numerous methods and classes for discretizing
data in a convienant way that makes sense for our spatially referenced
data/models.
"""

__all__ = [
    'meshgrid',
    'transpose',
    'saveUBC',
    'GriddedData',
]

import numpy as np
import properties


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


def saveUBC(fname, x, y, z, models, header='Data', widths=False, origin=(0.0, 0.0, 0.0)):
    """Saves a 3D gridded array with spatail reference to the UBC mesh/model format.
    Use `PVGeo`_ to visualize this data. For more information on the UBC mesh
    format, reference the `GIFtoolsCookbook`_ website.

    Warning:
        This method assumes your mesh and data are defined on a normal cartesian
        system: <x,y,z>

    .. _PVGeo: http://pvgeo.org
    .. _GIFtoolsCookbook: https://giftoolscookbook.readthedocs.io/en/latest/content/fileFormats/mesh3Dfile.html

    Args:
        fname (str): the string file name of the mesh file. Model files will be
            saved next to this file.
        x (ndarray or float): a 1D array of unique coordinates along the X axis,
            float for uniform cell widths, or an array with ``widths==True``
            to treat as cell spacing on X axis
        y (ndarray or float): a 1D array of unique coordinates along the Y axis,
            float for uniform cell widths, or an array with ``widths==True``
            to treat as cell spacing on Y axis
        z (ndarray or float): a 1D array of unique coordinates along the Z axis,
            float for uniform cell widths, or an array with ``widths==True``
            to treat as cell spacing on Z axis
        models (dict): a dictionary of models. Key is model name and value is a
            3D array with dimensions <x,y,z> containing cell data.
        header (str): a string header for your mesh/model files
        widths (bool): flag for whether to treat the (``x``, ``y``, ``z``) args as
            cell sizes/widths
        origin (tuple(float)): optional origin value used if ``widths==True``,
            or used on a component basis if any of the ``x``, ``y``, or ``z``
            args are scalars.

    Yields:
        Saves out a mesh file named {``fname``}.msh and a model file for every
        key/value pair in the ``models`` argument (key is file extension for model
        file and value is the data.

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

        >>> import numpy as np
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


class GriddedData(properties.HasProperties):
    """A data structure to store a model space discretization and different
    attributes of that model space.

    Example:
        >>> import wtools
        >>> import numpy as np
        >>> models = {
            'rand': np.random.random(1000).reshape((10,10,10)),
            'spatial': np.arange(1000).reshape((10,10,10)),
            }
        >>> grid = wtools.GriddedData(models=models)
        >>> grid.validate() # Make sure the data object was created successfully
        True

        >>> # Or you could create a model with a defined spatial reference
        >>> nx, ny, nz = 18, 24, 20
        >>> x = np.linspace(20, 200, nx)
        >>> y = np.linspace(20, 500, ny)
        >>> z = np.linspace(0, 1000, nz)
        >>> grid = wtools.GriddedData(models={
                                'density': np.random.rand(nx,ny,nz)
                                },
                                xcoords=x,
                                ycoords=y,
                                zcoords=z)


    """
    models = properties.Dictionary(
        'The volumetric data as a 3D NumPy arrays in <X,Y,Z> or <i,j,k> ' +
        'coordinates. Each key value pair represents a different model for ' +
        'the gridded model space. Keys will be treated as the string name of ' +
        'the model.',
        key_prop=properties.String('Model name'),
        value_prop=properties.Array(
            'The volumetric data as a 3D NumPy array in <X,Y,Z> or <i,j,k> coordinates.',
            shape=('*','*','*'))
    )
    xcoords = properties.Array(
        'The cell center coordinates along the X axis.',
        shape=('*',),
        required=False,
    )
    ycoords = properties.Array(
        'The cell center coordinates along the Y axis.',
        shape=('*',),
        required=False,
    )
    zcoords = properties.Array(
        'The cell center coordinates along the Z axis.',
        shape=('*',),
        required=False,
    )

    def saveUBC(self, fname):
        """Save the grid in the UBC mesh format.
        """
        self.validate()
        # Half cell widths
        dx = np.unique(np.diff(self.xcoords))[0] / 2.
        dy = np.unique(np.diff(self.ycoords))[0] / 2.
        dz = np.unique(np.diff(self.zcoords))[0] / 2.
        # Nodes
        x = np.arange(self.xcoords.min()-dx, self.xcoords.max()+2*dx, 2*dx)
        y = np.arange(self.ycoords.min()-dy, self.ycoords.max()+2*dy, 2*dy)
        z = np.arange(self.zcoords.min()-dz, self.zcoords.max()+2*dz, 2*dz)
        return saveUBC(fname, x, y, z, self.models)

    def todict(self, key):
        """Export this object as a dictionary compatible with the interactive
        plotting routine. Only exports a single model attribute.

        Args:
            key (str): the model key to export

        Return:
            dict:
                Dictionary with key/value pairs ready for interactive
                plotting routines.
        """
        self.validate()
        d = {'xc': self.xcoords,
             'yc': self.ycoords,
             'zc': self.zcoords,
             'data': self.models[key],
            }
        return d

    def validate(self):
        properties.HasProperties.validate(self)
        # Check the models
        shp = list(self.models.values())[0].shape
        for k, d in self.models.items():
            if d.shape != shp:
                raise properties.ValidationError('Validation Failed: dimesnion mismatch between models.')
        # Now create tensors if not present
        if self.xcoords is None:
            self.xcoords = np.arange(shp[0])
        if self.ycoords is None:
            self.ycoords = np.arange(shp[1])
        if self.zcoords is None:
            self.zcoords = np.arange(shp[2])
        return True
