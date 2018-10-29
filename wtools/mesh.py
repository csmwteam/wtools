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

__displayname__ = 'Mesh Tools'

import numpy as np
import properties

from .plots import display

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

    def _getWidths(nw, w, widths=False, z=False):
        # Convert scalars if necessary
        if isinstance(w, (float, int)):
            dw = np.full(nw, w)
            return dw, None
        # Now get cell widths
        if widths:
            dw = w
            return dw, None
        dw = np.diff(w)
        if z:
            o = np.max(w)
        else:
            o = np.min(w)
        return dw, o

    # Get proper cell widths
    dx, ox = _getWidths(nx, x, widths=widths)
    dy, oy = _getWidths(ny, y, widths=widths)
    dz, oz = _getWidths(nz, z, widths=widths, z=True)


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

    Note:
        See Jupyter notebooks under the ``examples`` directory

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

    xtensor = properties.Array(
        'Tensor cell widths, x-direction',
        shape=('*',),
        dtype=(float, int)
    )

    ytensor = properties.Array(
        'Tensor cell widths, y-direction',
        shape=('*',),
        dtype=(float, int)
    )

    ztensor = properties.Array(
        'Tensor cell widths, z-direction',
        shape=('*',),
        dtype=(float, int)
    )

    origin = properties.Vector3(
        'The lower southwest corner of the data volume.',
        default=[0., 0., 0.],
    )

    @property
    def keys(self):
        """List of the string names for each of the models"""
        return list(self.models.keys())

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return ((len(self.xtensor)+1) * (len(self.ytensor)+1) *
                (len(self.ztensor)+1))

    @property
    def num_cells(self):
        """Number of cells"""
        return len(self.xtensor) * len(self.ytensor) * len(self.ztensor)

    @property
    def shape(self):
        """3D shape of the grid (number of cells in all three directions)"""
        return ( len(self.xtensor), len(self.ytensor), len(self.ztensor))

    @property
    def nx(self):
        """Number of cells in the X direction"""
        return len(self.xtensor)

    @property
    def ny(self):
        """Number of cells in the Y direction"""
        return len(self.ytensor)

    @property
    def nz(self):
        """Number of cells in the Z direction"""
        return len(self.ztensor)

    @property
    def xnodes(self):
        """The node coordinates along the X-axis"""
        ox, oy, oz = self.origin
        x = ox + np.cumsum(self.xtensor)
        return np.insert(x, 0, ox)

    @property
    def ynodes(self):
        """The node coordinates along the Y-axis"""
        ox, oy, oz = self.origin
        y = oy + np.cumsum(self.ytensor)
        return np.insert(y, 0, oy)

    @property
    def znodes(self):
        """The node coordinates along the Z-axis"""
        ox, oy, oz = self.origin
        z = oz + np.cumsum(self.ztensor)
        return np.insert(z, 0, oz)

    @property
    def bounds(self):
        """The bounds of the grid"""
        return (self.xnodes.min(), self.xnodes.max(),
                self.ynodes.min(), self.ynodes.max(),
                self.znodes.min(), self.znodes.max())

    def getNodePoints(self):
        """Get ALL nodes in the gridded volume as an XYZ point set"""
        # Build out all nodes in the mesh
        xx, yy, zz = np.meshgrid(self.xnodes, self.ynodes, self.znodes, indexing='ij')
        return np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T

    @property
    def xcenters(self):
        """The cell center coordinates along the X-axis"""
        xn = self.xnodes
        xc = [xn[i] + self.xtensor[i]/2. for i in range(len(self.xtensor))]
        return np.array(xc)

    @property
    def ycenters(self):
        """The cell center coordinates along the Y-axis"""
        yn = self.ynodes
        yc = [yn[i] + self.ytensor[i]/2. for i in range(len(self.ytensor))]
        return np.array(yc)

    @property
    def zcenters(self):
        """The cell center coordinates along the Z-axis"""
        zn = self.znodes
        zc = [zn[i] + self.ztensor[i]/2. for i in range(len(self.ztensor))]
        return np.array(zc)

    def saveUBC(self, fname):
        """Save the grid in the UBC mesh format."""
        self.validate()
        return saveUBC(fname, self.xnodes, self.ynodes, self.znodes, self.models)

    def getDataRange(self, key):
        """Get the data range for a given model"""
        data = self.models[key]
        dmin = np.nanmin(data)
        dmax = np.nanmax(data)
        return (dmin, dmax)


    def validate(self):
        # Check the models
        shp = list(self.models.values())[0].shape
        for k, d in self.models.items():
            if d.shape != shp:
                raise properties.ValidationError('Validation Failed: dimesnion mismatch between models.')
        # Now create tensors if not present
        if self.xtensor is None:
            self.xtensor = np.ones(shp[0])
        if self.ytensor is None:
            self.ytensor = np.ones(shp[1])
        if self.ztensor is None:
            self.ztensor = np.ones(shp[2])
        return properties.HasProperties.validate(self)

    def __str__(self):
        """Print this onject as a human readable string"""
        self.validate()
        fmt = ["<%s instance at %s>" % (self.__class__.__name__, id(self))]
        fmt.append("  Shape: {}".format(self.shape))
        fmt.append("  Origin: {}".format(tuple(self.origin)))
        bds = self.bounds
        fmt.append("  X Bounds: {}".format((bds[0], bds[1])))
        fmt.append("  Y Bounds: {}".format((bds[2], bds[3])))
        fmt.append("  Z Bounds: {}".format((bds[4], bds[5])))
        if self.models is not None:
            fmt.append("  Models: ({})".format(len(self.models.keys())))
            for key, val in self.models.items():
                dl, dh = self.getDataRange(key)
                fmt.append("    '{}' ({}): ({:.3e}, {:.3e})".format(key, val.dtype, dl, dh))
        return '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        """Get a model of this grid by its string name"""
        return self.models[key]

    def display(self, plt, key, plane='xy', slc=None, showit=True, **kwargs):
        """Display a 2D slice of this grid.

        Args:
            plt (handle): the active plotting handle to use
            key (str): the string name of the model to view
            plane (``'xy'``, ``'xz'``, ``'yz'``): The plane to slice upon
            slc (float): the coordinate along the sliced dimension
            showit (bool): A flag for whether or not to call ``plt.show()``
        """
        plane = plane.lower()
        # Now exract the plane
        data = self.models[key]
        if plane == 'xz' or plane == 'zx':
            if slc:
                ind = np.argmin(np.abs(self.ycenters - slc))
            else:
                ind = self.ynodes.size // 2 - 1
            data = data[:, ind, :]
            pos = 'Y-Slice @ {}'.format(self.ycenters[ind])
            plt.xlabel('X')
            plt.ylabel('Z')
            disp = display(plt, data, x=self.xnodes, y=self.znodes, **kwargs)
        elif plane == 'xy' or plane == 'yx':
            if slc:
                ind = np.argmin(np.abs(self.zcenters - slc))
            else:
                ind = self.znodes.size // 2 - 1
            data = data[:, :, ind]
            pos = 'Z-Slice @ {}'.format(self.zcenters[ind])
            plt.xlabel('X')
            plt.ylabel('Y')
            disp = display(plt, data, x=self.xnodes, y=self.ynodes, **kwargs)
        elif plane == 'yz'  or plane == 'zy':
            if slc:
                ind = np.argmin(np.abs(self.xcenters - slc))
            else:
                ind = self.xnodes.size // 2 - 1
            data = data[ind, :, :]
            pos = 'X-Slice @ {}'.format(self.xcenters[ind])
            plt.xlabel('Y')
            plt.ylabel('Z')
            disp = display(plt, data, x=self.ynodes, y=self.znodes, **kwargs)
        else:
            raise RuntimeError('Plane ({}) not understood.'.format(plane))
        plt.axis('image')
        plt.title('Data Array: {}\n{}'.format(key, pos))
        if showit: return plt.show()
        return disp
