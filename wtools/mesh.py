"""This provides a class for discretizing data in a convienant way that makes
sense for our spatially referenced data/models.
"""

__all__ = [
    'Grid',
]

__displayname__ = 'Mesh Tools'

import numpy as np
import pandas as pd
import properties

from .plots import display
from .fileio import GridFileIO


class Grid(properties.HasProperties, GridFileIO):
    """A data structure to store a model space discretization and different
    attributes of that model space.

    Example:
        >>> import wtools
        >>> import numpy as np
        >>> models = {
            'rand': np.random.random(1000).reshape((10,10,10)),
            'spatial': np.arange(1000).reshape((10,10,10)),
            }
        >>> grid = wtools.Grid(models=models)
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

    def getDataRange(self, key):
        """Get the data range for a given model"""
        data = self.models[key]
        dmin = np.nanmin(data)
        dmax = np.nanmax(data)
        return (dmin, dmax)


    def validate(self):
        # Check the models
        if self.models is not None:
            shp = list(self.models.values())[0].shape
            for k, d in self.models.items():
                if d.shape != shp:
                    raise properties.ValidationError('Validation Failed: dimesnion mismatch between models.')
        else:
            return properties.HasProperties.validate(self)
        # Now create tensors if not present
        if self.xtensor is None:
            self.xtensor = np.ones(shp[0])
        if self.ytensor is None:
            self.ytensor = np.ones(shp[1])
        if self.ztensor is None:
            self.ztensor = np.ones(shp[2])
        return properties.HasProperties.validate(self)

    def equal(self, other):
        """Compare this Grid to another Grid"""
        return properties.equal(self, other)

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

    def toDataFrame(self, order='C'):
        """Returns the models in this Grid to a Pandas DataFrame with all arrays
        flattened in the specified order. A header attribute is added to the
        DataFrame to specified the grid extents. Much metadata is lost in this
        conversion.
        """
        self.validate()
        tits = self.models.keys()
        data = {k: v.flatten(order=order) for k, v in self.models.items()}
        df = pd.DataFrame.from_dict(data)
        df.header = '{} {} {}'.format(self.nx, self.ny, self.nz)
        return df

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
