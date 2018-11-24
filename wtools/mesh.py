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
import discretize

from .plots import display
from .fileio import GridFileIO


def get_data_range(data):
    """Get the data range for a given ndarray"""
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    return (dmin, dmax)



class Grid(discretize.TensorMesh, GridFileIO):
    """
    A data structure to store a model space discretization and different
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
    def __init__(self, h=None, x0=(0.,0.,0.), models=None, **kwargs):
        if models is not None:
            self.models = models
            if h is None:
                h = []
                shp = list(models.values())[0].shape
                # Now create tensors if not present
                if len(shp) > 0:
                    h.append(np.ones(shp[0]))
                if len(shp) > 1:
                    h.append(np.ones(shp[1]))
                if len(shp) > 2:
                    h.append(np.ones(shp[2]))
        discretize.TensorMesh.__init__(self, h=h, x0=x0, **kwargs)



    models = properties.Dictionary(
        'The volumetric data as a 3D NumPy arrays in <X,Y,Z> or <i,j,k> ' +
        'coordinates. Each key value pair represents a different model for ' +
        'the gridded model space. Keys will be treated as the string name of ' +
        'the model.',
        key_prop=properties.String('Model name'),
        value_prop=properties.Array(
            'The volumetric data as a 3D NumPy array in <X,Y,Z> or <i,j,k> coordinates.',
            shape=('*','*','*')),
        required=False
    )

    @properties.validator
    def _validate_models(self):
        # Check the models
        if self.models is not None:
            shp = list(self.models.values())[0].shape
            for k, d in self.models.items():
                if d.shape != shp:
                    raise RuntimeError('Validation Failed: dimesnion mismatch between models.')
        return True

    @property
    def keys(self):
        """List of the string names for each of the models"""
        return list(self.models.keys())

    @property
    def shape(self):
        """3D shape of the grid (number of cells in all three directions)"""
        return ( self.nCx, self.nCy, self.nCz)

    @property
    def bounds(self):
        """The bounds of the grid"""
        grid = self.gridN
        try:
            x0, x1 = np.min(grid[:,0]), np.max(grid[:,0])
        except:
            x0, x1 =  0., 0.
        try:
            y0, y1 = np.min(grid[:,1]), np.max(grid[:,1])
        except:
            y0, y1 =  0., 0.
        try:
            z0, z1 = np.min(grid[:,2]), np.max(grid[:,2])
        except:
            z0, z1 =  0., 0.
        return (x0,x1, y0,y1, z0,z1)


    def get_data_range(self, key):
        """Get the data range for a given model"""
        data = self.models[key]
        return get_data_range(data)

    def equal(self, other):
        """Compare this Grid to another Grid"""
        return properties.equal(self, other)

    def __str__(self):
        """Print this onject as a human readable string"""
        self.validate()
        fmt = ["<%s instance>" % (self.__class__.__name__)]
        fmt.append("  Shape: {}".format(self.shape))
        fmt.append("  Origin: {}".format(tuple(self.x0)))
        bds = self.bounds
        fmt.append("  X Bounds: {}".format((bds[0], bds[1])))
        fmt.append("  Y Bounds: {}".format((bds[2], bds[3])))
        fmt.append("  Z Bounds: {}".format((bds[4], bds[5])))
        if self.models is not None:
            fmt.append("  Models: ({})".format(len(self.models.keys())))
            for key, val in self.models.items():
                dl, dh = self.get_data_range(key)
                fmt.append("    '{}' ({}): ({:.3e}, {:.3e})".format(key, val.dtype, dl, dh))
        return '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        self.validate()
        fmt = ""
        if self.models is not None:
            fmt += "<table>"
            fmt += "<tr><th>Grid Attributes</th><th>Models</th></tr>"
            fmt += "<tr><td>"
        fmt += "\n"
        fmt += "<table>\n"
        fmt += "<tr><th>Attribute</th><th>Values</th></tr>\n"
        row = "<tr><td>{}</td><td>{}</td></tr>\n"
        fmt += row.format("Shape", self.shape)
        fmt += row.format('Origin', tuple(self.x0))
        bds = self.bounds
        fmt += row.format("X Bounds", (bds[0], bds[1]))
        fmt += row.format("Y Bounds", (bds[2], bds[3]))
        fmt += row.format("Z Bounds", (bds[4], bds[5]))
        num = 0
        if self.models is not None:
            num = len(self.models.keys())
        fmt += row.format("Models", num)
        fmt += "</table>\n"
        fmt += "\n"
        if self.models is not None:
            fmt += "</td><td>"
            fmt += "\n"
            fmt += "<table>\n"
            row = "<tr><th>{}</th><th>{}</th><th>{}</th><th>{}</th></tr>\n"
            fmt += row.format("Name", "Type", "Min", "Max")
            row = "<tr><td>{}</td><td>{}</td><td>{:.3e}</td><td>{:.3e}</td></tr>\n"
            for key, val in self.models.items():
                dl, dh = self.get_data_range(key)
                fmt += row.format(key, val.dtype, dl, dh)
            fmt += "</table>\n"
            fmt += "\n"
            fmt += "</td></tr> </table>"
        return fmt


    def __getitem__(self, key):
        """Get a model of this grid by its string name"""
        return self.models[key]

    def to_data_frame(self, order='C'):
        """Returns the models in this Grid to a Pandas DataFrame with all arrays
        flattened in the specified order. A header attribute is added to the
        DataFrame to specified the grid extents. Much metadata is lost in this
        conversion.
        """
        self.validate()
        tits = self.models.keys()
        data = {k: v.flatten(order=order) for k, v in self.models.items()}
        df = pd.DataFrame.from_dict(data)
        df.header = '{} {} {}'.format(self.nCx, self.nCy, self.nCz)
        return df


    def plot_3d_slicer(self, key, **kwargs):
        model = self.models[key]
        return discretize.TensorMesh.plot_3d_slicer(self, model, **kwargs)

    def plotSlice(self, key, **kwargs):
        """Plots a 2D slice of the mesh

        Args:
            key (str): the model name to plot

        Note:
            See the `discretize code docs`_ for more details.

        .. _discretize code docs: http://discretize.simpeg.xyz/en/latest/content/mesh_tensor.html?highlight=plotSlice#discretize.View.TensorView.plotSlice
        """
        return discretize.TensorMesh.plotSlice(self, v=self.models[key], **kwargs)

    @property
    def models_flat(self):
        """Returns flattened model dictionary in Fortran ordering"""
        return {k:v.flatten(order='F') for k,v in self.models.items()}

    def toVTK(self):
        return discretize.TensorMesh.toVTK(self, models=self.models_flat)

    def writeVTK(self):
        return discretize.TensorMesh.writeVTK(self, models=self.models_flat)
