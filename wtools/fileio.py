"""This module holds several methods for standard file I/O for the data formats
that we work with regularly. Much of this regarding `Grid` objects in inherrited
directly into the `Grid` class.
"""

__all__ = [
    'read_gslib',
    'save_gslib',
    'GridFileIO',
    'load_models',
    'save_pickle',
    'load_pickle',
]

__displayname__ = 'File I/O'

import pandas as pd
import numpy as np

import glob
import os
import warnings
import datetime
import json
import pickle

import discretize

from .transform import transpose
from .models import Models


def save_pickle(filename, data):
    """Pickles a data object in a Python 2 AND 3 friendly manner"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=2)

def load_pickle(filename):
    """Reads a pickled data object"""
    with open(filename, 'rb' ) as f:
        obj = pickle.load(f)
    return obj


def read_gslib(filename):
    """This will read a standard GSLib or GeoEAS data file to a pandas
    ``DataFrame``.

    Args:
        filename (str): the string file name of the data to load. This can be a
            relative or abslute file path.

    Return:
        pandas.DataFrame:
            A table containing the all data arrays. Note that an attribute
            called ``header`` is added to the data frame contianing the string
            header line of the file.
    """
    with open(filename, 'r') as f:
        head = f.readline().strip()
        num = int(f.readline().strip())
        ts = []
        for i in range(num):
            ts.append(f.readline().strip())
        df = pd.read_csv(f,
                         names=ts,
                         delim_whitespace=True)
        df.header = head
    return df


def save_gslib(filename, dataframe, header=None):
    """This will save a pandas dataframe to a GSLib file"""
    if header is None:
        try:
            header = dataframe.header
        except AttributeError:
            warnings.warn('Header not defined. Using date')
            header = str(datetime.datetime.now())
    if '\n' in header:
        raise RuntimeError('`header` can only be 1 line.')
    datanames = '\n'.join(dataframe.columns)
    with open(filename, 'w') as f:
        f.write('%s\n' % header)
        f.write('%d\n' % len(dataframe.columns))
        f.write(datanames)
        f.write('\n')
        dataframe.to_csv(f, sep=' ', header=None, index=False, float_format='%.9e')
    return 1



class GridFileIO(object):
    """
    This class is inherrited by the :class:`~wtools.mesh.Grid` class and all
    these methods should be called from :class:`~wtools.mesh.Grid`.
    For example, If you have a file to read:

    Example:
        >>> import wtools
        >>> grid = wtools.Grid.read_sgems_grid('path/to/data/file.sgems')
        >>> grid.validate()
        True

    """

    @classmethod
    def table_to_grid(Grid, df, shp, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0], order='F'):
        """Converts a pandas ``DataFrame`` table to a ``Grid`` object.

        Args:
            shp (tuple(int)): length 3 tuple of integers sizes for the data grid
                dimensions.
            origin (iter(float)): the southwest-bottom corner of the grid.
            spacing (iter(float)): the cell spacings for each axial direction.
            order (``'C'``, ``'F'``, ``'A'``), optional: the reshape order.

        Return:
            Grid:
                The data table loaded onto a ``Grid`` object.
        """
        if not isinstance(shp, (list, tuple)) or len(shp) != 3:
            raise RuntimeError('`shp` must be a length 3 tuple.')
        for i, n in enumerate(shp):
            if not isinstance(n, int):
                raise RuntimeError('`shp` index ({}) must be an integer: ({}) is invalid'.format(i, n))
        nx, ny, nz = shp
        # Now make a dictionary of the models
        d = {}
        for k in df.keys():
            # Be sure to reshape using fortran ordering as SGeMS using <z,y,x> order
            d[k] = df[k].values.reshape(shp, order='F')
        grid = Grid(models=d,
                    x0=origin,
                    h=[np.full(nx, spacing[0], dtype=float),
                       np.full(ny, spacing[1], dtype=float),
                        np.full(nz, spacing[2], dtype=float),]
                    )
        grid.validate()
        return grid

    @classmethod
    def read_sgems_grid(Grid, fname, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0]):
        """Reads an SGeMS grid file where grid shape is defined in the header as
        three integers seperated by whitespace. Data arrays are treated as 3D and
        given in <x, y, z> indexing to a ``Grid`` object.

        Args:
            fname (str): the string file name of the data to load. This can be a
                relative or abslute file path.
            origin (iter(float)): the southwest-bottom corner of the grid.
            spacing (iter(float)): the cell spacings for each axial direction

        Return:
            Grid:
                The SGeMS data loaded onto a ``Grid`` object.

        """
        df = read_gslib(fname)
        shp = df.header.split()
        shp = [int(i) for i in shp]
        return Grid.table_to_grid(df, shp, origin=origin, spacing=spacing)


    def save_sgems(self, filename):
        """This will save the grid in the SGeMS gridded data file format"""
        df = self.to_data_frame(order='F')
        return save_gslib(filename, df)


    @classmethod
    def load_mesh(Grid, filename):
        """
        Open a json file and load the mesh into the ``Grid`` class

        :param str filename: name of file to read in
        """
        with open(filename, 'r') as outfile:
            jsondict = json.load(outfile)
            data = Grid.deserialize(jsondict, trusted=True)
        return data

    def writeUBC(self, fileName, directory='', comment_lines=''):
        ext = os.path.splitext(fileName)[1]
        if ext is '':
            ext = '.msh'
            fileName = fileName + ext
        d = {}
        for k,v in self.models.items():
            d['%s_%s.ubc' % (fileName.replace(ext, ''), k)] = v
        return discretize.TensorMesh.writeUBC(self, fileName, models=d, directory=directory, comment_lines=comment_lines)



def load_models(filename):
    """
    Open a json file and loads the models into the target class
    As long as there are no namespace conflicts, the target __class__
    will be stored on the properties.HasProperties registry and may be
    fetched from there.
    :param str filename: name of file to read in
    """
    with open(filename, 'r') as infile:
        jsondict = json.load(infile)
        data = Models.deserialize(jsondict, trusted=True)
    return data
