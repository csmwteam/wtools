"""This module holds several methods for standard file I/O for the data formats
that we work with regularly.
"""
__all__ = [
    'readGSLib',
    'tableToGrid',
    'readSGeMSGrid',
]

__displayname__ = 'File I/O'

import pandas as pd
import numpy as np

from .mesh import GriddedData

def readGSLib(fname):
    """This will read a standard GSLib or GeoEAS data file to a pandas
    ``DataFrame``.

    Args:
        fname (str): the string file name of the data to load. This can be a
            relative or abslute file path.

    Return:
        pandas.DataFrame:
            A table containing the all data arrays. Note that an attribute
            called ``header`` is added to the data frame contianing the string
            header line of the file.
    """
    with open(fname, 'r') as f:
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


def tableToGrid(df, shp, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0], order='F'):
    """Converts a pandas ``DataFrame`` table to a ``GriddedData`` object.
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
    grid = GriddedData(models=d,
                       origin=origin,
                       xtensor=np.full(nx, spacing[0], dtype=float),
                       ytensor=np.full(ny, spacing[1], dtype=float),
                       ztensor=np.full(nz, spacing[2], dtype=float),
                      )
    grid.validate()
    return grid


def readSGeMSGrid(fname, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0]):
    """Reads an SGeMS grid file where grid shape is defined in the header as
    three integers seperated by whitespace. Data arrays are treated as 3D and
    given in <x, y, z> indexing to a ``GriddedData`` object.

    Args:
        fname (str): the string file name of the data to load. This can be a
            relative or abslute file path.

    Return:
        GriddedData:
            The SGeMS data loaded onto a ``GriddedData`` object.

    """
    df = readGSLib(fname)
    shp = df.header.split()
    shp = [int(i) for i in shp]
    return tableToGrid(df, shp, origin=origin, spacing=spacing)
