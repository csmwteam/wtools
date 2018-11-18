__all__ = [
    'GridSpec',
    'geoeas_to_np',
    'geoeas_to_npGS',
]

__displayname__ = 'Grids'

import properties
import numpy as np


################################################################################


class GridSpec(properties.HasProperties):
    """A ``GridSpec`` object provides the details of a single axis along a grid.
    If you have a 3D grid then you will have 3 ``GridSpec`` objects.
    """
    n = properties.Integer('The number of components along this dimension.')
    min = properties.Integer('The minimum value along this dimension. The origin.', default=0)
    sz = properties.Integer('The uniform cell size along this dimension.', default=1)
    nnodes = properties.Integer('The number of grid nodes to consider on either \
        side of the origin in the output map', required=False)

    @properties.validator
    def _validate_nnodes(self):
        if self.nnodes is None:
            self.nnodes = self.n // 2
        return True


################################################################################


def geoeas_to_np(datain, nx, ny=None, nz=None):
    """Transform GeoEas array into np.ndarray to be treated like image.
    Function to transform a SINGLE GoeEas-formatted raster (datain)
    i.e., a single column, to a NumPy array that can be viewed using
    imshow (in 2D) or slice (in 3D).

    Args:
        datain (np.ndarray): 1D input GeoEas-formatted raster of dimensions:
        nx (int): the number of dimensions along the 1st axis
        ny (int, optional): the number of dimensions along the 2nd axis
        nz (int, optional): the number of dimensions along the 3rd axis

    Return:
        np.ndarray: If only nx given: 1D array.
            If only nx and ny given: 2D array.
            If nx, ny, and nz given: 3D array.

    Note:
      In 3D, z increases upwards

    References:
        Originally implemented in MATLAB by:
            Phaedon Kyriakidis,
            Department of Geography,
            University of California Santa Barbara,
            May 2005

        Reimplemented into Python by:
            Bane Sullivan and Jonah Bartrand,
            Department of Geophysics,
            Colorado School of Mines,
            October 2018
    """
    # 1D
    if ny is None and nz is None and isinstance(nx, int):
        return datain # 1D so it does nothing!
    # 2D
    elif nz is None and isinstance(nx, int) and isinstance(ny, int):
        tmp = np.reshape(datain, (nx, ny))
        tmp = np.swapaxes(tmp, 2,1) # TODO: should we do this???
        return tmp[ny:1:-1,:] # TODO: should we do this???
    # 3D
    elif isinstance(nx, int) and isinstance(ny, int) and isinstance(nz, int):
        tmp = np.reshape(datain, (nx, ny, nz))
        tmp = np.swapaxes(tmp, 1,0,2) # TODO: should we do this???
        return tmp[ny:1:-1,:,:] # TODO: should we do this???
    # Uh-oh.
    raise RuntimeError('``geoeas_to_np``: arguments not understood.')


def geoeas_to_npGS(datain, gridspecs):
    """A wrapper for ``geoeas_to_np`` to handle a list of ``GridSpec`` objects

    Args:
        gridspecs (list(GridSpec)): array with grid specifications using
            ``GridSpec`` objects
    """
    # Check that gridspecs is a list of ``GridSpec`` objects
    if not isinstance(gridspecs, list):
        if not isinstance(gridspecs, GridSpec):
            raise RuntimeError('gridspecs arguments ({}) improperly defined.'. format(gridspecs))
        gridspecs = [gridspecs] # Make sure we have a list to index if only 1D
    if len(gridspecs) == 1:
        return geoeas_to_np(datain, nx=gridspecs[0].n)
    elif len(gridspecs) == 2:
        return geoeas_to_np(datain, nx=gridspecs[0].n, ny=gridspecs[1].n)
    elif len(gridspecs) == 3:
        return geoeas_to_np(datain, nx=gridspecs[0].n, ny=gridspecs[1].n, nz=gridspecs[2].n)
    raise RuntimeError('gridspecs must be max of length 3 for geoas2numpy.')
