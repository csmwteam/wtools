"""This module holds several methods for standard file I/O for the data formats
that we work with regularly. Much of this regarding `Grid` objects in inherrited
directly into the `Grid` class.
"""

__all__ = [
    'readGSLib',
]

__displayname__ = 'File I/O'

import pandas as pd
import numpy as np
import glob, os

from .transform import transpose


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


class GridFileIO(object):

    @classmethod
    def tableToGrid(Grid, df, shp, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0], order='F'):
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
                           origin=origin,
                           xtensor=np.full(nx, spacing[0], dtype=float),
                           ytensor=np.full(ny, spacing[1], dtype=float),
                           ztensor=np.full(nz, spacing[2], dtype=float),
                          )
        grid.validate()
        return grid

    @classmethod
    def readSGeMSGrid(Grid, fname, origin=[0.0, 0.0, 0.0], spacing=[1.0, 1.0, 1.0]):
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
        df = readGSLib(fname)
        shp = df.header.split()
        shp = [int(i) for i in shp]
        return Grid.tableToGrid(df, shp, origin=origin, spacing=spacing)


    @staticmethod
    def _saveUBC(fname, x, y, z, models, header='Data', widths=False, origin=(0.0, 0.0, 0.0)):
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


    def saveUBC(self, fname):
        """Save the grid in the UBC mesh format."""
        self.validate()
        return GridFileIO._saveUBC(fname, self.xnodes, self.ynodes, self.znodes, self.models)


    @classmethod
    def _readMeshUBC(Grid, FileName):
        """This method reads a UBC 3D Mesh file and builds an empty
        ``Grid`` object for data to be inserted into.

        Args:
            FileName (str) : The mesh filename as an absolute path for the input
                mesh file in UBC 3D Mesh Format.

        Return:
            Grid :
                No data attributes here, simply an empty mesh.
        """

        #--- Read in the mesh ---#
        fileLines = np.genfromtxt(FileName, dtype=str,
            delimiter='\n', comments='!')

        # Get mesh dimensions
        dim = np.array(fileLines[0].split('!')[0].split(), dtype=int)
        dim = (dim[0]+1, dim[1]+1, dim[2]+1)

        # The origin corner (Southwest-top)
        #- Remember UBC format specifies down as the positive Z
        #- Easting, Northing, Altitude
        oo = np.array(
            fileLines[1].split('!')[0].split(),
            dtype=float
        )
        ox,oy,oz = oo[0],oo[1],oo[2]

        # Read cell sizes for each line in the UBC mesh files
        def _readCellLine(line):
            line_list = []
            for seg in line.split():
                if '*' in seg:
                    sp = seg.split('*')
                    seg_arr = np.ones((int(sp[0]),), dtype=float) * float(sp[1])
                else:
                    seg_arr = np.array([float(seg)], dtype=float)
                line_list.append(seg_arr)
            return np.concatenate(line_list)

        # Read the cell sizes
        cx = _readCellLine(fileLines[2].split('!')[0])
        cy = _readCellLine(fileLines[3].split('!')[0])
        cz = _readCellLine(fileLines[4].split('!')[0])
        # Invert the indexing of the vector to start from the bottom.
        #cz = cz[::-1]
        # Adjust the reference point to the bottom south west corner
        oz = oz - np.sum(cz)

        # Set the dims and coordinates for the output
        grid = Grid(xtensor=cx,
                           ytensor=cy,
                           ztensor=cz,
                           origin=(ox, oy, oz),)
        return grid

    @staticmethod
    def _readModelUBC(FileName):
            """Reads the 3D model file and returns a 1D NumPy float array. Be sure
            to associate with a ``Grid`` object.

            Args:
                FileName (str) : The model file name(s) as a relative or absolute
                    path for the input model file in UBC 3D Model Model Format.
                    Also accepts a `list` of string file names.

            Return:
                np.array :
                    Returns a NumPy float array that holds the model data
                    read from the file. Use the ``PlaceModelOnMesh()`` method to
                    associate with a grid. If a list of file names is given then it
                    will return a dictionary of NumPy float array with keys as the
                    basenames of the files.
            """
            # Check if recurssion needed
            if type(FileName) is list:
                out = {}
                for f in FileName:
                    out[os.path.splitext(f)[1][1:]] = GridFileIO._readModelUBC(f)
                return out
            # Perform IO
            try:
                data = np.genfromtxt(FileName, dtype=np.float, comments='!')
            except (IOError, OSError) as fe:
                raise _helpers.PVGeoError(str(fe))
            return data

    @classmethod
    def readUBC(Grid, name, directory=''):
        # Find the files
        # Check the extension of the grid name
        if directory == '':
            directory = os.getcwd()
        meshname = os.path.join(directory, name)
        directory = os.path.dirname(meshname)
        ext = os.path.splitext(meshname)[1]
        if ext is '':
            meshname = meshname + '.msh'
        elif ext not in '.msh' and ext not in '.mesh':
            raise IOError('{:s} is an incorrect extension; has to be `.msh` or `.mesh`.'.format(ext))
        # Create the grid
        grid = Grid._readMeshUBC(meshname)
        # Now walk the directory and find models
        modelFileNames = []
        for file in os.listdir(os.path.dirname(meshname)):
            fn = os.path.basename(os.path.splitext(file)[0])
            ext = os.path.splitext(file)[1]
            if fn == os.path.splitext(os.path.basename(meshname))[0] and ext not in '.msh' and ext not in '.mesh':
                modelFileNames.append(file)
        if len(modelFileNames) < 1:
            raise RuntimeError('Model files not found.')
        # Now add the models
        for i, m in enumerate(modelFileNames):
            modelFileNames[i] = os.path.join(directory, m)
        # Load them models
        n1, n2, n3 = grid.shape
        models = GridFileIO._readModelUBC(modelFileNames)
        for name, model in models.items():
            model = np.reshape(model, (n3,n1,n2), order='F')
            model = model[::-1, :, :]
            model = np.transpose(model, (1, 2, 0))
            models[name] = model
        grid.models = models
        grid.validate()
        return grid
