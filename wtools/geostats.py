__all__ = [
    'GridSpec',
    'geoeas2numpy',
    'geoeas2numpyGS',
    'raster2structgrid',
    'suprts2modelcovFFT',
]


import properties
import numpy as np


################################################################################


class GridSpec(properties.HasProperties):
    """A ``GridSpec`` object provides the details of a single axis along a grid.
    If you have a 3D grid then you will have 3 ``GridSpec`` objects.
    """
    n = properties.Integer('The number of components along this dimension.')
    min = properties.Integer('The minimum value along this dimension. The origin.')
    sz = properties.Integer('The uniform cell size along this dimension.')
    nnodes = properties.Integer('The number of grid nodes to consider on either \
        side of the origin in the output map', required=False)


################################################################################


def geoeas2numpy(datain, nx, ny=None, nz=None):
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

    Important:
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
    raise RuntimeError('``geoeas2numpy``: arguments not understood.')


def geoeas2numpyGS(datain, gridspecs):
    """A wrapper for ``geoeas2numpy`` to handle a list of ``GridSpec`` objects

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
        return geoeas2numpy(datain, nx=gridspecs[0].n)
    elif len(gridspecs) == 2:
        return geoeas2numpy(datain, nx=gridspecs[0].n, ny=gridspecs[1].n)
    elif len(gridspecs) == 3:
        return geoeas2numpy(datain, nx=gridspecs[0].n, ny=gridspecs[1].n, nz=gridspecs[2].n)
    raise RuntimeError('gridspecs must be max of length 3 for geoas2numpy.')


################################################################################


def raster2structgrid(datain, gridspecs, imeas='covariogram', idisp=False):
    """Create an auto-variogram or auto-covariance map from 1D or 2D rasters.
    This computes auto-variogram or auto-covariance maps from
    1D or 2D rasters. This function computes variograms/covariances in the
    frequency domain via the Fast Fourier Transform (``np.fft``).

    Note this only handles one dataset and we removed the ``icolV`` argument.

    Note:
        Missing values, flagged as ``np.nan``, are allowed.

    Args:
        datain (np.ndarray): input arrray with raster in GeoEas format
        gridspecs (list(GridSpec)): array with grid specifications using
            ``GridSpec`` objects
        imeas (str): key indicating which structural measure to compute:
            semi-variogram or covariogram
        idisp (bool): flag for whether to display results using an internal
            plotting routine

    Return:
        np.ndarray: output array with variogram or covariogram map, depending
            on imeas, with size: in 1D: ( 2*nxOutHalf+1 ) or in 2D:
            ( 2*nxOutHalf+1 x 2*nxOutHalf+1 )
        np.ndarray: output array with number of pairs available in each lag,
            of same size as outStruct


    Note:
      Author: Dennis Marcotte: Computers & Geosciences,
      > Vol. 22, No. 10, pp. 1175-1186, 1996.

    Important:
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

        Algorith based on:
            Marcotte, D. (1996): Fast Variogram Computation with FFT,
            Computers & Geosciences, 22(10), 1175-1186.
    """
    # Check that gridspecs is a list of ``GridSpec`` objects
    if not isinstance(gridspecs, list):
        if not isinstance(gridspecs, GridSpec):
            raise RuntimeError('gridspecs arguments ({}) improperly defined.'. format(gridspecs))
        gridspecs = [gridspecs] # Make sure we have a list to index if only 1D

    TINY = 1e-10;

    #DataIn = np.reshape(DataIn, (-1,1))
    #nPix,nCol = DataIn.shape

    # Check the `GridSpec` objects and ensure they hava an nnodes
    nDim = len(gridspecs)
    for i, gs in enumerate(gridspecs):
        gs.validate()
        if gs.nnodes is None:
            raise RuntimeError('GridSpec object at index %d does not have an nnodes property.' % i)
        if gs.nnodes > gs.n*0.5:
            raise RuntimeError('For GridSpec at index %d nnodesOff > # of gridnodes/2.' % i);


    # Check imeas
    itypes = ['covariogram', 'semi-variance']
    if isinstance(imeas, int) and imeas < 2 and imeas > -1:
        imeas = itypes[imeas]
    if imeas not in itypes:
        raise RuntimeError("imeas argument must be one of 'covariogram' or 'semi-variance'. Not {}".format(imeas))

    ## Extract columns of DataIn

    ## Convert to Matlab format
    Data = geoeas2numpyGS(datain, gridspecs);
    print('just ran `geoeas2numpyGS`...')

    ####################################

    shp = Data.shape
    n = shp[0]
    nrows = 2*n-1
    ncols = 1;
    if nDim == 2:
        p = shp[1]
        ncols = 2*p-1;

    ## Get appropriate dimensions
    # find the closest multiple of 8 to obtain a good compromise between
    # speed (a power of 2) and memory required
    nr2 = int(np.ceil(nrows/8)*8)
    if nDim == 2:
        nc2 = np.ceil(ncols/8)*8
    else:
        nc2 = 1

    ## Form an indicator  matrix:
    # 0's for all data values, 1's for missing values
    DataInd = ~np.isnan(Data);
    # In data matrix, replace missing values by 0;
    Data[DataInd] = 0  # missing replaced by 0

    def fft2(data):
        return np.fft.fft2(data, s=[nr2,nc2])

    def fft1(data):
        return np.fft.fft(data)

    def ifft2(data):
        return np.fft.ifft2(data)

    def ifft1(data):
        return np.fft.ifft(data)

    if nDim == 2:
        fft = fft2
        ifft = ifft2
    else:
        fft = fft1
        ifft = ifft1

    ## FFT of Data
    fData = fft(Data);

    ## FFT of Data*Data
    if imeas == itypes[1]: # semi-variance
        fDataData = fft(Data*Data)

    ## FFT of the indicator matrix
    fDataInd = fft(DataInd)

    ## Compute number of pairs at all lags
    outNpairs = np.real(ifft(np.abs(fDataInd)**2)).astype(np.int)
    #Edit remove single formating for matlab v6
    #outNpairs = single(outNpairs);

    ## Compute the different structural functions according to imeas
    if imeas == itypes[1]: # semi-variance
        outStruct = np.real(ifft(np.conj(fDataInd)*
                                    fDataData+conj(fDataData)*fDataInd-
                                         2*np.conj(fData)*fData))
        outStruct /= np.max(outNpairs, axis=0) / 2;
    else: # covariogram
        # tail mean
        m1 = np.real(ifft(np.abs(fData)**2)) / np.max(outNpairs,axis=0);
        # head mean
        m2 = np.real(ifft(np.abs(fDataInd)**2)) / np.max(outNpairs,axis=0)
        outStruct = np.real(ifft(np.abs(fData)**2));
        outStruct /= np.max(outNpairs, axis=0) - m1 * m2

    ## Reduce matrix to required size and shift,
    print(outNpairs.shape, outStruct.shape)
    # so that the 0 lag appears at the center of each matrix
    if nDim == 2:
        outNpairs=[outNpairs[0:n,0:p], outNpairs[0:n,nc2-p+1:nc2],
                   outNpairs[nr2-n+1:nr2,0:p], outNpairs[nr2-n+1:nr2, nc2-p+1:nc2]]
        outStruct=[outStruct[0:n,0:p], outStruct[0:n,nc2-p+1:nc2],
                   outStruct[nr2-n+1:nr2,0:p], outStruct[nr2-n+1:nr2,nc2-
                                                         +1:nc2]]
    else:
        outNpairs = [outNpairs[0:n], outNpairs[0:n],
                     outNpairs[nr2-n+1:nr2], outNpairs[nr2-n+1:nr2] ]
        outStruct = [outStruct[0:n], outStruct[0:n],
                     outStruct[nr2-n+1:nr2], outStruct[nr2-n+1:nr2]]


    outStruct = np.fft.fftshift(outStruct);
    outNpairs = np.fft.fftshift(outNpairs);

    # TODO: check this....
    for i, arr in enumerate(outNpairs):
        ind = arr < TINY
        outStruct[i][ind] = np.nan

    # ## Addition by Phaedon Kyriakidis, to crop image to a desired size
    # nxcurr, nycurr=outStruct.shape
    # nnodesOffN = 2*nnodesOff + 1;
    # if nDim == 1:
    #     ngridx = nnodesOffN[0];
    #     tmpx = np.floor((nxcurr-ngridx)/2);
    #     isx  = tmpx;
    #     iex  = isx + ngridx
    #     outStruct = outStruct[isx:iex]
    #     outNpairs = outNpairs[isx:iex]
    # elif nDim == 2:
    #     ngridx = nnodesOffN[1];
    #     ngridy = nnodesOffN[0];
    #     tmpx = np.floor((nxcurr-ngridx)/2);
    #     tmpy = np.floor((nycurr-ngridy)/2);
    #     isx  = tmpx;
    #     iex  = isx + ngridx
    #     isy  = tmpy
    #     iey  = isy + ngridy
    #     outStruct = outStruct[isx:iex,isy:iey];
    #     outNpairs = outNpairs[isx:iex,isy:iey];

    ####################################

    ## Convert to geoeas format
    # if nDim == 2:
    #     outStruct = matlab2geoeas(outStruct)
    #     outNpairs = matlab2geoeas(outNpairs)

    # TODO: this plotting routine needs to be handled in the ``plots`` module
    # ## Display results, if requested
    # if len(args) > 5:
    #     if nDim == 1:  ### 1D case
    #         xsiz = Gridspecs[2];
    #         xmin = nnodesOff[0]*xsiz;
    #         xmin = -xmin;
    #         nx = nnodesOff[0]*2+1;
    #         xax = np.arange(xmin,xmin+nx*xsiz,xsiz).T;
    #         plt.plot(xax,outStruct,'.-');
    #         plt.xlim([xmin, -xmin]);
    #         plt.xlabel('lag distance h');
    #         if imeas == 1:
    #             plt.title('Sample semivariogram (col # %d)' %(icolV))
    #             plt.ylabel('semivariance  \gamma (h)')
    #         else:
    #             plt.title('Covariogram (col # %d)' %icolV)
    #             plt.ylabel('covariance  \sigma (h)')
    #
    #     elif nDim == 2: ### 2D case
    #         gsiz = Gridspecs[:,2];
    #         gmin = nnodesOff[:]*gsiz
    #         gmin = -gmin;
    #         ng = nnodesOff[:]*2+1;
    #         rastermap(outStruct,1,[ng, gmin, gsiz]);
    #         if imeas == 1:
    #             plt.title('Sample semivariogram map (col # %d)' %icolV)
    #         else:
    #             plt.title('Sample covariogram map (col # %d)' %icolV)
    #
    #         plt.xlabel('h_x')
    #         plt.ylabel('h_y')

    ## FINISHED
    print('Finished RASTER2STRUCTMAP: Version #1');

    return(outStruct,outNpairs)






################################################################################


def suprts2modelcovFFT(CovMapExtFFT, ind1Ext, sf1Ext, ind2Ext, sf2Ext):
    """Integrated model covariances between 1 or 2 sets of arbitrary supports.
    Function to calculate array of TOTAL or AVERAGE model covariances
    between 1 or 2 sets of irregular supports, using convolution in
    the frequency domain (FFT-based). Integration or averaging is
    IMPLICIT in the pre-computed sampling functions (from discrsuprtsFFT).

    Args:
        CovMapExtFFT (np.ndarray): Fourier transform of model covariance map
            evaluated at nodes of an extended MATLAB grid
        ind1Ext: (nSup1 x 1) cell array with MATLAB indices of non-zero
            sampling function values for support set #1 in extended MATLAB grid
        sf1Ext: (nSup1 x 1) cell array with sampling function values for support set #1
        ind2Ext: Optional (nSup2 x 1) cell array with MATLAB indices of
            non-zero sampling function values for support set #2 in extended
            MATLAB grid
        sf2Ext: Optional (nSup2 x 1) cell array with sampling function values
            for support set #2

    Return:
       np.ndarray: (nSup1 x nSup[1,2]) array with integrated covariances


    Important:
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
    # ## Get some input parameters
    # nSup1 = len(ind1Ext);
    # ngExtTot = np.prod(np.array(CovMapExtFFT.shape));
    #
    # ## Proceed according to whether nargin <4 or not
    # if nargin < 4:  #### Single set of supports
    #
    #     Out = zeros(nSup1,nSup1);
    #     # Loop over # of supports
    #     # First, loop over rows
    #     for ii in range(nSup1):
    #         # Construct array of sampling functions for TAIL support
    #         u1 = np.zeros(CovMapExtFFT.shape);
    #         # TODO: u1[ind1Ext{ii}] = sf1Ext{ii};
    #         # Compute convolution in frequency domain
    #         v1 = np.fft.fft(u1);
    #         v2 = v1;
    #         v1 = conj(v1);
    #         v1Lv2 = v1*CovMapExtFFT*v2;
    #         covOut = np.sum(v1Lv2)/ngExtTot;
    #         # Fill in diagonal elements of output array
    #         Out[ii,ii] = np.real(covOut);
    #         # Now loop over columns with jj>ii (upper triangular part)
    #         for jj in range(ii+1,nSup1);
    #             # Construct array of sampling functions for HEAD support
    #             u2 = np.zeros(CovMapExtFFT.shape);
    #             # TODO: u2[ind1Ext{jj}] = sf1Ext{jj};
    #             v2 = np.fft.fft(u2);
    #             # Compute integrated model covariance value
    #             v1Lv2 = v1*CovMapExtFFT*v2;
    #             covOut = np.sum(v1Lv2)/ngExtTot;
    #             covOut = np.real(covOut);
    #             # Place value in output array accounting for symmetry
    #             Out[ii,jj] = covOut;
    #             Out[jj,ii] = covOut;
    #
    # else: #### Two sets of supports
    #
    #     nSup2 = len(ind2Ext);
    #     Out   = np.zeros([nSup1,nSup2]);
    #     # First, loop over supports of set #1
    #     for ii = range(1, Nsup1+1);
    #         # Construct array of sampling functions for TAIL support
    #         u1 = np.zeros(CovMapExtFFT.shape);
    #         # TODO: u1[ind1Ext{ii}] = sf1Ext{ii};
    #         # Compute fft of u1
    #         v1 = conj(fftn(u1));
    #         # Now loop over supports of set #2
    #         for jj in 1:nSup2;
    #             # Construct array of sampling functions for HEAD support
    #             u2 = np.zeros(CovMapExtFFT.shape);
    #             # TODO: u2[ind2Ext{jj}] = sf2Ext{jj};
    #             v2 = np.fft.fft(u2);
    #             # Compute integrated model covariance value
    #             v1Lv2 = v1*CovMapExtFFT*v2;
    #             covOut = np.sum(v1Lv2)/ngExtTot;
    #             # Place value in output array
    #             Out[ii,jj] = np.real(covOut)

    # return(Out)
    pass



################################################################################
