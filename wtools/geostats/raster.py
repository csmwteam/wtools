__all__ = ['raster2structgrid']


def raster2structgrid(datain, variogram=True, visual=True, spacing=[1], rtol=1e-10):
    """Create an auto-variogram or auto-covariance map from 1D or 2D rasters.
    This computes auto-variogram or auto-covariance maps from
    1D or 2D rasters. This function computes variograms/covariances in the
    frequency domain via the Fast Fourier Transform (``np.fftn``).

    Todo:
        * Remove plotting code and place into the ``plots`` module
        * Remove spcaing arg as unneccesary

    Note:
        Missing values, flagged as ``np.nan``, are allowed.

    Args:
        datain (np.ndarray): input arrray with raster in GeoEas format
        variogram (bool): a flag on whether to produce a variogram or a
            covariance map
        gridspecs (list(GridSpec)): array with grid specifications using
            ``GridSpec`` objects
        rtol (float): the tolerance. Default is 1e-10

    Return:
        tuple(np.ndarray, np.ndarray):
            output array with variogram or covariogram map, depending
            on imeas, with size: in 1D: ( 2*nxOutHalf+1 ) or in 2D:
            ( 2*nxOutHalf+1 x 2*nxOutHalf+1 ).

            output array with number of pairs available in each lag,
            of same size as outStruct

    References:
        Originally implemented in MATLAB by:
            Phaedon Kyriakidis,
            Department of Geography,
            University of California Santa Barbara,
            May 2005

        Reimplemented into Python by:
            Jonah Bartrand,
            Department of Geophysics,
            Colorado School of Mines,
            October 2018

        Algorith based on:
            Marcotte, D. (1996): Fast Variogram Computation with FFT,
            Computers & Geosciences, 22(10), 1175-1186.
    """

    # Import required modules
    import numpy as np
    import matplotlib.pyplot as plt

    data_dims = datain.shape
    nDim = len(data_dims)

    if spacing == [1]:
        spacing = [1]*nDim

    print('Assuming input to be %dD array of values.' %len(datain.shape))

    ## Get appropriate dimensions
    # find the closest multiple of 8 to obtain a good compromise between
    # speed (a power of 2) and memory required
    out_dims = [2*d-1 for d in data_dims]#[int(np.ceil((2*d-1)/8)*8) for d in data_dims]

    ## Form an indicator  matrix:
    # 0's for all data values, 1's for missing values
    missing_data_ind = np.isnan(datain);
    data_loc_ind = np.logical_not(missing_data_ind)
    # In data matrix, replace missing values by 0;
    datain[missing_data_ind] = 0  # missing replaced by 0

    ## FFT of datain
    fD = np.fft.fftn(datain,s=out_dims)

    ## FFT of datain*datain
    fDD = np.fft.fftn(datain*datain,s=out_dims)

    ## FFT of the indicator matrix
    fI = np.fft.fftn(data_loc_ind, s=out_dims)

    ## FFT of datain*indicator
    fID = np.fft.fftn(datain*data_loc_ind, s=out_dims)

    ## Compute number of pairs at all lags
    outNpairs = np.real(np.fft.ifftn(np.abs(fI)**2)).astype(int)
    #Edit remove single formating for matlab v6
    #outNpairs = single(outNpairs);

    cov = np.real(np.fft.ifftn(np.abs(fD)**2)/np.fft.ifftn(np.abs(fI)**2) - np.fft.ifftn(np.conj(fD)*fI)*np.fft.ifftn(np.conj(fI)*fD)/(np.fft.ifftn(np.abs(fI)**2))**2)

    if variogram:
        outStruct = np.max(cov)-cov
    else:
        outStruct = cov

    ## Reduce matrix to required size and shift,
    # so that the 0 lag appears at the center of each matrix

    unpad_ind = [[int(d/2),int(3*d/2)] for d in data_dims]
    unpad_list = [np.arange(*l) for l in unpad_ind]
    unpad_coord = np.meshgrid(*unpad_list)

    outStruct=np.fft.fftshift(outStruct)[unpad_coord]
    outNpairs=np.fft.fftshift(outNpairs)[unpad_coord]

    indzeros = outNpairs<(np.max(outNpairs)*rtol)
    outStruct[indzeros] = np.nan

    ## Display results, if requested
    if visual:
        if nDim == 1:  ### 1D case
            xax = spacing[0]*np.arange(-data_dims[0]/2, data_dims[0]/2)
            plt.plot(xax,outStruct,'.-')
            plt.xlabel('lag distance h')

            if variogram:
                plt.title('Sample semivariogram')
                plt.ylabel('semivariance  \gamma (h)')
            else:
                plt.title('Covariogram')
                plt.ylabel('covariance  \sigma (h)')

        elif nDim == 2: ### 2D case
            xax = spacing[0]*np.arange(-data_dims[0]/2, data_dims[0]/2)
            yax = spacing[1]*np.arange(-data_dims[1]/2, data_dims[1]/2)

            Y_show, X_show = np.meshgrid(xax, yax)

            plt.pcolormesh(outStruct, X_show, Y_show, cmap='nipy_spectral')

            if imeas == 1:
                plt.title('Sample semivariogram map' )
            else:
                plt.title('Sample covariogram map')

            plt.xlabel('h_x')
            plt.ylabel('h_y')

    return(outStruct,outNpairs)
