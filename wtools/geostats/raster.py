"""This module provides useful methods for operating on 1D and 2D rasters such
as making variogram or covariograms.
"""

__all__ = [
    'raster2structgrid',
    'suprts2modelcovFFT',
]

__displayname__ = 'Rasters'

import numpy as np


def raster2structgrid(datain, imeas='covar', rtol=1e-10):
    """Create an auto-variogram or auto-covariance map from 1D or 2D rasters.
    This computes auto-variogram or auto-covariance maps from
    1D or 2D rasters. This function computes variograms/covariances in the
    frequency domain via the Fast Fourier Transform (``np.fftn``).

    Note:
        For viewing the results, please use the ``plotStructGrid`` method
        from the ``plots`` module.

    Note:
        Missing values, flagged as ``np.nan``, are allowed.

    Args:
        datain (np.ndarray): input arrray with raster in GeoEas format
        imeas (str): key indicating which structural measure to compute:
            ``'var'`` for semi-variogram or ``'covar'`` for covariogram.
        gridspecs (list(GridSpec)): array with grid specifications using
            ``GridSpec`` objects
        rtol (float): the tolerance. Default is 1e-10

    Return:
        tuple(np.ndarray, np.ndarray):
            output array with variogram or covariogram map, depending
            on variogram choice, with size: in 1D: ( 2*nxOutHalf+1 ) or in 2D:
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
    # Check imeas
    itypes = ['covar', 'var']
    if isinstance(imeas, int) and imeas < 2 and imeas > -1:
        imeas = itypes[imeas]
    if imeas not in itypes:
        raise RuntimeError("imeas argument must be one of 'covar' for covariogram or 'var' for semi-variance. Not {}".format(imeas))

    data_dims = datain.shape
    nDim = len(data_dims)

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
    fD = np.fft.fftn(datain, s=out_dims)

    ## FFT of datain*datain
    fDD = np.fft.fftn(datain*datain, s=out_dims)

    ## FFT of the indicator matrix
    fI = np.fft.fftn(data_loc_ind, s=out_dims)

    ## FFT of datain*indicator
    fID = np.fft.fftn(datain*data_loc_ind, s=out_dims)

    ## Compute number of pairs at all lags
    outNpairs = np.real(np.fft.ifftn(np.abs(fI)**2)).astype(int)
    #Edit remove single formating for matlab v6
    #outNpairs = single(outNpairs);

    cov = np.real(  np.fft.ifftn(np.abs(fD)**2) /
                    np.fft.ifftn(np.abs(fI)**2) -
                    np.fft.ifftn(np.conj(fD)*fI) *
                    np.fft.ifftn(np.conj(fI)*fD) /
                    (np.fft.ifftn(np.abs(fI)**2))**2
                )

    if imeas == 'var':
        outStruct = np.max(cov)-cov
    else:
        outStruct = cov

    ## Reduce matrix to required size and shift,
    # so that the 0 lag appears at the center of each matrix

    unpad_ind = [[int(d/2),int(3*d/2)] for d in data_dims]
    unpad_list = [np.arange(*l) for l in unpad_ind]
    unpad_coord = np.meshgrid(*unpad_list, indexing='ij')

    outStruct=np.fft.fftshift(outStruct)[unpad_coord]
    outNpairs=np.fft.fftshift(outNpairs)[unpad_coord]

    indzeros = outNpairs<(np.max(outNpairs)*rtol)
    outStruct[indzeros] = np.nan

    return outStruct, outNpairs




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
