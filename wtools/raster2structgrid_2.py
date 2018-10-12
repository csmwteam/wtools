def raster2structgrid(Data, 
                      variogram=True, 
                      visual=True, 
                      spacing=[1],
                      TINY=1e-10):
    
	# Auto-variogram or auto-covariance map from 1D or 2D rasters
	#
	## DESCRIPTION: raster2structgrid.m
	# Function to compute auto-variogram or auto-covariance maps from 
	# 1D or 2D rasters. Missing values, flagged as NaNs, are allowed.
	# This function computes variograms/covariances in the
	# frequency domain via the Fast Fourier Transform (FFT).
	#
	## SYNTAX:
	#   [outStruct,outNpairs]=raster2structgrid(Data);
	#
	## INPUTS:
	#   Data = input arrray
	#   variogram=True => output semi-variogram. If False, covariogram
    #   visual=True    => Display graphing    
	#   spacing        => spatial distance between adjacent points, per dimension
    #   TINY           =  fraction of maximum number of contributing pairs
    #
	## OUTPUTS:
	#   outStruct   = output array with variogram or covariogram map,
	#                 depending on imeas, with size:
	#                 in 1D: ( 2*nxOutHalf+1 )
	#                 in 2D: ( 2*nxOutHalf+1 x 2*nxOutHalf+1 )
	#   outNpairs:  = output array with # of pairs available in each lag,
	#                 of same size as outStruct
	#
	## NOTES:
	#   Author: Dennis Marcotte: Computers & Geosciences,
	#   Vol. 22, No. 10, pp. 1175-1186, 1996.
	#
	###########################################################################
	#
	## SYNTAX:
	#   [outStruct,outNpairs]= raster2structgrid(DataIn,icolV,Gridspecs,nnodesOff,imeas,idisp);

	## CREDITS:
	###########################################################################
    #                                                                         #
    #                            Jonah Bartrand                               #
    #                       Department of Geophysics                          #
    #                       Colorado Schol of Mines                           #
    #                              Fall 2018                                  #
    #                                                                         #
    # BASED ON:  Marcotte, D. (1996): Fast Variogram Computation with FFT,    #
    #            Computers & Geosciences, 22(10), 1175-1186.                  #                                      
    ###########################################################################

    # Import required modules
    import numpy as np
    import matplotlib.pyplot as plt
                
    data_dims = Data.shape
    nDim = len(data_dims)
    
    if spacing == [1]:
        spacing = [1]*nDim    
    
    print('Assuming input to be %dD array of values.' %len(Data.shape))
    
    ## Get appropriate dimensions
    # find the closest multiple of 8 to obtain a good compromise between
    # speed (a power of 2) and memory required
    out_dims = [2*d-1 for d in data_dims]#[int(np.ceil((2*d-1)/8)*8) for d in data_dims]

    ## Form an indicator  matrix:
    # 0's for all data values, 1's for missing values
    missing_data_ind = np.isnan(Data);
    data_loc_ind = np.logical_not(missing_data_ind)
    # In data matrix, replace missing values by 0;
    Data[missing_data_ind] = 0  # missing replaced by 0
    
    ## FFT of Data
    fD = np.fft.fftn(Data,s=out_dims)
    
    ## FFT of Data*Data
    fDD = np.fft.fftn(Data*Data,s=out_dims)

    ## FFT of the indicator matrix
    fI = np.fft.fftn(data_loc_ind, s=out_dims)
    
    ## FFT of Data*indicator
    fID = np.fft.fftn(Data*data_loc_ind, s=out_dims)
    
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
    
    indzeros = outNpairs<(np.max(outNpairs)*TINY)
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
