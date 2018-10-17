"""``plots``: This module provides various plotting routines that ensure we display
our spatially referenced data in logical, consistant ways across projects.
"""

__all__ = [
    'display',
    'plotStructGrid',
    'OrthographicSlicer',
]

__displayname__ = 'Plotting Routines'

import numpy as np

from .geostats.grids import GridSpec

def display(plt, arr, x=None, y=None, **kwargs):
    """This provides a convienant class for plotting 2D arrays that avoids
    treating our data like images. Since most datasets we work with are defined
    on Cartesian coordinates, <i,j,k> == <x,y,z>, we need to transpose our arrays
    before plotting in image plotting libraries like ``matplotlib``.

    Args:
        plt (handle): your active plotting handle
        arr (np.ndarray): A 2D array to plot
        kwargs (dict): Any kwargs to pass to the ``pcolormesh`` plotting routine

    Return:
        plt.pcolormesh

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> arr = np.arange(1000).reshape((10,100))
        >>> wtools.display(plt, arr)
        >>> plt.title('What we actually want')
        >>> plt.colorbar()
        >>> plt.show()

    """
    if x is None or y is None:
        return plt.pcolormesh(arr.T, **kwargs)
    return plt.pcolormesh(x, y, arr.T, **kwargs)



def plotStructGrid(plt, outStruct, gridspecs, imeas=None):
    """Plot a semivariogram or covariogram produced from raster2structgrid

    Args:
        plt (handle): An active plotting handle. This allows us to use the
            plotted result after the routine.
        outStruct (np.ndarray): the data to plot
        gridspecs (list(GridSpec)): the spatial reference of your gdata
        imeas (str): key indicating which structural measure to label:
            ``'var'`` for semi-variogram or ``'covar'`` for covariogram.
            This simply adds a few labels to the active figure.
            If semi-variance use ``True``. If covariance, use ``False``.

    Return:
        plt.plot or plt.pcolor

    """
    # Check that gridspecs is a list of ``GridSpec`` objects
    if not isinstance(gridspecs, list):
        if not isinstance(gridspecs, GridSpec):
            raise RuntimeError('gridspecs arguments ({}) improperly defined.'. format(gridspecs))
        gridspecs = [gridspecs] # Make sure we have a list to index if only 1D

    # Check the `GridSpec` objects and ensure they have an ``nnodes``
    nDim = outStruct.ndim
    if nDim != len(gridspecs):
        raise RuntimeError('Number of data dimensions does not match given gridspecs.')
    for i, gs in enumerate(gridspecs):
        gs.validate()
        if gs.nnodes is None:
            raise RuntimeError('GridSpec object at index %d does not have an nnodes property.' % i)
        if gs.nnodes > gs.n*0.5:
            raise RuntimeError('For GridSpec at index %d nnodesOff > # of gridnodes/2.' % i);

    if nDim > 3:
        raise RuntimeError('Plotting routine can only handle 1D or 2D grids. Please extract a 2D slice.')

    # Check imeas
    variogram = None
    if imeas is not None:
        itypes = ['covar', 'var']
        if isinstance(imeas, int) and imeas < 2 and imeas > -1:
            imeas = itypes[imeas]
        if imeas not in itypes:
            raise RuntimeError("imeas argument must be one of 'covar' for covariogram or 'var' for semi-variance. Not {}".format(imeas))
        if imeas == 'var':
            variogram = True
        else:
            variogram = False

    if nDim == 1:  ### 1D case
        gs = gridspecs[0]

        if gs.n != outStruct.shape[0]:
            raise RuntimeError('gridspecs do not match input data')

        xax = gs.sz * np.arange(-gs.n / 2, gs.n / 2)

        if variogram is not None:
            plt.xlabel(r'lag distance $h$')
            if variogram:
                plt.title('Sample semivariogram')
                plt.ylabel(r'Semivariance  $\gamma (h)$')
            else:
                plt.title('Covariogram')
                plt.ylabel(r'Covariance  $\sigma (h)$')

        return plt.plot(xax, outStruct,'.-')

    # Otherwise it is 2D
    gsx = gridspecs[0]
    gsy = gridspecs[1]

    if gsx.n != outStruct.shape[0] or gsy.n != outStruct.shape[1]:
        raise RuntimeError('gridspecs do not match input data')

    xax = gsx.sz * np.arange(-gsx.n / 2, gsx.n / 2)
    yax = gsy.sz * np.arange(-gsy.n / 2, gsy.n / 2)

    Y_show, X_show = np.meshgrid(xax, yax)

    if variogram is not None:
        if variogram:
            plt.title('Sample Semivariogram Map' )
        else:
            plt.title('Sample Covariogram Map')

        plt.xlabel('h_x')
        plt.ylabel('h_y')
    return display(plt, outStruct, x=X_show, y=Y_show, cmap='nipy_spectral')




class OrthographicSlicer():
    """Plot slices of the 3D volume.

    Use ``x``, ``y``, and ``z`` keyword arguments to specify the constant value
    to plot agianst. If none given, will use center of volume.

    Args:
        grid (GriddedData): The grid to plot
        plt (handle): An active plotting handle. This allows us to use the
            plotted result after the routine.
        model (str): The model name to plot (the attribute data)
        x, y, or z (int or float): the constant values to slice against.

    Return:
        None

    """
    def __init__(self, plt, grid, model, xslice=None, yslice=None, zslice=None):
        from ipywidgets import interact
        import ipywidgets as widgets
        # Figure
        #plt.clf()  # Just in case it exists already

        self.model = model
        self.grid = grid
        # Get slicing locations
        if xslice:
            self.xind = np.argmin(np.abs(grid.xcenters - xslice))
        else:
            self.xind = grid.xnodes.size // 2
        if yslice:
            self.yind = np.argmin(np.abs(grid.ycenters - yslice))
        else:
            self.yind = grid.ynodes.size // 2
        if zslice:
            self.zind = np.argmin(np.abs(grid.zcenters - zslice))
        else:
            self.zind = grid.znodes.size // 2

        # 2. Start figure

        # Create subplots
        plt.subplots_adjust(wspace=.075, hspace=.1)

        # X-Y
        self.ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        plt.ylabel('y-axis (units)')
        self.ax1.xaxis.set_ticks_position('top')
        plt.setp(self.ax1.get_xticklabels(), visible=False)

        # X-Z
        self.ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2,
                                    sharex=self.ax1)
        self.ax2.yaxis.set_ticks_position('both')
        plt.gca().invert_yaxis()
        plt.xlabel('x-axis (units)')
        plt.ylabel('z-axis (units)')

        # Z-Y
        self.ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2,
                                    sharey=self.ax1)
        self.ax3.yaxis.set_ticks_position('right')
        self.ax3.xaxis.set_ticks_position('both')
        plt.setp(self.ax3.get_yticklabels(), visible=False)

        # Title
        plt.suptitle('Use slider bars to control slice locations.')

        # Cross-line properties
        self.clprops = {'c': 'w', 'ls': '--', 'lw': 1, 'zorder': 10}

        # Store min and max of all data
        self.pcm_props = {'vmin': np.nanmin(grid.models[model]),
                          'vmax': np.nanmax(grid.models[model]),
                          'edgecolor': 'k'}

        # Create colorbar
        plt.sca(self.ax3)
        plt.pcolormesh(grid.znodes, grid.ynodes, grid.models[model][self.xind, :, :],
                       **self.pcm_props)
        plt.colorbar(label='Colorbar legend (units)', pad=0.15)

        # Initial draw
        self.update_xy(plt)
        self.update_xz(plt)
        self.update_zy(plt)

        # 3. Keep depth in X-Z and Z-Y in sync

        def do_adjust():
            """Return True if z-axis in X-Z and Z-Y are different."""
            one = np.array(self.ax2.get_ylim())
            two = np.array(self.ax3.get_xlim())[::-1]
            return sum(abs(one - two)) > 0.001  # Difference at least 1 m.

        def on_ylims_changed(ax):
            """Adjust Z-Y if X-Z changed."""
            if do_adjust():
                self.ax3.set_xlim([self.ax2.get_ylim()[1],
                                   self.ax2.get_ylim()[0]])

        def on_xlims_changed(ax):
            """Adjust X-Z if Z-Y changed."""
            if do_adjust():
                self.ax2.set_ylim([self.ax3.get_xlim()[1],
                                   self.ax3.get_xlim()[0]])

        self.ax3.callbacks.connect('xlim_changed', on_xlims_changed)
        self.ax2.callbacks.connect('ylim_changed', on_ylims_changed)

        # # Add inteeract widgets
        # interact(self.adjust_x, plt=plt, x=widgets.FloatSlider(
        #                 min=grid.xcenters.min(),
        #                 max=grid.xcenters.max(), value=grid.xcenters[self.xind] ))
        # interact(self.adjust_y,
        #                 plt=plt,
        #                 y=widgets.FloatSlider(
        #                     min=grid.ycenters.min(),
        #                     max=grid.ycenters.max(), value=grid.ycenters[self.yind] ))
        # interact(self.adjust_z,
        #                 plt=plt,
        #                 z=widgets.FloatSlider(
        #                     min=grid.zcenters.min(),
        #                     max=grid.zcenters.max(), value=grid.zcenters[self.zind] ))


    @staticmethod
    def find_nearest_idx(a, a0):
        return np.abs(a - a0).argmin()

    def adjust_x(self, plt, x):
        self.xind = self.find_nearest_idx(self.grid.xcenters, x)
        self.update_zy(plt)
        plt.draw()

    def adjust_y(self, plt, y):
        self.yind = self.find_nearest_idx(self.grid.ycenters, y)
        self.update_xz(plt)
        plt.draw()

    def adjust_z(self, plt, z):
        self.zind = self.find_nearest_idx(self.grid.zcenters, z)
        self.update_xy(plt)
        plt.draw()


    def update_xy(self, plt):
        """Update plot for change in Z-index."""

        # Clean up
        self.clear_element('xy_pc')
        self.clear_element('xz_ah')
        self.clear_element('zy_av')

        # Draw X-Y slice
        plt.sca(self.ax1)
        zdat = self.grid.models[self.model][:, :, self.zind].transpose()
        self.xy_pc = plt.pcolormesh(self.grid.xnodes, self.grid.ynodes, zdat, **self.pcm_props)

        # Draw Z-slice intersection in X-Z plot
        plt.sca(self.ax2)
        print(self.grid.xcenters[0], self.grid.xcenters[-1])
        self.xz_ah = plt.axhline(self.grid.zcenters[self.zind],
                                 self.grid.xcenters[0], self.grid.xcenters[-1],
                                 **self.clprops)

        # Draw Z-slice intersection in Z-Y plot
        plt.sca(self.ax3)
        self.zy_av = plt.axvline(self.grid.zcenters[self.zind],
                                 self.grid.ycenters[0], self.grid.ycenters[-1],
                                 **self.clprops)

    def update_xz(self, plt):
        """Update plot for change in Y-index."""

        # Clean up
        self.clear_element('xz_pc')
        self.clear_element('zy_ah')
        self.clear_element('xy_ah')

        # Draw X-Z slice
        plt.sca(self.ax2)
        ydat = self.grid.models[self.model][:, self.yind, :].transpose()
        self.xz_pc = plt.pcolormesh(self.grid.xnodes, self.grid.znodes, ydat, **self.pcm_props)

        # Draw X-slice intersection in X-Y plot
        plt.sca(self.ax1)
        # SUSPECT
        self.xy_ah = plt.axhline(self.grid.ycenters[self.yind],
                                 self.grid.xcenters[0], self.grid.xcenters[-1],
                                 **self.clprops)

        # Draw X-slice intersection in Z-Y plot
        plt.sca(self.ax3)
        self.zy_ah = plt.axhline(self.grid.ycenters[self.yind],
                                 self.grid.zcenters[0], self.grid.zcenters[-1],
                                 **self.clprops)

    def update_zy(self, plt):
        """Update plot for change in X-index."""

        # Clean up
        self.clear_element('zy_pc')
        self.clear_element('xz_av')
        self.clear_element('xy_av')

        # Draw Z-Y slice
        plt.sca(self.ax3)
        xdat = self.grid.models[self.model][self.xind, :, :]
        self.zy_pc = plt.pcolormesh(self.grid.znodes, self.grid.ynodes, xdat, **self.pcm_props)

        # Draw Y-slice intersection in X-Y plot
        plt.sca(self.ax1)
        self.xy_av = plt.axvline(self.grid.xcenters[self.xind],
                                 self.grid.ycenters[0], self.grid.ycenters[-1],
                                 **self.clprops)

        # Draw Y-slice intersection in X-Z plot
        plt.sca(self.ax2)
        self.xz_av = plt.axvline(self.grid.xcenters[self.xind],
                                 self.grid.zcenters[0], self.grid.zcenters[-1],
                                 **self.clprops)

    def clear_element(self, name):
        """Remove element <name> from plot if it exists."""
        if hasattr(self, name):
            getattr(self, name).remove()
