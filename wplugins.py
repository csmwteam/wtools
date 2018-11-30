paraview_plugin_version = '0.0.4'
"""
These are ParaView plugins based on PVGeo for the wtools package
"""
from paraview.util.vtkAlgorithm import *
import PVGeo
from PVGeo.base import InterfacedBaseReader, ReaderBase
from PVGeo import _helpers

import sys
import os
sys.path.append(os.path.dirname(__file__))
import wtools


class _gridreader(InterfacedBaseReader):
    """A general reader for all ``discretize`` mesh objects saved to the
    ``.json`` serialized format"""
    extensions = 'json'
    __displayname__ = 'W Tools Grid Reader'
    description = 'Serialized W Tools Grids'
    def __init__(self, **kwargs):
        InterfacedBaseReader.__init__(self, **kwargs)

    @staticmethod
    def _readFile(filename):
        """Reads a mesh object from the serialized format"""
        return wtools.Grid.load_mesh(filename)

    @staticmethod
    def _getVTKObject(obj):
        """Returns the mesh's proper VTK data object"""
        return obj.toVTK()


@smproxy.reader(name="WtoolsGridReader",
       label='PVGeo: %s'%_gridreader.__displayname__,
       extensions=_gridreader.extensions,
       file_description=_gridreader.description)
class WtoolsGridReader(InterfacedBaseReader):
    """Wrapped for import to ParaView"""

    @smproperty.xml(_helpers.getFileReaderXml(_gridreader.extensions, readerDescription=_gridreader.description))
    def AddFileName(self, fname):
        InterfacedBaseReader.AddFileName(self, fname)

    @smproperty.doublevector(name="TimeDelta", default_values=1.0, panel_visibility="advanced")
    def SetTimeDelta(self, dt):
        InterfacedBaseReader.SetTimeDelta(self, dt)

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        """This is critical for registering the timesteps"""
        return InterfacedBaseReader.GetTimestepValues(self)







class _TimeModelsReader(ReaderBase):
    """Reade time model objects saved to the ``.json`` serialized format to a table"""
    extensions = 'pickle'
    __displayname__ = 'W Tools Time Models Reader'
    description = 'Serialized W Tools Time Models'
    def __init__(self, outputType='vtkTable', **kwargs):
        ReaderBase.__init__(self, **kwargs)
        self._timesteps = 1
        self.__models = None
        self.__dt = 1.0

    @staticmethod
    def _readFile(filename):
        """Reads a mesh object from the serialized format"""
        return wtools.load_pickle(filename)

    def _ReadUpFront(self):
        """Do not override. A predifiened routine for reading the files up front."""
        fileNames = self.GetFileNames()
        if len(fileNames) != 1:
            raise _helpers.PVGeoError('WTools time models reader can only handle 1 input file. Not ({}) files.'.format(len(fileNames)))
        self.__models = self._readFile(fileNames[0])
        self.NeedToRead(flag=False) # Only meta data has been read
        return 1

    def _UpdateTimeSteps(self):
        """For internal use only: appropriately sets the timesteps.
        """
        if self.NeedToRead():
            self._ReadUpFront()
        if self.__models.nt > 1:
            self._timesteps = _helpers.updateTimeSteps(self, self.__models.nt, self.__dt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        """Do not override. Used by pipeline to get data for current timestep
        and populate the output data object.
        """
        # Get requested time index
        i = _helpers.getRequestedTime(self, outInfo)
        # Get the models from the dict at that timestep
        table = self.__models.getTable(i)
        # Get output:
        output = self.GetOutputData(outInfo, 0)
        PVGeo.dataFrameToTable(table, pdo=output)
        return 1


@smproxy.reader(name="WtoolsTimeModelReader",
       label='PVGeo: %s'%_TimeModelsReader.__displayname__,
       extensions=_TimeModelsReader.extensions,
       file_description=_TimeModelsReader.description)
class WtoolsTimeModelReader(_TimeModelsReader):

    @smproperty.xml(_helpers.getFileReaderXml(_TimeModelsReader.extensions, readerDescription=_TimeModelsReader.description))
    def AddFileName(self, fname):
        _TimeModelsReader.AddFileName(self, fname)

    @smproperty.doublevector(name="TimeDelta", default_values=1.0, panel_visibility="advanced")
    def SetTimeDelta(self, dt):
        _TimeModelsReader.SetTimeDelta(self, dt)

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        """Use this in ParaView decorator to register timesteps on the pipeline."""
        if self.NeedToRead():
            self._ReadUpFront()
        print(self._timesteps)
        return self._timesteps if self._timesteps is not None else None
