"""
These are ParaView plugins based on PVGeo for the wtools package
"""

from PVGeo.base import InterfacedBaseReader

import sys
import os
sys.path.append(os.path.dirname(__file__))
import wtools


@smproxy.reader(name="WtoolsGridReader",
       label='PVGeo: %s'%WtoolsGridReader.__displayname__,
       extensions=WtoolsGridReader.extensions,
       file_description=WtoolsGridReader.description)
class WtoolsGridReader(InterfacedBaseReader):
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

    @smproperty.xml(_helpers.getFileReaderXml(WtoolsGridReader.extensions, readerDescription=WtoolsGridReader.description))
    def AddFileName(self, fname):
        InterfacedBaseReader.AddFileName(self, fname)

    @smproperty.doublevector(name="TimeDelta", default_values=1.0, panel_visibility="advanced")
    def SetTimeDelta(self, dt):
        InterfacedBaseReader.SetTimeDelta(self, dt)

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        """This is critical for registering the timesteps"""
        return InterfacedBaseReader.GetTimestepValues(self)
