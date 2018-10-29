import unittest
import shutil
import tempfile
import os
import numpy as np


# Functionality to test:
from wtools.mesh import *

RTOL = 0.000001

class TestsaveUBC(unittest.TestCase):
    """
    Test `saveUBC`: A file I/O method
    """
    def setUp(self):
        unittest.TestCase.setUp(self)
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.fname = os.path.join(self.test_dir, 'test')
        return

    def tearDown(self):
        # Remove the test data directory after the test
        shutil.rmtree(self.test_dir)
        unittest.TestCase.tearDown(self)

    def test_coords(self):
        # Create the unique coordinates along each axis
        x = np.linspace(0, 100, 11)
        y = np.linspace(220, 500, 11)
        z = np.linspace(0, 50, 11)
        # Create some model data
        arr = np.array([i*j*k for i in range(10) for j in range(10) for k in range(10)]).reshape(10, 10, 10)
        models = dict( foo=arr )
        # Perfrom the write out
        saveUBC(self.fname + 'c', x, y, z, models, header='A simple model')
        # Two files saved: 'test.msh' and 'test.foo'
        # TODO: make sure the files were generated correctly:
        #       - Bane did this maually
        return

    def test_uniform(self):
        # Uniform cell sizes
        d = np.random.random(1000).reshape((10, 10, 10))
        v = np.random.random(1000).reshape((10, 10, 10))
        models = dict(den=d, vel=v)
        saveUBC(self.fname + 'u', 25, 25, 2, models, widths=True, origin=(200.0, 100.0, 500.0))
        # Three files saved: 'volume.msh', 'volume.den', and 'volume.vel'
        # TODO: make sure the files were generated correctly:
        #       - Bane did this maually
        return


class TestTranspose(unittest.TestCase):
    """
    Test `transpose`: A coordinate transform method
    """

    def test(self):
        """Test a forward and back transpose"""
        d = np.random.random(1000).reshape((10, 10, 10))
        t = transpose(d)
        self.assertTrue(np.allclose(np.flip(d.swapaxes(0, 1), 2), t))
        t = transpose(t)
        self.assertTrue(np.allclose(d, t))
        return





###############################################################################
###############################################################################
###############################################################################
if __name__ == '__main__':
    import unittest
    unittest.main()
###############################################################################
###############################################################################
###############################################################################
