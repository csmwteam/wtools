import unittest
import shutil
import tempfile
import os
import numpy as np


# Functionality to test:
import wtools

RTOL = 0.000001


class TestTranspose(unittest.TestCase):
    """
    Test `transpose`: A coordinate transform method
    """

    def test(self):
        """Test a forward and back transpose"""
        d = np.random.random(1000).reshape((10, 10, 10))
        t = wtools.transpose(d)
        self.assertTrue(np.allclose(np.flip(d.swapaxes(0, 1), 2), t))
        t = wtools.transpose(t)
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
