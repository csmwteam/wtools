"""A set of conveinant methods for model transforms"""

__all__ = [
    'mapTable',
    'Models',
    'TimeModels',
]


import numpy as np
import pandas as pd
import properties
import json
import pickle


def mapTable(table, arr, to_dict=True, index=None):
    """Map the values defined by ``table`` dataframe to the values in ``arr``.
    If an index key is not given, then the first column (name should be
    ``Index``) will be used for the mapping.
    """
    # TODO: check that table table contains all indexs found in arr
    if index is None:
        index = table.keys()[0]
    table.set_index(index)
    ntbl = table[table.keys()].iloc[arr]
    if to_dict:
        d = {}
        for k in ntbl.columns:
            d[k] = np.array(ntbl[k].values)
        return d
    return  ntbl


class Models(properties.HasProperties):
    """A container for static models"""

    _models = properties.Dictionary('The data dictionary.',
                key_prop=properties.String('The model names'),
                value_prop=properties.Array('The data values', dtype=(float, int, bool), shape=None)
            )

    @property
    def shape(self):
        return list(self._models.values())[0].shape

    @properties.validator
    def _validate_models(self):
        shp = self.shape
        for k, d in self._models.items():
            if d.shape != shp:
                raise RuntimeError('Validation Failed: dimesnion mismatch between models.')
        return True

    def __getitem__(self, key):
        """Get a model by its string name and time index"""
        return self._models[key]

    def __setitem__(self, key, value):
        if self._models is None:
            self._models = {}
        self._models[key] = value

    def keys(self):
        return self._models.keys()

    def values(self):
        return self._models.values()

    def items(self):
        return self._models.items()

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.serialize(), f)
        return filename

    def pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filename):
        return pickle.load( open(filename, 'rb' ) )




class TimeModels(Models):
    """A container for time varying models"""

    _models = properties.Dictionary('The data dictionary.',
                key_prop=properties.String('The model names'),
                value_prop=properties.List('The model data as a time series list of arrays',
                    properties.Array('The data values', dtype=(float, int, bool), shape=None))
            )

    @property
    def shape(self):
        return np.array(list(self._models.values())[0][0]).shape

    @property
    def nt(self):
        return len(list(self._models.values())[0])

    @properties.validator
    def _validate_models(self):
        nt = self.nt
        shp = self.shape
        for key, data in self._models.items():
            if len(data) != nt:
                raise RuntimeError('Validation Failed: time step mismatch on `{}`.'.format(key))
            for arr in data:
                if arr.shape != shp:
                    raise RuntimeError('Validation Failed: dimesnion mismatch between models on `{}`.'.format(key))
        return True

    def __getitem__(self, pos):
        """Get a model by its string name and time index"""
        if not isinstance(pos, (list, tuple)):
            pos = [pos]
        if len(pos) == 1:
            return self._models[pos[0]]
        elif len(pos) == 2:
            return self._models[pos[0]][pos[1]]
        else:
            raise RuntimeError('Get at [{}] not understood.'.format(pos))

    def getTable(self, idx):
        """Returns a pandas dataframe table of all the models at a specified timestep"""
        df = pd.DataFrame()
        for k in self.keys():
            df[k] = self[k, idx].flatten(order='f')
        return df
