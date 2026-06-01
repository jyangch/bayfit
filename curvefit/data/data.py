import json
import inspect
import numpy as np
import pandas as pd
from ..util.info import Info
from collections import OrderedDict



class Data(object):

    def __init__(self, data=None):

        self.data = data


    @property
    def data(self):

        return self._data


    @data.setter
    def data(self, new_data):

        self._data = OrderedDict()

        if new_data is None:
            pass

        elif isinstance(new_data, list):
            for item in new_data:
                if isinstance(item, tuple):
                    self._setitem(*item)

            self._extract()

        elif isinstance(new_data, dict):
            for item in new_data.items():
                self._setitem(*item)

            self._extract()

        else:
            raise ValueError('unsupported data type')


    def _setitem(self, key, value):

        if not isinstance(value, DataUnit):
            raise ValueError('value parameter should be DataUnit type')

        value.name = key
        self._data[key] = value


    def _extract(self):

        if self.data is None:
            raise ValueError('data is None')

        self.exprs = [key for key in self.data.keys()]
        self.stats = [unit.stat for unit in self.data.values()]
        self.npoints = np.array([unit.npoint for unit in self.data.values()])


    @property
    def expr(self):

        return self.get_obj_name() or 'data'


    @property
    def pdicts(self):

        return OrderedDict()


    @property
    def info(self):

        info_dict = OrderedDict()
        info_dict['Name'] = [key for key in self.data.keys()]
        info_dict['Npoint'] = [unit.npoint for unit in self.data.values()]
        info_dict['Statistic'] = [unit.stat for unit in self.data.values()]
        info_dict['Upperlimit'] = [int(np.sum(unit.up)) for unit in self.data.values()]

        return Info.from_dict(info_dict)


    @property
    def fit_with(self):

        try:
            return self._fit_with
        except AttributeError:
            raise AttributeError('no model fit with')


    @fit_with.setter
    def fit_with(self, new_model):

        from ..model.model import Model

        self._fit_with = new_model

        if not isinstance(self._fit_with, Model):
            raise ValueError('fit_with argument should be Model type!')

        try:
            self._fit_with.fit_to
        except AttributeError:
            self._fit_with.fit_to = self
        else:
            if self._fit_with.fit_to != self:
                self._fit_with.fit_to = self


    def get_obj_name(self):

        frame = inspect.currentframe()

        possible_var_names = []

        while frame:
            local_vars = frame.f_locals.items()
            var_names = [var_name for var_name, var_val in local_vars if var_val is self]
            if var_names:
                possible_var_names.extend(var_names)
            frame = frame.f_back

        if possible_var_names:
            return possible_var_names[-1]

        return None


    def __getitem__(self, key):

        return self._data[key]


    def __setitem__(self, key, value):

        self._setitem(key, value)
        self._extract()


    def __delitem__(self, key):

        del self._data[key]
        self._extract()


    def __contains__(self, key):

        return key in self._data


    def __str__(self):

        print(self.info.table)

        return ''



class DataUnit(object):

    def __init__(
        self,
        x,
        y,
        xerr=None,
        yerr=None,
        weight=1,
        up=None,
        stat='chi^2',
        name=None
        ):

        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.npoint = self.x.shape[0]

        self.xerr = self._normalize_err(xerr, default=0.0)
        self.yerr = self._normalize_err(yerr, default=1.0)
        self.weight = self._normalize_weight(weight)
        self.up = self._normalize_up(up)
        self.stat = stat

        if name is not None:
            self.name = name


    def _normalize_err(self, err, default):

        n = self.npoint

        if err is None:
            arr = np.full(n, default, dtype=float)
            return np.vstack([arr, arr])

        err = np.asarray(err, dtype=float)

        if err.ndim == 0:
            arr = np.full(n, float(err))
            return np.vstack([arr, arr])

        if err.ndim == 1:
            if err.shape[0] != n:
                raise ValueError('error length does not match data')
            return np.vstack([err, err])

        if err.ndim == 2:
            if err.shape == (2, n):
                return err.astype(float)
            if err.shape == (n, 2):
                return err.T.astype(float)
            raise ValueError('2D error should have shape (2, npoint) or (npoint, 2)')

        raise ValueError('unsupported error shape')


    def _normalize_weight(self, weight):

        n = self.npoint

        if np.isscalar(weight):
            return np.full(n, float(weight))

        weight = np.asarray(weight, dtype=float)

        if weight.ndim == 0:
            return np.full(n, float(weight))

        if weight.shape[0] != n:
            raise ValueError('weight length does not match data')

        return weight


    def _normalize_up(self, up):

        n = self.npoint

        if up is None:
            return np.zeros(n, dtype=bool)

        up = np.asarray(up, dtype=bool)
        if up.shape[0] != n:
            raise ValueError('up length does not match data')

        return up


    @classmethod
    def from_dict(cls, d, **kwargs):

        return cls(
            d['x'],
            d['y'],
            xerr=d.get('xerr'),
            yerr=d.get('yerr'),
            weight=d.get('weight', 1),
            up=d.get('up'),
            stat=d.get('stat', 'chi^2'),
            name=d.get('name'),
            **kwargs)


    @classmethod
    def from_dataframe(
        cls,
        df,
        x='x',
        y='y',
        xerr=None,
        yerr=None,
        xerr_low=None,
        xerr_high=None,
        yerr_low=None,
        yerr_high=None,
        weight=None,
        up=None,
        stat='chi^2',
        name=None
        ):

        def col(c):
            return None if c is None else np.asarray(df[c], dtype=float)

        # auto-detect standard column names when not explicitly specified
        if xerr is None and xerr_low is None and xerr_high is None:
            if 'xerr' in df.columns:
                xerr = 'xerr'
        if yerr is None and yerr_low is None and yerr_high is None:
            if 'yerr' in df.columns:
                yerr = 'yerr'

        xe = cls._cols_to_err(col(xerr), col(xerr_low), col(xerr_high))
        ye = cls._cols_to_err(col(yerr), col(yerr_low), col(yerr_high))
        w = 1 if weight is None else np.asarray(df[weight], dtype=float)
        u = None if up is None else np.asarray(df[up], dtype=bool)

        return cls(col(x), col(y), xerr=xe, yerr=ye, weight=w, up=u, stat=stat, name=name)


    @staticmethod
    def _cols_to_err(sym, low, high):

        if low is not None and high is not None:
            return np.vstack([low, high])
        if sym is not None:
            return sym
        return None


    @classmethod
    def from_json(cls, path, **kwargs):

        with open(path) as f:
            d = json.load(f)

        return cls.from_dict(d, **kwargs)


    @classmethod
    def from_csv(cls, path, **kwargs):

        df = pd.read_csv(path)

        return cls.from_dataframe(df, **kwargs)


    @property
    def name(self):

        try:
            return self._name
        except AttributeError:
            return self.get_obj_name()


    @name.setter
    def name(self, new_name):

        self._name = new_name


    @property
    def info(self):

        info_dict = OrderedDict()
        info_dict['npoint'] = self.npoint
        info_dict['stat'] = self.stat
        info_dict['upperlimit'] = int(np.sum(self.up))

        info_dict = OrderedDict([('property', list(info_dict.keys())),
                                 (self.name, list(info_dict.values()))])

        return Info.from_dict(info_dict)


    def get_obj_name(self):

        frame = inspect.currentframe()

        possible_var_names = []

        while frame:
            local_vars = frame.f_locals.items()
            var_names = [var_name for var_name, var_val in local_vars if var_val is self]
            if var_names:
                possible_var_names.extend(var_names)
            frame = frame.f_back

        if possible_var_names:
            return possible_var_names[-1]

        return None


    def __str__(self):

        print(self.info.table)

        return ''
