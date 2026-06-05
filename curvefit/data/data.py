"""Container types for generic curve-fitting data.

A :class:`DataUnit` bundles one set of curve data -- x/y values, optional
x/y errors, a per-unit weight, upper-limit flags, and a statistic choice --
into a single unit. Errors are normalised to a ``(2, npoint)`` array of
``[low, high]`` rows regardless of whether the caller supplies a scalar,
a symmetric 1-D array, or explicit asymmetric columns.  A :class:`Data`
holds an ordered collection of ``DataUnit`` instances and exposes
aggregated views of their counts, point numbers, and statistics, which
the inference layer consumes.
"""

from collections import OrderedDict
import inspect
import json
import os

import numpy as np
import pandas as pd

from curvefit.util.tools import json_dump

from ..util.info import Info


class Data:
    """Ordered collection of :class:`DataUnit` objects.

    Indexing with a key returns the stored :class:`DataUnit`; assignment
    or deletion re-runs the aggregate extraction so cached views stay
    consistent.

    Attributes:
        data: ``OrderedDict`` mapping names to :class:`DataUnit` instances.
        names: Ordered list of registered data-unit names.
        stats: Per-unit statistic strings in insertion order.
        npoints: Integer array of per-unit point counts.
    """

    def __init__(self, data=None):
        """Build a container from a list of ``(name, unit)`` tuples or a dict.

        Args:
            data: ``None``, a list of ``(name, DataUnit)`` tuples, or a
                dict mapping names to :class:`DataUnit` instances.

        Raises:
            ValueError: If ``data`` is not one of the supported input types.
        """

        self.data = data

    @property
    def data(self):

        return self._data

    @data.setter
    def data(self, new_data):
        """Replace the contents from a list of tuples or a dict and re-extract.

        Raises:
            ValueError: If ``new_data`` is not ``None``, a list, or a dict.
        """

        self._data = OrderedDict()

        if new_data is None:
            pass

        elif isinstance(new_data, list):
            for item in new_data:
                if isinstance(item, tuple):
                    self._setitem(*item)

            self._update()

        elif isinstance(new_data, dict):
            for item in new_data.items():
                self._setitem(*item)

            self._update()

        else:
            raise ValueError('unsupported data type')

    def _setitem(self, key, value):

        if not isinstance(value, DataUnit):
            raise ValueError('value parameter should be DataUnit type')

        value.name = key
        self._data[key] = value

    def _update(self):

        if self.data is None:
            raise ValueError('data is None')

        self.names = [key for key in self.data]
        self.stats = [unit.stat for unit in self.data.values()]
        self.npoints = np.array([unit.npoint for unit in self.data.values()])

    @property
    def info(self):
        """Tabular :class:`Info` view of per-unit name, point count, statistic, and upper limits."""

        info_dict = OrderedDict()
        info_dict['Name'] = [key for key in self.data]
        info_dict['Npoint'] = [unit.npoint for unit in self.data.values()]
        info_dict['Statistic'] = [unit.stat for unit in self.data.values()]
        info_dict['Weight'] = [unit.weight for unit in self.data.values()]

        return Info.from_dict(info_dict)

    def save(self, savepath):
        """Dump the per-unit info table to ``<savepath>/data.json``.

        Args:
            savepath: Directory path. Created if missing.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.info.data_list_dict, savepath + '/data.json')

    @property
    def fit_with(self):
        """Return the ``Model`` this data is bound to, or raise if unset.

        Raises:
            AttributeError: If no model has been assigned yet.
        """

        try:
            return self._fit_with
        except AttributeError:
            raise AttributeError('no model fit with') from None

    @fit_with.setter
    def fit_with(self, new_model):
        """Bind a ``Model`` to this data and keep the back-reference in sync.

        Raises:
            ValueError: If ``new_model`` is not a ``Model`` instance.
        """

        from ..model.model import Model

        self._fit_with = new_model

        if not isinstance(self._fit_with, Model):
            raise ValueError('fit_with argument should be Model type!')

        try:
            _ = self._fit_with.fit_to
        except AttributeError:
            self._fit_with.fit_to = self
        else:
            if self._fit_with.fit_to != self:
                self._fit_with.fit_to = self

    @property
    def xs(self):
        """Per-unit ``x`` arrays (``float64`` by construction)."""

        return [unit.xs for unit in self.data.values()]

    @property
    def ys(self):
        """Per-unit ``y`` arrays (``float64`` by construction) for the kernels."""

        return [unit.ys for unit in self.data.values()]

    @property
    def xerr(self):
        """Per-unit ``(2, n)`` x-error arrays (``float64`` by construction)."""

        return [unit.xerr for unit in self.data.values()]

    @property
    def yerr(self):
        """Per-unit ``(2, n)`` y-error arrays (``float64`` by construction)."""

        return [unit.yerr for unit in self.data.values()]

    @property
    def weights(self):
        """Per-unit scalar weights as a ``float64`` array (one per data unit)."""

        return np.array([unit.weight for unit in self.data.values()], dtype=float)

    @property
    def ups(self):
        """Per-unit upper-limit boolean masks."""

        return [unit.ups for unit in self.data.values()]

    @property
    def los(self):
        """Per-unit lower-limit boolean masks."""

        return [unit.los for unit in self.data.values()]

    @property
    def expr(self):
        """Best-effort identifier for this container inferred from caller scope."""

        return self.get_obj_name()

    def get_obj_name(self):
        """Walk call frames and return the outermost local name bound to ``self``.

        Returns ``None`` if no binding is found.
        """

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
        """Return the stored :class:`DataUnit` registered under ``key``."""

        return self._data[key]

    def __setitem__(self, key, value):
        """Register ``value`` under ``key`` and re-run aggregate extraction."""

        self._setitem(key, value)
        self._update()

    def __delitem__(self, key):
        """Remove the entry under ``key`` and re-run aggregate extraction."""

        del self._data[key]
        self._update()

    def __contains__(self, key):
        """Return ``True`` when ``key`` is registered in this container."""

        return key in self._data

    def __str__(self):

        return f'*** Data ***\n{self.info.text_table}\n'

    def __repr__(self):

        return self.__str__()

    def _repr_html_(self):

        return (
            f'{self.info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>Data</b></summary>'
            f'{self.info.html_table}'
            f'</details>'
        )


class DataUnit:
    """One curve-data record with errors, weights, upper limits, and a statistic.

    ``x`` and ``y`` are cast to ``float64`` arrays of length ``npoint``.
    Both ``xerr`` and ``yerr`` are normalised to a ``(2, npoint)`` array
    with rows ``[low, high]``; a scalar or ``None`` produces a uniform
    array, a 1-D array is broadcast symmetrically, and a 2-D array is
    accepted in either ``(2, npoint)`` or ``(npoint, 2)`` orientation.
    The per-unit weight defaults to ``1`` and upper-/lower-limit flags
    default to ``False``.

    Attributes:
        x: Independent-variable values, shape ``(npoint,)``.
        y: Dependent-variable values, shape ``(npoint,)``.
        npoint: Number of data points.
        xerr: X error array, shape ``(2, npoint)``, rows are ``[low, high]``.
        yerr: Y error array, shape ``(2, npoint)``, rows are ``[low, high]``.
        weight: Per-unit scalar fit weight.
        up: Boolean upper-limit mask, shape ``(npoint,)``.
        lo: Boolean lower-limit mask, shape ``(npoint,)``.
        stat: Statistic identifier string (e.g. ``'chi2'``).
    """

    def __init__(self, xs, ys, yerr, xerr=None, ups=None, los=None, stat='chi2', weight=1.0):
        """Initialise a data unit and normalise all error and weight arrays.

        Args:
            x: Independent-variable values; any array-like of length ``n``.
            y: Dependent-variable values; any array-like of length ``n``.
            yerr: Y errors; required.  Scalar, 1-D array of length ``n``, or
                2-D array of shape ``(2, n)`` or ``(n, 2)``.  For an
                unweighted (OLS) fit pass an explicit constant such as ``1``.
            xerr: X errors; same shapes as ``yerr``.  ``None`` defaults to
                ``0`` (no x uncertainty).
            weight: Per-unit scalar fit weight for this whole dataset.
                Defaults to ``1``.
            up: Boolean upper-limit flag per point; array of length ``n``
                or ``None`` (all ``False``).
            lo: Boolean lower-limit flag per point; array of length ``n``
                or ``None`` (all ``False``).
            stat: Statistic string used during fitting.  Defaults to
                ``'chi2'``.

        Raises:
            ValueError: If ``yerr`` is ``None``, if any error, weight, or
                limit array length does not match the data, or an error has
                an unsupported shape.
        """

        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)
        self.npoint = self.xs.shape[0]

        self.xerr = self._normalize_err(xerr, default=0.0)
        self.yerr = self._normalize_err(yerr, default=1.0)
        self.ups = self._normalize_mask(ups, 'up')
        self.los = self._normalize_mask(los, 'lo')
        self.stat = stat
        self.weight = float(weight)

    def _normalize_err(self, err, default=0.0):

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

    def _normalize_mask(self, mask, label):

        n = self.npoint

        if mask is None:
            return np.zeros(n, dtype=bool)

        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != n:
            raise ValueError(f'{label} length does not match data')

        return mask

    @classmethod
    def from_dict(
        cls,
        dict,
        xs='xs',
        ys='ys',
        xerr='xerr',
        yerr='yerr',
        ups='ups',
        los='los',
        stat='stat',
        weight='weight',
    ):
        """Construct a :class:`DataUnit` from a plain dictionary.

        Recognised keys are ``'xs'``, ``'ys'``, ``'yerr'``, ``'xerr'``,
        ``'weight'``, ``'ups'``, ``'los'``, ``'stat'``, and ``'name'``; all
        except ``'xs'``, ``'ys'``, and ``'yerr'`` are optional and fall back to
        the :class:`DataUnit` defaults.  Extra keyword arguments are forwarded
        to the constructor.

        Args:
            d: Mapping with at least ``'xs'``, ``'ys'``, and ``'yerr'`` keys.
            **kwargs: Additional keyword arguments passed to ``__init__``.

        Raises:
            KeyError: If ``d`` is missing ``'xs'``, ``'ys'``, or ``'yerr'``.

        Example:
            >>> du = DataUnit.from_dict({'xs': [1, 2], 'ys': [3, 4], 'yerr': [0.1, 0.2]})
        """

        return cls(
            dict[xs],
            dict[ys],
            yerr=dict[yerr],
            xerr=dict.get(xerr),
            ups=dict.get(ups),
            los=dict.get(los),
            stat=dict.get(stat, 'chi2'),
            weight=dict.get(weight, 1),
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        xs='xs',
        ys='ys',
        xerr='xerr',
        yerr='yerr',
        xerr_low='xerr_low',
        xerr_high='xerr_high',
        yerr_low='yerr_low',
        yerr_high='yerr_high',
        ups='ups',
        los='los',
        stat='chi2',
        weight=1.0,
    ):
        """Construct a :class:`DataUnit` from a ``pandas`` ``DataFrame``.

        Column names for symmetric errors (``xerr``, ``yerr``) and
        asymmetric error pairs (``xerr_low``/``xerr_high``,
        ``yerr_low``/``yerr_high``) are resolved in this order:
        explicit keyword arguments take precedence, otherwise columns
        named ``'xerr'`` and ``'yerr'`` are detected automatically.
        When both ``_low`` and ``_high`` columns are given, the result
        is a ``(2, npoint)`` asymmetric error array; when only the
        symmetric column is given, errors are broadcast symmetrically.

        Args:
            df: Source ``DataFrame``; must contain columns named by ``x``
                and ``y``.
            x: Column name for the independent variable.  Defaults to
                ``'x'``.
            y: Column name for the dependent variable.  Defaults to
                ``'y'``.
            xerr: Column name for symmetric x errors, or ``None``.
            yerr: Column name for symmetric y errors, or ``None``.
            xerr_low: Column name for the low x error bound, or ``None``.
            xerr_high: Column name for the high x error bound, or ``None``.
            yerr_low: Column name for the low y error bound, or ``None``.
            yerr_high: Column name for the high y error bound, or ``None``.
            weight: Per-unit scalar weight for this whole dataset.  Defaults
                to ``1``.
            up: Column name for the boolean upper-limit mask, or ``None``
                (all ``False``).
            lo: Column name for the boolean lower-limit mask, or ``None``
                (all ``False``).
            stat: Statistic identifier.  Defaults to ``'chi2'``.
            name: Optional label for the constructed unit.

        Raises:
            KeyError: If a specified column name is absent from ``df``.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.2]})
            >>> du = DataUnit.from_dataframe(df)
            >>> du_asym = DataUnit.from_dataframe(df, yerr_low='yerr', yerr_high='yerr')
        """

        def col(c, dtype=float):
            return None if c not in df.columns else np.asarray(df[c], dtype=dtype)

        def err_col(sym, low, high):
            if low is not None and high is not None:
                return np.vstack([low, high])
            if sym is not None:
                return sym
            return None

        return cls(
            col(xs),
            col(ys),
            yerr=err_col(col(yerr), col(yerr_low), col(yerr_high)),
            xerr=err_col(col(xerr), col(xerr_low), col(xerr_high)),
            ups=col(ups, dtype=bool),
            los=col(los, dtype=bool),
            stat=stat,
            weight=weight,
        )

    @classmethod
    def from_json(cls, path):
        """Construct a :class:`DataUnit` by reading a JSON file.

        The file must contain a JSON object with at least ``'x'`` and
        ``'y'`` keys; the same optional keys accepted by
        :meth:`from_dict` are also honoured.

        Args:
            path: Path to the JSON file.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`from_dict`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If the object is missing ``'x'`` or ``'y'``.
        """

        with open(path) as f:
            dict = json.load(f)

        return cls.from_dict(dict)

    @classmethod
    def from_csv(cls, path):
        """Construct a :class:`DataUnit` by reading a CSV file.

        Column discovery follows the same rules as :meth:`from_dataframe`:
        columns named ``'xerr'`` and ``'yerr'`` are auto-detected as
        symmetric errors when no explicit error column names are given via
        ``**kwargs``.

        Args:
            path: Path to the CSV file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If the required ``'x'`` or ``'y'`` column is absent.
        """

        df = pd.read_csv(path)

        return cls.from_dataframe(df)

    @property
    def info(self):
        """Tabular :class:`Info` view of the unit's point count, statistic, and limit counts."""

        info_dict = OrderedDict()
        info_dict['npoint'] = self.npoint
        info_dict['stat'] = self.stat
        info_dict['weight'] = self.weight

        info_dict = OrderedDict(
            [('property', list(info_dict.keys())), (self.name, list(info_dict.values()))]
        )

        return Info.from_dict(info_dict)

    def save(self, savepath):
        """Dump the unit's info table to ``<savepath>/dataunit.json``.

        Args:
            savepath: Directory path. Created if missing.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.info.data_list_dict, savepath + '/dataunit.json')

    @property
    def name(self):
        """User-assigned name if set, otherwise a best-effort caller-scope name."""

        try:
            return self._name
        except AttributeError:
            return self.get_obj_name()

    @name.setter
    def name(self, new_name):

        self._name = new_name

    def get_obj_name(self):
        """Walk call frames and return the outermost local name bound to ``self``.

        Returns ``None`` if no binding is found.
        """

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

        return f'*** DataUnit ***\n{self.info.text_table}'

    def __repr__(self):

        return self.__str__()

    def _repr_html_(self):

        return (
            f'{self.info.html_style}'
            f'<details open>'
            f'<summary style="margin-bottom: 10px;"><b>DataUnit</b></summary>'
            f'{self.info.html_table}'
            f'</details>'
        )
