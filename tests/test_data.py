import json

import numpy as np
import pandas as pd

from curvefit.data.data import Data, DataUnit


def test_defaults_and_shapes():
    u = DataUnit([1, 2, 3], [10, 20, 30], yerr=1.0)
    assert u.npoint == 3
    assert u.xerr.shape == (2, 3)
    assert u.yerr.shape == (2, 3)
    assert np.all(u.xerr == 0)  # no xerr -> no x uncertainty
    assert np.all(u.yerr == 1)
    assert np.all(~u.up)
    assert np.all(u.weight == 1)
    assert u.stat == 'chi2'


def test_yerr_is_required():
    import pytest

    # yerr has no default: omitting it is a TypeError, passing None a ValueError
    with pytest.raises(TypeError):
        DataUnit([1, 2, 3], [10, 20, 30])
    with pytest.raises(ValueError, match='yerr must be provided'):
        DataUnit([1, 2, 3], [10, 20, 30], yerr=None)
    with pytest.raises(KeyError):
        DataUnit.from_dict({'x': [1, 2], 'y': [3, 4]})


def test_symmetric_and_asymmetric_errors():
    u = DataUnit([1, 2], [3, 4], yerr=[0.5, 0.5])
    assert np.allclose(u.yerr, [[0.5, 0.5], [0.5, 0.5]])
    a = DataUnit([1, 2], [3, 4], yerr=[[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(a.yerr, [[0.1, 0.2], [0.3, 0.4]])  # (2, n) low/high


def test_from_dict():
    u = DataUnit.from_dict({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.2], 'stat': 'chi2f'})
    assert u.stat == 'chi2f'
    assert np.allclose(u.y, [3, 4])


def test_from_dataframe_asymmetric():
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'yl': [0.1, 0.2], 'yh': [0.3, 0.4]})
    u = DataUnit.from_dataframe(df, yerr_low='yl', yerr_high='yh')
    assert np.allclose(u.yerr, [[0.1, 0.2], [0.3, 0.4]])


def test_from_csv(tmp_path):
    p = tmp_path / 'd.csv'
    pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6], 'yerr': [0.1, 0.1, 0.1]}).to_csv(p, index=False)
    u = DataUnit.from_csv(str(p))
    assert u.npoint == 3
    assert np.allclose(u.yerr[0], 0.1)


def test_from_json(tmp_path):
    p = tmp_path / 'd.json'
    with open(p, 'w') as f:
        json.dump({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.1]}, f)
    u = DataUnit.from_json(str(p))
    assert u.npoint == 2


def test_data_container():
    u1 = DataUnit([1, 2], [3, 4], yerr=1.0)
    u2 = DataUnit([1, 2], [3, 4], yerr=1.0, stat='vdr', up=[False, True])
    data = Data([('a', u1), ('b', u2)])
    assert data.names == ['a', 'b']
    assert data.stats == ['chi2', 'vdr']
    assert list(data.npoints) == [2, 2]
    assert 'a' in data
    assert data['b'].stat == 'vdr'


def test_lower_limit_defaults_and_normalization():
    u = DataUnit([1, 2, 3], [10, 20, 30], yerr=1.0)
    assert np.all(~u.lo)
    assert u.lo.shape == (3,)
    u2 = DataUnit([1, 2, 3], [10, 20, 30], yerr=1.0, lo=[True, False, True])
    assert list(u2.lo) == [True, False, True]


def test_lower_limit_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match='lo length does not match data'):
        DataUnit([1, 2, 3], [10, 20, 30], yerr=1.0, lo=[True, False])


def test_from_dict_lower_limit():
    u = DataUnit.from_dict({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.1], 'lo': [True, False]})
    assert list(u.lo) == [True, False]


def test_data_container_reports_lowerlimit_count():
    u = DataUnit([1, 2, 3], [3, 4, 5], yerr=1.0, lo=[True, True, False])
    data = Data([('a', u)])
    assert data.info.data_dict['Lowerlimit'] == [2]


def test_weight_is_per_unit_scalar():
    # weight is a single per-unit scalar (not a per-point array)
    u = DataUnit([1, 2, 3], [4, 5, 6], yerr=1.0, weight=2.0)
    assert u.weight == 2.0
    # 0-d numpy scalars/arrays are accepted and stored as a float
    assert DataUnit([1, 2], [3, 4], yerr=1.0, weight=np.float64(3.0)).weight == 3.0
    assert DataUnit([1, 2], [3, 4], yerr=1.0, weight=np.array(5.0)).weight == 5.0


def test_data_weights_are_per_unit():
    u1 = DataUnit([1, 2], [3, 4], yerr=1.0, weight=1.0)
    u2 = DataUnit([1, 2], [3, 4], yerr=1.0, weight=2.5)
    data = Data([('a', u1), ('b', u2)])
    assert np.allclose(data.weights, [1.0, 2.5])
