import json
import numpy as np
import pandas as pd
from curvefit.data.data import DataUnit, Data


def test_defaults_and_shapes():
    u = DataUnit([1, 2, 3], [10, 20, 30])
    assert u.npoint == 3
    assert u.xerr.shape == (2, 3)
    assert u.yerr.shape == (2, 3)
    assert np.all(u.xerr == 0)
    assert np.all(u.yerr == 1)        # no yerr -> OLS
    assert np.all(u.up == False)
    assert np.all(u.weight == 1)
    assert u.stat == 'chi^2'


def test_symmetric_and_asymmetric_errors():
    u = DataUnit([1, 2], [3, 4], yerr=[0.5, 0.5])
    assert np.allclose(u.yerr, [[0.5, 0.5], [0.5, 0.5]])
    a = DataUnit([1, 2], [3, 4], yerr=[[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(a.yerr, [[0.1, 0.2], [0.3, 0.4]])   # (2, n) low/high


def test_from_dict():
    u = DataUnit.from_dict({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.2], 'stat': 'chi^2f'})
    assert u.stat == 'chi^2f'
    assert np.allclose(u.y, [3, 4])


def test_from_dataframe_asymmetric():
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4],
                       'yl': [0.1, 0.2], 'yh': [0.3, 0.4]})
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
    json.dump({'x': [1, 2], 'y': [3, 4], 'yerr': [0.1, 0.1]}, open(p, 'w'))
    u = DataUnit.from_json(str(p))
    assert u.npoint == 2


def test_data_container():
    u1 = DataUnit([1, 2], [3, 4])
    u2 = DataUnit([1, 2], [3, 4], stat='vdr', up=[False, True])
    data = Data([('a', u1), ('b', u2)])
    assert data.exprs == ['a', 'b']
    assert data.stats == ['chi^2', 'vdr']
    assert list(data.npoints) == [2, 2]
    assert 'a' in data
    assert data['b'].stat == 'vdr'
