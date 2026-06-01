import numpy as np

from curvefit.model.local import const, expcut, list_local_models, ln, pl


def test_all_models_registered():
    names = set(list_local_models())
    expected = {
        'ln',
        'pl',
        'expd',
        'bln',
        'bpl',
        'sbpl',
        'spindown',
        'psd',
        'const',
        'expcut',
    }
    assert expected <= names


def test_const_returns_constant_array():
    m = const()
    m.params['C'].val = 3.5
    X = np.array([1.0, 2.0, 3.0])[:, None]
    y = m.func(X)
    assert np.allclose(y, [3.5, 3.5, 3.5])
    assert m.type == 'math'


def test_expcut_is_unit_at_zero_and_decreasing():
    m = expcut()
    m.params['logxc'].val = 0.0  # xc = 1
    X = np.array([0.0, 1.0, 2.0])[:, None]
    y = m.func(X)
    assert np.isclose(y[0], 1.0)
    assert y[1] < y[0] and y[2] < y[1]
    assert m.type == 'mul'


def test_existing_models_are_additive():
    assert ln().type == 'add'
    assert pl().type == 'add'
