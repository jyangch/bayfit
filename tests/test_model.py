import numpy as np

from curvefit.model.local import ln, pl


def test_single_model_eval():
    m = ln()
    m.params['k'].val = 2.0
    m.params['b'].val = 1.0
    X = np.array([0.0, 1.0, 2.0])[:, None]
    assert np.allclose(m.func(X), [1.0, 3.0, 5.0])


def test_composite_add_eval():
    a = ln()
    a.params['k'].val = 1.0
    a.params['b'].val = 0.0
    b = pl()
    b.params['alpha'].val = 1.0
    b.params['logA'].val = 0.0  # amplitude 1
    model = a + b
    X = np.array([1.0, 2.0, 3.0])[:, None]
    # ln = x ; pl = x ; sum = 2x
    assert np.allclose(model.func(X), [2.0, 4.0, 6.0])
