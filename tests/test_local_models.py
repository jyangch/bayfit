import numpy as np

from curvefit.model import local

# Golden outputs captured before the ruff cleanup (lambda -> def refactor guard).
# Each model is built, every parameter set to val = 0.3 + 0.17 * j (j = param index),
# then evaluated on X = linspace(1, 9, 7)[:, None].
GOLDEN = {
    'ln': [0.77, 1.17, 1.57, 1.97, 2.37, 2.77, 3.17],
    'pl': [2.951209, 3.805336, 4.357942, 4.782897, 5.1344, 5.437283, 5.705225],
    'expd': [0.105281, 0.001236, 1.5e-05, 0.0, 0.0, 0.0, 0.0],
    'bln1': [0.77, 1.17, 1.57, 1.97, 2.37, 2.77, 3.17],
    'bln2': [1.1712, 1.797867, 2.424533, 3.0512, 3.677867, 4.304533, 4.9312],
    'bln3': [1.4857, 2.339033, 3.192367, 4.0457, 4.899033, 5.752367, 6.6057],
    'bln4': [1.7934, 2.7935, 3.8735, 4.9535, 6.0335, 7.1135, 8.1935],
    'bpl1': [2.951209, 3.805336, 4.357942, 4.782897, 5.1344, 5.437283, 5.705225],
    'bpl2': [6.965454, 10.372859, 12.827942, 14.841073, 16.585043, 18.143225, 19.563357],
    'bpl3': [14.690924, 25.267034, 33.742933, 41.151902, 47.87335, 54.100008, 59.946567],
    'bpl4': [31.009272, 57.376832, 82.743755, 106.375367, 128.824233, 150.385754, 171.242594],
    'sbpl2': [10.139091, 13.613661, 15.78303, 17.423586, 18.767635, 19.918647, 20.932511],
    'sbpl3': [30.882457, 42.211844, 48.520424, 53.303885, 57.24329, 60.631388, 63.625709],
    'sbpl4': [86.52084, 129.681584, 148.559317, 163.049882, 175.033471, 185.359081, 194.493373],
}


def _build(name):
    if name.startswith('bln'):
        return local.bln(int(name[3:]))
    if name.startswith('bpl'):
        return local.bpl(int(name[3:]))
    if name.startswith('sbpl'):
        return local.sbpl(int(name[4:]))
    return getattr(local, name)()


def test_local_models_match_golden():
    X = np.linspace(1, 9, 7)[:, None]
    for name, expected in GOLDEN.items():
        m = _build(name)
        for j, p in enumerate(m.params.values()):
            p.val = 0.3 + 0.17 * j
        y = np.asarray(m.func(X), dtype=float)
        assert np.allclose(y, expected, rtol=1e-4, atol=1e-6), name
