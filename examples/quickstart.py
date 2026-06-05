import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from curvefit import BayesInfer, Data, DataUnit, Plot
from curvefit.model.local import line

savepath = './quickstart'

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 30)
yerr = np.full(x.size, 0.5)
y = 2.0 * x + 1.0 + rng.normal(0, 0.5, x.size)

unit = DataUnit(x, y, yerr=yerr, stat='chi2')
data = Data([('d', unit)])
print(data)

model = line()
model.params['logv'].frozen = True
print(model)

infer = BayesInfer([(data, model)])
print(infer)

post = infer.emcee(nstep=2000, discard=500, resume=False, savepath=savepath)
print(post)

fig = Plot.infer(post)
fig.savefig(f'{savepath}/fit.png', dpi=150)

cfig = Plot.post_corner(post)
cfig.savefig(f'{savepath}/corner.png', dpi=150)
