"""curvefit -- generic x/y curve-fitting framework.

Re-exports the four functional layers -- ``util`` (priors, parameters,
plot helpers), ``data`` (data containers and units), ``model`` (model
algebra and local components), and ``infer`` (pairs, samplers, analyzers) --
so that downstream code can ``from curvefit import ...`` directly.
"""

from .util import *  # noqa: F403
from .data import *  # noqa: F403
from .model import *  # noqa: F403
from .infer import *  # noqa: F403
from .__info__ import __version__
