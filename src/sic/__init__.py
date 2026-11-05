"""SIC: Survival In-Context.

Amortized Bayesian survival analysis via prior-fitted networks. The headline
entry point is the :class:`SIC` estimator::

    from sic import SIC
    surv = SIC().fit_predict_surv(X_train, y_train, X_test)
"""

from .__about__ import __version__
from .api import SIC, make_y
from .config import SURVIVAL_HEADS
from .train.datasets import load_dataset, list_datasets

# DUE TO A VERSION ISSUE WITH SCIPY, WE NEED TO MONKEY PATCH THE simps FUNCTION 
import scipy.integrate
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

__all__ = [
    "SIC",
    "make_y",
    "SURVIVAL_HEADS",
    "load_dataset",
    "list_datasets",
    "__version__",
]
