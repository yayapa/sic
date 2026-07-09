"""Vendored copy of TorchSurv (https://github.com/Novartis/torchsurv).

This is a lightly modified vendored fork of TorchSurv (Novartis, MIT license,
see LICENSE in this directory). It is bundled inside SIC because the released
PyPI version lacked functionality SIC relies on (e.g. the CoxPH baseline /
survival-function helpers used by the CoxPH head). Kept under ``sic._vendor`` so
it never collides with a separately installed ``torchsurv``.
"""

__version__ = "0.1.6+sic"

