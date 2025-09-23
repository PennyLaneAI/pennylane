# pylint: disable = missing-module-docstring
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser

import numpy as base_numpy
import pennylane as qml

try:
    import jax
except ImportError:
    jax = None

try:
    import torch
except ImportError:
    torch = None

namespace = {"qml": qml, "np": base_numpy, "jax": jax, "torch": torch, "jnp": jax.numpy}

pytest_collect_file = Sybil(
    setup=lambda ns: ns.update(namespace),
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
    ],
    patterns=["*.rst", "*.py"],
).pytest()
