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

namespace = {
    "qml": qml,
    "np": base_numpy,
    "pnp": qml.numpy,
    "jax": jax,
    "torch": torch,
    "jnp": getattr(jax, "numpy", None),
}


# pylint: disable=unused-argument, redefined-outer-name
def reset_pennylane_state(namespace):
    """
    A teardown function for Sybil to reset PennyLane's global state
    after testing a document.
    """
    qml.capture.disable()
    qml.decomposition.disable_graph()


pytest_collect_file = Sybil(
    setup=lambda ns: ns.update(namespace),
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
    ],
    patterns=["*.rst", "*.py"],
    teardown=reset_pennylane_state,
).pytest()
