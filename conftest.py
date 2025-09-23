# pylint: disable = missing-module-docstring
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
import pytest

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
    "jax": jax,
    "torch": torch,
    "jnp": getattr(jax, "numpy", None),
}


@pytest.fixture(autouse=True)
def reset_global_state_fixture():
    """
    This autouse fixture automatically runs for every test. It resets
    any global state that might have been changed.
    """
    # The 'yield' keyword passes control to the test.
    # Code before 'yield' is the setup phase.
    yield
    # Code after 'yield' is the teardown/cleanup phase.
    # This runs after every test, even if it fails.
    qml.capture.disable()
    qml.decomposition.disable_graph()


pytest_collect_file = Sybil(
    setup=lambda ns: ns.update(namespace),
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
    ],
    patterns=["*.rst", "*.py"],
    fixtures=["reset_global_state_fixture"],
).pytest()
