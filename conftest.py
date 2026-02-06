# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable = missing-module-docstring
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser

import numpy as base_numpy
import scipy as base_scipy
import pennylane as qp

try:
    import jax
except ImportError:
    jax = None

try:
    import torch
except ImportError:
    torch = None


namespace = {
    "qp": qp,
    "np": base_numpy,
    "sp": base_scipy,
    "pnp": qp.numpy,
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
    qp.capture.disable()
    qp.decomposition.disable_graph()
    jax.config.update("jax_dynamic_shapes", False)
    # jax.config.update("jax_enable_x64", False)


pytest_collect_file = Sybil(
    setup=lambda ns: ns.update(namespace),
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
    ],
    patterns=["*.rst", "*.py"],
    teardown=reset_pennylane_state,
).pytest()
