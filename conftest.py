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
import re
from collections.abc import Iterable
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

import numpy as base_numpy
import pytest
import scipy as base_scipy
from sybil import Document, Example, Region, Sybil
from sybil.parsers.abstract.lexers import LexerCollection
from sybil.parsers.markdown import PythonCodeBlockParser as MarkDownPythonCodeBlockParser
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser
from sybil.parsers.rest.lexers import DirectiveInCommentLexer

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
    if jax is not None:
        jax.config.update("jax_dynamic_shapes", False)
    # jax.config.update("jax_enable_x64", False)
    base_numpy.set_printoptions(precision=8)


@pytest.fixture(scope="module")
def local_decomp_context():
    """enable and disable graph-decomposition around each test."""
    with qp.decomposition.local_decomps():
        yield


def pytest_configure(config):
    """Used to amend to the pytest.ini used for testing."""
    config.addinivalue_line(
        "filterwarnings", "error::pennylane.exceptions.PennyLaneDeprecationWarning"
    )


class XfailParser:
    directive = "xfail"

    def __init__(self):
        self.lexers = LexerCollection(
            [DirectiveInCommentLexer(directive=re.escape(self.directive))]
        )

    def __call__(self, document: Document) -> Iterable[Region]:
        for lexed in self.lexers(document):
            reason = (lexed.lexemes.get("arguments") or "").lstrip(
                ": "
            ).strip() or "expected failure"
            yield Region(lexed.start, lexed.end, reason, self._install)

    def _install(self, example: Example):
        example.document.push_evaluator(_XfailInterceptor(example.parsed))


class _XfailInterceptor:
    def __init__(self, reason):
        self.reason = reason

    def __call__(self, example: Example):
        example.document.pop_evaluator(self)  # one-shot: wrap only the next example
        try:
            result = example.region.evaluator(example)  # run the real doctest
        except Exception:
            return None  # expected failure -> xfailed
        if result:  # evaluator returned a failure string
            return None  # xfailed
        raise AssertionError(
            f"[xfail strict] example unexpectedly PASSED (marked xfail: {self.reason})"
        )


pytest_collect_file = Sybil(
    setup=lambda ns: ns.update(namespace),
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
        MarkDownPythonCodeBlockParser(),
        XfailParser(),
    ],
    fixtures=["local_decomp_context"],
    patterns=["*.rst", "*.py", "*.md"],
    teardown=reset_pennylane_state,
).pytest()
