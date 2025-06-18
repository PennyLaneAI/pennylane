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
"""Pytest configuration for tests for the pennylane.compiler.python_compiler submodule."""

import io

import pytest

catalyst_available = True
filecheck_available = True
xdsl_available = True

try:
    from catalyst.compiler import _quantum_opt  # pylint: disable=protected-access
except ImportError:
    catalyst_available = False

try:
    from filecheck.finput import FInput
    from filecheck.matcher import Matcher
    from filecheck.options import parse_argv_options
    from filecheck.parser import Parser, pattern_for_opts
except ImportError:
    filecheck_available = False

try:
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func, linalg, scf, stablehlo, tensor, transform
    from xdsl.parser import Parser

    from pennylane.compiler.python_compiler import Quantum
except ImportError:
    xdsl_available = False


def _run_filecheck_impl(program_str, xdsl_module):
    """Run filecheck on an xDSL module, comparing it to a program string containing
    filecheck directives."""
    if not filecheck_available:
        return

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(xdsl_module)),
        Parser(opts, io.StringIO(program_str), *pattern_for_opts(opts)),
    )
    assert matcher.run() == 0


@pytest.fixture(scope="function")
def run_filecheck():
    """Fixture to run filecheck on an xDSL module."""
    if not filecheck_available:
        pytest.skip("Cannot run lit tests without filecheck.")

    yield _run_filecheck_impl


def _from_qjit_impl(qjit_fn):
    """Create a wrapper around a QJIT-ed function that returns an xDSL module."""

    def wrapper(*args, **kwargs):
        _ = qjit_fn(*args, **kwargs)
        mod = qjit_fn.mlir
        mod_generic = _quantum_opt(
            _quantum_opt(
                ("--pass-pipeline", "builtin.module(canonicalize)"),
                "-mlir-print-op-generic",
                stdin=mod,
            )
        )

        ctx = Context(allow_unregistered=True)
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(linalg.Linalg)
        ctx.load_dialect(scf.Scf)
        ctx.load_dialect(stablehlo.StableHLO)
        ctx.load_dialect(tensor.Tensor)
        ctx.load_dialect(transform.Transform)
        ctx.load_dialect(Quantum)

        xmod = Parser(ctx, mod_generic).parse_module()
        return xmod

    return wrapper


@pytest.fixture(scope="function")
def from_qjit():
    """Fixture to create an xDSL module from a QJIT-ed function"""
    if not catalyst_available or xdsl_available:
        pytest.skip("Cannot use the 'from_catalyst' fixture without Catalyst and xDSL installed.")

    yield _from_qjit_impl
