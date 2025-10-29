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

from inspect import getsource
from io import StringIO

import pytest

deps_available = True

try:
    from filecheck.finput import FInput
    from filecheck.matcher import Matcher
    from filecheck.options import parse_argv_options
    from filecheck.parser import Parser, pattern_for_opts
    from xdsl.context import Context
    from xdsl.dialects import test
    from xdsl.passes import PassPipeline
    from xdsl.printer import Printer

    from pennylane.compiler.python_compiler import Compiler, QuantumParser
    from pennylane.compiler.python_compiler.conversion import parse_generic_to_xdsl_module
except (ImportError, ModuleNotFoundError):
    deps_available = False


def _run_filecheck_impl(program_str, pipeline=(), verify=False, roundtrip=False):
    """Run filecheck on an xDSL module, comparing it to a program string containing
    filecheck directives."""
    if not deps_available:
        return

    ctx = Context()
    xdsl_module = QuantumParser(ctx, program_str, extra_dialects=(test.Test,)).parse_module()

    if roundtrip:
        # Print generic format
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(xdsl_module)
        xdsl_module = QuantumParser(ctx, stream.getvalue()).parse_module()

    if verify:
        xdsl_module.verify()

    pipeline = PassPipeline(pipeline)
    pipeline.apply(ctx, xdsl_module)

    if verify:
        xdsl_module.verify()

    stream = StringIO()
    Printer(stream).print_op(xdsl_module)
    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", stream.getvalue()),
        Parser(opts, StringIO(program_str), *pattern_for_opts(opts)),
    )

    exit_code = matcher.run()
    assert (
        exit_code == 0
    ), f"""
        filecheck failed with exit code {exit_code}.

        Original program string:
        {program_str}

        Parsed module:
        {stream.getvalue()}
    """


@pytest.fixture(scope="function")
def run_filecheck():
    """Fixture to run filecheck on an xDSL module.

    This fixture uses FileCheck to verify the correctness of a parsed MLIR string. Testers
    can provide a pass pipeline to transform the IR, and verify correctness by including
    FileCheck directives as comments in the input program string.

    Args:
        program_str (str): The MLIR string containing the input program and FileCheck directives
        pipeline (tuple[ModulePass]): A sequence containing all passes that should be applied
            before running FileCheck
        verify (bool): Whether or not to verify the IR after parsing and transforming.
            ``False`` by default.
        roundtrip (bool): Whether or not to use round-trip testing. This is useful for dialect
            tests to verify that xDSL both parses and prints the IR correctly. If ``True``, we parse
            the program string into an xDSL module, print it in generic format, and then parse the
            generic program string back to an xDSL module. ``False`` by default.
    """
    if not deps_available:
        pytest.skip("Cannot run lit tests without xDSL and filecheck.")

    yield _run_filecheck_impl


def _get_filecheck_directives(qjit_fn):
    """Return a string containing all FileCheck directives in the source function."""
    try:
        src = getsource(qjit_fn)
    except Exception as e:
        raise RuntimeError(f"Could not get source for {qjit_fn}") from e

    filecheck_directives = []
    for line in src.splitlines():
        line = line.strip()
        if line[0] != "#":
            continue

        line = line[1:].strip()
        if line.startswith("CHECK"):
            filecheck_directives.append("// " + line)

    return "\n".join(filecheck_directives)


def _run_filecheck_qjit_impl(qjit_fn, verify=False):
    """Run filecheck on a qjit-ed function, using FileCheck directives in its inline
    comments to assert correctness."""
    if not deps_available:
        return

    checks = _get_filecheck_directives(qjit_fn)
    compiler = Compiler()
    mlir_module = compiler.run(qjit_fn.mlir_module)

    # The following is done because ``mlir_module`` will be in the generic syntax, and
    # we want as many ops to be pretty printed as possible.
    mod_str = mlir_module.operation.get_asm(
        binary=False, print_generic_op_form=True, assume_verified=True
    )
    xdsl_module = parse_generic_to_xdsl_module(mod_str)

    if verify:
        xdsl_module.verify()

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(xdsl_module)),
        Parser(opts, StringIO(checks), *pattern_for_opts(opts)),
    )

    exit_code = matcher.run()
    assert exit_code == 0, f"filecheck failed with exit code {exit_code}"


@pytest.fixture(scope="function")
def run_filecheck_qjit():
    """Fixture to run filecheck on a qjit-ed function.

    This fixture yields a function that takes a QJIT object as input, parses its
    MLIR, applies any passes that are present, and uses FileCheck to check the
    output IR against FileCheck directives that may be present in the source
    function as inline comments.

    Args:
        qjit_fn (Callable): The QJIT object on which we want to run lit tests
        verify (bool): Whether or not to verify the IR after parsing and transforming.
            ``False`` by default.

    An example showing how to use the fixture is shown below. We apply the
    ``merge_rotations_pass`` and check that there is only one rotation in
    the final IR:

    .. code-block:: python

        def test_qjit(self, run_filecheck_qjit):
            # Test that the merge_rotations_pass works as expected when used with `qjit`
            dev = qml.device("lightning.qubit", wires=2)

            @qml.qjit(target="mlir", pass_plugins=[getXDSLPluginAbsolutePath()])
            @merge_rotations_pass
            @qml.qnode(dev)
            def circuit(x: float, y: float):
                # CHECK: [[phi:%.*]] = arith.addf
                # CHECK: quantum.custom "RX"([[phi]])
                # CHECK-NOT: quantum.custom
                qml.RX(x, 0)
                qml.RX(y, 0)
                return qml.state()

            run_filecheck_qjit(circuit)

    """
    if not deps_available:
        pytest.skip("Cannot run lit tests without xDSL and filecheck.")

    yield _run_filecheck_qjit_impl
