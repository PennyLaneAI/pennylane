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

"""Unit tests for the Python compiler `measurements_from_samples` transform."""

# pylint: disable=wrong-import-position

import io
import pytest

# pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

# For lit tests of xdsl passes, we use https://github.com/AntonLydike/filecheck/,
# a Python re-implementation of FileCheck 
filecheck = pytest.importorskip("filecheck")

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import MeasurementsFromSamplesPass

def test_transform():
    program = """
    // CHECK: identity
    func.func @identity(%arg0 : i32) -> i32 {
      return %arg0 : i32
    }
    """
    ctx = xdsl.context.Context()
    ctx.load_dialect(xdsl.dialects.builtin.Builtin)
    ctx.load_dialect(xdsl.dialects.func.Func)
    module = xdsl.parser.Parser(ctx, program).parse_module()
    pipeline = xdsl.passes.PipelinePass((MeasurementsFromSamplesPass(),))
    pipeline.apply(ctx, module)
    from filecheck.matcher import Matcher
    from filecheck.options import parse_argv_options
    from filecheck.parser import Parser, pattern_for_opts
    from filecheck.finput import FInput
    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(opts, FInput("no-name", str(module)), Parser(opts, io.StringIO(program),*pattern_for_opts(opts)))
    assert matcher.run() == 0


class TestMeasurementsFromSamplesExecution:
    """TODO"""
    def test_measurements_from_samples_basic(self):
        """TODO"""
        from catalyst.passes import xdsl_plugin

        qml.capture.enable()

        dev = qml.device("lightning.qubit", wires=2, shots=10)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        # @qml.qjit(keep_intermediate=True, pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @MeasurementsFromSamplesPass
        @qml.qnode(dev)
        def deleteme():
            qml.H(0)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))
            # return qml.sample(wires=0), qml.sample(wires=1)

        print("\n")
        print(deleteme())
        print("\n")
        print(deleteme.mlir)

        qml.capture.disable()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
