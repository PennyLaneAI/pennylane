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

import numpy as np
import pytest

# pytestmark = pytest.mark.external
import pennylane as qml
from pennylane.compiler.python_compiler.transforms import MeasurementsFromSamplesPass

xdsl = pytest.importorskip("xdsl")

# For lit tests of xdsl passes, we use https://github.com/AntonLydike/filecheck/,
# a Python re-implementation of FileCheck
filecheck = pytest.importorskip("filecheck")

catalyst = pytest.importorskip("catalyst")
from catalyst.passes import xdsl_plugin


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
    from filecheck.finput import FInput
    from filecheck.matcher import Matcher
    from filecheck.options import parse_argv_options
    from filecheck.parser import Parser, pattern_for_opts

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(module)),
        Parser(opts, io.StringIO(program), *pattern_for_opts(opts)),
    )
    assert matcher.run() == 0


class TestMeasurementsFromSamplesExecution:
    """Tests of the execution of simple workloads with the xDSL-based MeasurementsFromSamplesPass
    transform.
    """

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "measurement_process, initial_op, expected_res",
        [
            (qml.expval, qml.I, 1.0),
            (qml.expval, qml.X, -1.0),
            (qml.var, qml.I, 0.0),
            (qml.var, qml.X, 0.0),
        ],
    )
    def test_1_wire_pauliZ_basis(self, shots, measurement_process, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements in the PauliZ (computational) basis."""

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @MeasurementsFromSamplesPass
        @qml.qnode(dev)
        def circuit():
            initial_op(wires=0)
            return measurement_process(qml.Z(wires=0))

        result = circuit()
        assert result == expected_res

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Have not yet implemented transform for probs",
        strict=True,
    )
    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "measurement_process, initial_op, expected_res",
        [
            (qml.probs, qml.I, [1.0, 0.0]),
            (qml.probs, qml.X, [0.0, 1.0]),
        ],
    )
    def test_1_wire_pauliZ_basis_probs(self, shots, measurement_process, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements in the PauliZ (computational) basis.

        TODO: for probs
        """

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @MeasurementsFromSamplesPass
        @qml.qnode(dev)
        def circuit():
            initial_op(wires=0)
            return measurement_process(wires=0)

        result = circuit()
        assert np.array_equal(result, expected_res)

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Have currently only implemented support for computation-basis measurements",
        strict=False,
    )
    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "measurement_process, initial_op, expected_res",
        [
            (qml.expval, qml.I, 1.0),
            (qml.expval, qml.Z, -1.0),
            (qml.var, qml.I, 0.0),
            (qml.var, qml.Z, 0.0),
        ],
    )
    def test_1_wire_pauliX_basis(self, shots, measurement_process, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements in the PauliX basis.

        A measurement process in the PauliX basis should trigger diagonalize_measurements.
        """

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @MeasurementsFromSamplesPass
        @qml.qnode(dev)
        def circuit():
            qml.H(wires=0)
            initial_op(wires=0)
            return measurement_process(qml.Z(wires=0))

        result = circuit()
        assert result == expected_res

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Have currently only implemented support for computation-basis measurements",
        strict=False,  # Allow xpass; there is some probability that we return the expected result
    )
    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "measurement_process, initial_op, expected_res",
        [
            (qml.expval, qml.I, 1.0),
            (qml.expval, qml.Z, -1.0),
            (qml.var, qml.I, 0.0),
            (qml.var, qml.Z, 0.0),
        ],
    )
    def test_1_wire_pauliY_basis(self, shots, measurement_process, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements in the PauliY basis.

        A measurement process in the PauliY basis should trigger diagonalize_measurements.
        """

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @MeasurementsFromSamplesPass
        @qml.qnode(dev)
        def circuit():
            qml.H(wires=0)
            qml.S(wires=0)
            initial_op(wires=0)
            return measurement_process(qml.Z(wires=0))

        result = circuit()
        assert result == expected_res

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "measurement_process, initial_ops, expected_res",
        [
            (qml.expval, (qml.I, qml.I), (1.0, 1.0)),
            (qml.expval, (qml.I, qml.X), (1.0, -1.0)),
            (qml.expval, (qml.X, qml.I), (-1.0, 1.0)),
            (qml.expval, (qml.X, qml.X), (-1.0, -1.0)),
            (qml.var, (qml.I, qml.I), (0.0, 0.0)),
            (qml.var, (qml.I, qml.X), (0.0, 0.0)),
            (qml.var, (qml.X, qml.I), (0.0, 0.0)),
            (qml.var, (qml.X, qml.X), (0.0, 0.0)),
        ],
    )
    def test_2_wires_pauliZ_basis_separate(
        self, shots, measurement_process, initial_ops, expected_res
    ):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements in the PauliZ (computational) basis.

        In this test, the terminal measurements are performed separately per wire.
        """

        dev = qml.device("lightning.qubit", wires=2, shots=shots)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @MeasurementsFromSamplesPass
        @qml.qnode(dev)
        def circuit():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return measurement_process(qml.Z(wires=0)), measurement_process(qml.Z(wires=1))

        result = circuit()
        assert result == expected_res

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Operator arithmetic not yet supported with capture enabled", strict=True
    )
    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "measurement_process, initial_ops, expected_res",
        [
            (qml.expval, (qml.I, qml.I), 1.0),
            (qml.expval, (qml.I, qml.X), -1.0),
            (qml.expval, (qml.X, qml.I), -1.0),
            (qml.expval, (qml.X, qml.X), 1.0),
            (qml.var, (qml.I, qml.I), 0.0),
            (qml.var, (qml.I, qml.X), 0.0),
            (qml.var, (qml.X, qml.I), 0.0),
            (qml.var, (qml.X, qml.X), 0.0),
        ],
    )
    def test_2_wires_pauliZ_basis_combined(
        self, shots, measurement_process, initial_ops, expected_res
    ):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements in the PauliZ (computational) basis.

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2, shots=shots)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @MeasurementsFromSamplesPass
        @qml.qnode(dev)
        def circuit():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return measurement_process(qml.Z(wires=0) @ qml.Z(wires=1))

        result = circuit()
        assert result == expected_res


if __name__ == "__main__":
    pytest.main(["-x", __file__])
