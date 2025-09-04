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
"""Unit test module for the xDSL implementation of the diagonalize_final_measurements pass"""


# pylint: disable=wrong-import-position

import numpy as np
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

catalyst = pytest.importorskip("catalyst")
from catalyst.passes import xdsl_plugin

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    DiagonalizeFinalMeasurementsPass,
    diagonalize_final_measurements_pass,
)


class TestDiagonalizeFinalMeasurementsPass:
    """Unit tests for the diagonalize-final-measurements pass."""

    def test_with_pauli_z(self, run_filecheck):
        """Test that a PauliZ observable is not affected by diagonalization"""

        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: quantum.namedobs %0[PauliZ] : !quantum.obs
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs

                // CHECK: quantum.expval %1 : f64
                %2 = quantum.expval %1 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_identity(self, run_filecheck):
        """Test that an Identity observable is not affected by diagonalization."""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: quantum.namedobs %0[Identity] : !quantum.obs
                %1 = quantum.namedobs %0[Identity] : !quantum.obs

                // CHECK: quantum.var %1 : f64
                %2 = quantum.var %1 : f64
                return
            }
            """
        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_pauli_x(self, run_filecheck):
        """Test that when diagonalizing a PauliX observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "Hadamard"() [[q0]]
                // CHECK-NEXT: [[q0_2:%.*]] =  quantum.namedobs [[q0_1]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliX]
                %1 = quantum.namedobs %0[PauliX] : !quantum.obs

                // CHECK: quantum.expval [[q0_2]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_pauli_y(self, run_filecheck):
        """Test that when diagonalizing a PauliY observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "PauliZ"() [[q0]]
                // CHECK-NEXT: [[q0_2:%.*]] = quantum.custom "S"() [[q0_1]]
                // CHECK-NEXT: [[q0_3:%.*]] = quantum.custom "Hadamard"() [[q0_2]]
                // CHECK-NEXT: [[q0_4:%.*]] =  quantum.namedobs [[q0_3]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliY]
                %1 = quantum.namedobs %0[PauliY] : !quantum.obs

                // CHECK: quantum.expval [[q0_4]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_hadamard(self, run_filecheck):
        """Test that when diagonalizing a Hadamard observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[quarter_pi:%.*]] = arith.constant -0.78539816339744828 : f64
                // CHECK-NEXT: [[q0_1:%.*]] = quantum.custom "RY"([[quarter_pi]]) [[q0]]
                // CHECK-NEXT: [[q0_2:%.*]] =  quantum.namedobs [[q0_1]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][Hadamard]
                %1 = quantum.namedobs %0[Hadamard] : !quantum.obs

                // CHECK: quantum.expval [[q0_2]]
                %2 = quantum.expval %1 : f64
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_composite_observable(self, run_filecheck):
        """Test transform on a measurement process with a composite observable. In this
        case, the simplified program is based on the MLIR generated by the circuit

        @qml.qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            return qml.expval(qml.Y(0)@qml.X(1) + qml.Z(2))
        """

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "PauliZ"() [[q0]]
                // CHECK: [[q0_2:%.*]] = quantum.custom "S"() [[q0_1]]
                // CHECK: [[q0_3:%.*]] = quantum.custom "Hadamard"() [[q0_2]]
                // CHECK: [[q_y:%.*]] =  quantum.namedobs [[q0_3]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliY]
                %3 = quantum.namedobs %0[PauliY] : !quantum.obs
                
                // CHECK: [[q1_1:%.*]] = quantum.custom "Hadamard"() [[q1]]
                // CHECK: [[q_x:%.*]] = quantum.namedobs [[q1_1]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliX]
                %4 = quantum.namedobs %1[PauliX] : !quantum.obs
                
                // CHECK: [[tensor0:%.*]] = quantum.tensor [[q_y]], [[q_x]] : !quantum.obs
                %5 = quantum.tensor %3, %4 : !quantum.obs
                
                // CHECK: [[q_z:%.*]] = quantum.namedobs [[q2]][PauliZ] : !quantum.obs
                %6 = quantum.namedobs %2[PauliZ] : !quantum.obs
                
                // CHECK: [[size:%.*]] = "test.op"() : () -> tensor<2xf64>
                %size_info = "test.op"() : () -> tensor<2xf64>
                
                // CHECK: quantum.hamiltonian([[size]] : tensor<2xf64>) [[tensor0]], [[q_z]] : !quantum.obs
                %7 = quantum.hamiltonian(%size_info : tensor<2xf64>) %5, %6 : !quantum.obs

                // CHECK: quantum.expval
                %8 = quantum.expval %7 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_multiple_measurements(self, run_filecheck):
        """Test diagonalizing a circuit with multiple measurements. The simplified program
        for this test is based on the circuit

        @qml.qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            return qml.var(qml.Y(0)), qml.var(qml.X(1))
        """

        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
    
                // CHECK: quantum.custom "PauliZ"()
                // CHECK-NEXT: quantum.custom "S"()
                // CHECK-NEXT: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %2 = quantum.namedobs %0[PauliY] : !quantum.obs
                // CHECK: quantum.var
                %3 = quantum.var %2 : f64
                
    
                // CHECK: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %4 = quantum.namedobs %1[PauliX] : !quantum.obs
    
                // CHECK: quantum.expval
                %5 = quantum.expval %4 : f64
                return
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)

    def test_with_overlapping_observables(self, run_filecheck):
        """Test diagonalizing a circuit with overlapping observables (i.e. the requested
        measurements are on the same wire, but commute). The simplified program
        for this test is based on the circuit

        @qml.qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            return qml.var(qml.X(0)), qml.var(qml.X(0))
        """

        # Note: ideally, we would like to see this only contain a single Hadamard, that's used by
        # both branches of measurements. The HLOLoweringPass currently merges these two branches, but
        # without that, this would return incorrect results (which would be captured in integration tests
        # below). This also shouldn't occur as long as split_non_commuting is a part of the pipeline before
        # diagonalization.

        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %1 = quantum.namedobs %0[PauliX] : !quantum.obs
                %2 = quantum.var %1 : f64
                // CHECK: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %3 = quantum.namedobs %0[PauliX] : !quantum.obs
                %4 = quantum.var %3 : f64
            }
            """

        pipeline = (DiagonalizeFinalMeasurementsPass(),)
        run_filecheck(program, pipeline)


class TestDiagonalizeFinalMeasurementsProgramCaptureExecution:
    """Integration tests going through plxpr (program capture enabled)"""

    # pylint: disable=unnecessary-lambda
    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize(
        "mp, obs, expected_res",
        [
            (qml.expval, qml.Identity, lambda x: 1),
            (qml.var, qml.Identity, lambda x: 0),
            (qml.expval, qml.X, lambda x: 0),
            (qml.var, qml.X, lambda x: 1),
            (qml.expval, qml.Y, lambda x: -np.sin(x)),
            (qml.var, qml.Y, lambda x: 1 - np.sin(x) ** 2),
            (qml.expval, qml.Z, lambda x: np.cos(x)),
            (qml.var, qml.Z, lambda x: 1 - np.cos(x) ** 2),
            (qml.expval, qml.Hadamard, lambda x: np.cos(x) / np.sqrt(2)),
            (qml.var, qml.Hadamard, lambda x: (2 - np.cos(x) ** 2) / 2),
        ],
    )
    def test_with_single_obs(self, mp, obs, expected_res):
        """Test the diagonalization transform for a circuit with a single measurement
        of a single supported observable"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit_ref(phi):
            qml.RX(phi, 0)
            return mp(obs(0))

        angle = 0.7692

        assert np.allclose(
            expected_res(angle), circuit_ref(angle)
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.allclose(expected_res(angle), circuit_compiled(angle))

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_with_composite_observables(self):
        """Test the transform works for an observable built using operator arithmetic
        (sprod, prod, sum)"""

        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def circuit_ref(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            qml.RY(y / 2, 2)
            return qml.expval(qml.Y(0) @ qml.X(1) + 3 * qml.X(2))

        def expected_res(x, y):
            y0_res = -np.sin(x)
            x1_res = np.sin(y)
            x2_res = np.sin(y / 2)
            return y0_res * x1_res + 3 * x2_res

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_with_multiple_measurements(self):
        """Test that the transform runs and returns the expected results for
        a circuit with multiple measurements"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_ref(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            return qml.expval(qml.Y(0)), qml.var(qml.X(1))

        def expected_res(x, y):
            return -np.sin(x), 1 - np.sin(y) ** 2

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_with_overlapping_observables(self):
        """Test the case where multiple overlapping (commuting) observables exist in
        the same circuit. Here we want to be sure not to diagonalize the Y(0)
        observable twice (once for each occurrence in measurements)"""

        # Note that this seems to work without split_non_commuting ONLY because the
        # HLOLoweringPass merges the identical branches into one branch; otherwise
        # the diagonalizing gate would be applied twice before calculating the output
        # for the second MeasurementProcess

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_ref(x):
            qml.RX(x, 0)
            return qml.expval(qml.Y(0)), qml.var(qml.Y(0))

        def expected_res(x):
            return -np.sin(x), 1 - np.sin(x) ** 2

        phi = 0.7316

        qml.capture.disable()
        assert np.allclose(
            expected_res(phi), circuit_ref(phi)
        ), "Sanity check failed, is expected_res correct?"
        qml.capture.enable()

        circuit_compiled = qml.qjit(
            diagonalize_final_measurements_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.allclose(expected_res(phi), circuit_compiled(phi))

    @pytest.mark.xfail(reason="for now, assume split_non_commuting is always applied")
    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_non_commuting_observables_raise_error(self):
        """Check that an error is raised if we try to diagonalize a circuit that contains
        non-commuting observables."""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @diagonalize_final_measurements_pass
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Y(0)), qml.expval(qml.X(0))

        with pytest.raises(
            RuntimeError, match="cannot diagonalize circuit with non-commuting observables"
        ):
            _ = circuit(0.7)


class TestDiagonalizeFinalMeasurementsCatalystFrontend:
    """Integration tests going through the catalyst frontend (program capture disabled)"""

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "mp, obs, expected_res",
        [
            (qml.expval, qml.Identity, lambda x: 1),
            (qml.var, qml.Identity, lambda x: 0),
            (qml.expval, qml.X, lambda x: 0),
            (qml.var, qml.X, lambda x: 1),
            (qml.expval, qml.Y, lambda x: -np.sin(x)),
            (qml.var, qml.Y, lambda x: 1 - np.sin(x) ** 2),
            (qml.expval, qml.Z, lambda x: np.cos(x)),
            (qml.var, qml.Z, lambda x: 1 - np.cos(x) ** 2),
            (qml.expval, qml.Hadamard, lambda x: np.cos(x) / np.sqrt(2)),
            (qml.var, qml.Hadamard, lambda x: (2 - np.cos(x) ** 2) / 2),
        ],
    )
    def test_with_single_obs(self, mp, obs, expected_res):
        """Test the diagonalization transform for a circuit with a single measurement
        of a single supported observable"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit_ref(phi):
            qml.RX(phi, 0)
            return mp(obs(0))

        angle = 0.7692

        assert np.allclose(
            expected_res(angle), circuit_ref(angle)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            catalyst.passes.apply_pass("catalyst_xdsl_plugin.diagonalize-final-measurements")(
                circuit_ref
            ),
        )

        np.allclose(expected_res(angle), circuit_compiled(angle))

    def test_with_composite_observables(self):
        """Test the transform works for an observable built using operator arithmetic
        (sprod, prod, sum)"""

        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def circuit_ref(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            qml.RY(y / 2, 2)
            return qml.expval(qml.Y(0) @ qml.X(1) + 3 * qml.X(2))

        def expected_res(x, y):
            y0_res = -np.sin(x)
            x1_res = np.sin(y)
            x2_res = np.sin(y / 2)
            return y0_res * x1_res + 3 * x2_res

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            catalyst.passes.apply_pass("catalyst_xdsl_plugin.diagonalize-final-measurements")(
                circuit_ref
            ),
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))

    def test_with_multiple_measurements(self):
        """Test that the transform runs and returns the expected results for
        a circuit with multiple measurements"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_ref(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            return qml.expval(qml.Y(0)), qml.var(qml.X(1))

        def expected_res(x, y):
            return -np.sin(x), 1 - np.sin(y) ** 2

        phi = 0.3867
        theta = 1.394

        assert np.allclose(
            expected_res(phi, theta), circuit_ref(phi, theta)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            catalyst.passes.apply_pass("catalyst_xdsl_plugin.diagonalize-final-measurements")(
                circuit_ref
            ),
        )

        assert np.allclose(expected_res(phi, theta), circuit_compiled(phi, theta))

    def test_with_overlapping_observables(self):
        """Test the case where multiple overlapping (commuting) observables exist in
        the same circuit."""

        # Note that this seems to work without split_non_commuting ONLY because the
        # HLOLoweringPass merges the identical branches into one branch; otherwise
        # the diagonalizing gate would be applied twice before calculating the output
        # for the second MeasurementProcess

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_ref(x):
            qml.RX(x, 0)
            return qml.expval(qml.Y(0)), qml.var(qml.Y(0))

        def expected_res(x):
            return -np.sin(x), 1 - np.sin(x) ** 2

        phi = 0.7316

        assert np.allclose(
            expected_res(phi), circuit_ref(phi)
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            catalyst.passes.apply_pass("catalyst_xdsl_plugin.diagonalize-final-measurements")(
                circuit_ref
            ),
        )

        assert np.allclose(expected_res(phi), circuit_compiled(phi))

    @pytest.mark.xfail(reason="for now, assume split_non_commuting is always applied")
    def test_non_commuting_observables_raise_error(self):
        """Check that an error is raised if we try to diagonalize a circuit that contains
        non-commuting observables."""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit()
        @catalyst.passes.apply_pass("catalyst_xdsl_plugin.diagonalize-final-measurements")
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Y(0)), qml.expval(qml.X(0))

        with pytest.raises(
            RuntimeError, match="cannot diagonalize circuit with non-commuting observables"
        ):
            _ = circuit(0.7)
