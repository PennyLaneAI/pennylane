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

"""Unit and integration tests for the Python compiler `measurements_from_samples` transform."""

# pylint: disable=wrong-import-position

from functools import partial

import numpy as np
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")
from catalyst.passes import xdsl_plugin

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    MeasurementsFromSamplesPass,
    measurements_from_samples_pass,
)


class TestMeasurementsFromSamplesPass:
    """Unit tests for the measurements-from-samples pass."""

    def test_1_wire_expval(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with an expval(Z)
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %3 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.expval
                %4 = quantum.expval %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>

                // CHECK: func.return [[res]] : tensor<f64>
                func.return %5 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_var(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a var(Z)
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %3 = quantum.namedobs %2[PauliZ] : !quantum.obs

                // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res:%.+]] = func.call @var_from_samples.tensor.1x1xf64([[samples]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.var
                %4 = quantum.var %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>

                // CHECK: func.return [[res]] : tensor<f64>
                func.return %5 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @var_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_probs(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a probs
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg
                %2 = "test.op"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<1x1xf64>
                // CHECK: [[res:%.+]] = func.call @probs_from_samples.tensor.1x1xf64([[samples]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>
                // CHECK-NOT: quantum.probs
                %4 = quantum.probs %3 : tensor<2xf64>

                // CHECK: func.return [[res]] : tensor<2xf64>
                func.return %4 : tensor<2xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_1_wire_sample(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a sample
        measurement.

        This pass should be a no-op.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.reg
                %2 = "test.op"() : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<1x1xf64>
                %4 = quantum.sample %3 : tensor<1x1xf64>

                // CHECK: func.return [[samples]] : tensor<1x1xf64>
                func.return %4 : tensor<1x1xf64>
            }
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    @pytest.mark.xfail(reason="Counts not supported", strict=True, raises=NotImplementedError)
    def test_1_wire_counts(self, run_filecheck):
        """Test the measurements-from-samples pass on a 1-wire circuit terminating with a counts
        measurement.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.reg
                %2 = "test.op"() : () -> !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[q0]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample %3 : tensor<1x1xf64>
                // CHECK: [[eigvals:%.+]], [[counts:%.+]] = func.call @counts_from_samples.tensor.1x1xf64([[samples]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>, tensor<2xi64>
                // CHECK-NOT: quantum.counts
                %eigvals, %counts = quantum.counts %3 : tensor<2xf64>, tensor<2xi64>

                // CHECK: [[eigvals_converted:%.+]] = {{.*}}stablehlo.convert{{.+}}[[eigvals]] :
                // CHECK-SAME: (tensor<2xf64>) -> tensor<2xi64>
                %4 = "stablehlo.convert"(%eigvals) : (tensor<2xf64>) -> tensor<2xi64>

                // CHECK: func.return [[eigvals_converted]], [[counts]] : tensor<1x1xf64>
                func.return %4, %counts : tensor<2xi64>, tensor<2xi64>
            }
            // CHECK-LABEL: func.func public @counts_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_2_wire_expval(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with an expval(Z)
        measurement on each wire.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %4 = quantum.namedobs %2[PauliZ] : !quantum.obs
                %5 = quantum.namedobs %3[PauliZ] : !quantum.obs

                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples0:%.+]] = quantum.sample [[obs0]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[obs1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                // CHECK: [[samples1:%.+]] = quantum.sample [[obs1]] : tensor<1x1xf64>
                // CHECK: [[c1:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res0:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples0]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK: [[res1:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples1]], [[c1]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.expval
                %6 = quantum.expval %4 : f64
                %7 = "tensor.from_elements"(%6) : (f64) -> tensor<f64>
                %8 = quantum.expval %5 : f64
                %9 = "tensor.from_elements"(%8) : (f64) -> tensor<f64>

                // CHECK: func.return [[res0]], [[res1]] : tensor<f64>, tensor<f64>
                func.return %7, %9 : tensor<f64>, tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_2_wire_var(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with a var(Z)
        measurement on each wire.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit

                // CHECK-NOT: quantum.namedobs
                %4 = quantum.namedobs %2[PauliZ] : !quantum.obs
                %5 = quantum.namedobs %3[PauliZ] : !quantum.obs

                // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                // CHECK: [[samples0:%.+]] = quantum.sample [[obs0]] : tensor<1x1xf64>
                // CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[obs1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                // CHECK: [[samples1:%.+]] = quantum.sample [[obs1]] : tensor<1x1xf64>
                // CHECK: [[c1:%.+]] = arith.constant dense<0> : tensor<i64>
                // CHECK: [[res0:%.+]] = func.call @var_from_samples.tensor.1x1xf64([[samples0]], [[c0]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK: [[res1:%.+]] = func.call @var_from_samples.tensor.1x1xf64([[samples1]], [[c1]]) :
                // CHECK-SAME: (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.var
                %6 = quantum.var %4 : f64
                %7 = "tensor.from_elements"(%6) : (f64) -> tensor<f64>
                %8 = quantum.var %5 : f64
                %9 = "tensor.from_elements"(%8) : (f64) -> tensor<f64>

                // CHECK: func.return [[res0]], [[res1]] : tensor<f64>, tensor<f64>
                func.return %7, %9 : tensor<f64>, tensor<f64>
            }
            // CHECK-LABEL: func.func public @var_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_2_wire_probs_global(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with a "global"
        probs measurement (one that implicitly acts on all wires).
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[qreg:%.+]] = quantum.alloc
                %2 = quantum.alloc(2) : !quantum.reg

                // CHECK: [[compbasis:%.+]] = quantum.compbasis qreg [[qreg]] : !quantum.obs
                %3 = quantum.compbasis qreg %2 : !quantum.obs

                // CHECK: [[samples:%.+]] = quantum.sample [[compbasis]] : tensor<1x2xf64>
                // CHECK: [[res:%.+]] = func.call @probs_from_samples.tensor.1x2xf64([[samples]]) :
                // CHECK-SAME: (tensor<1x2xf64>) -> tensor<4xf64>
                // CHECK-NOT: quantum.probs
                %4 = quantum.probs %3 : tensor<4xf64>

                // CHECK: func.return [[res]] : tensor<4xf64>
                func.return %4 : tensor<4xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.1x2xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)

    def test_2_wire_probs_per_wire(self, run_filecheck):
        """Test the measurements-from-samples pass on a 2-wire circuit terminating with separate
        probs measurements per wire.
        """
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> (tensor<f64>) {
                %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[qreg:%.+]] = quantum.alloc
                %2 = quantum.alloc(2) : !quantum.reg

                // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][0]
                %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit

                // CHECK: [[compbasis0:%.+]] = quantum.compbasis qubits [[q0]] : !quantum.obs
                %4 = quantum.compbasis qubits %3 : !quantum.obs

                // CHECK: [[samples0:%.+]] = quantum.sample [[compbasis0]] : tensor<1x1xf64>
                // CHECK: [[res0:%.+]] = func.call @probs_from_samples.tensor.1x1xf64([[samples0]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>
                // CHECK-NOT: quantum.probs
                %5 = quantum.probs %4 : tensor<2xf64>

                // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][1]
                %6 = quantum.extract %2[1] : !quantum.reg -> !quantum.bit

                // CHECK: [[compbasis1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
                %7 = quantum.compbasis qubits %6 : !quantum.obs

                // CHECK: [[samples1:%.+]] = quantum.sample [[compbasis1]] : tensor<1x1xf64>
                // CHECK: [[res1:%.+]] = func.call @probs_from_samples.tensor.1x1xf64([[samples1]]) :
                // CHECK-SAME: (tensor<1x1xf64>) -> tensor<2xf64>
                // CHECK-NOT: quantum.probs
                %8 = quantum.probs %7 : tensor<2xf64>

                // CHECK: func.return [[res0]], [[res1]] : tensor<2xf64>, tensor<2xf64>
                func.return %5, %8 : tensor<2xf64>, tensor<2xf64>
            }
            // CHECK-LABEL: func.func public @probs_from_samples.tensor.1x1xf64
        }
        """

        pipeline = (MeasurementsFromSamplesPass(),)
        run_filecheck(program, pipeline)


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestMeasurementsFromSamplesIntegration:
    """Tests of the execution of simple workloads with the xDSL-based MeasurementsFromSamplesPass
    transform.
    """

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, mp, obs, expected_res",
        [
            # PauliZ observables
            (qml.I, qml.expval, qml.Z, 1.0),
            (qml.X, qml.expval, qml.Z, -1.0),
            (qml.I, qml.var, qml.Z, 0.0),
            (qml.X, qml.var, qml.Z, 0.0),
            # PauliX observables
            pytest.param(
                partial(qml.RY, phi=np.pi / 2),
                qml.expval,
                qml.X,
                1.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
            pytest.param(
                partial(qml.RY, phi=-np.pi / 2),
                qml.expval,
                qml.X,
                -1.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
            pytest.param(
                partial(qml.RY, phi=np.pi / 2),
                qml.var,
                qml.X,
                0.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
            pytest.param(
                partial(qml.RY, phi=-np.pi / 2),
                qml.var,
                qml.X,
                0.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
            # PauliY observables
            pytest.param(
                partial(qml.RX, phi=-np.pi / 2),
                qml.expval,
                qml.Y,
                1.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
            pytest.param(
                partial(qml.RX, phi=np.pi / 2),
                qml.expval,
                qml.Y,
                -1.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
            pytest.param(
                partial(qml.RX, phi=-np.pi / 2),
                qml.var,
                qml.Y,
                0.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
            pytest.param(
                partial(qml.RX, phi=np.pi / 2),
                qml.var,
                qml.Y,
                0.0,
                marks=pytest.mark.xfail(
                    reason="Only PauliZ-basis measurements supported",
                    strict=True,
                    raises=NotImplementedError,
                ),
            ),
        ],
    )
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_exec_1_wire_mp_with_obs(self, shots, initial_op, mp, obs, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements that require an observable (i.e. expval and var).
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return mp(obs(wires=0))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qml.I, [1.0, 0.0]),
            (qml.X, [0.0, 1.0]),
        ],
    )
    def test_exec_1_wire_probs(self, shots, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        probs measurements.
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.probs(wires=0)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Counts not supported in Catalyst with program capture",
        strict=True,
        raises=NotImplementedError,
    )
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qml.I, {"0": 10, "1": 0}),
            (qml.X, {"0": 0, "1": 10}),
        ],
    )
    def test_exec_1_wire_counts(self, shots, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        counts measurements.
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.counts(wires=0)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.array_equal(expected_res, _counts_catalyst_to_pl(*circuit_compiled()))

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res_base",
        [
            (qml.I, 0),
            (qml.X, 1),
        ],
    )
    def test_exec_1_wire_sample(self, shots, initial_op, expected_res_base):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        sample measurements.

        In this case, the measurements_from_samples pass should effectively be a no-op.
        """
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.sample(wires=0)

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        expected_res = expected_res_base * np.ones(shape=(shots, 1), dtype=int)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, mp, obs, expected_res",
        [
            ((qml.I, qml.I), qml.expval, qml.Z, (1.0, 1.0)),
            ((qml.I, qml.X), qml.expval, qml.Z, (1.0, -1.0)),
            ((qml.X, qml.I), qml.expval, qml.Z, (-1.0, 1.0)),
            ((qml.X, qml.X), qml.expval, qml.Z, (-1.0, -1.0)),
            ((qml.I, qml.I), qml.var, qml.Z, (0.0, 0.0)),
            ((qml.I, qml.X), qml.var, qml.Z, (0.0, 0.0)),
            ((qml.X, qml.I), qml.var, qml.Z, (0.0, 0.0)),
            ((qml.X, qml.X), qml.var, qml.Z, (0.0, 0.0)),
        ],
    )
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_exec_2_wire_with_obs_separate(self, shots, initial_ops, mp, obs, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed separately per wire.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return mp(obs(wires=0)), mp(obs(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Operator arithmetic not yet supported with capture enabled", strict=True
    )
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, mp, expected_res",
        [
            ((qml.I, qml.I), qml.expval, 1.0),
            ((qml.I, qml.X), qml.expval, -1.0),
            ((qml.X, qml.I), qml.expval, -1.0),
            ((qml.X, qml.X), qml.expval, 1.0),
            ((qml.I, qml.I), qml.var, 0.0),
            ((qml.I, qml.X), qml.var, 0.0),
            ((qml.X, qml.I), qml.var, 0.0),
            ((qml.X, qml.X), qml.var, 0.0),
        ],
    )
    def test_exec_2_wire_with_obs_combined(self, shots, initial_ops, mp, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return mp(qml.Z(wires=0) @ qml.Z(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), [1.0, 0.0, 0.0, 0.0]),
            ((qml.I, qml.X), [0.0, 1.0, 0.0, 0.0]),
            ((qml.X, qml.I), [0.0, 0.0, 1.0, 0.0]),
            ((qml.X, qml.X), [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_exec_2_wire_probs_global(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.probs()

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), ([1.0, 0.0], [1.0, 0.0])),
            ((qml.I, qml.X), ([1.0, 0.0], [0.0, 1.0])),
            ((qml.X, qml.I), ([0.0, 1.0], [1.0, 0.0])),
            ((qml.X, qml.X), ([0.0, 1.0], [0.0, 1.0])),
        ],
    )
    def test_exec_2_wire_probs_per_wire(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.probs(wires=0), qml.probs(wires=1)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(reason="Dynamic shots not supported")
    def test_exec_expval_dynamic_shots(self):
        """Test the measurements_from_samples transform where the number of shots is dynamic.

        This use case is not currently supported.
        """

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        def workload(shots):
            dev = qml.device("lightning.qubit", wires=1)

            @measurements_from_samples_pass
            @qml.qnode(dev, shots=shots)
            def circuit():
                return qml.expval(qml.Z(wires=0))

            return circuit()

        result = workload(2)
        assert result == 1.0

    def test_qjit_filecheck(self, run_filecheck_qjit):
        """Test that the measurements_from_samples_pass works correctly with qjit."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir", pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        @measurements_from_samples_pass
        @qml.qnode(dev, shots=25)
        def circuit():
            # CHECK-NOT: quantum.namedobs
            # CHECK: [[obs:%.+]] = quantum.compbasis
            # CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<25x1xf64>
            # CHECK: [[c0:%.+]] = arith.constant dense<0> : tensor<i64>
            # CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.25x1xf64([[samples]], [[c0]]) :
            # CHECK-SAME: (tensor<25x1xf64>, tensor<i64>) -> tensor<f64>
            # CHECK-NOT: quantum.expval
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)


def _counts_catalyst_to_pl(basis_states, counts):
    """Helper function to convert counts in the Catalyst format to the PennyLane format.

    Example:

    >>> basis_states, counts = ([0, 1], [6, 4])
    >>> _counts_catalyst_to_pl(basis_states, counts)
    {'0': 6, '1': 4}
    """
    return {format(int(state), "01b"): count for state, count in zip(basis_states, counts)}


if __name__ == "__main__":
    pytest.main(["-x", __file__])
