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

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position

from xdsl.dialects import arith, builtin, func, tensor

import pennylane as qml
from pennylane.compiler.python_compiler import quantum_dialect as quantum
from pennylane.compiler.python_compiler.transforms import (
    MeasurementsFromSamplesPass,
    measurements_from_samples_pass,
)

catalyst = pytest.importorskip("catalyst")
from catalyst.passes import xdsl_plugin


class TestMeasurementsFromSamplesPass:
    """Unit tests for the measurements-from-samples pass."""

    def test_1_wire_expval(run_filecheck):
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
                // CHECK: [[res:%.+]] = func.call @expval_from_samples.tensor.1x1xf64([[samples]], [[c0]]) : (tensor<1x1xf64>, tensor<i64>) -> tensor<f64>
                // CHECK-NOT: quantum.expval
                %4 = quantum.expval %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>
                // CHECK: func.return [[res]] : tensor<f64>
                func.return %5 : tensor<f64>
            }
            // CHECK-LABEL: func.func public @expval_from_samples.tensor.1x1xf64
        }
        """
        ctx = xdsl.context.Context(allow_unregistered=True)
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(tensor.Tensor)
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(quantum.QuantumDialect)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MeasurementsFromSamplesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)


class TestMeasurementsFromSamplesExecution:
    """Tests of the execution of simple workloads with the xDSL-based MeasurementsFromSamplesPass
    transform.
    """

    @pytest.mark.usefixtures("enable_disable_plxpr")
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
    def test_1_wire_mp_with_obs(self, shots, initial_op, mp, obs, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements that require an observable (i.e. expval and var).
        """

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit_ref():
            initial_op(wires=0)
            return mp(obs(wires=0))

        assert expected_res == circuit_ref(), f"Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qml.I, [1.0, 0.0]),
            (qml.X, [0.0, 1.0]),
        ],
    )
    def test_1_wire_probs(self, shots, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        probs measurements.
        """

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit_ref():
            initial_op(wires=0)
            return qml.probs(wires=0)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), f"Sanity check failed, is expected_res correct?"

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
    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qml.I, {"0": 10, "1": 0}),
            (qml.X, {"0": 0, "1": 10}),
        ],
    )
    def test_1_wire_counts(self, shots, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        counts measurements.
        """

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit_ref():
            initial_op(wires=0)
            return qml.counts(wires=0)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), f"Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.array_equal(expected_res, _counts_catalyst_to_pl(circuit_compiled()))

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res_base",
        [
            (qml.I, 0),
            (qml.X, 1),
        ],
    )
    def test_1_wire_sample(self, shots, initial_op, expected_res_base):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        sample measurements.

        In this case, the measurements_from_samples pass should effectively be a no-op.
        """

        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit_ref():
            initial_op(wires=0)
            return qml.sample(wires=0)

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        expected_res = expected_res_base * jnp.ones(shape=(shots, 1), dtype=int)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.usefixtures("enable_disable_plxpr")
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
    def test_2_wires_with_obs_separate(self, shots, initial_ops, mp, obs, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed separately per wire.
        """

        dev = qml.device("lightning.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return mp(obs(wires=0)), mp(obs(wires=1))

        assert expected_res == circuit_ref(), f"Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Operator arithmetic not yet supported with capture enabled", strict=True
    )
    @pytest.mark.usefixtures("enable_disable_plxpr")
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
    def test_2_wires_with_obs_combined(self, shots, initial_ops, mp, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return mp(qml.Z(wires=0) @ qml.Z(wires=1))

        assert expected_res == circuit_ref(), f"Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
            pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()],
        )

        assert np.array_equal(expected_res, _counts_catalyst_to_pl(circuit_compiled()))

        assert expected_res == circuit_compiled()

    @pytest.mark.xfail(reason="Dynamic shots not supported")
    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_expval_dynamic_shots(self):

        @qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
        def workload(shots):
            dev = qml.device("lightning.qubit", wires=1, shots=shots)

            @measurements_from_samples_pass
            @qml.qnode(dev)
            def circuit():
                return qml.expval(qml.Z(wires=0))

            return circuit()

        result = workload(2)
        assert result == 1.0


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
