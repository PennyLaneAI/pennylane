# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the jacobian product calculator classes.
"""
import numpy as np

# pylint: disable=protected-access
import pytest
from cachetools import LRUCache
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.workflow.jacobian_products import (
    DeviceDerivatives,
    DeviceJacobianProducts,
    JacobianProductCalculator,
    NoGradients,
    TransformJacobianProducts,
)

dev = qml.device("default.qubit")
dev_lightning = qml.device("lightning.qubit", wires=5)
adjoint_config = qml.devices.ExecutionConfig(gradient_method="adjoint")
dev_ps = ParamShiftDerivativesDevice()
ps_config = qml.devices.ExecutionConfig(gradient_method="parameter-shift")


def inner_execute_numpy(tapes):
    return dev.execute(tapes)


param_shift_jpc = TransformJacobianProducts(inner_execute_numpy, qml.gradients.param_shift)
param_shift_cached_jpc = TransformJacobianProducts(
    inner_execute_numpy, qml.gradients.param_shift, cache_full_jacobian=True
)
hadamard_grad_jpc = TransformJacobianProducts(
    inner_execute_numpy, qml.gradients.hadamard_grad, {"aux_wire": "aux"}
)
device_jacs = DeviceDerivatives(dev, adjoint_config)
device_ps_jacs = DeviceDerivatives(dev_ps, ps_config)
device_native_jps = DeviceJacobianProducts(dev, adjoint_config)
device_ps_native_jps = DeviceJacobianProducts(dev_ps, ps_config)
lightning_vjps = DeviceJacobianProducts(dev_lightning, execution_config=adjoint_config)

transform_jpc_matrix = [param_shift_jpc, param_shift_cached_jpc, hadamard_grad_jpc]
dev_jpc_matrix = [device_jacs, device_ps_jacs]
jpc_matrix = [
    param_shift_jpc,
    param_shift_cached_jpc,
    hadamard_grad_jpc,
    device_jacs,
    device_ps_jacs,
    device_native_jps,
    device_ps_native_jps,
    lightning_vjps,
]


def _accepts_finite_shots(jpc):
    if isinstance(jpc, TransformJacobianProducts):
        return True
    if isinstance(jpc, (DeviceDerivatives, DeviceJacobianProducts)):
        return isinstance(jpc._device, ParamShiftDerivativesDevice)
    return False


def _tol_for_shots(shots):
    return 0.05 if shots else 1e-6


def test_no_gradients():
    """Test that errors are raised when derivatives are requested from `NoGradients`."""

    jpc = NoGradients()

    with pytest.raises(QuantumFunctionError, match="cannot be calculated with diff_method=None"):
        jpc.compute_jacobian(())

    with pytest.raises(QuantumFunctionError, match="cannot be calculated with diff_method=None"):
        jpc.compute_vjp((), ())

    with pytest.raises(QuantumFunctionError, match="cannot be calculated with diff_method=None"):
        jpc.execute_and_compute_jvp((), ())

    with pytest.raises(QuantumFunctionError, match="cannot be calculated with diff_method=None"):
        jpc.execute_and_compute_jacobian(())


# pylint: disable=too-few-public-methods
class TestBasics:
    """Test initialization and repr for jacobian product calculator classes."""

    def test_transform_jacobian_product_basics(self):
        """Test the initialization and basic properties of a TransformJacobianProduct class."""
        jpc = TransformJacobianProducts(
            inner_execute_numpy, qml.gradients.hadamard_grad, {"aux_wire": "aux"}
        )

        assert isinstance(jpc, JacobianProductCalculator)
        assert jpc._inner_execute is inner_execute_numpy
        assert jpc._gradient_transform is qml.gradients.hadamard_grad
        assert jpc._gradient_kwargs == {"aux_wire": "aux"}

        expected_repr = (
            f"TransformJacobianProducts({repr(inner_execute_numpy)}, "
            "gradient_transform=<transform: hadamard_grad>, "
            "gradient_kwargs={'aux_wire': 'aux'}, cache_full_jacobian=False)"
        )
        assert repr(jpc) == expected_repr

    def test_device_derivatives_initialization_without_config(self):
        """Test that not providing an execution config sets it to None."""
        device = qml.device("default.qubit")

        jpc = DeviceDerivatives(device)

        assert jpc._execution_config is None

    def test_device_jacobians_initialization_new_dev(self):
        """Tests the private attributes are set during initialization of a DeviceDerivatives class."""

        device = qml.device("default.qubit")
        config = qml.devices.ExecutionConfig(gradient_method="adjoint")

        jpc = DeviceDerivatives(device, config)

        assert jpc._device is device
        assert jpc._execution_config is config
        assert isinstance(jpc._results_cache, LRUCache)
        assert len(jpc._results_cache) == 0
        assert isinstance(jpc._jacs_cache, LRUCache)
        assert len(jpc._jacs_cache) == 0

    def test_device_jacobians_repr(self):
        """Test the repr method for device jacobians."""
        device = qml.device("default.qubit")
        config = qml.devices.ExecutionConfig(gradient_method="adjoint")

        jpc = DeviceDerivatives(device, config)

        expected = (
            r"<DeviceDerivatives: default.qubit,"
            r" ExecutionConfig(grad_on_execution=None, use_device_gradient=None,"
            r" use_device_jacobian_product=None,"
            r" gradient_method='adjoint', gradient_keyword_arguments={},"
            r" device_options={}, interface=<Interface.NUMPY: 'numpy'>, derivative_order=1,"
            r" mcm_config=MCMConfig(mcm_method=None, postselect_mode=None), convert_to_numpy=True,"
            r" executor_backend=<class 'pennylane.concurrency.executors.native.multiproc.MPPoolExec'>)>"
        )

        assert repr(jpc) == expected

    def test_device_jacobian_products_repr(self):
        """Test the repr method for device jacobian products."""

        device = qml.device("default.qubit")
        config = qml.devices.ExecutionConfig(gradient_method="adjoint")

        jpc = DeviceJacobianProducts(device, config)

        expected = (
            r"<DeviceJacobianProducts: default.qubit,"
            r" ExecutionConfig(grad_on_execution=None, use_device_gradient=None,"
            r" use_device_jacobian_product=None,"
            r" gradient_method='adjoint', gradient_keyword_arguments={}, device_options={},"
            r" interface=<Interface.NUMPY: 'numpy'>, derivative_order=1,"
            r" mcm_config=MCMConfig(mcm_method=None, postselect_mode=None), convert_to_numpy=True,"
            r" executor_backend=<class 'pennylane.concurrency.executors.native.multiproc.MPPoolExec'>)>"
        )

        assert repr(jpc) == expected


@pytest.mark.parametrize("jpc", jpc_matrix)
@pytest.mark.parametrize("shots", (None, 10000, (10000, 10000)))
class TestJacobianProductResults:
    """Test first order results for the matrix of jpc options."""

    def test_execute_jvp_basic(self, jpc, shots):
        """Test execute_and_compute_jvp for a simple single input single output."""
        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")
        if isinstance(jpc, DeviceJacobianProducts) and "lightning" in jpc._device.name:
            pytest.xfail("Lightning devices don't have JVP method")

        x = 0.92
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
        tangents = ((0.5,),)
        res, jvp = jpc.execute_and_compute_jvp((tape,), tangents)

        if tape.shots.has_partitioned_shots:
            assert len(res[0]) == 2
            assert len(jvp[0]) == 2
        else:
            assert qml.math.shape(res[0]) == tuple()
            assert qml.math.shape(jvp[0]) == tuple()

        assert qml.math.allclose(res[0], np.cos(x), atol=_tol_for_shots(shots))
        assert qml.math.allclose(jvp[0], -0.5 * np.sin(x), atol=_tol_for_shots(shots))

        if tape.shots.has_partitioned_shots:
            assert qml.math.allclose(res[0][1], np.cos(x), atol=_tol_for_shots(shots))
            assert qml.math.allclose(jvp[0][1], -0.5 * np.sin(x), atol=_tol_for_shots(shots))

    def test_vjp_basic(self, jpc, shots, seed):
        """Test compute_vjp for a simple single input single output."""

        np.random.seed(seed)

        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")

        x = -0.294
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)

        dy = ((1.1, 0.7),) if tape.shots.has_partitioned_shots else (1.8,)
        vjp = jpc.compute_vjp((tape,), dy)

        assert qml.math.allclose(vjp[0], -1.8 * np.sin(x), atol=_tol_for_shots(shots))

    def test_jacobian_basic(self, jpc, shots):
        """Test compute_jacobian for a simple single input single output."""
        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")

        x = 1.62
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
        jac = jpc.compute_jacobian((tape,))

        assert qml.math.allclose(jac, -np.sin(x), atol=_tol_for_shots(shots))

    def test_execute_jacobian_basic(self, jpc, shots):
        """Test execute_and_compute_jacobian for a simple single input single output."""
        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")

        x = 1.62
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
        results, jacs = jpc.execute_and_compute_jacobian((tape,))
        assert qml.math.allclose(results[0], np.cos(x), atol=_tol_for_shots(shots))
        assert qml.math.allclose(jacs, -np.sin(x), atol=_tol_for_shots(shots))

    def test_batch_execute_jvp(self, jpc, shots):
        """Test execute_and_compute_jvp on a batch with ragged observables and parameters.."""
        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")
        if isinstance(jpc, DeviceJacobianProducts) and "lightning" in jpc._device.name:
            pytest.skip("Lightning devices don't have JVP method")
        x = -0.92
        y = 0.84
        phi = 1.62

        tape1 = qml.tape.QuantumScript(
            [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(0))],
        )
        tape2 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))],
            [qml.expval(qml.PauliZ(1))],
            shots=shots,
        )

        tangents = ((2.0, 3.0), (0.5,))

        res, jvps = jpc.execute_and_compute_jvp((tape1, tape2), tangents)

        assert qml.math.allclose(res[0][0], np.sin(y), atol=_tol_for_shots(shots))
        assert qml.math.allclose(res[0][1], -np.sin(x) * np.sin(y), atol=_tol_for_shots(shots))
        assert qml.math.allclose(res[1], np.cos(phi), atol=_tol_for_shots(shots))

        assert qml.math.allclose(jvps[0][0], 3.0 * np.cos(y), atol=_tol_for_shots(shots))
        assert qml.math.allclose(
            jvps[0][1],
            -2.0 * np.cos(x) * np.sin(y) - 3.0 * np.sin(x) * np.cos(y),
            atol=_tol_for_shots(shots),
        )
        assert qml.math.allclose(jvps[1], -0.5 * np.sin(phi), atol=_tol_for_shots(shots))

    def test_batch_vjp(self, jpc, shots):
        """Test compute_vjp on a batch with ragged observables and parameters."""

        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")
        if jpc is hadamard_grad_jpc and qml.measurements.Shots(shots).has_partitioned_shots:
            pytest.skip(
                "hadamard gradient does not support multiple measurements with partitioned shots."
            )

        x = 0.385
        y = 1.92
        phi = -1.05

        tape1 = qml.tape.QuantumScript(
            [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(0))],
            shots=shots,
        )
        tape2 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))],
            [qml.expval(qml.PauliZ(1))],
            shots=shots,
        )

        if tape1.shots.has_partitioned_shots:
            dy1 = ((0.3, 0.2), (0.2, 0.4))
            dy2 = (0.4, 0.5)
            dy = (dy1, dy2)
        else:
            dy = ((0.5, 0.6), 0.9)

        vjps = jpc.compute_vjp((tape1, tape2), dy)

        assert qml.math.allclose(
            vjps[0][0], -0.6 * np.cos(x) * np.sin(y), atol=_tol_for_shots(shots)
        )  # dx
        assert qml.math.allclose(
            vjps[0][1], 0.5 * np.cos(y) - 0.6 * np.sin(x) * np.cos(y), atol=_tol_for_shots(shots)
        )  # dy
        assert qml.math.allclose(vjps[1], -0.9 * np.sin(phi), atol=_tol_for_shots(shots))

    def test_batch_jacobian(self, jpc, shots):
        """Test compute_jacobian on a batch with ragged observables and parameters."""

        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")
        if jpc is hadamard_grad_jpc and qml.measurements.Shots(shots).has_partitioned_shots:
            pytest.skip(
                "hadamard gradient does not work with partitioned shots and multiple measurements."
            )

        x = np.array(0.28)
        y = np.array(1.62)
        phi = np.array(0.6293)

        tape1 = qml.tape.QuantumScript(
            [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(0))],
            shots=shots,
        )
        tape2 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))],
            [qml.expval(qml.PauliZ(1))],
            shots=shots,
        )

        # note reversed order of tapes in this test
        jacs = jpc.compute_jacobian((tape2, tape1))

        if tape1.shots.has_partitioned_shots:
            for i in [0, 1]:
                assert qml.math.allclose(jacs[0][i], -np.sin(phi), atol=_tol_for_shots(shots))
                assert qml.math.allclose(jacs[1][i][0][0], 0, atol=_tol_for_shots(shots))
                assert qml.math.allclose(jacs[1][i][0][1], np.cos(y), atol=_tol_for_shots(shots))
                assert qml.math.allclose(
                    jacs[1][i][1][0], -np.cos(x) * np.sin(y), atol=_tol_for_shots(shots)
                )
                assert qml.math.allclose(
                    jacs[1][i][1][1], -np.sin(x) * np.cos(y), atol=_tol_for_shots(shots)
                )
        else:
            assert qml.math.allclose(jacs[0], -np.sin(phi), atol=_tol_for_shots(shots))
            assert qml.math.allclose(jacs[1][0][0], 0, atol=_tol_for_shots(shots))
            assert qml.math.allclose(jacs[1][0][1], np.cos(y), atol=_tol_for_shots(shots))
            assert qml.math.allclose(
                jacs[1][1][0], -np.cos(x) * np.sin(y), atol=_tol_for_shots(shots)
            )
            assert qml.math.allclose(
                jacs[1][1][1], -np.sin(x) * np.cos(y), atol=_tol_for_shots(shots)
            )

    def test_batch_execute_jacobian(self, jpc, shots):
        """Test execute_and_compute_jacobian on a batch with ragged observables and parameters."""

        if shots and not _accepts_finite_shots(jpc):
            pytest.skip("jpc does not work with finite shots.")
        if jpc is hadamard_grad_jpc and qml.measurements.Shots(shots).has_partitioned_shots:
            pytest.skip(
                "hadamard gradient does not work with partitioned shots and multiple measurements."
            )

        x = np.array(0.28)
        y = np.array(1.62)
        phi = np.array(0.6293)

        tape1 = qml.tape.QuantumScript(
            [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(0))],
            shots=shots,
        )
        tape2 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))],
            [qml.expval(qml.PauliZ(1))],
            shots=shots,
        )

        # note reversed order of tapes in this test
        res, jacs = jpc.execute_and_compute_jacobian((tape2, tape1))

        if tape1.shots.has_partitioned_shots:
            for i in [0, 1]:
                assert qml.math.allclose(res[1][i][0], np.sin(y), atol=_tol_for_shots(shots))
                assert qml.math.allclose(
                    res[1][i][1], -np.sin(x) * np.sin(y), atol=_tol_for_shots(shots)
                )
                assert qml.math.allclose(res[0][i], np.cos(phi), atol=_tol_for_shots(shots))

                assert qml.math.allclose(jacs[1][i][0][0], 0, atol=_tol_for_shots(shots))
                assert qml.math.allclose(jacs[1][i][0][1], np.cos(y), atol=_tol_for_shots(shots))
                assert qml.math.allclose(
                    jacs[1][i][1][0], -np.cos(x) * np.sin(y), atol=_tol_for_shots(shots)
                )
                assert qml.math.allclose(
                    jacs[1][i][1][1], -np.sin(x) * np.cos(y), atol=_tol_for_shots(shots)
                )
                assert qml.math.allclose(jacs[0][i], -np.sin(phi), atol=_tol_for_shots(shots))
        else:
            assert qml.math.allclose(res[1][0], np.sin(y), atol=_tol_for_shots(shots))
            assert qml.math.allclose(res[1][1], -np.sin(x) * np.sin(y), atol=_tol_for_shots(shots))
            assert qml.math.allclose(res[0], np.cos(phi), atol=_tol_for_shots(shots))

            assert qml.math.allclose(jacs[0], -np.sin(phi), atol=_tol_for_shots(shots))
            assert qml.math.allclose(jacs[1][0][0], 0, atol=_tol_for_shots(shots))
            assert qml.math.allclose(jacs[1][0][1], np.cos(y), atol=_tol_for_shots(shots))
            assert qml.math.allclose(
                jacs[1][1][0], -np.cos(x) * np.sin(y), atol=_tol_for_shots(shots)
            )
            assert qml.math.allclose(
                jacs[1][1][1], -np.sin(x) * np.cos(y), atol=_tol_for_shots(shots)
            )


@pytest.mark.parametrize("jpc", dev_jpc_matrix)
class TestCachingDeviceDerivatives:
    """Test caching for device jacobians."""

    def test_execution_caching(self, jpc):
        """Test that results and jacobians are cached on calls to execute."""
        tape1 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        batch = (tape1,)

        with jpc._device.tracker:
            results = jpc.execute_and_cache_jacobian(batch)

        assert qml.math.allclose(results[0], np.cos(0.1))
        assert jpc._device.tracker.totals["execute_and_derivative_batches"] == 1
        assert jpc._device.tracker.totals["derivatives"] == 1

        if isinstance(jpc._device, ParamShiftDerivativesDevice):
            # extra execution since needs to do the forward pass again.
            expected_execs = 3
        elif isinstance(jpc._device, qml.devices.LegacyDevice):
            expected_execs = 2
        else:
            expected_execs = 1

        assert jpc._device.tracker.totals["executions"] == expected_execs

        # Test reuse with jacobian
        with jpc._device.tracker:
            jac = jpc.compute_jacobian(batch)

        assert qml.math.allclose(jac, -np.sin(0.1))
        assert jpc._device.tracker.totals.get("derivatives", 0) == 0
        assert jpc._device.tracker.totals.get("executions", 0) == 0

        # Test reuse with execute_and_compute_jvp
        with jpc._device.tracker:
            res2, jvp = jpc.execute_and_compute_jvp(batch, ((0.5,),))

        assert qml.math.allclose(res2, results)
        assert qml.math.allclose(jvp, 0.5 * -np.sin(0.1))
        assert jpc._device.tracker.totals.get("derivatives", 0) == 0
        assert jpc._device.tracker.totals.get("executions", 0) == 0

        # Test reuse with compute_vjp
        with jpc._device.tracker:
            vjp = jpc.compute_vjp(batch, ((1.5,),))

        assert qml.math.allclose(vjp, -1.5 * np.sin(0.1))
        assert jpc._device.tracker.totals.get("derivatives", 0) == 0
        assert jpc._device.tracker.totals.get("executions", 0) == 0

        # Test device called again if batch a new instance, even if identical
        tape2 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        batch2 = (tape2,)

        with jpc._device.tracker:
            jac2 = jpc.compute_jacobian(batch2)

        assert qml.math.allclose(jac, jac2)
        assert jpc._device.tracker.totals["derivatives"] == 1

        if isinstance(jpc._device, ParamShiftDerivativesDevice):
            # extra execution since needs to do the forward pass again.
            expected_execs = 2
        elif isinstance(jpc._device, qml.devices.LegacyDevice):
            expected_execs = 1
        else:
            expected_execs = 0

        assert jpc._device.tracker.totals.get("executions", 0) == expected_execs

    def test_cached_on_execute_and_compute_jvps(self, jpc):
        """Test that execute_and_compute_jvp caches results and Jacobians if they are not precalculated."""
        tape1 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(0.8, wires=(0, 1))], [qml.expval(qml.PauliZ(1))]
        )
        batch = (tape1,)
        tangents = ((0.5,),)

        with jpc._device.tracker:
            res, jvps = jpc.execute_and_compute_jvp(batch, tangents)

        assert jpc._device.tracker.totals["execute_and_derivative_batches"] == 1

        assert qml.math.allclose(res, np.cos(0.8))
        assert qml.math.allclose(jvps, -0.5 * np.sin(0.8))

        assert jpc._results_cache[batch] is res
        assert qml.math.allclose(jpc._jacs_cache[batch], (-np.sin(0.8)))

        with jpc._device.tracker:
            jpc.execute_and_compute_jvp(batch, tangents)

        assert jpc._device.tracker.totals.get("derivatives", 0) == 0
        assert jpc._device.tracker.totals.get("executions", 0) == 0

    def test_cached_on_execute_and_compute_jacobian(self, jpc):
        """Test that execute_and_compute_jacobians caches results and Jacobians if they are not precalculated."""
        x = 1.5
        tape1 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(x, wires=(0, 1))], [qml.expval(qml.PauliZ(1))]
        )
        batch = (tape1,)

        with jpc._device.tracker:
            res, jacs = jpc.execute_and_compute_jacobian(batch)

        assert jpc._device.tracker.totals["execute_and_derivative_batches"] == 1

        assert qml.math.allclose(res, np.cos(x))
        assert qml.math.allclose(jacs, -np.sin(x))

        assert jpc._results_cache[batch] is res
        assert qml.math.allclose(jpc._jacs_cache[batch], (-np.sin(x)))

        with jpc._device.tracker:
            jpc.execute_and_compute_jacobian(batch)

        assert jpc._device.tracker.totals.get("derivatives", 0) == 0
        assert jpc._device.tracker.totals.get("executions", 0) == 0

    def test_cached_on_vjps(self, jpc):
        """test that only jacs are cached on calls to compute_vjp."""

        tape1 = qml.tape.QuantumScript([qml.RZ(0.5, wires=0)], [qml.expval(qml.PauliX(0))])
        batch = (tape1,)
        dy = ((0.5,),)

        with jpc._device.tracker:
            jpc.compute_vjp(batch, dy)

        if isinstance(jpc._device, ParamShiftDerivativesDevice):
            expected = 2
        elif isinstance(jpc._device, qml.devices.LegacyDevice):
            expected = 1
        else:
            expected = 0

        assert jpc._device.tracker.totals.get("executions", 0) == expected

        assert batch not in jpc._results_cache
        assert qml.math.allclose(jpc._jacs_cache[batch], 0)

        with jpc._device.tracker:
            jpc.execute_and_compute_jvp(batch, ((0.5,),))

        assert jpc._device.tracker.totals["executions"] == 1
        assert jpc._device.tracker.totals.get("derivatives", 0) == 0
        assert qml.math.allclose(jpc._results_cache[batch], 0)

    def test_error_cant_cache_results_without_jac(self, jpc):
        """Test that a NotImplementedError is raised if somehow the results are cached
        without the jac being cached and execute_and_compute_jvp is called."""

        tape = qml.tape.QuantumScript([], [qml.state()])
        batch = (tape,)
        jpc._results_cache[batch] = "value"

        with pytest.raises(NotImplementedError):
            jpc.execute_and_compute_jacobian(batch)


@pytest.mark.parametrize("jpc", transform_jpc_matrix + [device_ps_jacs])
class TestProbsTransformJacobians:
    """Testing results when probabilities are returned. This only works with gradient transforms."""

    def test_execute_jvp_multi_params_multi_out(self, jpc):
        """Test execute_and_compute_jvp with multiple parameters and multiple outputs"""
        x = 0.62
        y = 2.64
        ops = [qml.RY(y, 0), qml.RX(x, 0)]
        measurements = [qml.probs(wires=0), qml.expval(qml.PauliZ(0))]
        tape1 = qml.tape.QuantumScript(ops, measurements)

        phi = 0.623
        ops2 = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        measurements2 = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tape2 = qml.tape.QuantumScript(ops2, measurements2)

        tangents = (1.5, 2.5)
        tangents2 = (0.6,)
        res, jvp = jpc.execute_and_compute_jvp((tape1, tape2), (tangents, tangents2))

        expected_res00 = 0.5 * np.array([1 + np.cos(x) * np.cos(y), 1 - np.cos(x) * np.cos(y)])
        assert qml.math.allclose(res[0][0], expected_res00)

        expected_res01 = np.cos(x) * np.cos(y)
        assert qml.math.allclose(res[0][1], expected_res01)

        assert qml.math.allclose(res[1][0], 0)
        assert qml.math.allclose(res[1][1], np.cos(phi))

        res0dx = 0.5 * np.array([-np.sin(x) * np.cos(y), np.sin(x) * np.cos(y)])
        res0dy = 0.5 * np.array([-np.cos(x) * np.sin(y), np.cos(x) * np.sin(y)])
        expected_jvp00 = 2.5 * res0dx + 1.5 * res0dy
        assert qml.math.allclose(expected_jvp00, jvp[0][0])

        expected_jvp01 = -2.5 * np.sin(x) * np.cos(y) - 1.5 * np.cos(x) * np.sin(y)
        assert qml.math.allclose(expected_jvp01, jvp[0][1])

        assert qml.math.allclose(jvp[1][0], 0)
        assert qml.math.allclose(jvp[1][1], -0.6 * np.sin(phi))

    def test_execute_jacobian_multi_params_multi_out(self, jpc):
        """Test execute_and_compute_jacobian with multiple parameters and multiple outputs"""
        x = 0.93
        y = -0.83
        ops = [qml.RY(y, 0), qml.RX(x, 0)]
        measurements = [qml.probs(wires=0), qml.expval(qml.PauliZ(0))]
        tape1 = qml.tape.QuantumScript(ops, measurements)

        phi = 0.545
        ops2 = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        measurements2 = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tape2 = qml.tape.QuantumScript(ops2, measurements2)

        res, jac = jpc.execute_and_compute_jacobian((tape1, tape2))

        expected_res00 = 0.5 * np.array([1 + np.cos(x) * np.cos(y), 1 - np.cos(x) * np.cos(y)])
        assert qml.math.allclose(res[0][0], expected_res00)

        expected_res01 = np.cos(x) * np.cos(y)
        assert qml.math.allclose(res[0][1], expected_res01)

        assert qml.math.allclose(res[1][0], 0)
        assert qml.math.allclose(res[1][1], np.cos(phi))

        # first tape, first measurement, first parameters (y)
        expected = 0.5 * np.array([-np.cos(x) * np.sin(y), np.cos(x) * np.sin(y)])
        assert qml.math.allclose(jac[0][0][0], expected)

        # first tape, first measurement, second parameter (x)
        expected = 0.5 * np.array([-np.sin(x) * np.cos(y), np.sin(x) * np.cos(y)])
        assert qml.math.allclose(jac[0][0][1], expected)

        # first tape, second measurement, first parameter(y)
        expected = -np.cos(x) * np.sin(y)
        assert qml.math.allclose(jac[0][1][0], expected)
        # first tape, second measurement, second parameter (x)
        expected = -np.sin(x) * np.cos(y)
        assert qml.math.allclose(jac[0][1][1], expected)

        # second tape, first measurement, only parameter
        assert qml.math.allclose(jac[1][0], 0)
        # second tape, second measurement, only parameter
        assert qml.math.allclose(jac[1][1], -np.sin(phi))

    def test_vjp_multi_params_multi_out(self, jpc):
        """Test compute_vjp with multiple parameters and multiple outputs."""

        x = 0.62
        y = 2.64
        ops = [qml.RY(y, 0), qml.RX(x, 0)]
        measurements = [qml.probs(wires=0), qml.expval(qml.PauliZ(0))]
        tape1 = qml.tape.QuantumScript(ops, measurements)

        phi = 0.623
        ops2 = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        measurements2 = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tape2 = qml.tape.QuantumScript(ops2, measurements2)

        dy = (np.array([0.25, 0.5]), 1.5)
        dy2 = (0.7, 0.8)
        vjps = jpc.compute_vjp((tape1, tape2), (dy, dy2))

        dy = (
            0.5 * 0.25 * np.cos(x) * -np.sin(y)
            + 0.5 * 0.5 * np.cos(x) * np.sin(y)
            + 1.5 * np.cos(x) * -np.sin(y)
        )
        assert qml.math.allclose(vjps[0][0], dy)

        dx = (
            0.5 * 0.25 * -np.sin(x) * np.cos(y)
            + 0.5 * 0.5 * np.sin(x) * np.cos(y)
            + 1.5 * -np.sin(x) * np.cos(y)
        )
        assert qml.math.allclose(vjps[0][1], dx)

        assert qml.math.allclose(vjps[1], -0.8 * np.sin(phi))

    def test_jac_multi_params_multi_out(self, jpc):
        """Test compute_jacobian with multiple parameters and multiple measurements."""

        x = 0.62
        y = 2.64
        ops = [qml.RY(y, 0), qml.RX(x, 0)]
        measurements = [qml.probs(wires=0), qml.expval(qml.PauliZ(0))]
        tape1 = qml.tape.QuantumScript(ops, measurements)

        phi = 0.623
        ops2 = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        measurements2 = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tape2 = qml.tape.QuantumScript(ops2, measurements2)

        jac = jpc.compute_jacobian((tape1, tape2))

        # first tape, first measurement, first parameters (y)
        expected = 0.5 * np.array([-np.cos(x) * np.sin(y), np.cos(x) * np.sin(y)])
        assert qml.math.allclose(jac[0][0][0], expected)

        # first tape, first measurement, second parameter (x)
        expected = 0.5 * np.array([-np.sin(x) * np.cos(y), np.sin(x) * np.cos(y)])
        assert qml.math.allclose(jac[0][0][1], expected)

        # first tape, second measurement, first parameter(y)
        expected = -np.cos(x) * np.sin(y)
        assert qml.math.allclose(jac[0][1][0], expected)
        # first tape, second measurement, second parameter (x)
        expected = -np.sin(x) * np.cos(y)
        assert qml.math.allclose(jac[0][1][1], expected)

        # second tape, first measurement, only parameter
        assert qml.math.allclose(jac[1][0], 0)
        # second tape, second measurement, only parameter
        assert qml.math.allclose(jac[1][1], -np.sin(phi))


class TestTransformsDifferentiability:
    """Tests that the transforms are differentiable if the inner execution is differentiable.

    Note that testing is only done for the required method for each ml framework.
    """

    @pytest.mark.jax
    def test_execute_jvp_jax(self):
        """Test that execute_and_compute_jvp is jittable and differentiable with jax."""
        import jax

        jpc = param_shift_jpc

        def f(x, tangents):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            return jpc.execute_and_compute_jvp((tape,), tangents)[1][0]

        x = jax.numpy.array(0.1)
        tangents = ((jax.numpy.array(0.5),),)

        res = f(x, tangents=tangents)
        assert qml.math.allclose(res, -tangents[0][0] * np.sin(x))

        jit_res = jax.jit(f)(x, tangents=tangents)
        assert qml.math.allclose(jit_res, -tangents[0][0] * np.sin(x))

        grad = jax.grad(f)(x, tangents=tangents)
        assert qml.math.allclose(grad, -tangents[0][0] * np.cos(x))

        tangent_grad = jax.grad(f, argnums=1)(x, tangents)
        assert qml.math.allclose(tangent_grad[0][0], -np.sin(x))

    @pytest.mark.autograd
    def test_vjp_autograd(self):
        """Test that the derivative of compute_vjp can be taken with autograd."""

        jpc = param_shift_jpc

        def f(x, dy):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            vjp = jpc.compute_vjp((tape,), (dy,))
            return vjp[0]

        x = qml.numpy.array(0.1)
        dy = qml.numpy.array(2.0)

        res = f(x, dy)
        assert qml.math.allclose(res, -dy * np.sin(x))

        dx, ddy = qml.grad(f)(x, dy)
        assert qml.math.allclose(dx, -dy * np.cos(x))
        assert qml.math.allclose(ddy, -np.sin(x))

    @pytest.mark.torch
    def test_vjp_torch(self):
        """Test that the derivative of compute_vjp can be taken with torch."""

        import torch

        jpc = param_shift_jpc

        def f(x, dy):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            vjp = jpc.compute_vjp((tape,), (dy,))
            return vjp[0]

        x = torch.tensor(0.1, requires_grad=True)
        dy = torch.tensor(2.0, requires_grad=True)

        res = f(x, dy)
        assert qml.math.allclose(res, -2.0 * np.sin(0.1))

        res.backward()
        assert qml.math.allclose(x.grad, -2.0 * np.cos(0.1))
        assert qml.math.allclose(dy.grad, -np.sin(0.1))

    @pytest.mark.tf
    def test_vjp_tf(self):
        """Test that the derivatives of compute_vjp can be taken with tensorflow."""

        import tensorflow as tf

        jpc = param_shift_jpc

        def f(x, dy):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            vjp = jpc.compute_vjp((tape,), (dy,))
            return vjp[0]

        x = tf.Variable(0.6, dtype=tf.float64)
        dy = tf.Variable(1.5, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = f(x, dy)

        assert qml.math.allclose(res, -1.5 * np.sin(0.6))

        dx, ddy = tape.gradient(res, (x, dy))

        assert qml.math.allclose(dx, -1.5 * np.cos(0.6))
        assert qml.math.allclose(ddy, -np.sin(0.6))
