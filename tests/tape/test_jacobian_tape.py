# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the JacobianTape"""
import pytest
import numpy as np

import pennylane as qml
from pennylane.tape import JacobianTape, QuantumTape
from pennylane.devices import DefaultQubit
from pennylane.operation import Observable
from pennylane.operation import AnyWires


class TestConstruction:
    """Test for queuing and construction"""

    @pytest.fixture
    def make_tape(self):
        ops = []
        obs = []

        with JacobianTape() as tape:
            ops += [qml.RX(0.432, wires=0)]
            ops += [qml.Rot(0.543, 0, 0.23, wires=0)]
            ops += [qml.CNOT(wires=[0, "a"])]
            ops += [qml.RX(0.133, wires=4)]
            obs += [qml.PauliX(wires="a")]
            qml.expval(obs[0])
            obs += [qml.probs(wires=[0, "a"])]

        return tape, ops, obs

    def test_parameter_info(self, make_tape):
        """Test that parameter information is correctly extracted"""
        tape, ops, obs = make_tape
        tape._update_gradient_info()
        assert tape._trainable_params == set(range(5))
        assert tape._par_info == {
            0: {"op": ops[0], "p_idx": 0, "grad_method": "F"},
            1: {"op": ops[1], "p_idx": 0, "grad_method": "F"},
            2: {"op": ops[1], "p_idx": 1, "grad_method": "F"},
            3: {"op": ops[1], "p_idx": 2, "grad_method": "F"},
            4: {"op": ops[3], "p_idx": 0, "grad_method": "0"},
        }


class TestGradMethod:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        assert tape._grad_method(0) is None
        assert tape._grad_method(1) == "F"
        assert tape._grad_method(2) == "F"

        tape._update_gradient_info()

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "F"
        assert tape._par_info[2]["grad_method"] == "F"

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliY(0))

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "0"

        tape._update_gradient_info()

        assert tape._par_info[0]["grad_method"] == "F"
        assert tape._par_info[1]["grad_method"] == "0"

        # in non-graph mode, it is impossible to determine
        # if a parameter is independent or not
        tape._graph = None
        assert tape._grad_method(1, use_graph=False) == "F"


class TestJacobian:
    """Unit tests for the jacobian method"""

    def test_unknown_grad_method_error(self):
        """Test error raised if gradient method is unknown"""
        tape = JacobianTape()
        with pytest.raises(ValueError, match="Unknown gradient method"):
            tape.jacobian(None, method="unknown method")

    def test_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a non-differentiable argument"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        # by default all parameters are assumed to be trainable
        with pytest.raises(
            ValueError, match=r"Cannot differentiate with respect to parameter\(s\) {0}"
        ):
            tape.jacobian(None)

        # setting trainable parameters avoids this
        tape.trainable_params = {1, 2}
        dev = qml.device("default.qubit", wires=2)
        res = tape.jacobian(dev)
        assert res.shape == (4, 2)

    def test_analytic_method_with_unsupported_params(self):
        """Test that an exception is raised if method="A" but a parameter
        only support finite differences"""
        with JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliY(0))

        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(ValueError, match=r"analytic gradient method cannot be used"):
            tape.jacobian(dev, method="analytic")

    def test_analytic_method(self, mocker):
        """Test that calling the Jacobian with method=analytic correctly
        calls the analytic_pd method"""
        mock = mocker.patch("pennylane.tape.JacobianTape._grad_method")
        mock.return_value = "A"

        with JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliY(0))

        dev = qml.device("default.qubit", wires=1)
        tape.analytic_pd = mocker.Mock()
        tape.analytic_pd.return_value = [[QuantumTape()], lambda res: np.array([1.0])]

        tape.jacobian(dev, method="analytic")
        assert len(tape.analytic_pd.call_args_list) == 2

    def test_device_method(self, mocker):
        """Test that calling the Jacobian with method=device correctly
        calls the device_pd method"""
        with JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliY(0))

        dev = qml.device("default.qubit", wires=1)

        dev.jacobian = mocker.Mock()
        tape.device_pd(dev)
        dev.jacobian.assert_called_once()

        dev.jacobian = mocker.Mock()
        tape.jacobian(dev, method="device")
        dev.jacobian.assert_called_once()

    def test_no_output_execute(self):
        """Test that tapes with no measurement process return
        an empty list."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])

        res = tape.jacobian(dev)
        assert res.size == 0

    def test_incorrect_inferred_output_dim(self):
        """Test that a quantum tape with an incorrect inferred output dimension
        corrects itself when computing the Jacobian."""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1])

        # inferred output dim should be correct
        assert tape.output_dim == sum([2, 2])

        # modify the output dim
        tape._output_dim = 2

        res = tape.jacobian(dev, order=2, method="numeric")

        # output dim should be correct
        assert tape.output_dim == sum([2, 2])
        assert res.shape == (4, 3)

    def test_incorrect_ragged_output_dim(self, mocker):
        """Test that a quantum tape with an incorrect inferred *ragged* output dimension
        corrects itself after evaluation."""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        # inferred output dim should be correct
        assert tape.output_dim == sum([2, 4])

        # modify the output dim
        tape._output_dim = 2

        res = tape.jacobian(dev, order=2, method="numeric")

        # output dim should be correct
        assert tape.output_dim == sum([2, 4])
        assert res.shape == (6, 3)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        numeric_spy = mocker.spy(JacobianTape, "numeric_pd")
        analytic_spy = mocker.spy(JacobianTape, "analytic_pd")

        with JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        res = tape.jacobian(dev)
        assert res.shape == (1, 2)

        # the numeric pd method is only called once
        assert len(numeric_spy.call_args_list) == 1

        # analytic pd should not be called at all
        assert len(analytic_spy.call_args_list) == 0

        # the numeric pd method is only called for parameter 0
        assert numeric_spy.call_args[0] == (tape, 0)

    def test_no_trainable_parameters(self, mocker):
        """Test that if the tape has no trainable parameters, no
        subroutines are called and the returned Jacobian is empty"""
        numeric_spy = mocker.spy(JacobianTape, "numeric_pd")
        analytic_spy = mocker.spy(JacobianTape, "analytic_pd")

        with JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        tape.trainable_params = {}

        res = tape.jacobian(dev)
        assert res.size == 0
        assert np.all(res == np.array([[]]))

        numeric_spy.assert_not_called()
        analytic_spy.assert_not_called()

    def test_y0(self, mocker):
        """Test that if first order finite differences is used, then
        the tape is executed only once using the current parameter
        values."""
        dev = qml.device("default.qubit", wires=2)
        execute_spy = mocker.spy(dev, "execute")
        numeric_spy = mocker.spy(JacobianTape, "numeric_pd")

        with JacobianTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            qml.expval(qml.PauliZ(0))

        res = tape.jacobian(dev, order=1)

        # the execute device method is called once per parameter,
        # plus one global call
        assert len(execute_spy.call_args_list) == tape.num_params + 1
        assert "y0" in numeric_spy.call_args_list[0][1]
        assert "y0" in numeric_spy.call_args_list[1][1]

    def test_parameters(self, tol):
        """Test Jacobian computation works when parameters are both passed and not passed."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        # test Jacobian with no parameters
        res1 = tape.jacobian(dev)
        assert tape.get_parameters() == params

        # test Jacobian with parameters
        res2 = tape.jacobian(dev, params=[0.5, 0.6])
        assert tape.get_parameters() == params

        # test setting parameters
        tape.set_parameters(params=[0.5, 0.6])
        res3 = tape.jacobian(dev)
        assert np.allclose(res2, res3, atol=tol, rtol=0)
        assert not np.allclose(res1, res2, atol=tol, rtol=0)
        assert tape.get_parameters() == [0.5, 0.6]

    def test_numeric_pd_no_y0(self, tol):
        """Test that, if y0 is not passed when calling the numeric_pd method,
        y0 is calculated."""
        dev = qml.device("default.qubit", wires=2)

        params = [0.1, 0.2]

        with JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        # compute numeric gradient of parameter 0, without passing y0
        tapes, fn = tape.numeric_pd(0)
        assert len(tapes) == 2

        res1 = fn([tape.execute(dev) for tape in tapes])

        # compute y0 in advance
        y0 = tape.execute(dev)
        tapes, fn = tape.numeric_pd(0, y0=y0)
        assert len(tapes) == 1

        res2 = fn([tape.execute(dev) for tape in tapes])

        assert np.allclose(res1, res2, atol=tol, rtol=0)

    def test_numeric_unknown_order(self):
        """Test that an exception is raised if the finite-difference
        order is not supported"""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with JacobianTape() as tape:
            qml.RX(1, wires=[0])
            qml.RY(1, wires=[1])
            qml.RZ(1, wires=[2])
            qml.CNOT(wires=[0, 1])

            qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliZ(2))

        with pytest.raises(ValueError, match="Order must be 1 or 2"):
            tape.jacobian(dev, order=3)

    def test_independent_parameters(self):
        """Test the case where expectation values are independent of some parameters. For those
        parameters, the gradient should be evaluated to zero without executing the device."""
        dev = qml.device("default.qubit", wires=2)

        with JacobianTape() as tape1:
            qml.RX(1, wires=[0])
            qml.RX(1, wires=[1])
            qml.expval(qml.PauliZ(0))

        with JacobianTape() as tape2:
            qml.RX(1, wires=[0])
            qml.RX(1, wires=[1])
            qml.expval(qml.PauliZ(1))

        j1 = tape1.jacobian(dev)

        # We should only be executing the device to differentiate 1 parameter (2 executions)
        assert dev.num_executions == 2

        j2 = tape2.jacobian(dev)

        exp = -np.sin(1)

        assert np.allclose(j1, [exp, 0])
        assert np.allclose(j2, [0, exp])


class TestJacobianIntegration:
    """Integration tests for the Jacobian method"""

    def test_ragged_output(self):
        """Test that the Jacobian is correctly returned for a tape
        with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with JacobianTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        res = tape.jacobian(dev)
        assert res.shape == (6, 3)

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        res = tape.jacobian(dev)
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        res = tape.jacobian(dev)
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        res = tape.jacobian(dev)
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with JacobianTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        res = tape.jacobian(dev)
        assert res.shape == (5, 2)

        expected = (
            np.array(
                [
                    [-2 * np.sin(x), 0],
                    [
                        -(np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        -(np.sin(x) * np.sin(y / 2) ** 2),
                        (np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.sin(x) * np.sin(y / 2) ** 2),
                        (np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                ]
            )
            / 2
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestJacobianCVIntegration:
    """Intgration tests for the Jacobian method and CV circuits"""

    def test_single_output_value(self, tol):
        """Tests correct Jacobian and output shape for a CV tape
        with a single output"""
        dev = qml.device("default.gaussian", wires=2)
        n = 0.543
        a = -0.654

        with JacobianTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            qml.var(qml.NumberOperator(0))

        tape.trainable_params = {0, 1}
        res = tape.jacobian(dev)
        assert res.shape == (1, 2)

        expected = np.array([2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_output_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple outputs"""
        dev = qml.device("default.gaussian", wires=2)
        n = 0.543
        a = -0.654

        with JacobianTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            qml.expval(qml.NumberOperator(1))
            qml.var(qml.NumberOperator(0))

        tape.trainable_params = {0, 1}
        res = tape.jacobian(dev)
        assert res.shape == (2, 2)

        expected = np.array([[0, 0], [2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_trainable_measurement(self, tol):
        """Test that a trainable measurement can be differentiated"""
        dev = qml.device("default.gaussian", wires=2)
        a = 0.32
        phi = 0.54

        with JacobianTape() as tape:
            qml.Displacement(a, 0, wires=0)
            qml.expval(qml.QuadOperator(phi, wires=0))

        tape.trainable_params = {2}
        res = tape.jacobian(dev)
        expected = np.array([[-2 * a * np.sin(phi)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestHessian:
    """Unit tests for the hessian method"""

    def test_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with respect to a
        non-differentiable argument"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        # by default all parameters are assumed to be trainable
        with pytest.raises(
            ValueError, match=r"Cannot differentiate with respect to parameter\(s\) {0}"
        ):
            tape.hessian(None)

    def test_unknown_hessian_method_error(self):
        """Test error raised if gradient method is unknown."""
        tape = JacobianTape()
        with pytest.raises(ValueError, match="Unknown Hessian method"):
            tape.hessian(None, method="unknown method")

    def test_return_state_hessian_error(self):
        """Test error raised if circuit returns the state."""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with JacobianTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.state()

        with pytest.raises(
            ValueError,
            match=r"The Hessian method does not support circuits that return the state",
        ):
            tape.hessian(None)


class TestObservableWithObjectReturnType:
    """Unit tests for differentiation of observables returning an object"""

    def test_special_observable_qnode_differentiation(self):
        """Test differentiation of a QNode on a device supporting a
        special observable that returns an object rathern than a nummber."""

        class SpecialObject:
            """SpecialObject

            A special object that conveniently encapsulates the return value of
            a special observable supported by a special device and which supports
            multiplication with scalars and addition.
            """

            def __init__(self, val):
                self.val = val

            def __mul__(self, other):
                new = SpecialObject(self.val)
                new *= other
                return new

            def __imul__(self, other):
                self.val *= other
                return self

            def __rmul__(self, other):
                return self * other

            def __iadd__(self, other):
                self.val += other.val if isinstance(other, self.__class__) else other
                return self

            def __add__(self, other):
                new = SpecialObject(self.val)
                new += other.val if isinstance(other, self.__class__) else other
                return new

            def __radd__(self, other):
                return self + other

        class SpecialObservable(Observable):
            """SpecialObservable"""

            num_wires = AnyWires
            num_params = 0
            par_domain = None

            def diagonalizing_gates(self):
                """Diagonalizing gates"""
                return []

        class DeviceSupporingSpecialObservable(DefaultQubit):
            name = "Device supporing SpecialObservable"
            short_name = "default.qibit.specialobservable"
            observables = DefaultQubit.observables.union({"SpecialObservable"})

            def expval(self, observable, **kwargs):
                if self.analytic and isinstance(observable, SpecialObservable):
                    val = super().expval(qml.PauliZ(wires=0), **kwargs)
                    return SpecialObject(val)

                return super().expval(observable, **kwargs)

        dev = DeviceSupporingSpecialObservable(wires=1, shots=None)

        # force diff_method='parameter-shift' because otherwise
        # PennyLane swaps out dev for default.qubit.autograd
        @qml.qnode(dev, diff_method="parameter-shift")
        def qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(SpecialObservable(wires=0))

        @qml.qnode(dev, diff_method="parameter-shift")
        def reference_qnode(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        assert np.isclose(qnode(0.2).item().val, reference_qnode(0.2))
        assert np.isclose(qml.jacobian(qnode)(0.2).item().val, qml.jacobian(reference_qnode)(0.2))
