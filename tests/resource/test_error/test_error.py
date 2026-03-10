# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Test base AlgorithmicError class and its associated methods.
"""

import numpy as np

# pylint: disable=too-few-public-methods, unused-argument
import pytest

import pennylane as qml
from pennylane.operation import Operation
from pennylane.resource.error import (
    AlgorithmicError,
    ErrorOperation,
    SpectralNormError,
)
from pennylane.resource.error.error import _compute_algo_error


class SimpleError(AlgorithmicError):
    def combine(self, other):
        return self.__class__(self.error + other.error)

    @staticmethod
    def get_error(approximate_op, exact_op):
        return 0.5  # get simple error is always 0.5


class ErrorNoGetError(AlgorithmicError):
    def combine(self, other):
        return self.__class__(self.error + other.error)


class TestAlgorithmicError:
    """Test the methods and attributes of the AlgorithmicError class"""

    @pytest.mark.parametrize("error", [1.23, 0.45, -6])
    def test_error_attribute(self, error):
        """Test that instantiation works"""
        ErrorObj = SimpleError(error)
        assert ErrorObj.error == error

    def test_combine_not_implemented(self):
        """Test can't instantiate Error if the combine method is not defined."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class ErrorNoCombine(AlgorithmicError):
                @staticmethod
                def get_error(approximate_op, exact_op):
                    return 0.5  # get simple error is always 0.5

            # pylint: disable=abstract-class-instantiated
            _ = ErrorNoCombine(1.23)

    @pytest.mark.parametrize("err1", [1.23, 0.45, -6])
    @pytest.mark.parametrize("err2", [1.23, 0.45, -6])
    def test_combine(self, err1, err2):
        """Test that combine works as expected"""
        ErrorObj1 = SimpleError(err1)
        ErrorObj2 = SimpleError(err2)

        res = ErrorObj1.combine(ErrorObj2)
        assert res.error == err1 + err2
        assert isinstance(res, type(ErrorObj1))

    def test_get_error_not_implemented(self):
        """Test NotImplementedError is raised if the method is not defined."""
        approx_op = qml.RZ(0.01, 0)
        exact_op = qml.PauliZ(0)

        with pytest.raises(NotImplementedError):
            _ = ErrorNoGetError.get_error(approx_op, exact_op)

    def test_get_error(self):
        """Test that get_error works as expected"""
        approx_op = qml.RZ(0.01, 0)
        exact_op = qml.PauliZ(0)

        res = SimpleError.get_error(approx_op, exact_op)
        assert res == 0.5


class TestSpectralNormError:
    """Test methods for the SpectralNormError class"""

    @pytest.mark.parametrize("err1", [0, 0.25, 0.75, 1.50, 2.50])
    @pytest.mark.parametrize("err2", [0, 0.25, 0.75, 1.50, 2.50])
    def test_combine(self, err1, err2):
        """Test that combine works as expected"""
        ErrorObj1 = SpectralNormError(err1)
        ErrorObj2 = SpectralNormError(err2)

        res = ErrorObj1.combine(ErrorObj2)
        assert res.error == err1 + err2
        assert isinstance(res, type(ErrorObj1))

    @pytest.mark.parametrize(
        "phi, expected",
        [
            [0, 2.0000000000000004],
            [0.25, 1.9980522880732308],
            [0.75, 1.9828661007943447],
            [1.50, 1.9370988373785705],
            [2.50, 1.8662406421959807],
        ],
    )
    def test_get_error(self, phi, expected):
        """Test that get_error works as expected"""
        approx_op = qml.Hadamard(0)
        exact_op = qml.RX(phi, 0)

        res = SpectralNormError.get_error(approx_op, exact_op)
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "phi, expected",
        [
            [0, 1.311891347309272],
            [0.25, 1.3182208123805488],
            [0.75, 1.3772695464365001],
            [1.50, 1.6078817482299055],
            [2.50, 2.0506044587737255],
        ],
    )
    def test_custom_operator(self, phi, expected):
        """Test that get_error for a custom operator"""

        class DummyOp(Operation):
            def compute_matrix(self):
                return np.array([[0.5, 1.0], [1.2, 1.3]])

        approx_op = DummyOp(1)
        exact_op = qml.RX(phi, 1)

        res = SpectralNormError.get_error(approx_op, exact_op)
        assert np.isclose(res, expected)

    def test_no_operator_matrix_defined(self):
        """Test that get_error fails if the operator matrix is not defined"""

        class MyOp(Operation):

            @property
            def name(self):
                return self.__class__.__name__

        approx_op = MyOp(0)
        exact_op = qml.RX(0.1, 0)

        with pytest.raises(qml.operation.MatrixUndefinedError):
            SpectralNormError.get_error(approx_op, exact_op)

    def test_repr(self):
        """Test that formal string representation is correct"""
        S1 = SpectralNormError(0.3)
        assert repr(S1) == f"SpectralNormError({0.3})"


class TestErrorOperation:
    """Test the base ErrorOperation class."""

    def test_error_method(self):
        """Test that error method works as expected"""

        class SimpleErrorOperation(ErrorOperation):
            def error(self):
                return len(self.wires)

        no_error_op = SimpleErrorOperation(wires=[1, 2, 3])
        assert no_error_op.error() == 3

    def test_no_error_method(self):
        """Test error is raised if the error method is not defined."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class NoErrorOp(ErrorOperation):
                num_wires = 3

            _ = NoErrorOp(wires=[1, 2, 3])


class MultiplicativeError(AlgorithmicError):
    """Multiplicative error object"""

    def combine(self, other):
        return self.__class__(self.error * other.error)

    def __repr__(self):
        """Return formal string representation."""
        return f"MultiplicativeError({self.error})"


class AdditiveError(AlgorithmicError):
    """Additive error object"""

    def combine(self, other):
        return self.__class__(self.error + other.error)

    def __repr__(self):
        """Return formal string representation."""
        return f"AdditiveError({self.error})"


class CustomErrorOp1(ErrorOperation):
    """Custome error operation with multiplicative error"""

    def __init__(self, phase, wires):
        self.phase = phase
        super().__init__(phase, wires=wires)

    def error(self, *args, **kwargs):
        return MultiplicativeError(self.phase)


class CustomErrorOp2(ErrorOperation):
    """Custome error with additive error"""

    def __init__(self, flips, wires):
        self.flips = flips
        super().__init__(flips, wires=wires)

    def error(self, *args, **kwargs):
        return AdditiveError(self.flips)


_HAMILTONIAN = qml.dot([1.0, -0.5], [qml.X(0) @ qml.Y(1), qml.Y(0) @ qml.Y(1)])


class TestSpecAndTracker:
    """Test capture of ErrorOperation in specs and tracker."""

    # TODO: remove this when support for below is present
    # little hack for stopping device-level decomposition for custom ops
    @staticmethod
    def preprocess(execution_config: qml.devices.ExecutionConfig | None = None):
        """A vanilla preprocesser"""
        return qml.CompilePipeline(), execution_config

    dev = qml.device("null.qubit", wires=2)
    dev.preprocess = preprocess.__func__

    @staticmethod
    @qml.qnode(dev)
    def circuit():
        """circuit with custom ops"""
        qml.TrotterProduct(_HAMILTONIAN, time=1.0, n=4, order=2)
        CustomErrorOp1(0.31, [0])
        CustomErrorOp2(0.12, [1])
        qml.TrotterProduct(_HAMILTONIAN, time=1.0, n=4, order=4)
        CustomErrorOp1(0.24, [1])
        CustomErrorOp2(0.73, [0])
        return qml.state()

    errors_types = ["MultiplicativeError", "AdditiveError", "SpectralNormError"]

    def test_computation(self):
        """Test that _compute_algo_error are adding up errors as expected."""

        tape = qml.workflow.construct_tape(self.circuit)()
        algo_errors = _compute_algo_error(tape)
        assert len(algo_errors) == 3
        assert all(error in algo_errors for error in self.errors_types)
        assert algo_errors["MultiplicativeError"].error == 0.31 * 0.24
        assert algo_errors["AdditiveError"].error == 0.73 + 0.12
        assert algo_errors["SpectralNormError"].error == 0.25 + 0.17998560822421455

    def test_tracker(self):
        """Test that tracker are tracking errors as expected."""

        with qml.Tracker(self.dev) as tracker:
            self.circuit()

        algo_errors = tracker.latest["errors"]
        assert len(algo_errors) == 3
        assert all(error in algo_errors for error in self.errors_types)
        assert algo_errors["MultiplicativeError"].error == 0.31 * 0.24
        assert algo_errors["AdditiveError"].error == 0.73 + 0.12
        assert algo_errors["SpectralNormError"].error == 0.25 + 0.17998560822421455


class TestAlgoError:
    """Test the qml.resource.algo_error function."""

    dev = qml.device("default.qubit", wires=2)

    def test_basic_usage(self):
        """Test basic usage of algo_error with TrotterProduct."""
        Hamiltonian = qml.dot([1.0, -0.5], [qml.X(0) @ qml.Y(1), qml.Y(0) @ qml.Y(1)])

        @qml.qnode(self.dev)
        def circuit():
            qml.TrotterProduct(Hamiltonian, time=1.0, n=4, order=2)
            return qml.state()

        errors = qml.resource.algo_error(circuit)()
        assert "SpectralNormError" in errors
        assert isinstance(errors["SpectralNormError"], SpectralNormError)
        assert errors["SpectralNormError"].error > 0

    def test_multiple_error_types(self):
        """Test algo_error with multiple error types."""

        @qml.qnode(self.dev)
        def circuit():
            qml.TrotterProduct(_HAMILTONIAN, time=1.0, n=4, order=2)
            CustomErrorOp1(0.31, [0])
            CustomErrorOp2(0.12, [1])
            qml.TrotterProduct(_HAMILTONIAN, time=1.0, n=4, order=4)
            CustomErrorOp1(0.24, [1])
            CustomErrorOp2(0.73, [0])
            return qml.state()

        errors = qml.resource.algo_error(circuit)()
        assert len(errors) == 3
        assert "MultiplicativeError" in errors
        assert "AdditiveError" in errors
        assert "SpectralNormError" in errors
        assert np.isclose(errors["MultiplicativeError"].error, 0.31 * 0.24)
        assert np.isclose(errors["AdditiveError"].error, 0.73 + 0.12)
        assert np.isclose(errors["SpectralNormError"].error, 0.25 + 0.17998560822421455)

    def test_no_error_operations(self):
        """Test algo_error with a circuit containing no ErrorOperations."""

        @qml.qnode(self.dev)
        def circuit():
            qml.RX(0.5, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        errors = qml.resource.algo_error(circuit)()
        assert errors == {}

    def test_with_arguments(self):
        """Test algo_error when the qnode takes arguments."""
        Hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])

        @qml.qnode(self.dev)
        def circuit(time, n):
            qml.TrotterProduct(Hamiltonian, time=time, n=n, order=2)
            return qml.state()

        errors1 = qml.resource.algo_error(circuit)(time=1.0, n=4)
        errors2 = qml.resource.algo_error(circuit)(time=2.0, n=4)

        # Different time values should give different errors
        assert errors1["SpectralNormError"].error != errors2["SpectralNormError"].error

    def test_level_argument(self):
        """Test that the level argument affects which operations are analyzed for errors."""
        Hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])

        @qml.qnode(self.dev)
        def circuit():
            qml.TrotterProduct(Hamiltonian, time=1.0, n=4, order=2)
            return qml.state()

        # At user level (before device decomposition), TrotterProduct ErrorOperation is present
        errors_user = qml.resource.algo_error(circuit, level="user")()
        assert "SpectralNormError" in errors_user
        assert errors_user["SpectralNormError"].error > 0

        # At device level (after full decomposition), TrotterProduct is decomposed
        # into basic gates, so the ErrorOperation is no longer present
        errors_device = qml.resource.algo_error(circuit, level="device")()
        assert errors_device == {}

        # Test with integer level as well
        errors_level_0 = qml.resource.algo_error(circuit, level=0)()
        assert "SpectralNormError" in errors_level_0
        assert errors_level_0["SpectralNormError"].error == errors_user["SpectralNormError"].error

    def test_invalid_input(self):
        """Test that algo_error raises an error for invalid input."""
        with pytest.raises(ValueError, match="can only be applied to a QNode"):
            qml.resource.algo_error("not_a_qnode")

    def test_with_parameters(self):
        """Test algo_error with parameterized circuit."""
        Hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])

        @qml.qnode(self.dev)
        def circuit(phi):
            qml.RX(phi, wires=0)
            qml.TrotterProduct(Hamiltonian, time=1.0, n=4, order=2)
            return qml.state()

        errors = qml.resource.algo_error(circuit)(0.5)
        assert "SpectralNormError" in errors
        assert errors["SpectralNormError"].error > 0

    def test_consistency_with_compute_algo_error(self):
        """Test that algo_error gives the same result as _compute_algo_error."""

        @qml.qnode(self.dev)
        def circuit():
            qml.TrotterProduct(_HAMILTONIAN, time=1.0, n=4, order=2)
            CustomErrorOp1(0.31, [0])
            CustomErrorOp2(0.12, [1])
            return qml.state()

        # Using algo_error
        algo_error_result = qml.resource.algo_error(circuit)()

        # Using _compute_algo_error directly
        tape = qml.workflow.construct_tape(circuit)()
        compute_result = _compute_algo_error(tape)

        # Results should match
        assert set(algo_error_result.keys()) == set(compute_result.keys())
        for key in algo_error_result:
            assert np.isclose(algo_error_result[key].error, compute_result[key].error)

    def test_multi_tape_batch_returns_list(self):
        """Test that algo_error returns a list when there are multiple tapes."""

        @qml.qnode(self.dev)
        def circuit():
            qml.TrotterProduct(_HAMILTONIAN, time=1.0, n=4, order=2)
            CustomErrorOp1(0.5, [0])
            return qml.sample()

        # Apply a transform that creates multiple tapes (e.g., split_non_commuting)
        @qml.transforms.split_non_commuting
        @qml.qnode(self.dev)
        def multi_tape_circuit():
            qml.TrotterProduct(_HAMILTONIAN, time=1.0, n=4, order=2)
            CustomErrorOp1(0.5, [0])
            # Non-commuting measurements create multiple tapes
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0))

        errors = qml.resource.algo_error(multi_tape_circuit)()

        # Should return a list of error dicts, one per tape
        assert isinstance(errors, list)
        assert len(errors) == 2
        for tape_errors in errors:
            assert isinstance(tape_errors, dict)
            assert "SpectralNormError" in tape_errors
            assert "MultiplicativeError" in tape_errors

    @pytest.mark.torch
    def test_torch_layer_support(self):
        """Test that algo_error works with TorchLayer."""
        torch = pytest.importorskip("torch")
        from pennylane.qnn.torch import TorchLayer

        Hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])

        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            qml.RX(inputs[0], wires=0)
            qml.RY(weights[0], wires=0)
            qml.TrotterProduct(Hamiltonian, time=1.0, n=4, order=2)
            return qml.expval(qml.Z(0))

        weight_shapes = {"weights": (1,)}
        layer = TorchLayer(circuit, weight_shapes)

        # Pass inputs only - weights are bound to the layer
        errors = qml.resource.algo_error(layer)(torch.tensor([0.5]))
        assert "SpectralNormError" in errors
        assert errors["SpectralNormError"].error > 0
