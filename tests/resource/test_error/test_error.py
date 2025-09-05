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
    _compute_algo_error,
)


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
        return qml.transforms.core.TransformProgram(), execution_config

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

    def test_specs(self):
        """Test that specs are tracking errors as expected."""

        algo_errors = qml.specs(self.circuit)()["errors"]
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
