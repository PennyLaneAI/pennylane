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
import pytest

import pennylane as qml
from pennylane.resource.error import AlgorithmicError, ErrorOperation


class SimpleError(AlgorithmicError):
    def combine(self, other):
        return self.__class__(self.error + other.error)

    @staticmethod
    def get_error(approx_op, other_op):  # pylint: disable=unused-argument
        return 0.5  # get simple error is always 0.5


class ErrorNoGetError(AlgorithmicError):  # pylint: disable=too-few-public-methods
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

            class ErrorNoCombine(AlgorithmicError):  # pylint: disable=too-few-public-methods
                @staticmethod
                def get_error(approx_op, other_op):  # pylint: disable=unused-argument
                    return 0.5  # get simple error is always 0.5

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


class TestErrorOperation:  # pylint: disable=too-few-public-methods
    """Test the base ErrorOperation class."""

    def test_error_method(self):
        """Test that error method works as expected"""

        class SimpleErrorOperation(ErrorOperation):  # pylint: disable=too-few-public-methods
            @property
            def error(self):
                return len(self.wires)

        no_error_op = SimpleErrorOperation(wires=[1, 2, 3])
        assert no_error_op.error == 3

    def test_no_error_method(self):
        """Test error is raised if the error method is not defined."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class NoErrorOp(ErrorOperation):  # pylint: disable=too-few-public-methods
                num_wires = 3

            _ = NoErrorOp(wires=[1, 2, 3])
