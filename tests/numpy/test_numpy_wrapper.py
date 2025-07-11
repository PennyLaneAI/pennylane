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
"""
Tests for the ``autograd.numpy`` wrapping functionality. This functionality
modifies Autograd NumPy arrays so that they have an additional property,
``requires_grad``, that marks them as trainable/non-trainable.
"""

import numpy as onp
import pytest
from autograd.numpy.numpy_boxes import ArrayBox

import pennylane as qml
import pennylane.exceptions
from pennylane import numpy as np
from pennylane.numpy.tensor import tensor_to_arraybox


@pytest.mark.unit
class TestExtractTensors:
    """Tests for the extract_tensors function"""

    def test_empty_iterable(self):
        """Test that an empty iterable returns nothing"""
        res = list(np.extract_tensors([]))
        # pylint: disable=use-implicit-booleaness-not-comparison
        assert res == []

    def test_iterable_with_strings(self):
        """Test that strings are not treated as a sequence"""
        arr1 = np.array([0.4, 0.1])
        arr2 = np.array([1])

        res = list(np.extract_tensors([arr1, ["abc", [arr2]]]))

        assert len(res) == 2
        assert res[0] is arr1
        assert res[1] is arr2

    def test_iterable_with_unpatched_numpy_arrays(self):
        """Test that the extraction ignores unpatched numpy arrays"""
        arr1 = np.array([0.4, 0.1])
        arr2 = np.array([1])

        res = list(np.extract_tensors([arr1, [onp.array([1, 2]), [arr2]]]))

        assert len(res) == 2
        assert res[0] is arr1
        assert res[1] is arr2


@pytest.mark.unit
class TestTensor:
    """Tests for the Tensor(ndarray) subclass"""

    def test_passing_requires_grad_arg(self):
        """Test that you can instantiate the Tensor class with the
        requires_grad argument"""
        # default value is true
        x = np.tensor([0, 1, 2])
        assert x.requires_grad

        x = np.tensor([0, 1, 2], requires_grad=True)
        assert x.requires_grad

        x = np.tensor([0, 1, 2], requires_grad=False)
        assert not x.requires_grad

    def test_requires_grad_setter(self):
        """Test that the value of requires_grad can be changed
        on an instantiated object"""
        # default value is true
        x = np.tensor([0, 1, 2])
        assert x.requires_grad

        x.requires_grad = False
        assert not x.requires_grad

    def test_string_representation(self, capsys):
        """Test the string representation is correct"""
        x = np.tensor([0, 1, 2])
        print(repr(x))
        captured = capsys.readouterr()
        assert "tensor([0, 1, 2], requires_grad=True)" in captured.out

        x.requires_grad = False
        print(repr(x))
        captured = capsys.readouterr()
        assert "tensor([0, 1, 2], requires_grad=False)" in captured.out

    @pytest.mark.parametrize("grad", [True, False])
    def test_indexing(self, grad):
        """Test that indexing into a tensor always returns a tensor"""
        x = np.tensor([[0, 1, 2], [3, 4, 5]], requires_grad=grad)

        assert isinstance(x[0], np.tensor)
        assert x[0].requires_grad is grad

        assert isinstance(x[0, 0], np.tensor)
        assert x[0, 0].requires_grad is grad
        assert x[0, 0].shape == tuple()
        assert x[0, 0].item() == 0

    def test_numpy_to_arraybox(self):
        """Test converting a PennyLane NumPy array to an ArrayBox object"""
        x = onp.array(0)
        res = tensor_to_arraybox(x, None, None)
        assert isinstance(res, ArrayBox)


# The following NumPy functions all create
# arrays based on list input. Additional keyword
# arguments required for the function are provided
# as an optional dictionary.
ARRAY_CREATION_FNS = [
    [np.array, {}],
    [np.asarray, {}],
    [np.fromiter, {"dtype": np.int64}],
    [np.empty_like, {}],
    [np.ones_like, {}],
    [np.zeros_like, {}],
    [np.full_like, {"fill_value": 5}],
]

# The following NumPy functions all create
# arrays based on shape input.
ARRAY_SHAPE_FNS = [
    [np.empty, {}],
    [np.identity, {}],
    [np.ones, {}],
    [np.zeros, {}],
    [np.full, {"fill_value": 5}],
    [np.arange, {}],
    [np.eye, {}],
]


@pytest.mark.unit
class TestNumpyIntegration:
    """Test that the wrapped NumPy functionality integrates well
    with standard NumPy functions."""

    @pytest.mark.parametrize("fn, kwargs", ARRAY_CREATION_FNS)
    def test_tensor_creation_from_list(self, fn, kwargs):
        """Test that you can create the tensor class from NumPy functions
        instantiated via lists with the requires_grad argument"""
        # default value is true
        x = fn([1, 1, 2], **kwargs)

        assert isinstance(x, np.tensor)
        assert x.requires_grad

        x = fn([1, 1, 2], requires_grad=True, **kwargs)
        assert x.requires_grad

        x.requires_grad = False
        assert not x.requires_grad

        x = fn([1, 1, 2], requires_grad=False, **kwargs)
        assert not x.requires_grad

    @pytest.mark.parametrize("fn, kwargs", ARRAY_SHAPE_FNS)
    def test_tensor_creation_from_shape(self, fn, kwargs):
        """Test that you can create the tensor class from NumPy functions
        instantiated via shapes with the requires_grad argument"""
        # default value is true
        shape = 4
        x = fn(shape, **kwargs)

        assert isinstance(x, np.tensor)
        assert x.requires_grad

        x = fn(shape, requires_grad=True, **kwargs)
        assert x.requires_grad

        x.requires_grad = False
        assert not x.requires_grad

        x = fn(shape, requires_grad=False, **kwargs)
        assert not x.requires_grad

    def test_tensor_creation_from_string(self):
        """Test that a tensor is properly created from a string."""
        string = "5, 4, 1, 2"
        x = np.fromstring(string, dtype=int, sep=",")

        assert isinstance(x, np.tensor)
        assert x.requires_grad

        x = np.fromstring(string, requires_grad=True, dtype=int, sep=",")
        assert x.requires_grad

        x.requires_grad = False
        assert not x.requires_grad

        x = np.fromstring(string, requires_grad=False, dtype=int, sep=",")
        assert not x.requires_grad

    def test_wrapped_docstring(self, capsys):
        """Test that wrapped NumPy functions retains the original
        docstring."""
        print(np.sin.__doc__)
        captured = capsys.readouterr()
        assert "Trigonometric sine, element-wise." in captured.out

    def test_wrapped_function_on_tensor(self):
        """Test that wrapped functions work correctly"""
        x = np.array([0, 1, 2], requires_grad=True)
        res = np.sin(x)
        expected = onp.sin(onp.array([0, 1, 2]))
        assert np.all(res == expected)
        assert res.requires_grad

        # since the wrapping is dynamic, even ``sin`` will
        # now accept the requires_grad keyword argument.
        x = np.array([0, 1, 2], requires_grad=True)
        res = np.sin(x, requires_grad=False)
        expected = onp.sin(onp.array([0, 1, 2]))
        assert np.all(res == expected)
        assert not res.requires_grad

    def test_wrapped_function_nontrainable_list_input(self):
        """Test that a wrapped function with signature of the form
        func([arr1, arr2, ...]) acting on non-trainable input returns non-trainable output"""
        arr1 = np.array([0, 1], requires_grad=False)
        arr2 = np.array([2, 3], requires_grad=False)
        arr3 = np.array([4, 5], requires_grad=False)

        res = np.vstack([arr1, arr2, arr3])
        assert not res.requires_grad

        # If one of the inputs is trainable, the output always is.
        arr1.requires_grad = True
        res = np.vstack([arr1, arr2, arr3])
        assert res.requires_grad

    def test_wrapped_function_nontrainable_variable_args(self):
        """Test that a wrapped function with signature of the form
        func(arr1, arr2, ...) acting on non-trainable input returns non-trainable output"""
        arr1 = np.array([0, 1], requires_grad=False)
        arr2 = np.array([2, 3], requires_grad=False)

        res = np.arctan2(arr1, arr2)
        assert not res.requires_grad

        # If one of the inputs is trainable, the output always is.
        arr1.requires_grad = True
        res = np.arctan2(arr1, arr2)
        assert res.requires_grad

    def test_wrapped_function_on_array(self):
        """Test behaviour of a wrapped function on a vanilla NumPy
        array."""
        res = np.sin(onp.array([0, 1, 2]))
        expected = onp.sin(onp.array([0, 1, 2]))
        assert np.all(res == expected)

        # the result has been converted into a tensor
        assert isinstance(res, np.tensor)
        assert res.requires_grad

    def test_classes_not_wrapped(self):
        """Test that NumPy classes are not wrapped"""
        x = np.ndarray([0, 1, 2])
        assert not isinstance(x, np.tensor)
        assert not hasattr(x, "requires_grad")

    def test_random_subpackage(self):
        """Test that the random subpackage is correctly wrapped"""
        x = np.random.normal(size=[2, 3])
        assert isinstance(x, np.tensor)

    def test_linalg_subpackage(self):
        """Test that the linalg subpackage is correctly wrapped"""
        x = np.linalg.eigvals([[1, 1], [1, 1]])
        assert isinstance(x, np.tensor)

    def test_fft_subpackage(self):
        """Test that the fft subpackage is correctly wrapped"""
        x = np.fft.fft(np.arange(8))
        assert isinstance(x, np.tensor)

    def test_unary_operators(self):
        """Test that unary operators (negate, power)
        correctly work on tensors."""
        x = np.array([[1, 2], [3, 4]])
        res = -x
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        x = np.array([[1, 2], [3, 4]], requires_grad=False)
        res = -x
        assert isinstance(res, np.tensor)
        assert not res.requires_grad

        x = np.array([[1, 2], [3, 4]])
        res = x**2
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        x = np.array([[1, 2], [3, 4]], requires_grad=False)
        res = x**2
        assert isinstance(res, np.tensor)
        assert not res.requires_grad

    def test_binary_operators(self):
        """Test that binary operators (add, subtract, divide, multiply, matmul)
        correctly work on tensors."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        res = x + y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x - y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x / y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x * y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x @ y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

    def test_binary_operator_nontrainable(self):
        """Test that binary operators on two non-trainable
        arrays result in non-trainable output."""
        x = np.array([[1, 2], [3, 4]], requires_grad=False)
        y = np.array([[5, 6], [7, 8]], requires_grad=False)
        res = x + y
        assert isinstance(res, np.tensor)
        assert not res.requires_grad

        res = x - y
        assert isinstance(res, np.tensor)
        assert not res.requires_grad

        res = x / y
        assert isinstance(res, np.tensor)
        assert not res.requires_grad

        res = x * y
        assert isinstance(res, np.tensor)
        assert not res.requires_grad

        res = x @ y
        assert isinstance(res, np.tensor)
        assert not res.requires_grad

    def test_binary_operator_mixed_trainable_left(self):
        """Test that binary operators on one trainable and one non-trainable
        arrays result in trainable output."""
        x = np.array([[1, 2], [3, 4]], requires_grad=True)
        y = np.array([[5, 6], [7, 8]], requires_grad=False)

        res = x + y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x - y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x / y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x * y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x @ y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

    def test_binary_operator_mixed_trainable_right(self):
        """Test that binary operators on one non-trainable and one trainable
        arrays result in trainable output."""
        x = np.array([[1, 2], [3, 4]], requires_grad=False)
        y = np.array([[5, 6], [7, 8]], requires_grad=True)

        res = x + y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x - y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x / y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x * y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

        res = x @ y
        assert isinstance(res, np.tensor)
        assert res.requires_grad

    def test_multi_output_array_ufunc(self):
        """Test that a tensor is returned if the array ufunc method returns
        multiple results."""

        class _ufunc:
            # pylint: disable=unused-argument,too-few-public-methods
            def __init__(self, *args, **kwargs):
                self.nout = 2

            def __call__(self, *args, **kwargs):
                return [True, True]

        x = np.array([[1, 2], [3, 4]], requires_grad=False)
        res = x.__array_ufunc__(_ufunc(), "__call__")
        assert isinstance(res, tuple)
        assert len(res) == 2


@pytest.mark.integration
class TestAutogradIntegration:
    """Test autograd works with the new tensor subclass"""

    def test_gradient(self):
        """Test gradient computations continue to work"""

        def cost(x):
            return np.sum(np.sin(x))

        grad_fn = qml.grad(cost, argnum=[0])
        arr1 = np.array([0.0, 1.0, 2.0])

        res = grad_fn(arr1)
        expected = np.cos(arr1)

        assert np.all(res == expected)

    def test_non_differentiable_gradient(self):
        """Test gradient computation with requires_grad=False raises an error"""

        def cost(x):
            return np.sum(np.sin(x))

        grad_fn = qml.grad(cost, argnum=[0])
        arr1 = np.array([0.0, 1.0, 2.0], requires_grad=False)

        with pytest.raises(pennylane.exceptions.NonDifferentiableError, match="non-differentiable"):
            grad_fn(arr1)


@pytest.mark.unit
class TestScalarHashing:
    """Test for the hashing capability of scalar arrays."""

    def test_create_set_scalar_arrays(self):
        """Test that a collection of scalar arrays can be properly hashed
        when creating a set"""
        data = [np.array(1), np.array(2), np.array(1), np.array(3)]
        res = set(data)
        expected = {np.array(1), np.array(2), np.array(3)}
        assert res == expected

    def test_requires_grad_hashing(self):
        """Test that the gradient information is correctly taken into account when hashing"""
        data = [np.array(1), np.array(2), np.array(1, requires_grad=False), np.array(3)]
        res = set(data)
        expected = {np.array(1, requires_grad=False), np.array(1), np.array(2), np.array(3)}
        assert res == expected

    def test_create_set_from_array_iteration(self):
        """Test that a one dimensional array correctly produces a set via iteration"""
        data = np.array([1, 2, 1, 3], requires_grad=True)
        res = set(data)
        expected = {
            np.array(1, requires_grad=True),
            np.array(2, requires_grad=True),
            np.array(3, requires_grad=True),
        }
        assert res == expected

        data = np.array([1, 2, 1, 3], requires_grad=False)
        res = set(data)
        expected = {
            np.array(1, requires_grad=False),
            np.array(2, requires_grad=False),
            np.array(3, requires_grad=False),
        }
        assert res == expected

    def test_nonzero_dim_arrays_non_hashable(self):
        """Test that a non-scalar array continues to remain non-hashable"""
        with pytest.raises(TypeError, match=r"unhashable type: 'numpy\.tensor'"):
            set(np.array([[1, 2], [3, 4]]))


class TestNumpyConversion:
    """Tests for the tensor.unwrap() and tensor.numpy() methods"""

    @pytest.mark.unit
    def test_convert_scalar_array(self):
        """Test that a scalar array converts to a python literal"""
        data = np.array(1.543)
        res = data.unwrap()
        assert res == data.item()
        assert isinstance(res, float)

    @pytest.mark.unit
    def test_convert_array(self):
        """Test that a numpy array successfully converts"""
        data = np.array([1, 2, 3])
        res = data.numpy()

        assert np.shares_memory(res, data)
        assert np.all(res == data)
        assert isinstance(res, np.ndarray)
        assert not isinstance(res, np.tensor)

    @pytest.mark.system
    def test_single_gate_parameter(self):
        """Test that when supplied a PennyLane tensor, a QNode passes an
        unwrapped tensor as the argument to a gate taking a single parameter"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(phi=None):
            for y in phi:
                for idx, x in enumerate(y):
                    qml.RX(x, wires=idx)
            return qml.expval(qml.PauliZ(0))

        phi = np.tensor([[0.04439891, 0.14490549, 3.29725643, 2.51240058]])

        tape = qml.workflow.construct_tape(circuit)(phi)

        ops = tape.operations
        assert len(ops) == 4
        for op, p in zip(ops, phi[0]):
            # Test each rotation applied
            assert op.name == "RX"
            assert op.parameters == [p]

    @pytest.mark.system
    def test_multiple_gate_parameter(self):
        """Test that when supplied a PennyLane tensor, a QNode passes arguments
        as unwrapped tensors to a gate taking multiple parameters"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(phi=None):
            for idx, x in enumerate(phi):
                qml.Rot(*x, wires=idx)
            return qml.expval(qml.PauliZ(0))

        phi = np.tensor([[0.04439891, 0.14490549, 3.29725643]])

        tape = qml.workflow.construct_tape(circuit)(phi)

        # Test the rotation applied
        ops = tape.operations
        assert len(ops) == 1
        assert ops[0].name == "Rot"
        assert np.array_equal(ops[0].parameters, phi[0])
