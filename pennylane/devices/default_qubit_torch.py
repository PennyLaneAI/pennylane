# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains a PyTorch implementation of the :class:`~.DefaultQubit`
reference plugin.
"""
import semantic_version

try:
    import torch

    VERSION_SUPPORT = semantic_version.match(">=1.8.1", torch.__version__)
    if not VERSION_SUPPORT:
        raise ImportError("default.qubit.torch device requires Torch>=1.8.1")

except ImportError as e:
    raise ImportError("default.qubit.torch device requires Torch>=1.8.1") from e

import numpy as np
from pennylane.operation import DiagonalOperation
from pennylane.devices import torch_ops
from . import DefaultQubit


class DefaultQubitTorch(DefaultQubit):
    """Simulator plugin based on ``"default.qubit"``, written using PyTorch.

    **Short name:** ``default.qubit.torch``

    This device provides a pure-state qubit simulator written using PyTorch.
    As a result, it supports classical backpropagation as a means to compute the Jacobian. This can
    be faster than the parameter-shift rule for analytic quantum gradients
    when the number of parameters to be optimized is large.

    To use this device, you will need to install PyTorch:

    .. code-block:: console

        pip install torch>=1.8.0

    **Example**

    The ``default.qubit.torch`` is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) and the PyTorch interface. This is the default method
    of differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your PyTorch computation:

    .. code-block:: python

        dev = qml.device("default.qubit.torch", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> weights = torch.tensor([0.2, 0.5, 0.1], requires_grad=True)
    >>> res = circuit(weights)
    >>> res.backward()
    >>> print(weights.grad)
    tensor([-2.2527e-01, -1.0086e+00,  1.3878e-17])

    Autograd mode will also work when using classical backpropagation:

    >>> def cost(weights):
    ...    return torch.sum(circuit(weights)**3) - 1
    >>> res = circuit(weights)
    >>> res.backward()
    >>> print(weights.grad)
    tensor([-4.5053e-01, -2.0173e+00,  5.9837e-17])

    Executing the pipeline in PyTorch will allow the whole computation to be run on the GPU,
    and therefore providing an acceleration. Your parameters need to be instantiated on the same
    device as the backend device.

    .. code-block:: python

        dev = qml.device("default.qubit.torch", wires=1, torch_device='cuda')

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> weights = torch.tensor([0.2, 0.5, 0.1], requires_grad=True, device='cuda')
    >>> res = circuit(weights)
    >>> res.backward()
    >>> print(weights.grad)
    tensor([-2.2527e-01, -1.0086e+00,  1.3878e-17])


    There are a couple of things to keep in mind when using the ``"backprop"``
    differentiation method for QNodes:

    * You must use the ``"torch"`` interface for classical backpropagation, as PyTorch is
      used as the device backend.

    * Only exact expectation values, variances, and probabilities are differentiable.
      When instantiating the device with ``shots!=None``, differentiating QNode
      outputs will result in an error.

    If you wish to use a different machine-learning interface, or prefer to calculate quantum
    gradients using the ``parameter-shift`` or ``finite-diff`` differentiation methods,
    consider using the ``default.qubit`` device instead.

    Args:
        wires (int, Iterable): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems. Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means
            that the device returns analytical results.
            If ``shots > 0`` is used, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
        torch_device='cpu' (str): the device on which the computation will be run, ``'cpu'`` or ``'cuda'``
    """

    name = "Default qubit (Torch) PennyLane plugin"
    short_name = "default.qubit.torch"

    parametric_ops = {
        "PhaseShift": torch_ops.PhaseShift,
        "ControlledPhaseShift": torch_ops.ControlledPhaseShift,
        "RX": torch_ops.RX,
        "RY": torch_ops.RY,
        "RZ": torch_ops.RZ,
        "MultiRZ": torch_ops.MultiRZ,
        "Rot": torch_ops.Rot,
        "CRX": torch_ops.CRX,
        "CRY": torch_ops.CRY,
        "CRZ": torch_ops.CRZ,
        "CRot": torch_ops.CRot,
        "IsingXX": torch_ops.IsingXX,
        "IsingYY": torch_ops.IsingYY,
        "IsingZZ": torch_ops.IsingZZ,
        "SingleExcitation": torch_ops.SingleExcitation,
        "SingleExcitationPlus": torch_ops.SingleExcitationPlus,
        "SingleExcitationMinus": torch_ops.SingleExcitationMinus,
        "DoubleExcitation": torch_ops.DoubleExcitation,
        "DoubleExcitationPlus": torch_ops.DoubleExcitationPlus,
        "DoubleExcitationMinus": torch_ops.DoubleExcitationMinus,
    }

    C_DTYPE = torch.complex128
    R_DTYPE = torch.float64

    _abs = staticmethod(torch.abs)
    _einsum = staticmethod(torch.einsum)
    _flatten = staticmethod(torch.flatten)
    _reshape = staticmethod(torch.reshape)
    _roll = staticmethod(torch.roll)
    _stack = staticmethod(lambda arrs, axis=0, out=None: torch.stack(arrs, axis=axis, out=out))
    _tensordot = staticmethod(
        lambda a, b, axes: torch.tensordot(
            a, b, axes if isinstance(axes, int) else tuple(map(list, axes))
        )
    )
    _transpose = staticmethod(lambda a, axes=None: a.permute(*axes))
    _asnumpy = staticmethod(lambda x: x.cpu().numpy())
    _conj = staticmethod(torch.conj)
    _imag = staticmethod(torch.imag)
    _norm = staticmethod(torch.norm)
    _flatten = staticmethod(torch.flatten)

    def __init__(self, wires, *, shots=None, analytic=None, torch_device="cpu"):
        self._torch_device = torch_device
        super().__init__(wires, shots=shots, cache=0, analytic=analytic)

        # Move state to torch device (e.g. CPU, GPU, XLA, ...)
        self._state.requires_grad = True
        self._state = self._state.to(self._torch_device)
        self._pre_rotated_state = self._state

    @staticmethod
    def _asarray(a, dtype=None):
        if isinstance(a, list):
            # Handle unexpected cases where we don't have a list of tensors
            if not isinstance(a[0], torch.Tensor):
                res = np.asarray(a)
                res = torch.from_numpy(res)
            else:
                res = torch.cat([torch.reshape(i, (-1,)) for i in a], dim=0)
            res = torch.cat([torch.reshape(i, (-1,)) for i in res], dim=0)
        else:
            res = torch.as_tensor(a, dtype=dtype)
        return res

    @staticmethod
    def _dot(x, y):
        if x.device != y.device:
            if x.device != "cpu":
                return torch.tensordot(x, y.to(x.device), dims=1)
            if y.device != "cpu":
                return torch.tensordot(x.to(y.device), y, dims=1)

        return torch.tensordot(x, y, dims=1)

    def _cast(self, a, dtype=None):
        return torch.as_tensor(self._asarray(a, dtype=dtype), device=self._torch_device)

    @staticmethod
    def _reduce_sum(array, axes):
        if not axes:
            return array
        return torch.sum(array, dim=axes)

    @staticmethod
    def _conj(array):
        if isinstance(array, torch.Tensor):
            return torch.conj(array)
        return np.conj(array)

    @staticmethod
    def _scatter(indices, array, new_dimensions):

        # `array` is now a torch tensor
        tensor = array
        new_tensor = torch.zeros(new_dimensions, dtype=tensor.dtype, device=tensor.device)
        new_tensor[indices] = tensor
        return new_tensor

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(passthru_interface="torch", supports_reversible_diff=False)
        return capabilities

    def _get_unitary_matrix(self, unitary):
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            torch.Tensor[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis, or, in the case of a diagonal unitary,
            a 1D array representing the matrix diagonal.
        """
        op_name = unitary.base_name
        if op_name in self.parametric_ops:
            if op_name == "MultiRZ":
                mat = self.parametric_ops[op_name](
                    *unitary.parameters, len(unitary.wires), device=self._torch_device
                )
            else:
                mat = self.parametric_ops[op_name](*unitary.parameters, device=self._torch_device)
            if unitary.inverse:
                if isinstance(unitary, DiagonalOperation):
                    mat = self._conj(mat)
                else:
                    mat = self._transpose(self._conj(mat), axes=[1, 0])
            return mat

        if isinstance(unitary, DiagonalOperation):
            return self._asarray(unitary.eigvals, dtype=self.C_DTYPE)
        return self._asarray(unitary.matrix, dtype=self.C_DTYPE)

    def sample_basis_states(self, number_of_states, state_probability):
        """Sample from the computational basis states based on the state
        probability.

        This is an auxiliary method to the ``generate_samples`` method.

        Args:
            number_of_states (int): the number of basis states to sample from
            state_probability (torch.Tensor[float]): the computational basis probability vector

        Returns:
            List[int]: the sampled basis states
        """
        return super().sample_basis_states(
            number_of_states, state_probability.cpu().detach().numpy()
        )

    def _apply_operation(self, state, operation):
        """Applies operations to the input state.

        Args:
            state (torch.Tensor[complex]): input state
            operation (~.Operation): operation to apply on the device

        Returns:
            torch.Tensor[complex]: output state
        """
        if state.device != self._torch_device:
            state = state.to(self._torch_device)
        return super()._apply_operation(state, operation)
