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
"""This module contains a PyTorch implementation of the :class:`~.DefaultQubitLegacy`
reference plugin.
"""
import warnings
import inspect
import logging
import semantic_version

try:
    import torch

    VERSION_SUPPORT = semantic_version.match(">=1.8.1", torch.__version__)
    if not VERSION_SUPPORT:  # pragma: no cover
        raise ImportError("default.qubit.torch device requires Torch>=1.8.1")

except ImportError as e:  # pragma: no cover
    raise ImportError("default.qubit.torch device requires Torch>=1.8.1") from e

import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from . import DefaultQubitLegacy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DefaultQubitTorch(DefaultQubitLegacy):
    """Simulator plugin based on ``"default.qubit.legacy"``, written using PyTorch.

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
            return qml.expval(qml.Z(0))

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
            return qml.expval(qml.Z(0))

    >>> weights = torch.tensor([0.2, 0.5, 0.1], requires_grad=True, device='cuda')
    >>> res = circuit(weights)
    >>> res.backward()
    >>> print(weights.grad)
    tensor([-2.2527e-01, -1.0086e+00,  2.9919e-17], device='cuda:0')


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
        torch_device='cpu' (str): the device on which the computation will be
        run, e.g., ``'cpu'`` or ``'cuda'``
    """

    name = "Default qubit (Torch) PennyLane plugin"
    short_name = "default.qubit.torch"

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
    _real = staticmethod(torch.real)
    _imag = staticmethod(torch.imag)
    _norm = staticmethod(torch.norm)
    _flatten = staticmethod(torch.flatten)
    _const_mul = staticmethod(torch.mul)
    _size = staticmethod(torch.numel)
    _ndim = staticmethod(lambda tensor: tensor.ndim)

    def __init__(self, wires, *, shots=None, analytic=None, torch_device=None):
        # Store if the user specified a Torch device. Otherwise the execute
        # method attempts to infer the Torch device from the gate parameters.
        self._torch_device_specified = torch_device is not None
        self._torch_device = torch_device

        r_dtype = torch.float64
        c_dtype = torch.complex128

        super().__init__(wires, r_dtype=r_dtype, c_dtype=c_dtype, shots=shots, analytic=analytic)

        # Move state to torch device (e.g. CPU, GPU, XLA, ...)
        self._state.requires_grad = True
        self._state = self._state.to(self._torch_device)
        self._pre_rotated_state = self._state

    @staticmethod
    def _get_parameter_torch_device(ops):
        """An auxiliary function to determine the Torch device specified for
        the gate parameters of the input operations.

        Returns the first CUDA Torch device found (if any) using a string
        format. Does not handle tensors put on multiple CUDA Torch devices.
        Such a case raises an error with Torch.

        If CUDA is not used with any of the parameters, then specifies the CPU
        if the parameters are on the CPU or None if there were no parametric
        operations.

        Args:
            ops (list[Operator]): list of operations to check

        Returns:
            str or None: The string of the Torch device determined or None if
            there is no data for any operations.
        """
        par_torch_device = None
        for op in ops:
            for data in op.data:
                # Using hasattr in case we don't have a Torch tensor as input
                if hasattr(data, "is_cuda"):
                    if data.is_cuda:  # pragma: no cover
                        return ":".join([data.device.type, str(data.device.index)])

                    par_torch_device = "cpu"

        return par_torch_device

    def execute(self, circuit, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Entry with args=(circuit=%s, kwargs=%s) called by=%s",
                circuit,
                kwargs,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        par_torch_device = self._get_parameter_torch_device(circuit.operations)

        if not self._torch_device_specified:
            self._torch_device = par_torch_device

            # If we've changed the device of the parameters between device
            # executions, need to move the state to the correct Torch device
            if self._state.device != self._torch_device:
                self._state = self._state.to(self._torch_device)
        else:
            if par_torch_device is not None:  # pragma: no cover
                params_cuda_device = "cuda" in par_torch_device
                specified_device_cuda = "cuda" in self._torch_device

                # Raise a warning if there's a mismatch between the specified and
                # used Torch devices
                if params_cuda_device != specified_device_cuda:
                    warnings.warn(
                        f"Torch device {self._torch_device} specified "
                        "upon PennyLane device creation does not match the "
                        "Torch device of the gate parameters; "
                        f"{self._torch_device} will be used."
                    )

        return super().execute(circuit, **kwargs)

    def _asarray(self, a, dtype=None):
        if isinstance(a, list):
            # Handle unexpected cases where we don't have a list of tensors
            if not isinstance(a[0], torch.Tensor):
                res = np.asarray(a)
                res = torch.from_numpy(res)
                res = torch.cat([torch.reshape(i, (-1,)) for i in res], dim=0)
            elif len(a) == 1 and len(a[0].shape) > 1:
                res = a[0]
            else:
                res = torch.cat([torch.reshape(i, (-1,)) for i in a], dim=0)
                res = torch.cat([torch.reshape(i, (-1,)) for i in res], dim=0)
        else:
            res = torch.as_tensor(a, dtype=dtype)

        res = torch.as_tensor(res, device=self._torch_device)
        return res

    _cast = _asarray

    @staticmethod
    def _dot(x, y):
        if x.device != y.device:
            # GPU-specific cases
            if x.device != "cpu":  # pragma: no cover
                return torch.tensordot(x, y.to(x.device), dims=1)
            if y.device != "cpu":  # pragma: no cover
                return torch.tensordot(x.to(y.device), y, dims=1)

        return torch.tensordot(x, y, dims=1)

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
        capabilities.update(passthru_interface="torch")
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
        if unitary in diagonal_in_z_basis:
            return self._asarray(unitary.eigvals(), dtype=self.C_DTYPE)
        return self._asarray(unitary.matrix(), dtype=self.C_DTYPE)

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
