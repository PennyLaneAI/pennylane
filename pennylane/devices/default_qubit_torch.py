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
"""This module contains a PyTorch implementation of the :class:`~.DefaultQubit`
reference plugin.
"""

try:
    import torch

    # TODO test version compatibility (see default_qubit_tf.py)

except ImportError as e:
    raise ImportError("default.qubit.torch device requires Torch") from e

from pennylane.operation import DiagonalOperation
from pennylane.devices import torch_ops
from . import DefaultQubit

class DefaultQubitTorch(DefaultQubit):
    # TODO docstring

    name = "Default qubit (Torch) PennyLane plugin"
    short_name = "default.qubit.torch"

    parametric_ops = {
        "PhaseShift": torch_ops.PhaseShift,
        "RX": torch_ops.RX,
        "RY": torch_ops.RY,
        "RZ": torch_ops.RZ,
        "Rot": torch_ops.Rot,
        "CRX": torch_ops.CRX,
        "CRY": torch_ops.CRY,
        "CRZ": torch_ops.CRZ,
    }

    C_DTYPE = torch.complex128
    R_DTYPE = torch.float64

    # TODO test numpy -> torch interface mappings for all kwargs
    _abs = staticmethod(torch.abs)
    _dot = staticmethod(torch.dot)
    _einsum = staticmethod(torch.einsum)
    _flatten = staticmethod(torch.flatten)
    _reduce_sum = staticmethod(torch.sum)
    _reshape = staticmethod(torch.reshape)
    _roll = staticmethod(torch.roll)
    _stack = staticmethod(lambda arrs, axis=0, out=None: torch.stack(arrs, axis=axis, out=out))
    _tensordot = staticmethod(lambda a, b, axes: torch.tensordot(a, b, dims=axes))
    _transpose = staticmethod(lambda a, axes=None: a.permute(*axes))
    _asnumpy = staticmethod(lambda x: x.cpu().numpy())

    def __init__(self, wires, *, shots=1000, analytic=True, torch_device=torch.device('cpu')):
        self._torch_device = torch_device
        super().__init__(wires, shots=shots, analytic=analytic, cache=0)
        
        # Move state to torch device (e.g. CPU, GPU, XLA, ...)
        self._state = self._state.to(self._torch_device)
        self._pre_rotated_state = self._state

    # TODO remove once torch.einsum fully supports compex valued tensors
    def _apply_unitary_einsum(self, mat, wires):
        self._apply_unitary(mat, wires)

    def _asarray(self, a, dtype=None):
        return torch.as_tensor(a, dtype=dtype, device=self._torch_device)

    def _cast(self, a, dtype=None):
        return self._asarray(a, dtype=dtype)

    def _conj(self, inputs):
        inputs = torch.as_tensor(inputs, dtype=self.C_DTYPE, device=self._torch_device)
        return torch.conj(inputs)

    def _zeros(self, shape, dtype=float):
        return torch.zeros(shape, dtype=dtype, device=self._torch_device)

    @staticmethod
    def _ravel_multi_index(multi_index, dims):
        # Idea: ravelling a multi-index can be expressed as a matrix-vector product
        flip = lambda  x: torch.flip(x, dims=[0])

        dims = torch.as_tensor(dims, device=multi_index.device)
        coeffs = torch.ones_like(dims, device=multi_index.device)
        coeffs[:-1] = dims[1:]
        coeffs = flip(torch.cumprod(flip(coeffs), dim=0))

        ravelled_indices = (multi_index.T.type(torch.float) @ coeffs.type(torch.float)).type(torch.long)
        return ravelled_indices
        
    @staticmethod
    def _scatter(indices, array, new_dimensions):

        # `array` is now a torch tensor
        tensor = array

        new_tensor = torch.zeros(new_dimensions, dtype=tensor.dtype, device=tensor.device)
        new_tensor[indices] = tensor
        return new_tensor

    @staticmethod
    def _norm(x, ord=None):

        # TODO consolidate with PyTorch implementation once this is fixed
        # Neither torch.norm nor torch.linalg.norm currently supports complex
        # vector arguments

        if ord and ord != 2:
            raise ValueError('Only 2-norm supported for now')

        return torch.sqrt(x @ x.conj())

    @staticmethod
    def _allclose(a, b, atol=1e-08):
        return torch.allclose(a, torch.as_tensor(b, dtype=a.dtype), atol=atol)

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            passthru_interface="torch",
            supports_reversible_diff=False
        )
        return capabilities

    
    def _get_unitary_matrix(self, unitary):
        # TODO docstring (see default_qubit_tf.py)

        if unitary.name in self.parametric_ops:
            return self.parametric_ops[unitary.name](
                *unitary.parameters,
                device=self._torch_device
            )

        if isinstance(unitary, DiagonalOperation):
            return self._asarray(unitary.eigvals, dtype=self.C_DTYPE)

        return self._asarray(unitary.matrix, dtype=self.C_DTYPE)
