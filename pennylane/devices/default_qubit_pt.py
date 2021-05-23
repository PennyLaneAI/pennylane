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
import numpy as np
import semantic_version

from pennylane.operation import DiagonalOperation

try:
    import torch

    if torch.__version__[0:3] != "1.8":
        raise ImportError("default.qubit.torch device requires PyTorch>=1.8")


    SUPPORTS_APPLY_OPS = semantic_version.match(">=1.8.1", torch.__version__)

except ImportError as e:
    raise ImportError("default.qubit.torch device requires PyTorch>=2.0") from e

try:
    from torch import einsum
except ImportError:
    pass

from . import DefaultQubit
from . import pt_ops


class DefaultQubitPT(DefaultQubit):
    """Simulator plugin based on ``"default.qubit"``, written using PyTorch.

    **Short name:** ``default.qubit.torch``

    This device provides a pure-state qubit simulator written using PyTorch.
    As a result, it supports classical backpropagation as a means to compute the Jacobian. This can
    be faster than the parameter-shift rule for analytic quantum gradients
    when the number of parameters to be optimized is large.

    To use this device, you will need to install PyTorch:

    .. code-block:: console

        pip3 install pytorch>=1.8

    **Example**

    The ``default.qubit.torch`` is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the PyTorch interface. This is the default method
    of differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your PyTorch computation:

    >>> dev = qml.device("default.qubit.torch", wires=1)
    >>> @qml.qnode(dev, interface="pt", diff_method="backprop")
    ... def circuit(x):
    ...     qml.RX(x[1], wires=0)
    ...     qml.Rot(x[0], x[1], x[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0))

    ## TODO:  need to test all code samples
    >>> weights = tf.Variable([0.2, 0.5, 0.1])
    >>> with tf.GradientTape() as tape:
    ...     res = circuit(weights)
    >>> print(tape.gradient(res, weights))
    tf.Tensor([-2.2526717e-01 -1.0086454e+00  1.3877788e-17], shape=(3,), dtype=float32)

    Autograph mode will also work when using classical backpropagation:

    >>> @tf.function
    ... def cost(weights):
    ...     return tf.reduce_sum(circuit(weights)**3) - 1
    >>> with tf.GradientTape() as tape:
    ...     res = cost(weights)
    >>> print(tape.gradient(res, weights))
    tf.Tensor([-3.5471588e-01 -1.5882589e+00  3.4694470e-17], shape=(3,), dtype=float32)

    If you wish to use a different machine-learning interface, or prefer to calculate quantum
    gradients using the ``parameter-shift`` or ``finite-diff`` differentiation methods,
    consider using the ``default.qubit`` device instead.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means
            that the device returns analytical results.
            If ``shots > 0`` is used, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
    """

    name = "Default qubit (PyTorch) PennyLane plugin"
    short_name = "default.qubit.torch"

    parametric_ops = {
        "PhaseShift": pt_ops.PhaseShift,
        "ControlledPhaseShift": pt_ops.ControlledPhaseShift,
        "RX": pt_ops.RX,
        "RY": pt_ops.RY,
        "RZ": pt_ops.RZ,
        "Rot": pt_ops.Rot,
        "MultiRZ": pt_ops.MultiRZ,
        "CRX": pt_ops.CRX,
        "CRY": pt_ops.CRY,
        "CRZ": pt_ops.CRZ,
        "CRot": pt_ops.CRot,
        "SingleExcitation": pt_ops.SingleExcitation,
        "SingleExcitationPlus": pt_ops.SingleExcitationPlus,
        "SingleExcitationMinus": pt_ops.SingleExcitationMinus,
        "DoubleExcitation": pt_ops.DoubleExcitation,
        "DoubleExcitationPlus": pt_ops.DoubleExcitationPlus,
        "DoubleExcitationMinus": pt_ops.DoubleExcitationMinus,
    }

    C_DTYPE = torch.complex128
    R_DTYPE = torch.float64
    _asarray = staticmethod(torch.as_tensor)
    _dot = staticmethod(lambda x, y: torch.tensordot(x, y, axis=1))
    _abs = staticmethod(torch.abs)
    _reduce_sum = staticmethod(torch.sum)
    _reshape = staticmethod(torch.reshape)
    _flatten = staticmethod(lambda tensor: torch.reshape(tensor, [-1]))
    _gather = staticmethod(torch.gather)
    _einsum = staticmethod(torch.einsum)
    #_cast = staticmethod(tf.cast)
    _transpose = staticmethod(torch.transpose)
    _tensordot = staticmethod(torch.tensordot)
    _conj = staticmethod(torch.conj)
    _imag = staticmethod(torch.imag)
    _roll = staticmethod(torch.roll)
    _stack = staticmethod(torch.stack)

    @staticmethod
    def _asarray(array, dtype=None):
        try:
            res = torch.as_tensor(array)
            if dtype is not None:
                res = res.type(dtype)
        except TypeError as e:
            print("Invalid Argument given to function call \n",e)
        return res

    def __init__(self, wires, *, shots=None, analytic=None):
        super().__init__(wires, shots=shots, cache=0, analytic=analytic)

        # # TODO: Need to test this in Torch Env.
        del self._apply_ops["CZ"]

        # # TODO: Need to test this in Torch Env.
        if not SUPPORTS_APPLY_OPS or self.num_wires > 8:
            self._apply_ops = {}

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            passthru_interface="pt",
            supports_reversible_diff=False,
        )
        return capabilities


    @staticmethod
    def _scatter(indices, array, new_dimensions):
        indices = np.expand_dims(indices, 1)
        return torch.sparse_coo_tensor(indices, array, new_dimensions)

    def _get_unitary_matrix(self, unitary):
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            torch.tensor[complex] or array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis, or, in the case of a diagonal unitary,
            a 1D array representing the matrix diagonal. For non-parametric unitaries,
            the return type will be a ``np.ndarray``. For parametric unitaries, a ``tf.Tensor``
            object will be returned.
        """
        op_name = unitary.name.split(".inv")[0]

        if op_name in self.parametric_ops:
            if op_name == "MultiRZ":
                mat = self.parametric_ops[op_name](*unitary.parameters, len(unitary.wires))
            else:
                mat = self.parametric_ops[op_name](*unitary.parameters)

            if unitary.inverse:
                mat = self._transpose(self._conj(mat))

            return mat

        if isinstance(unitary, DiagonalOperation):
            return unitary.eigvals

        return unitary.matrix
