# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule contains the qutrit quantum operations that
accept a unitary matrix as a parameter.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import warnings

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires


class QutritUnitary(Operation):
    r"""Apply an arbitrary, fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires(Sequence[int] or int): the wire(s) the operation acts on
    """
    # TODO: Add example to doc-string after devices are added

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, *params, wires, do_queue=True):
        wires = Wires(wires)

        # For pure QutritUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        U = params[0]
        U_shape = qml.math.shape(U)

        dim = 3 ** len(wires)

        if not (len(U_shape) in {2, 3} and U_shape[-2:] == (dim, dim)):
            raise ValueError(
                f"Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) "
                f"to act on {len(wires)} wires."
            )

        # Check for unitarity; due to variable precision across the different ML frameworks,
        # here we issue a warning to check the operation, instead of raising an error outright.
        if not (
            qml.math.is_abstract(U)
            or qml.math.allclose(
                qml.math.einsum("...ij,...kj->...ik", U, qml.math.conj(U)),
                qml.math.eye(dim),
                atol=1e-6,
            )
        ):
            warnings.warn(
                f"Operator {U}\n may not be unitary."
                "Verify unitarity of operation, or use a datatype with increased precision.",
                UserWarning,
            )

        super().__init__(*params, wires=wires, do_queue=do_queue)

    @staticmethod
    def compute_matrix(U):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.QutritUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix
        """
        return U

    def adjoint(self):
        U = self.matrix()
        return QutritUnitary(qml.math.conj(qml.math.moveaxis(U, -2, -1)), wires=self.wires)

    # TODO: Add `_controlled()` method once `ControlledQutritUnitary` is implemented.
    # TODO: Add compute_decomposition() once parametrized operations are added.

    def pow(self, z):
        if isinstance(z, int):
            return [QutritUnitary(qml.math.linalg.matrix_power(self.matrix(), z), wires=self.wires)]
        return super().pow(z)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)
