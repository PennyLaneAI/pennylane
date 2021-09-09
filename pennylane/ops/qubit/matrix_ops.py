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
"""
This submodule contains the discrete-variable quantum operations that
accept a hermitian or an unitary matrix as a parameter.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import warnings
import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.operation import AnyWires, DiagonalOperation, Operation
from pennylane.wires import Wires


class QubitUnitary(Operation):
    r"""QubitUnitary(U, wires)
    Apply an arbitrary fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"
    grad_method = None

    def __init__(self, *params, wires, do_queue=True):
        wires = Wires(wires)

        # For pure QubitUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        if not isinstance(self, ControlledQubitUnitary):
            U = params[0]

            dim = 2 ** len(wires)

            if U.shape != (dim, dim):
                raise ValueError(
                    f"Input unitary must be of shape {(dim, dim)} to act on {len(wires)} wires."
                )

            # Check for unitarity; due to variable precision across the different ML frameworks,
            # here we issue a warning to check the operation, instead of raising an error outright.
            if not qml.math.allclose(
                qml.math.dot(U, qml.math.T(qml.math.conj(U))),
                qml.math.eye(qml.math.shape(U)[0]),
            ):
                warnings.warn(
                    f"Operator {U}\n may not be unitary."
                    "Verify unitarity of operation, or use a datatype with increased precision.",
                    UserWarning,
                )

        super().__init__(*params, wires=wires, do_queue=do_queue)

    @classmethod
    def _matrix(cls, *params):
        return params[0]

    @staticmethod
    def decomposition(U, wires):
        # Decomposes arbitrary single-qubit unitaries as Rot gates (RZ - RY - RZ format),
        # or a single RZ for diagonal matrices.
        if qml.math.shape(U) == (2, 2):
            wire = Wires(wires)[0]
            decomp_ops = qml.transforms.decompositions.zyz_decomposition(U, wire)
            return decomp_ops

        raise NotImplementedError("Decompositions only supported for single-qubit unitaries")

    def adjoint(self):
        return QubitUnitary(qml.math.T(qml.math.conj(self.matrix)), wires=self.wires)

    def _controlled(self, wire):
        ControlledQubitUnitary(*self.parameters, control_wires=wire, wires=self.wires)


class ControlledQubitUnitary(QubitUnitary):
    r"""ControlledQubitUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQubitUnitary``:

    * ``control_wires``: wires that act as control for the operation
    * ``U``: unitary applied to the target wires

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the unitary acts on
        control_values (str): a string of bits representing the state of the control
            qubits to control on (default is the all 1s state)

    **Example**

    The following shows how a single-qubit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1], wires=2)

    Typically controlled operations apply a desired gate if the control qubits
    are all in the state :math:`\vert 1\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all qubits being in the
    :math:`\vert 0\rangle` state, or a mix of the two.

    The state on which to control can be changed by passing a string of bits to
    `control_values`. For example, if we want to apply a single-qubit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``1``, we can write:

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values='011')

    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"
    grad_method = None

    def __init__(
        self,
        *params,
        control_wires=None,
        wires=None,
        control_values=None,
        do_queue=True,
    ):
        if control_wires is None:
            raise ValueError("Must specify control wires")

        wires = Wires(wires)
        control_wires = Wires(control_wires)

        if Wires.shared_wires([wires, control_wires]):
            raise ValueError(
                "The control wires must be different from the wires specified to apply the unitary on."
            )

        U = params[0]
        target_dim = 2 ** len(wires)
        if len(U) != target_dim:
            raise ValueError(f"Input unitary must be of shape {(target_dim, target_dim)}")

        # Saving for the circuit drawer
        self._target_wires = wires
        self._control_wires = control_wires
        self.U = U

        wires = control_wires + wires

        # If control values unspecified, we control on the all-ones string
        if not control_values:
            control_values = "1" * len(control_wires)

        control_int = self._parse_control_values(control_wires, control_values)
        self.control_values = control_values

        # A multi-controlled operation is a block-diagonal matrix partitioned into
        # blocks where the operation being applied sits in the block positioned at
        # the integer value of the control string. For example, controlling a
        # unitary U with 2 qubits will produce matrices with block structure
        # (U, I, I, I) if the control is on bits '00', (I, U, I, I) if on bits '01',
        # etc. The positioning of the block is controlled by padding the block diagonal
        # to the left and right with the correct amount of identity blocks.

        self._padding_left = control_int * len(U)
        self._padding_right = 2 ** len(wires) - len(U) - self._padding_left
        self._CU = None

        super().__init__(*params, wires=wires, do_queue=do_queue)

    def _matrix(self, *params):
        if self._CU is None:
            self._CU = block_diag(np.eye(self._padding_left), self.U, np.eye(self._padding_right))

        params = list(params)
        params[0] = self._CU
        return super()._matrix(*params)

    @property
    def control_wires(self):
        return self._control_wires

    @staticmethod
    def _parse_control_values(control_wires, control_values):
        """Ensure any user-specified control strings have the right format."""
        if isinstance(control_values, str):
            if len(control_values) != len(control_wires):
                raise ValueError("Length of control bit string must equal number of control wires.")

            # Make sure all values are either 0 or 1
            if any(x not in ["0", "1"] for x in control_values):
                raise ValueError("String of control values can contain only '0' or '1'.")

            control_int = int(control_values, 2)
        else:
            raise ValueError("Alternative control values must be passed as a binary string.")

        return control_int

    def _controlled(self, wire):
        ctrl_wires = sorted(self.control_wires + wire)
        ControlledQubitUnitary(*self.parameters, control_wires=ctrl_wires, wires=self._target_wires)


class DiagonalQubitUnitary(DiagonalOperation):
    r"""DiagonalQubitUnitary(D, wires)
    Apply an arbitrary fixed diagonal unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        D (array[complex]): diagonal of unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"
    grad_method = None

    @classmethod
    def _eigvals(cls, *params):
        D = qml.math.asarray(params[0])

        if not qml.math.allclose(D * D.conj(), qml.math.ones_like(D)):
            raise ValueError("Operator must be unitary.")

        return D

    @staticmethod
    def decomposition(D, wires):
        return [QubitUnitary(qml.math.diag(D), wires=wires)]

    def adjoint(self):
        return DiagonalQubitUnitary(qml.math.conj(self.parameters[0]), wires=self.wires)

    def _controlled(self, control):
        DiagonalQubitUnitary(
            qml.math.concatenate([np.array([1, 1]), self.parameters[0]]),
            wires=Wires(control) + self.wires,
        )
