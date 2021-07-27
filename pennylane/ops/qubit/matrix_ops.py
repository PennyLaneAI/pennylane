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
import warnings

# pylint:disable=abstract-method,arguments-differ,protected-access
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix

import pennylane as qml
from pennylane.operation import AnyWires, AllWires, DiagonalOperation, Observable, Operation
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
        D = np.asarray(params[0])

        if not np.allclose(D * D.conj(), np.ones_like(D)):
            raise ValueError("Operator must be unitary.")

        return D

    @staticmethod
    def decomposition(D, wires):
        return [QubitUnitary(np.diag(D), wires=wires)]

    def adjoint(self):
        return DiagonalQubitUnitary(qml.math.conj(self.parameters[0]), wires=self.wires)

    def _controlled(self, control):
        DiagonalQubitUnitary(
            qml.math.concatenate([np.array([1, 1]), self.parameters[0]]),
            wires=Wires(control) + self.wires,
        )


# When this gate no longer depends on ``ControlledQubitUnitary
# please move it to ``non_parametric_ops.py``

class MultiControlledX(ControlledQubitUnitary):
    r"""MultiControlledX(control_wires, wires, control_values)
    Apply a Pauli X gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires or int]): a single target wire the operation acts on
        control_values (str): a string of bits representing the state of the control
            qubits to control on (default is the all 1s state)
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of Toffoli gates

    .. note::

        If ``MultiControlledX`` is not supported on the targeted device, PennyLane will decompose
        the operation into :class:`~.Toffoli` and/or :class:`~.CNOT` gates. When controlling on
        three or more wires, the Toffoli-based decompositions described in Lemmas 7.2 and 7.3 of
        `Barenco et al. <https://arxiv.org/abs/quant-ph/9503016>`__ will be used. These methods
        require at least one work wire.

        The number of work wires provided determines the decomposition method used and the resulting
        number of Toffoli gates required. When ``MultiControlledX`` is controlling on :math:`n`
        wires:

        #. If at least :math:`n - 2` work wires are provided, the decomposition in Lemma 7.2 will be
           applied using the first :math:`n - 2` work wires.
        #. If fewer than :math:`n - 2` work wires are provided, a combination of Lemmas 7.3 and 7.2
           will be applied using only the first work wire.

        These methods present a tradeoff between qubit number and depth. The method in point 1
        requires fewer Toffoli gates but a greater number of qubits.

        Note that the state of the work wires before and after the decomposition takes place is
        unchanged.

    **Example**

    The ``MultiControlledX`` operation (sometimes called a mixed-polarity
    multi-controlled Toffoli) is a commonly-encountered case of the
    :class:`~.pennylane.ControlledQubitUnitary` operation wherein the applied
    unitary is the Pauli X (NOT) gate. It can be used in the same manner as
    ``ControlledQubitUnitary``, but there is no need to specify a matrix
    argument:

    >>> qml.MultiControlledX(control_wires=[0, 1, 2, 3], wires=4, control_values='1110')

    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"
    grad_method = None

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        control_wires=None,
        wires=None,
        control_values=None,
        work_wires=None,
        do_queue=True,
    ):
        wires = Wires(wires)
        control_wires = Wires(control_wires)
        work_wires = Wires([]) if work_wires is None else Wires(work_wires)

        if len(wires) != 1:
            raise ValueError("MultiControlledX accepts a single target wire.")

        if Wires.shared_wires([wires, work_wires]) or Wires.shared_wires(
            [control_wires, work_wires]
        ):
            raise ValueError("The work wires must be different from the control and target wires")

        self._target_wire = wires[0]
        self._work_wires = work_wires

        super().__init__(
            np.array([[0, 1], [1, 0]]),
            control_wires=control_wires,
            wires=wires,
            control_values=control_values,
            do_queue=do_queue,
        )

    # pylint: disable=unused-argument
    def decomposition(self, *args, **kwargs):

        if len(self.control_wires) > 2 and len(self._work_wires) == 0:
            raise ValueError(f"At least one work wire is required to decompose operation: {self}")

        flips1 = [
            qml.PauliX(self.control_wires[i])
            for i, val in enumerate(self.control_values)
            if val == "0"
        ]

        if len(self.control_wires) == 1:
            decomp = [qml.CNOT(wires=[self.control_wires[0], self._target_wire])]
        elif len(self.control_wires) == 2:
            decomp = [qml.Toffoli(wires=[*self.control_wires, self._target_wire])]
        else:
            num_work_wires_needed = len(self.control_wires) - 2

            if len(self._work_wires) >= num_work_wires_needed:
                decomp = self._decomposition_with_many_workers(
                    self.control_wires, self._target_wire, self._work_wires
                )
            else:
                work_wire = self._work_wires[0]
                decomp = self._decomposition_with_one_worker(
                    self.control_wires, self._target_wire, work_wire
                )

        flips2 = [
            qml.PauliX(self.control_wires[i])
            for i, val in enumerate(self.control_values)
            if val == "0"
        ]

        return flips1 + decomp + flips2

    @staticmethod
    def _decomposition_with_many_workers(control_wires, target_wire, work_wires):
        """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
        https://arxiv.org/pdf/quant-ph/9503016.pdf, which requires a suitably large register of
        work wires"""
        num_work_wires_needed = len(control_wires) - 2
        work_wires = work_wires[:num_work_wires_needed]

        work_wires_reversed = list(reversed(work_wires))
        control_wires_reversed = list(reversed(control_wires))

        gates = []

        for i in range(len(work_wires)):
            ctrl1 = control_wires_reversed[i]
            ctrl2 = work_wires_reversed[i]
            t = target_wire if i == 0 else work_wires_reversed[i - 1]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

        for i in reversed(range(len(work_wires))):
            ctrl1 = control_wires_reversed[i]
            ctrl2 = work_wires_reversed[i]
            t = target_wire if i == 0 else work_wires_reversed[i - 1]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        for i in range(len(work_wires) - 1):
            ctrl1 = control_wires_reversed[i + 1]
            ctrl2 = work_wires_reversed[i + 1]
            t = work_wires_reversed[i]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

        for i in reversed(range(len(work_wires) - 1)):
            ctrl1 = control_wires_reversed[i + 1]
            ctrl2 = work_wires_reversed[i + 1]
            t = work_wires_reversed[i]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        return gates

    @staticmethod
    def _decomposition_with_one_worker(control_wires, target_wire, work_wire):
        """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
        https://arxiv.org/pdf/quant-ph/9503016.pdf, which requires a single work wire"""
        tot_wires = len(control_wires) + 2
        partition = int(np.ceil(tot_wires / 2))

        first_part = control_wires[:partition]
        second_part = control_wires[partition:]

        gates = [
            MultiControlledX(
                control_wires=first_part,
                wires=work_wire,
                work_wires=second_part + target_wire,
            ),
            MultiControlledX(
                control_wires=second_part + work_wire,
                wires=target_wire,
                work_wires=first_part,
            ),
            MultiControlledX(
                control_wires=first_part,
                wires=work_wire,
                work_wires=second_part + target_wire,
            ),
            MultiControlledX(
                control_wires=second_part + work_wire,
                wires=target_wire,
                work_wires=first_part,
            ),
        ]

        return gates
