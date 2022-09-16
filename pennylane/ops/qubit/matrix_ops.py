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
# pylint:disable=arguments-differ
import warnings

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, DecompositionUndefinedError, Operation
from pennylane.wires import Wires


class QubitUnitary(Operation):
    r"""QubitUnitary(U, wires)
    Apply an arbitrary fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QubitUnitary(U, wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(example_circuit())
    0.0
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, *params, wires, do_queue=True, unitary_check=True):
        wires = Wires(wires)

        # For pure QubitUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        if not isinstance(self, ControlledQubitUnitary):
            U = params[0]
            U_shape = qml.math.shape(U)

            dim = 2 ** len(wires)

            if len(U_shape) not in {2, 3} or U_shape[-2:] != (dim, dim):
                raise ValueError(
                    f"Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) "
                    + "to act on {len(wires)} wires."
                )

            # Check for unitarity; due to variable precision across the different ML frameworks,
            # here we issue a warning to check the operation, instead of raising an error outright.
            if unitary_check and not (
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

        .. seealso:: :meth:`~.QubitUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[0.98877108+0.j, 0.-0.14943813j], [0.-0.14943813j, 0.98877108+0.j]])
        >>> qml.QubitUnitary.compute_matrix(U)
        [[0.98877108+0.j, 0.-0.14943813j],
        [0.-0.14943813j, 0.98877108+0.j]]
        """
        return U

    @staticmethod
    def compute_decomposition(U, wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        A decomposition is only defined for matrices that act on either one or two wires. For more
        than two wires, this method raises a ``DecompositionUndefined``.

        See :func:`~.transforms.zyz_decomposition` and :func:`~.transforms.two_qubit_decomposition`
        for more information on how the decompositions are computed.

        .. seealso:: :meth:`~.QubitUnitary.decomposition`.

        Args:
            U (array[complex]): square unitary matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        >>> qml.QubitUnitary.compute_decomposition(U, 0)
        [Rot(tensor(3.14159265, requires_grad=True), tensor(1.57079633, requires_grad=True), tensor(0., requires_grad=True), wires=[0])]

        """
        # Decomposes arbitrary single-qubit unitaries as Rot gates (RZ - RY - RZ format),
        # or a single RZ for diagonal matrices.
        shape = qml.math.shape(U)
        if shape == (2, 2):
            return qml.transforms.decompositions.zyz_decomposition(U, Wires(wires)[0])

        if shape == (4, 4):
            return qml.transforms.two_qubit_decomposition(U, Wires(wires))

        # TODO[dwierichs]: Implement decomposition of broadcasted unitary
        if len(shape) == 3:
            raise DecompositionUndefinedError(
                "The decomposition of QubitUnitary does not support broadcasting."
            )

        return super(QubitUnitary, QubitUnitary).compute_decomposition(U, wires=wires)

    def adjoint(self):
        U = self.matrix()
        return QubitUnitary(qml.math.moveaxis(qml.math.conj(U), -2, -1), wires=self.wires)

    def pow(self, z):
        if isinstance(z, int):
            return [QubitUnitary(qml.math.linalg.matrix_power(self.matrix(), z), wires=self.wires)]
        return super().pow(z)

    def _controlled(self, wire):
        new_op = ControlledQubitUnitary(*self.parameters, control_wires=wire, wires=self.wires)
        return new_op.inv() if self.inverse else new_op

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


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
    * Number of dimensions per parameter: (2,)
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
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

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

        self._hyperparameters = {
            "u_wires": wires,
            "control_wires": control_wires,
            "control_values": control_values,
        }

        total_wires = control_wires + wires
        super().__init__(*params, wires=total_wires, do_queue=do_queue)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        raise DecompositionUndefinedError

    @staticmethod
    def compute_matrix(
        U, control_wires, u_wires, control_values=None
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ControlledQubitUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix
            control_wires (Iterable): the control wire(s)
            u_wires (Iterable): the wire(s) the unitary acts on
            control_values (str or None): a string of bits representing the state of the control
                qubits to control on (default is the all 1s state)

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
        >>> qml.ControlledQubitUnitary.compute_matrix(U, control_wires=[1], u_wires=[0], control_values="1")
        [[ 1.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j]
         [ 0.        +0.j  1.        +0.j  0.        +0.j  0.        +0.j]
         [ 0.        +0.j  0.        +0.j  0.94877869+0.j  0.31594146+0.j]
         [ 0.        +0.j  0.        +0.j -0.31594146+0.j  0.94877869+0.j]]
        """
        target_dim = 2 ** len(u_wires)
        shape = qml.math.shape(U)
        if not (len(shape) in {2, 3} and shape[-2:] == (target_dim, target_dim)):
            raise ValueError(
                f"Input unitary must be of shape {(target_dim, target_dim)} or "
                f"(batch_size, {target_dim}, {target_dim})."
            )

        # A multi-controlled operation is a block-diagonal matrix partitioned into
        # blocks where the operation being applied sits in the block positioned at
        # the integer value of the control string. For example, controlling a
        # unitary U with 2 qubits will produce matrices with block structure
        # (U, I, I, I) if the control is on bits '00', (I, U, I, I) if on bits '01',
        # etc. The positioning of the block is controlled by padding the block diagonal
        # to the left and right with the correct amount of identity blocks.

        total_wires = qml.wires.Wires(control_wires) + qml.wires.Wires(u_wires)

        # if control values unspecified, we control on the all-ones string
        if not control_values:
            control_values = "1" * len(control_wires)

        if isinstance(control_values, str):
            if len(control_values) != len(control_wires):
                raise ValueError("Length of control bit string must equal number of control wires.")

            # Make sure all values are either 0 or 1
            if not set(control_values).issubset({"0", "1"}):
                raise ValueError("String of control values can contain only '0' or '1'.")

            control_int = int(control_values, 2)
        else:
            raise ValueError("Alternative control values must be passed as a binary string.")

        padding_left = control_int * target_dim
        padding_right = 2 ** len(total_wires) - target_dim - padding_left

        interface = qml.math.get_interface(U)
        left_pad = qml.math.cast_like(qml.math.eye(padding_left, like=interface), 1j)
        right_pad = qml.math.cast_like(qml.math.eye(padding_right, like=interface), 1j)
        if len(qml.math.shape(U)) == 3:
            return qml.math.stack([qml.math.block_diag([left_pad, _U, right_pad]) for _U in U])
        return qml.math.block_diag([left_pad, U, right_pad])

    @property
    def control_wires(self):
        return self.hyperparameters["control_wires"]

    def pow(self, z):
        if isinstance(z, int):
            return [
                ControlledQubitUnitary(
                    qml.math.linalg.matrix_power(self.data[0], z),
                    control_wires=self.control_wires,
                    wires=self.hyperparameters["u_wires"],
                )
            ]
        return super().pow(z)

    def _controlled(self, wire):
        ctrl_wires = self.control_wires + wire
        new_op = ControlledQubitUnitary(
            *self.parameters, control_wires=ctrl_wires, wires=self.hyperparameters["u_wires"]
        )
        return new_op.inv() if self.inverse else new_op


class DiagonalQubitUnitary(Operation):
    r"""DiagonalQubitUnitary(D, wires)
    Apply an arbitrary fixed diagonal unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe: None

    Args:
        D (array[complex]): diagonal of unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    @staticmethod
    def compute_matrix(D):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.matrix`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_matrix(torch.tensor([1, -1]))
        tensor([[ 1,  0],
                [ 0, -1]])
        """
        D = qml.math.asarray(D)

        if not qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D)):
            raise ValueError("Operator must be unitary.")

        # The diagonal is supposed to have one-dimension. If it is broadcasted, it has two
        if qml.math.ndim(D) == 2:
            return qml.math.stack([qml.math.diag(_D) for _D in D])

        return qml.math.diag(D)

    @staticmethod
    def compute_eigvals(D):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.eigvals`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_eigvals(torch.tensor([1, -1]))
        tensor([ 1, -1])
        """
        D = qml.math.asarray(D)

        if not (
            qml.math.is_abstract(D)
            or qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D))
        ):
            raise ValueError("Operator must be unitary.")

        return D

    @staticmethod
    def compute_decomposition(D, wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        ``DiagonalQubitUnitary`` decomposes into :class:`~.QubitUnitary`, which has further
        decompositions for one and two qubit matrices.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.decomposition`.

        Args:
            U (array[complex]): square unitary matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.DiagonalQubitUnitary.compute_decomposition([1, 1], wires=0)
        [QubitUnitary(array([[1, 0], [0, 1]]), wires=[0])]

        """
        return [QubitUnitary(DiagonalQubitUnitary.compute_matrix(D), wires=wires)]

    def adjoint(self):
        return DiagonalQubitUnitary(qml.math.conj(self.parameters[0]), wires=self.wires)

    def pow(self, z):
        if isinstance(self.data[0], list):
            if isinstance(self.data[0][0], list):
                # Support broadcasted list
                new_data = [[(el + 0j) ** z for el in x] for x in self.data[0]]
            else:
                new_data = [(x + 0.0j) ** z for x in self.data[0]]
            return [DiagonalQubitUnitary(new_data, wires=self.wires)]
        casted_data = qml.math.cast(self.data[0], np.complex128)
        return [DiagonalQubitUnitary(casted_data**z, wires=self.wires)]

    def _controlled(self, control):
        new_op = DiagonalQubitUnitary(
            qml.math.hstack([np.ones_like(self.parameters[0]), self.parameters[0]]),
            wires=control + self.wires,
        )
        return new_op.inv() if self.inverse else new_op

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)
