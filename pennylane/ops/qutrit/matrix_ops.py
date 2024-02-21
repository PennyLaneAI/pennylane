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
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    **Example**

    >>> dev = qml.device('default.qutrit', wires=1)
    >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QutritUnitary(U, wires=0)
    ...     return qml.state()
    >>> print(example_circuit())
    [0.70710678+0.j 0.70710678+0.j 0.        +0.j]
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, *params, wires):
        wires = Wires(wires)

        # For pure QutritUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        if not isinstance(self, ControlledQutritUnitary):
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
                    f"Operator {U}\n may not be unitary. "
                    "Verify unitarity of operation, or use a datatype with increased precision.",
                    UserWarning,
                )

        super().__init__(*params, wires=wires)

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

        **Example**

        >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        >>> qml.QutritUnitary.compute_matrix(U)
        array([[ 0.70710678,  0.70710678,  0.        ],
               [ 0.70710678, -0.70710678,  0.        ],
               [ 0.        ,  0.        ,  1.        ]])
        """
        return U

    def adjoint(self):
        U = self.matrix()
        return QutritUnitary(qml.math.conj(qml.math.moveaxis(U, -2, -1)), wires=self.wires)

    # TODO: Add compute_decomposition() once parametrized operations are added.

    def pow(self, z):
        if isinstance(z, int):
            return [QutritUnitary(qml.math.linalg.matrix_power(self.matrix(), z), wires=self.wires)]
        return super().pow(z)

    def _controlled(self, wire):
        return ControlledQutritUnitary(*self.parameters, control_wires=wire, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


class ControlledQutritUnitary(QutritUnitary):
    r"""ControlledQutritUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQutritUnitary``:

    * ``control_wires``: wires that act as control for the operation
    * ``U``: unitary applied to the target wires. Accessible via ``op.parameters[0]``
    * ``control_values``: a string of trits representing the state of the control
      qutrits to control on (default is the all 2s state)

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the unitary acts on
        control_values (str): a string of trits representing the state of the control
            qutrits to control on (default is the all 2s state)

    **Example**

    The following shows how a single-qutrit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    >>> qml.ControlledQutritUnitary(U, control_wires=[0, 1], wires=2)

    By default, controlled operations apply the desired gate if the control qutrit(s)
    are all in the state :math:`\vert 2\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all control qutrits being in the
    :math:`\vert 0\rangle` or :math:`\vert 1\rangle` state, or a mix of the three.

    The state on which to control can be changed by passing a string of trits to
    ``control_values``. For example, if we want to apply a single-qutrit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``2``, we can write:

    >>> qml.ControlledQutritUnitary(U, control_wires=[0, 1, 2], wires=3, control_values='012')
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, *params, control_wires=None, wires=None, control_values=None):
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
        super().__init__(*params, wires=total_wires)

    @staticmethod
    def compute_matrix(
        U, control_wires, u_wires, control_values=None
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Args:
            U (tensor_like): unitary matrix
            control_wires (Iterable): the control wire(s)
            u_wires (Iterable): the wire(s) the unitary acts on
            control_values (str or None): a string of trits representing the state of the control
                qutrits to control on (default is the all 2s state)

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        >>> qml.ControlledQutritUnitary.compute_matrix(U, control_wires=[0], u_wires=[1], control_values="1")
        array([[ 1.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  1.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  1.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,  0.70710678+0.j,  0.70710678+0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,  0.70710678+0.j, -0.70710678+0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  1.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  1.        +0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  1.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  0.        +0.j,  1.        +0.j]])
        """
        target_dim = 3 ** len(u_wires)
        shape = qml.math.shape(U)
        if not (len(shape) in {2, 3} and shape[-2:] == (target_dim, target_dim)):
            raise ValueError(
                f"Input unitary must be of shape {(target_dim, target_dim)} or "
                f"(batch_size, {target_dim}, {target_dim})."
            )

        # A multi-controlled operation is a block-diagonal matrix partitioned into
        # blocks where the operation being applied sits in the block positioned at
        # the integer value of the control string.

        total_wires = qml.wires.Wires(control_wires) + qml.wires.Wires(u_wires)

        # if control values unspecified, we control on the all-twos string
        if not control_values:
            control_values = "2" * len(control_wires)

        if isinstance(control_values, str):
            if len(control_values) != len(control_wires):
                raise ValueError(
                    "Length of control trit string must equal number of control wires."
                )

            # Make sure all values are either 0 or 1 or 2
            if not set(control_values).issubset({"0", "1", "2"}):
                raise ValueError("String of control values can contain only '0' or '1' or '2'.")

            control_int = int(control_values, 3)
        else:
            raise ValueError("Alternative control values must be passed as a ternary string.")

        padding_left = control_int * target_dim
        padding_right = 3 ** len(total_wires) - target_dim - padding_left

        interface = qml.math.get_interface(U)
        left_pad = qml.math.cast_like(qml.math.eye(padding_left, like=interface), 1j)
        right_pad = qml.math.cast_like(qml.math.eye(padding_right, like=interface), 1j)
        if len(qml.math.shape(U)) == 3:
            return qml.math.stack([qml.math.block_diag([left_pad, _U, right_pad]) for _U in U])
        return qml.math.block_diag([left_pad, U, right_pad])

    @property
    def control_wires(self):
        return self.hyperparameters["control_wires"]

    @property
    def control_values(self):
        """str. Specifies whether or not to control on zero "0", one "1", or two "2" for each
        control wire."""
        return self.hyperparameters["control_values"]

    def pow(self, z):
        if isinstance(z, int):
            return [
                ControlledQutritUnitary(
                    qml.math.linalg.matrix_power(self.data[0], z),
                    control_wires=self.control_wires,
                    wires=self.hyperparameters["u_wires"],
                    control_values=self.hyperparameters["control_values"],
                )
            ]
        return super().pow(z)

    def adjoint(self):
        return ControlledQutritUnitary(
            qml.math.conj(qml.math.moveaxis(self.data[0], -2, -1)),
            control_wires=self.control_wires,
            wires=self.hyperparameters["u_wires"],
            control_values=self.hyperparameters["control_values"],
        )

    def _controlled(self, wire):
        ctrl_wires = self.control_wires + wire
        old_control_values = self.hyperparameters["control_values"]
        values = None if old_control_values is None else f"{old_control_values}2"
        return ControlledQutritUnitary(
            *self.parameters,
            control_wires=ctrl_wires,
            wires=self.hyperparameters["u_wires"],
            control_values=values,
        )
