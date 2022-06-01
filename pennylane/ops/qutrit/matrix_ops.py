import warnings
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation, DecompositionUndefinedError
from pennylane.wires import Wires


class QutritUnitary(Operation):
    r"""
    Apply and arbitrary fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires(Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, *params, wires, do_queue=True):
        wires = Wires(wires)

        # For pure QutritUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        if not isinstance(self, ControlledQutritUnitary):
            U = params[0]

            dim = 3 ** len(wires)

            if qml.math.shape(U) != (dim, dim):
                raise ValueError(
                    f"Input unitary must be of shape {(dim, dim)} to act on {len(wires)} wires."
                )

            # Check for unitarity; due to variable precision across the different ML frameworks,
            # here we issue a warning to check the operation, instead of raising an error outright.
            if not qml.math.is_abstract(U) and not qml.math.allclose(
                qml.math.dot(U, qml.math.T(qml.math.conj(U))),
                qml.math.eye(qml.math.shape(U)[0]),
                atol=1e-6,
            ):
                warnings.warn(
                    f"Operator {U}\n may not be unitary."
                    "Verify unitarity of operation, or use a datatype with increased precision.",
                    UserWarning,
                )

        super().__init__(*params, wires=wires, do_queue=do_queue)

    @staticmethod
    def compute_matrix(U):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix
        """
        return U

    @staticmethod
    def compute_decomposition(U, wires):
        raise DecompositionUndefinedError

    def adjoint(self):
        return QutritUnitary(qml.math.T(qml.math.conj(self.matrix())), wires=self.wires)

    def _controlled(self, wire):
        ControlledQutritUnitary(*self.parameters, control_wires=wire, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


class ControlledQutritUnitary(QutritUnitary):
    r"""ControlledQutritUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQutritUnitary``:

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
        control_values (str): a string of trits representing the state of the control
            qutrits to control on (default is the all 2s state)
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

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

    #############################################################################
    #############################################################################
    ########################## TODO: Check correctness ##########################
    #############################################################################
    #############################################################################
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
        """
        target_dim = 3 ** len(u_wires)
        if len(U) != target_dim:
            raise ValueError(f"Input unitary must be of shape {(target_dim, target_dim)}")

        # A multi-controlled operation is a block-diagonal matrix partitioned into
        # blocks where the operation being applied sits in the block positioned at
        # the integer value of the control string.

        total_wires = qml.wires.Wires(control_wires) + qml.wires.Wires(u_wires)

        # if control values unspecified, we control on the all-ones string
        if not control_values:
            control_values = "2" * len(control_wires)

        if isinstance(control_values, str):
            if len(control_values) != len(control_wires):
                raise ValueError("Length of control bit string must equal number of control wires.")

            # Make sure all values are either 0 or 1
            if any(x not in ["0", "1", "2"] for x in control_values):
                raise ValueError("String of control values can contain only '0' or '1' or '2'.")

            control_int = int(control_values, 3)
        else:
            raise ValueError("Alternative control values must be passed as a ternary string.")

        padding_left = control_int * len(U)
        padding_right = 3 ** len(total_wires) - len(U) - padding_left

        interface = qml.math.get_interface(U)
        left_pad = qml.math.cast_like(qml.math.eye(padding_left, like=interface), 1j)
        right_pad = qml.math.cast_like(qml.math.eye(padding_right, like=interface), 1j)
        return qml.math.block_diag([left_pad, U, right_pad])

    @property
    def control_wires(self):
        return self.hyperparameters["control_wires"]

    def _controlled(self, wire):
        ctrl_wires = sorted(self.control_wires + wire)
        ControlledQutritUnitary(
            *self.parameters, control_wires=ctrl_wires, wires=self.hyperparameters["u_wires"]
        )