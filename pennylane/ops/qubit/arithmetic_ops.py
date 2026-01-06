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
This submodule contains the discrete-variable quantum operations that perform
arithmetic operations on their input states.
"""
from collections import Counter

# pylint: disable=arguments-differ
from copy import copy

import numpy as np

import pennylane as qml
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import pow_involutory, self_adjoint
from pennylane.operation import FlatPytree, Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike


class QubitCarry(Operation):
    r"""QubitCarry(wires)
    Apply the ``QubitCarry`` operation to four input wires.

    This operation performs the transformation:

    .. math::
        |a\rangle |b\rangle |c\rangle |d\rangle \rightarrow |a\rangle |b\rangle |b\oplus c\rangle |bc \oplus d\oplus (b\oplus c)a\rangle

    .. figure:: ../../_static/ops/QubitCarry.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    See `here <https://arxiv.org/abs/quant-ph/0008033v1>`__ for more information.

    .. note::
        The first wire should be used to input a carry bit from previous operations. The final wire
        holds the carry bit of this operation and the input state on this wire should be
        :math:`|0\rangle`.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on

    **Example**

    The ``QubitCarry`` operation maps the state :math:`|0110\rangle` to :math:`|0101\rangle`, where
    the last qubit denotes the carry value:

    .. code-block:: python

        import itertools

        input_bitstring = (0, 1, 1, 0)

        @qml.qnode(qml.device("default.qubit"))
        def circuit(basis_state):
            qml.BasisState(basis_state, wires=[0, 1, 2, 3])
            qml.QubitCarry(wires=[0, 1, 2, 3])
            return qml.probs(wires=[0, 1, 2, 3])

        probs =  circuit(input_bitstring)
        probs_indx = np.argwhere(probs == 1).flatten()[0]
        bitstrings = list(itertools.product(range(2), repeat=4))
        output_bitstring = bitstrings[probs_indx]

    The output bitstring is

    >>> output_bitstring
    (0, 1, 0, 1)

    The action of ``QubitCarry`` is to add wires ``1`` and ``2``. The modulo-two result is output
    in wire ``2`` with a carry value output in wire ``3``. In this case, :math:`1 \oplus 1 = 0` with
    a carry, so we have:

    >>> bc_sum = output_bitstring[2]
    >>> bc_sum
    0
    >>> carry = output_bitstring[3]
    >>> carry
    1
    """

    num_wires: int = 4
    """int: Number of wires that the operator acts on."""

    num_params: int = 0
    """int: Number of trainable parameters that the operator depends on."""

    resource_keys = set()

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.QubitCarry.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.QubitCarry.compute_matrix())
        [[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.QubitCarry.decomposition`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> qml.QubitCarry.compute_decomposition((0,1,2,4))
        [Toffoli(wires=[1, 2, 4]), CNOT(wires=[1, 2]), Toffoli(wires=[0, 2, 4])]

        """
        return [
            qml.Toffoli(wires=wires[1:]),
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.Toffoli(wires=[wires[0], wires[2], wires[3]]),
        ]


def _qubitcarry_to_cnot_toffoli_resources():
    return {qml.CNOT: 1, qml.Toffoli: 2}


@register_resources(_qubitcarry_to_cnot_toffoli_resources)
def _qubitcarry_to_cnot_toffolis(wires: WiresLike, **__):
    qml.Toffoli(wires=wires[1:])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.Toffoli(wires=[wires[0], wires[2], wires[3]])


add_decomps(QubitCarry, _qubitcarry_to_cnot_toffolis)


class QubitSum(Operation):
    r"""QubitSum(wires)
    Apply a ``QubitSum`` operation on three input wires.

    This operation performs the following transformation:

    .. math::
        |a\rangle |b\rangle |c\rangle \rightarrow |a\rangle |b\rangle |a\oplus b\oplus c\rangle


    .. figure:: ../../_static/ops/QubitSum.svg
        :align: center
        :width: 40%
        :target: javascript:void(0);

    See `here <https://arxiv.org/abs/quant-ph/0008033v1>`__ for more information.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on

    **Example**

    The ``QubitSum`` operation maps the state :math:`|010\rangle` to :math:`|011\rangle`, with the
    final wire holding the modulo-two sum of the first two wires:

    .. code-block:: python

        import itertools

        input_bitstring = (0, 1, 0)

        @qml.qnode(qml.device("default.qubit"))
        def circuit(basis_state):
            qml.BasisState(basis_state, wires = [0, 1, 2])
            qml.QubitSum(wires=[0, 1, 2])
            return qml.probs(wires=[0, 1, 2])

        probs = circuit(input_bitstring)
        probs_indx = np.argwhere(probs == 1).flatten()[0]
        bitstrings = list(itertools.product(range(2), repeat=3))
        output_bitstring = bitstrings[probs_indx]

    The output bitstring is

    >>> output_bitstring
    (0, 1, 1)

    The action of ``QubitSum`` is to add wires ``0``, ``1``, and ``2``. The modulo-two result is
    output in wire ``2``. In this case, :math:`0 \oplus 1 \oplus 0 = 1`, so we have:

    >>> output_bitstring[2]
    1
    """

    num_wires: int = 3
    """int: Number of wires that the operator acts on."""

    num_params: int = 0
    """int: Number of trainable parameters that the operator depends on."""

    resource_keys = set()

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label=base_label or "Σ", cache=cache)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.QubitSum.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.QubitSum.compute_matrix())
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 1 0]
         [0 0 0 0 0 0 0 1]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> qml.operation.Operator:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.QubitSum.decomposition`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> qml.QubitSum.compute_decomposition((0,1,2))
        [CNOT(wires=[1, 2]), CNOT(wires=[0, 2])]

        """
        decomp_ops = [
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.CNOT(wires=[wires[0], wires[2]]),
        ]
        return decomp_ops

    def adjoint(self):
        return QubitSum(wires=self.wires)


def _qubitsum_to_cnots_resources():
    return {qml.CNOT: 2}


@register_resources(_qubitsum_to_cnots_resources)
def _qubitsum_to_cnots(wires: WiresLike, **__):
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[0], wires[2]])


add_decomps(QubitSum, _qubitsum_to_cnots)
add_decomps("Adjoint(QubitSum)", self_adjoint)
add_decomps("Pow(QubitSum)", pow_involutory)


class IntegerComparator(Operation):
    r"""IntegerComparator(value, geq, wires)
    Apply a controlled Pauli X gate using integer comparison as the condition.

    Given a basis state :math:`\vert n \rangle`, where :math:`n` is a positive integer, and a fixed positive
    integer :math:`L`, flip a target qubit if :math:`n \geq L`. Alternatively, the flipping condition can
    be :math:`n < L`.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        This operation has one parameter: ``value``. However, ``value`` is simply an integer that is required to define
        the condition upon which a Pauli X gate is applied to the target wire. Given that, IntegerComparator has a
        gradient of zero; ``value`` is a non-differentiable parameter.

    Args:
        value (int): The value :math:`L` that the state's decimal representation is compared against.
        geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If ``False``, the comparison
            made will be :math:`n < L`.
        wires (Union[Wires, Sequence[int], or int]): Control wire(s) followed by a single target wire where
            the operation acts on.

    **Example**

    >>> dev = qml.device("default.qubit", wires=3)
    >>> @qml.qnode(dev)
    ... def circuit(state, value, geq):
    ...     qml.BasisState(np.array(state), wires=range(3))
    ...     qml.IntegerComparator(value, geq=geq, wires=range(3))
    ...     return qml.state()
    >>> circuit([1, 0, 1], 1, True).reshape(2, 2, 2)[1, 0, 0]
    np.complex128(1+0j)
    >>> circuit([0, 1, 0], 3, False).reshape(2, 2, 2)[0, 1, 1]
    np.complex128(1+0j)
    """

    is_self_inverse: bool = True
    num_params: int = 0
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None

    resource_keys = {"num_wires", "value", "geq", "num_work_wires"}

    def _flatten(self) -> FlatPytree:
        hp = self.hyperparameters
        metadata = (
            ("work_wires", hp["work_wires"]),
            ("value", hp["value"]),
            ("geq", hp["geq"]),
        )
        return tuple(), (hp["control_wires"] + hp["target_wires"], metadata)

    def __init__(
        self,
        value: int,
        wires: WiresLike,
        geq: bool = True,
        work_wires: WiresLike | None = None,
    ):
        if not isinstance(value, int):
            raise ValueError(f"The compared value must be an int. Got {type(value)}.")

        if wires is None:
            raise ValueError("Must specify wires that the operation acts on.")

        if len(wires) > 1:
            control_wires = Wires(wires[:-1])
            wires = Wires(wires[-1])
        else:
            raise ValueError(
                "IntegerComparator: wrong number of wires. "
                f"{len(wires)} wire(s) given. Need at least 2."
            )

        work_wires = Wires([]) if work_wires is None else Wires(work_wires)
        total_wires = control_wires + wires

        if Wires.shared_wires([total_wires, work_wires]):
            raise ValueError("The work wires must be different from the control and target wires")

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["target_wires"] = wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["value"] = value
        self.hyperparameters["geq"] = geq
        self.geq = geq
        self.value = value

        super().__init__(wires=total_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_wires": len(self.wires),
            "value": self.value,
            "geq": self.geq,
            "num_work_wires": len(self.hyperparameters["work_wires"]),
        }

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ):
        return base_label or f">={self.value}" if self.geq else f"<{self.value}"

    # pylint: disable=unused-argument
    @staticmethod
    def compute_matrix(
        control_wires: WiresLike, value: int | None = None, geq: bool = True, **kwargs
    ) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IntegerComparator.matrix`

        Args:
            control_wires (Union[Wires, Sequence[int], or int]): wires to place controls on
            value (int): The value :math:`L` that the state's decimal representation is compared against.
            geq (bool): If set to `True`, the comparison made will be :math:`n \geq L`. If `False`, the comparison
                made will be :math:`n < L`.

        Returns:
           tensor_like: matrix representation

        **Example**

        >>> print(qml.IntegerComparator.compute_matrix(control_wires=[0, 1], value=2))
        [[1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 0. 1. 0.]]
        >>> print(qml.IntegerComparator.compute_matrix(control_wires=[0, 1], value=2, geq=False))
        [[0. 1. 0. 0. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]]
        """

        if value is None:
            raise ValueError("The value to compare to must be specified.")

        if control_wires is None:
            raise ValueError("Must specify the control wires.")

        if not isinstance(value, int):
            raise ValueError(f"The compared value must be an int. Got {type(value)}.")

        small_val = not geq and value == 0
        large_val = geq and value > 2 ** len(control_wires) - 1
        if small_val or large_val:
            mat = np.eye(2 ** (len(control_wires) + 1))

        else:
            values = range(value, 2 ** (len(control_wires))) if geq else range(value)
            binary = "0" + str(len(control_wires)) + "b"
            control_values_list = [format(n, binary) for n in values]
            mat = np.eye(2 ** (len(control_wires) + 1))
            for control_values in control_values_list:
                control_values = [int(n) for n in control_values]
                mat = mat @ qml.MultiControlledX.compute_matrix(
                    control_wires, control_values=control_values
                )

        return mat

    @staticmethod
    def compute_decomposition(
        value: int,
        wires: WiresLike,
        geq: bool = True,
        work_wires: WiresLike | None = None,
        **kwargs,
    ) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.IntegerComparator.decomposition`.

        Args:
            value (int): The value :math:`L` that the state's decimal representation is compared against.
            geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If ``False``, the comparison
                made will be :math:`n < L`.
            wires (Union[Wires, Sequence[int], or int]): Control wire(s) followed by a single target wire where
                the operation acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.draw(qml.IntegerComparator.compute_decomposition)(4, wires=[0, 1, 2, 3]))
        0: ─╭●────╭●────╭●────┤
        1: ─├●──X─├●────├●──X─┤
        2: ─│─────├●──X─├●──X─┤
        3: ─╰X────╰X────╰X────┤

        """

        if not isinstance(value, int):
            raise ValueError(f"The compared value must be an int. Got {type(value)}.")

        if wires is None:
            raise ValueError("Must specify the wires that the operation acts on.")

        if len(wires) < 2:
            raise ValueError(
                f"IntegerComparator: wrong number of wires. {len(wires)} wire(s) given. Need at least 2."
            )

        work_wires = Wires([]) if work_wires is None else Wires(work_wires)
        with qml.queuing.AnnotatedQueue() as q:
            if geq:
                _integer_comparator_ge_decomposition(wires, value, work_wires)
            else:
                _integer_comparator_lt_decomposition(wires, value, work_wires)

        if qml.queuing.QueuingManager.recording():
            for op in q.queue:
                qml.apply(op)

        return q.queue

    @property
    def control_wires(self) -> Wires:
        return self.wires[:~0]

    def adjoint(self) -> "IntegerComparator":
        return copy(self).queue()

    def pow(self, z: int) -> list["IntegerComparator"]:
        return super().pow(z % 2)


def _integer_comparator_lt_resource(num_wires, value, num_work_wires, **_):

    if value == 0:
        return {}

    if value > 2 ** (num_wires - 1) - 1:
        return {qml.X: 1}

    num_controls = num_wires - 1
    binary_str = format(value, f"0{num_controls}b")
    last_significant = binary_str.rfind("1")
    gate_counts = {resource_rep(qml.X): (last_significant + 1) * 2}

    first_significant = binary_str.find("1")
    gate_counts[
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=first_significant + 1,
            num_work_wires=num_work_wires + num_wires - 2 - first_significant,
            num_zero_control_values=0,
            work_wire_type="borrowed",
        )
    ] = 1

    while (first_significant := binary_str.find("1", first_significant + 1)) != -1:
        gate_counts[
            resource_rep(
                qml.MultiControlledX,
                num_control_wires=first_significant + 1,
                num_work_wires=num_work_wires + num_wires - 2 - first_significant,
                num_zero_control_values=0,
                work_wire_type="borrowed",
            )
        ] = 1

    return gate_counts


@register_condition(lambda geq, **_: not geq)
@register_resources(_integer_comparator_lt_resource)
def _integer_comparator_lt_decomposition(wires, value, work_wires, **_):
    """Decompose the ``IntegerComparator`` for when the flipping condition is ``n < value``.

    This decomposition uses the minimum number of ``MultiControlledX`` gates. For a given value,
    we first convert it to binary, and iteratively look for the significant bits. For example,
    with 6 control wires, if the value is 22, which is 010110 in 6-bit binary, we observe
    that all 6-bit numbers that start with 00 will satisfy the flipping condition, so we apply
    a ``MultiControlledX`` with only the first two wires as controls, and 00 as the control values.
    Then we look for the next significant bit, and observe that 22 starts with 0101. Therefore,
    all 6-bit numbers that start with 0100 will also satisfy the flipping condition, so we apply
    a ``MultiControlledX`` with the first four wires as controls, and 0100 as the control values.
    This continues until we add a ``MultiControlledX`` for every significant bit in the value.

    .. code-block:: pycon

        0: ─╭○─╭○─╭○─┤
        1: ─├○─├●─├●─┤
        2: ─│──├○─├○─┤
        3: ─│──├○─├●─┤
        4: ─│──│──├○─┤
        6: ─╰X─╰X─╰X─┤

    If we decompose this circuit one level further, we get

    .. code-block:: pycon

        0: ──X─╭●──X──X─╭●──X──X─╭●──X─┤
        1: ──X─├●──X────├●───────├●────┤
        2: ────│───X────├●──X──X─├●──X─┤
        3: ────│───X────├●──X────├●────┤
        4: ────│────────│───X────├●──X─┤
        6: ────╰X───────╰X───────╰X────┤

    And we observe that the ``PauliX`` gates used to flip the control values can be merged:

    .. code-block:: pycon

        0: ──X─╭●────╭●────╭●──X─┤
        1: ──X─├●──X─├●────├●────┤
        2: ──X─│─────├●────├●──X─┤
        3: ──X─│─────├●──X─├●────┤
        4: ──X─│─────│─────├●──X─┤
        6: ────╰X────╰X────╰X────┤

    """

    # If the value is zero, the flipping condition is never satisfied.
    if value == 0:
        return

    num_controls = len(wires) - 1

    # If the value is larger than the maximum value that can be represented by
    # the number of control bits, the flipping condition is always satisfied, in
    # which case we apply an X gate to the target wire and terminate.
    if value > 2**num_controls - 1:
        qml.X(wires[-1])
        return

    # Track which control bits have been flipped back
    control_value_tracker = [0] * num_controls

    # First apply X to all wires until the last significant bit to flip control values to 1.
    binary_str = format(value, f"0{num_controls}b")
    last_significant = binary_str.rfind("1")
    for i in range(last_significant + 1):
        qml.X(wires[i])

    # The flipping condition is satisfied if all bits from the first bit to the
    # first non-zero bit of the value are zeroes.
    first_significant = binary_str.find("1")
    qml.MultiControlledX(
        wires=wires[: first_significant + 1] + wires[-1:],
        work_wires=wires[first_significant + 1 : -1] + work_wires,
    )
    control_value_tracker[first_significant] = 1
    qml.X(wires[first_significant])

    # If the wire corresponding to the first significant bit of the value is 1, then we
    # iteratively look for the next significant bit, and apply a flip conditioned on all
    # bits from the last significant bit to this next significant bit being zeroes.
    while (first_significant := binary_str.find("1", first_significant + 1)) != -1:
        qml.MultiControlledX(
            wires=wires[: first_significant + 1] + wires[-1:],
            work_wires=wires[first_significant + 1 : -1] + work_wires,
        )
        control_value_tracker[first_significant] = 1
        qml.X(wires[first_significant])

    for i in range(last_significant + 1):
        if control_value_tracker[i] == 0:
            qml.X(wires[i])


def _integer_comparator_ge_resource(num_wires, value, num_work_wires, **_):

    # If the value is 0, the flipping condition is always satisfied.
    if value == 0:
        return {qml.X: 1}

    num_controls = num_wires - 1

    if value > 2**num_controls - 1:
        return {}

    binary_str = format(value, f"0{num_controls}b")
    first_zero = binary_str.find("0")

    if first_zero == -1:
        return {
            resource_rep(
                qml.MultiControlledX,
                num_control_wires=num_controls,
                num_work_wires=num_work_wires,
                num_zero_control_values=0,
                work_wire_type="borrowed",
            ): 1
        }

    gate_set = Counter()

    gate_set[
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=first_zero + 1,
            num_work_wires=num_work_wires + num_wires - 2 - first_zero,
            num_zero_control_values=0,
            work_wire_type="borrowed",
        )
    ] = 1
    gate_set[resource_rep(qml.X)] = 2

    while (first_zero := binary_str.find("0", first_zero + 1)) != -1:
        gate_set[
            resource_rep(
                qml.MultiControlledX,
                num_control_wires=first_zero + 1,
                num_work_wires=num_work_wires + num_wires - 2 - first_zero,
                num_zero_control_values=0,
                work_wire_type="borrowed",
            )
        ] = 1
        gate_set[resource_rep(qml.X)] += 2

    gate_set[
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=num_controls,
            num_work_wires=num_work_wires,
            num_zero_control_values=0,
            work_wire_type="borrowed",
        )
    ] += 1

    return dict(gate_set)


@register_condition(lambda geq, **_: geq)
@register_resources(_integer_comparator_ge_resource)
def _integer_comparator_ge_decomposition(wires, value, work_wires, **_):
    """Decompose the ``IntegerComparator`` for when the flipping condition is ``n >= value``.

    This decomposition rule mirrors the implementation for the ``n < value`` case.

    """

    # If the value is 0, the flipping condition is always satisfied.
    if value == 0:
        qml.X(wires[-1])
        return

    num_controls = len(wires) - 1

    # If the value is larger than the maximum value that can be represented by
    # the number of control bits, the flipping condition is never satisfied,
    if value > 2**num_controls - 1:
        return

    # Track which control bits have been flipped
    control_value_tracker = [0] * num_controls

    binary_str = format(value, f"0{num_controls}b")
    first_zero = binary_str.find("0")

    if first_zero == -1:
        # If the value happens to be the all-one state, then we apply a single MCX
        qml.MultiControlledX(wires=wires, work_wires=work_wires)
        return

    qml.MultiControlledX(
        wires=wires[: first_zero + 1] + wires[-1:],
        work_wires=wires[first_zero + 1 : -1] + work_wires,
    )
    control_value_tracker[first_zero] = 1
    qml.X(wires[first_zero])

    while (first_zero := binary_str.find("0", first_zero + 1)) != -1:
        qml.MultiControlledX(
            wires=wires[: first_zero + 1] + wires[-1:],
            work_wires=wires[first_zero + 1 : -1] + work_wires,
        )
        control_value_tracker[first_zero] = 1
        qml.X(wires[first_zero])

    # The last MCX corresponds to the equal case.
    qml.MultiControlledX(wires=wires, work_wires=work_wires)

    for i in range(num_controls):
        if control_value_tracker[i]:
            qml.X(wires[i])


def _integer_comparator_flip_geq_resource(num_wires, value, num_work_wires, geq, **_):
    """Resource estimation for flipping the geq condition."""
    return {
        qml.X: 1,
        resource_rep(
            qml.IntegerComparator,
            num_wires=num_wires,
            value=value,
            geq=not geq,
            num_work_wires=num_work_wires,
        ): 1,
    }


@register_resources(_integer_comparator_flip_geq_resource)
def _integer_comparator_flip_geq(value, geq, wires, work_wires, **_):
    """Decompose the IntegerComparator by flipping geq to lt or vice versa."""
    qml.X(wires[-1])
    IntegerComparator(value, wires, geq=not geq, work_wires=work_wires)


add_decomps(
    IntegerComparator,
    _integer_comparator_lt_decomposition,
    _integer_comparator_ge_decomposition,
    _integer_comparator_flip_geq,
)
