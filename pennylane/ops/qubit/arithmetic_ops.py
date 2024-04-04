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
# pylint:disable=abstract-method,arguments-differ,protected-access
from copy import copy

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires
from pennylane.ops import Identity


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

    .. code-block::

        input_bitstring = (0, 1, 1, 0)

        @qml.qnode(dev)
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

    num_wires = 4
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
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
    def compute_decomposition(wires):
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

    .. code-block::

        input_bitstring = (0, 1, 0)

        @qml.qnode(dev)
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

    >>> abc_sum = output_bitstring[2]
    >>> abc_sum
    1
    """

    num_wires = 3
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "Σ", cache=cache)

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
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
    def compute_decomposition(wires):
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
    tensor(1.+0.j, requires_grad=True)
    >>> circuit([0, 1, 0], 3, False).reshape(2, 2, 2)[0, 1, 1]
    tensor(1.+0.j, requires_grad=True)
    """

    is_self_inverse = True
    num_wires = AnyWires
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None

    def _flatten(self):
        hp = self.hyperparameters
        metadata = (
            ("work_wires", hp["work_wires"]),
            ("value", hp["value"]),
            ("geq", hp["geq"]),
        )
        return tuple(), (hp["control_wires"] + hp["target_wires"], metadata)

    # pylint: disable=too-many-arguments
    def __init__(self, value, geq=True, wires=None, work_wires=None):
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

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or f">={self.value}" if self.geq else f"<{self.value}"

    # pylint: disable=unused-argument
    @staticmethod
    def compute_matrix(
        value=None, control_wires=None, geq=True, **kwargs
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IntegerComparator.matrix`

        Args:
            value (int): The value :math:`L` that the state's decimal representation is compared against.
            control_wires (Union[Wires, Sequence[int], or int]): wires to place controls on
            geq (bool): If set to `True`, the comparison made will be :math:`n \geq L`. If `False`, the comparison
                made will be :math:`n < L`.

        Returns:
           tensor_like: matrix representation

        **Example**

        >>> print(qml.IntegerComparator.compute_matrix(2, [0, 1]))
        [[1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 0. 1. 0.]]
        >>> print(qml.IntegerComparator.compute_matrix(2, [0, 1], geq=False))
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
    def compute_decomposition(value, geq=True, wires=None, work_wires=None, **kwargs):
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

        >>> print(qml.IntegerComparator.compute_decomposition(4, wires=[0, 1, 2, 3]))
        [MultiControlledX(wires=[0, 1, 2, 3], control_values=[1, 0, 0]),
         MultiControlledX(wires=[0, 1, 2, 3], control_values=[1, 0, 1]),
         MultiControlledX(wires=[0, 1, 2, 3], control_values=[1, 1, 0]),
         MultiControlledX(wires=[0, 1, 2, 3], control_values=[1, 1, 1])]
        """

        if not isinstance(value, int):
            raise ValueError(f"The compared value must be an int. Got {type(value)}.")
        if wires is None:
            raise ValueError("Must specify the wires that the operation acts on.")
        if len(wires) > 1:
            control_wires = Wires(wires[:-1])
            wires = Wires(wires[-1])
        else:
            raise ValueError(
                f"IntegerComparator: wrong number of wires. {len(wires)} wire(s) given. Need at least 2."
            )

        small_val = not geq and value == 0
        large_val = geq and value > 2 ** len(control_wires) - 1
        if small_val or large_val:
            gates = [Identity(0)]

        else:
            values = range(value, 2 ** (len(control_wires))) if geq else range(value)
            binary = "0" + str(len(control_wires)) + "b"
            control_values_list = [format(n, binary) for n in values]
            gates = []
            for control_values in control_values_list:
                control_values = [int(n) for n in control_values]
                gates.append(
                    qml.MultiControlledX(
                        wires=control_wires + wires,
                        control_values=control_values,
                        work_wires=work_wires,
                    )
                )

        return gates

    @property
    def control_wires(self):
        return self.wires[:~0]

    def adjoint(self):
        return copy(self).queue()

    def pow(self, z):
        return super().pow(z % 2)
