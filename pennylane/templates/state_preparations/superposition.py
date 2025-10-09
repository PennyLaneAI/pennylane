# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the Superposition template.
"""
from collections import Counter
from functools import reduce

import pennylane as qml
from pennylane.control_flow import for_loop
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation


def order_states(basis_states: list[list[int]]) -> dict[tuple[int], tuple[int]]:
    r"""
    This function maps a given list of :math:`m` computational basis states to the first
    :math:`m` computational basis states, except for input states that are among the first
    :math:`m` computational basis states, which are mapped to themselves.

    Args:
        basis_states (list[list[int]]): sequence of :math:`m` basis states to be mapped.
            Each state is a sequence of 0s and 1s.

    Returns:
        dict[tuple[int], tuple[int]]: dictionary mapping basis states to the first :math:`m` basis
        states, except for fixed points (states in the input that already were among the
        first :math:`m` basis states).

    **Example**

    For instance, a given list of :math:`[s_0, s_1, ..., s_m]` where :math:`s` is a basis
    state of length :math:`4` will be mapped as
    :math:`\{s_0: |0000\rangle, s_1: |0001\rangle, s_2: |0010\rangle, \dots\}`.

    >>> basis_states = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]]
    >>> order_states(basis_states)
    {(1, 1, 0, 0): (0, 0, 0, 0),
        (1, 0, 1, 0): (0, 0, 0, 1),
        (0, 1, 0, 1): (0, 0, 1, 0),
        (1, 0, 0, 1): (0, 0, 1, 1)}

    If a state in ``basis_states`` is one of the first :math:`m` basis states,
    this state will be mapped to itself, i.e. it will be a fixed point of the mapping.

    >>> basis_states = [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]]
    >>> order_states(basis_states)
    {(0, 0, 0, 1): (0, 0, 0, 1),
        (1, 1, 0, 0): (0, 0, 0, 0),
        (0, 1, 0, 1): (0, 0, 1, 0),
        (1, 0, 0, 1): (0, 0, 1, 1)}

    """

    m = len(basis_states)
    length = len(basis_states[0])
    # Create the integers corresponding to the input basis states
    basis_ints = [int("".join(map(str, state)), 2) for state in basis_states]

    basis_states = [tuple(s) for s in basis_states]  # Need hashable objects, so we use tuples
    state_map = {}  # The map for basis states to be populated
    unmapped_states = []  # Will collect non-fixed point states
    unmapped_ints = {i: None for i in range(m)}  # Will remove fixed point states
    # Map fixed-point states to themselves and collect states and target ints still to be paired
    for b_int, state in zip(basis_ints, basis_states):
        if b_int < m:
            state_map[state] = state
            unmapped_ints.pop(b_int)
        else:
            unmapped_states.append(state)

    # Map non-fixed point states
    for state, new_b_int in zip(unmapped_states, unmapped_ints):
        # Convert the index of the state to be mapped into a state itself
        state_map[state] = tuple(map(int, f"{new_b_int:0{length}b}"))

    return state_map


def _permutation_operator(basis1, basis2, wires, work_wire):
    r"""
    Creates operations that map an initial basis state to a target basis state using an auxiliary qubit.

    Args:
        basis1 (List): The initial basis state, represented as a list of binary digits.
        basis2 (List): The target basis state, represented as a list of binary digits.
        wires (Sequence[int]): The list of wires that the operator acts on.
        work_wire (Union[Wires, int, str]): The auxiliary wire used for the permutation.

    Returns:
        list: A list of operators that map :math:`|\text{basis1}\rangle` to :math:`|\text{basis2}\rangle`.
    """

    ops = []
    ops.append(qml.ctrl(qml.PauliX(work_wire), control=wires, control_values=basis1))

    for i, b in enumerate(basis1):
        if b != basis2[i]:
            ops.append(qml.CNOT(wires=work_wire + wires[i]))

    ops.append(qml.ctrl(qml.PauliX(work_wire), control=wires, control_values=basis2))

    return ops


def _permutation_operator_qfunc(basis1, basis2, wires, work_wire):
    r"""
    Creates operations that map an initial basis state to a target basis state using an auxiliary qubit.

    Args:
        basis1 (List): The initial basis state, represented as a list of binary digits.
        basis2 (List): The target basis state, represented as a list of binary digits.
        wires (Sequence[int]): The list of wires that the operator acts on.
        work_wire (Union[Wires, int, str]): The auxiliary wire used for the permutation.
    """

    qml.ctrl(qml.PauliX(work_wire), control=wires, control_values=basis1)

    @for_loop(len(basis1))
    def apply_cnots(i):
        b = basis1[i]

        def apply_cnot():
            qml.CNOT(wires=work_wire + wires[i])

        qml.cond(b != basis2[i], apply_cnot)()

    apply_cnots()  # pylint: disable=no-value-for-parameter

    qml.ctrl(qml.PauliX(work_wire), control=wires, control_values=basis2)


class Superposition(Operation):
    r"""
    Prepare a superposition of computational basis states.

    Given a list of :math:`m` coefficients :math:`c_i` and basic states :math:`|b_i\rangle`,
    this operator prepares the state:

    .. math::

        |\phi\rangle = \sum_i^m c_i |b_i\rangle.

    See the Details section for more information about the decomposition.

    Args:
        coeffs (tensor-like[float]): normalized coefficients of the superposition
        bases (tensor-like[int]): basis states of the superposition
        wires (Sequence[int]): wires that the operator acts on
        work_wire (Union[Wires, int, str]): the auxiliary wire used for the permutation

    **Example**

    .. code-block:: python

        import pennylane as qml
        import numpy as np

        coeffs = np.sqrt(np.array([1/3, 1/3, 1/3]))
        bases = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        wires = [0, 1, 2]
        work_wire = 3

        dev = qml.device('default.qubit')
        @qml.qnode(dev)
        def circuit():
            qml.Superposition(coeffs, bases, wires, work_wire)
            return qml.probs(wires)

    >>> print(circuit()) # doctest: +SKIP
    [0.3333 0.     0.3333 0.     0.     0.     0.     0.3333]


    .. details::
        :title: Details

        The input superposition state , :math:`|\phi\rangle = \sum_i^m c_i |b_i\rangle`, is implemented in two steps. First, the coefficients :math:`c_i` are used to prepares the state:

        .. math::

            |\phi\rangle = \sum_i^m c_i |i\rangle,

        where :math:`|i\rangle` is a computational basis states and :math:`m` is the number of terms
        in the superposition. This is done using the
        :class:`~.StatePrep` template in the fisrt :math:`\lceil \log_2 m \rceil` qubits. Note that the number of qubits depends on the number of terms in the superposition, which helps to reduce the complexity of the operation.

        The second step permutes the basis states prepared previously to
        the target basis states:

        .. math::

            |i\rangle \rightarrow |b_i\rangle.

        This block maps the elements one by one using an auxiliary qubit.
        This can be done in three separate steps:

        1. By using a multi-controlled NOT gate, check if the input state is :math:`|i\rangle` and
        store the information in the auxiliary qubit. If the state is :math:`|i\rangle` the auxiliary
        qubit will be in the :math:`|1\rangle` state.

        2. If the auxiliary qubit is in the :math:`|1\rangle` state, the input state is modified by applying
        ``X`` gates to the bits that are different between :math:`|i\rangle` and :math:`|b_i\rangle`.

        3. By using a multi-controlled ``NOT`` gate, check if the final state is :math:`|b_i\rangle` and
        return the auxiliary qubit back to :math:`|0\rangle` state.

        Applying all these together prepares the desired superposition:

        .. math::

            |\phi\rangle = \sum_i^m c_i |b_i\rangle.

        The decomposition has a complexity that grows linearly with the number of terms in the superposition,
        unlike other methods such as :class:`~.MottonenStatePreparation` that grows exponentially
        with the number of qubits.
    """

    grad_method = None
    ndim_params = (1,)

    resource_keys = {"num_wires", "num_coeffs", "bases"}

    def __init__(
        self, coeffs, bases, wires, work_wire, id=None
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments

        if not all(
            all(qml.math.isclose(i, 0.0) or qml.math.isclose(i, 1.0) for i in b) for b in bases
        ):
            raise ValueError("The elements of the basis states must be either 0 or 1.")

        basis_lengths = {len(b) for b in bases}
        if len(basis_lengths) > 1:
            raise ValueError("All basis states must have the same length.")

        if not qml.math.is_abstract(coeffs):
            coeffs_norm = qml.math.linalg.norm(coeffs)
            if not qml.math.allclose(coeffs_norm, qml.math.array(1.0)):
                raise ValueError("The input superposition must be normalized.")

        unique_basis = qml.math.unique(qml.math.array([tuple(b) for b in bases]), axis=0)

        if len(unique_basis) != len(bases):
            raise ValueError("The basis states must be unique.")

        self.hyperparameters["bases"] = tuple(tuple(int(i) for i in b) for b in bases)
        self.hyperparameters["target_wires"] = qml.wires.Wires(wires)
        self.hyperparameters["work_wire"] = qml.wires.Wires(work_wire)

        all_wires = self.hyperparameters["target_wires"] + self.hyperparameters["work_wire"]

        super().__init__(coeffs, wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_wires": len(self.hyperparameters["target_wires"]),
            "num_coeffs": len(self.data[0]),
            "bases": self.hyperparameters["bases"],
        }

    @property
    def num_params(self):
        return 1

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def _flatten(self):
        metadata = tuple(
            (key, value) for key, value in self.hyperparameters.items() if key != "target_wires"
        )
        return tuple(self.parameters), (self.hyperparameters["target_wires"], metadata)

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata[1])
        return cls(*data, wires=metadata[0], **hyperparams_dict)

    def decomposition(self):
        return self.compute_decomposition(
            *self.parameters,
            bases=self.hyperparameters["bases"],
            wires=self.hyperparameters["target_wires"],
            work_wire=self.hyperparameters["work_wire"],
        )

    @staticmethod
    def compute_decomposition(coeffs, bases, wires, work_wire):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            coeffs (tensor-like[float]): normalized coefficients of the superposition
            bases (tensor-like[int]): basis states of the superposition
            wires (Sequence[int]): wires that the operator acts on
            work_wire (Union[Wires, int, str]): the auxiliary wire used for the permutation

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> ops = qml.Superposition(np.sqrt([1/2, 1/2]), [[1, 1], [0, 0]], [0, 1], 2).decomposition()
        >>> from pprint import pprint
        >>> pprint(ops)
        [StatePrep(array([0.707..., 0.707...]), wires=[1]),
        MultiControlledX(wires=[0, 1, 2], control_values=[False, True]),
        CNOT(wires=[2, 0]),
        Toffoli(wires=[0, 1, 2])]

        """

        dic_state = dict(zip(bases, coeffs))
        perms = order_states(bases)
        new_dic_state = {perms[key]: dic_state[key] for key in dic_state if key in perms}

        sorted_coefficients = [
            value
            for key, value in sorted(
                new_dic_state.items(), key=lambda item: int("".join(map(str, item[0])), 2)
            )
        ]

        op_list = []
        op_list.append(
            qml.StatePrep(
                qml.math.stack(sorted_coefficients),
                wires=wires[-int(qml.math.ceil(qml.math.log2(len(coeffs)))) :],
                pad_with=0,
            )
        )

        for basis2, basis1 in perms.items():
            if not qml.math.allclose(basis1, basis2):
                op_list += _permutation_operator(basis1, basis2, wires, work_wire)

        return op_list

    @property
    def bases(self):
        r"""List of basis states :math:`|b_i\rangle`."""
        return self.hyperparameters["bases"]

    @property
    def work_wire(self):
        r"""The auxiliary wire used for the permutation."""
        return self.hyperparameters["work_wire"]

    @property
    def coeffs(self):
        r"""List of coefficients :math:`c_i`."""
        return self.parameters[0]

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["target_wires", "work_wire"]
        }

        return Superposition(
            self.coeffs,
            bases=self.bases,
            wires=new_dict["target_wires"],
            work_wire=new_dict["work_wire"],
        )


def _suerposition_resources(num_wires, num_coeffs, bases):
    perms = order_states(bases)

    resources = Counter()

    resources[
        resource_rep(qml.StatePrep, num_wires=int(qml.math.ceil(qml.math.log2(num_coeffs))))
    ] += 1

    for basis2, basis1 in perms.items():
        if not qml.math.allclose(basis1, basis2):
            resources[
                controlled_resource_rep(
                    base_class=qml.PauliX,
                    base_params={},
                    num_control_wires=num_wires,
                    num_zero_control_values=reduce(lambda acc, nxt: acc + int(nxt == 0), basis1, 0),
                )
            ] += 1

            resources[qml.CNOT] += reduce(
                lambda acc, ib: acc
                + int(ib[1] != basis2[ib[0]]),  # pylint: disable=cell-var-from-loop
                enumerate(basis1),
                0,
            )

            resources[
                controlled_resource_rep(
                    base_class=qml.PauliX,
                    base_params={},
                    num_control_wires=num_wires,
                    num_zero_control_values=reduce(lambda acc, nxt: acc + int(nxt == 0), basis2, 0),
                )
            ] += 1

    return dict(resources)


@register_resources(_suerposition_resources)
def _superposition_decomposition(coeffs, bases, target_wires, work_wire, wires=None):
    dic_state = dict(zip(bases, coeffs))
    perms = order_states(bases)
    new_dic_state = {perms[key]: dic_state[key] for key in dic_state if key in perms}

    sorted_coefficients = [
        value
        for key, value in sorted(
            new_dic_state.items(), key=lambda item: int("".join(map(str, item[0])), 2)
        )
    ]

    qml.StatePrep(
        qml.math.stack(sorted_coefficients),
        wires=wires[-int(qml.math.ceil(qml.math.log2(len(coeffs)))) :],
        pad_with=0,
    )

    bas = [(b2, b1) for b1, b2 in perms.items()]

    @for_loop(len(list(perms.keys())))
    def apply_permutations(i):
        basis2, basis1 = bas[i][0], bas[i][1]
        if not qml.math.allclose(basis1, basis2):
            _permutation_operator_qfunc(basis1, basis2, target_wires, work_wire)

    apply_permutations()  # pylint: disable=no-value-for-parameter


add_decomps(Superposition, _superposition_decomposition)
