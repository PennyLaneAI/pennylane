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

import pennylane as qml
from pennylane.operation import AnyWires, Operation


def _get_permutation(basis_list):
    r"""
    Given a list of :math:`m` basis states, this function generates a dictionary assigning to each of them
    the :math:`m`-th basis states in the computational base. Also, if a state within ``basis_list`` is one
    of the first :math:`m` basis states, this state will be assigned to itself.

    ** Example **

    .. code-block:: pycon

        >>> basis_list = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]]
        >>> _get_permutation(basis_list)
        {
        [1, 1, 0, 0]: [0, 0, 0, 0],
        [1, 0, 1, 0]: [0, 0, 0, 1],
        [0, 1, 0, 1]: [0, 0, 1, 0],
        [1, 0, 0, 1]: [0, 0, 1, 1]
        }


    .. code-block:: pycon

        >>>> basis_list = [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]]
        >>> _get_permutation(basis_list)
        {
        [1, 1, 0, 0]: [0, 0, 0, 0],
        [0, 1, 0, 1]: [0, 0, 1, 0],
        [0, 0, 0, 1]: [0, 0, 0, 1],
        [1, 0, 0, 1]: [0, 0, 1, 1]
        }

    """

    length = len(basis_list[0])
    smallest_basis_lists = [tuple(map(int, f"{i:0{length}b}")) for i in range(len(basis_list))]

    binary_dict = {}
    used_smallest = set()

    # Assign keys that can map to themselves
    for original in basis_list:

        if original in smallest_basis_lists and tuple(original) not in used_smallest:

            binary_dict[tuple(original)] = original
            used_smallest.add(tuple(original))

    # Assign remaining keys to unused binary lists
    remaining_keys = [key for key in basis_list if tuple(key) not in binary_dict]
    remaining_values = [
        value for value in smallest_basis_lists if tuple(value) not in used_smallest
    ]

    for key, value in zip(remaining_keys, remaining_values):
        binary_dict[tuple(key)] = value
        used_smallest.add(tuple(value))

    return binary_dict


def _permutation_operator(basis1, basis2, wires, work_wire):
    r"""
    Function that takes two basis states, ``basis1`` and ``basis2``, and creates an operator that
    maps :math:`|\text{basis1}\rangle` to :math:`|\text{basis2}\rangle`. To achieve this, it uses
    an auxiliary qubit.

    Args:
        basis1 (List): The initial basis state, represented as a list of binary digits.
        basis2 (List): The target basis state, represented as a list of binary digits.
        wires (Sequence[int]): The list of wires that the operator acts on
        work_wire (Union[Wires, int, str]): The auxiliary wire used for the permutation

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


class Superposition(Operation):
    r"""
    Prepare a superposition of computational basis states.

    Given a list of :math:`m` coefficients :math:`c_i` and a list of basic states :math:`|b_i\rangle`,
    this operator prepares the state:

    .. math::

        |\phi\rangle = \sum_i^m c_i |b_i\rangle

    The decomposition has a complexity that grows linearly with the number of terms,
    unlike other methods such as :class:`~.MottonenStatePreparation`, that grows exponentially
    with the number of qubits. More information on the decomposition can be
    found below in Implementation Details.

    Args:
        coeffs (List[float]): List of coefficients :math:`c_i`
        basis (List[List[int]]): List of basis states :math:`|b_i\rangle`
        wires (Sequence[int]): List of wires that the operator acts on
        work_wire (Union[Wires, int, str]): The auxiliary wire used for the permutation

    **Example**

    .. code-block::

        coeffs = np.sqrt([1/3, 1/3, 1/3])
        basis = [[1, 1, 1], [0, 1, 0], [0, 0, 0]]
        wires = [0, 1, 2]
        work_wire = 3

        dev = qml.device('default.qubit')


        @qml.qnode(dev)
        def circuit():
            qml.Superposition(coeffs, basis, wires, work_wire)
            return qml.probs(wires)

    .. code-block:: pycon

        >>> print(circuit())
        [0.33333333 0.         0.33333333 0.         0.         0.
        0.         0.33333333]


    .. details::
        :title: Implementation Details

        The construction of this template is divided into two blocks.
        The first block takes the list of coefficients :math:`c_i` and prepares the state:

        .. math::

            |\phi\rangle = \sum_i^m c_i |i\rangle,

        where :math:`|i\rangle` are the computational basis states. This is done using the
        :class:`~.StatePrep` template.

        The second block is responsible for the permutation of the basis states to the target basis states.

        .. math::

            |i\rangle \rightarrow |b_i\rangle.

        Appliying these two blocks together results in the desired superposition:

        .. math::

            |\phi\rangle = \sum_i^m c_i |b_i\rangle.
    """

    num_wires = AnyWires
    grad_method = None
    ndim_params = (1,)

    def __init__(
        self, coeffs, basis, wires, work_wire, id=None
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments

        self.hyperparameters["basis"] = tuple(tuple(b) for b in basis)
        self.hyperparameters["target_wires"] = qml.wires.Wires(wires)
        self.hyperparameters["work_wire"] = qml.wires.Wires(work_wire)

        all_wires = self.hyperparameters["target_wires"] + self.hyperparameters["work_wire"]

        super().__init__(coeffs, wires=all_wires, id=id)

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

    def decomposition(self):  # pylint: disable=arguments-differ
        return self.compute_decomposition(
            *self.parameters,
            basis=self.hyperparameters["basis"],
            wires=self.hyperparameters["target_wires"],
            work_wire=self.hyperparameters["work_wire"],
        )

    @staticmethod
    def compute_decomposition(coefs, basis, wires, work_wire):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            coefs (List[float]): List of coefficients :math:`c_i`
            basis (List[List[int]]): List of basis states :math:`|b_i\rangle`
            wires (Sequence[int]): List of wires that the operator acts on
            work_wire (Union[Wires, int, str]): The auxiliary wire used for the permutation

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        .. code-block:: pycon

            >>> qml.Superposition(np.sqrt([1/2, 1/2]), [[1, 1], [0, 0]], [0, 1], 2).decomposition()
            [StatePrep(array([0.70710678, 0.70710678]), wires=[1]),
            MultiControlledX(wires=[0, 1, 2], control_values=[False, True]),
            CNOT(wires=[2, 0]),
            Toffoli(wires=[0, 1, 2])]

        """

        dic_state = dict(zip(basis, coefs))
        perms = _get_permutation(basis)
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
                wires=wires[-int(qml.math.ceil(qml.math.log2(len(coefs)))) :],
                pad_with=0,
            )
        )

        for basis2, basis1 in perms.items():
            if not qml.math.allclose(basis1, basis2):
                op_list += _permutation_operator(basis1, basis2, wires, work_wire)

        return op_list

    @property
    def basis(self):
        r"""List of basis states :math:`|b_i\rangle`."""
        return self.hyperparameters["basis"]

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
            basis=self.basis,
            wires=new_dict["target_wires"],
            work_wire=new_dict["work_wire"],
        )
