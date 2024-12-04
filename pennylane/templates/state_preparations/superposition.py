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
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation


def _get_permutation(basis_list):
    r"""
    Given a list of :math:`m` basis states, this function generates a dictionary assigning to each of them
    the :math:`m`-th basis states in the computational base.

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

    ** Details **

    If a state within ``basis_list`` is one of the first :math:`m` basis states, this value
    will be removed from the dictionary.

    .. code-block:: pycon

        >>>> basis_list = [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]]
        >>> _get_permutation(basis_list)
        {
        [1, 1, 0, 0]: [0, 0, 0, 0],
        [0, 1, 0, 1]: [0, 0, 1, 0],
        [1, 0, 0, 1]: [0, 0, 1, 1]
        }

    Note that `[0, 0, 0, 1]` has been deleted in the previous example.

    """

    length = len(basis_list[0])  # All binary lists have the same length
    smallest_basis_lists = [list(map(int, f"{i:0{length}b}")) for i in range(len(basis_list))]

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

    filtered_dict = {key: value for key, value in binary_dict.items() if list(key) != value}
    return filtered_dict


def _permutation_operator(basis1, basis2, wires, work_wire):
    """
    Function that takes two basis states, ``basis1`` and ``basis2``, and creates an operator that
    maps :math:`|\text{basis1}\rangle` to :math:`|\text{basis2}\rangle`. To achieve this, it utilizes
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

    for i in range(len(basis1)):
        if basis1[i] != basis2[i]:
            ops.append(qml.CNOT(wires=work_wire + wires[i]))

    ops.append(qml.ctrl(qml.PauliX(work_wire), control=wires, control_values=basis2))

    return ops


class Superposition(Operation):
    r""" """

    num_wires = AnyWires
    grad_method = None
    ndim_params = (1,)

    def __init__(self, coeffs, basis, wires, work_wire, id=None):

        self.hyperparameters["basis"] = tuple(tuple(b) for b in basis)
        self.hyperparameters["target_wires"] = qml.wires.Wires(wires)
        self.hyperparameters["work_wire"] = qml.wires.Wires(work_wire)

        all_wires = self.hyperparameters["target_wires"] + self.hyperparameters["work_wire"]

        super().__init__(coeffs, wires=all_wires, id=id)

    @property
    def num_params(self):
        return 1

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
        return self.compute_decomposition(*self.parameters, **self.hyperparameters)

    @staticmethod
    def compute_decomposition(
        coefs, basis, target_wires, work_wire
    ):  # pylint: disable=arguments-differ
        r""" """

        op_list = []
        op_list.append(
            qml.StatePrep(
                coefs,
                wires=target_wires[-int(np.ceil(np.log2(len(coefs)))) :],
                pad_with=0,
                normalize=True,
            )
        )
        perms = _get_permutation(basis)
        for basis2, basis1 in perms.items():
            op_list += _permutation_operator(basis1, basis2, target_wires, work_wire)

        return op_list

    @property
    def basis(self):
        return self.hyperparameters["basis"]

    @property
    def work_wire(self):
        return self.hyperparameters["work_wire"]

    @property
    def coeffs(self):
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
