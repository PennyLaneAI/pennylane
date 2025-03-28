# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Compact input classes for resource estimation."""
# pylint: disable=too-many-arguments,
import math


class CompactState:
    r"""A compact representation for the state of a quantum system.

    Args:
        num_qubits (int): number of qubits used to represent the state
        num_coeffs (int): The number of coefficients in the linear combination of
            computational basis states representation of the target state.
        precision (float): a tolerance for approximation when preparing the state
        num_work_wires (int): number of additional work qubits available to prepare state
        num_bit_flips (int): The number of qubits in the :math:`|1\rangle` state for
            preparing the target basis state.
        positive_and_real (bool): A flag which is :code:`True` when all coefficients are
            real and positive for the target state.

    .. details::
        :title: Usage Details

        The :code:`CompactState` class is designed to be an alternative input to preparing
        a full statevector. It should be used in combination with a statepreparation template
        for more efficient state preparation when performing resource estimation.

        .. code-block:: python

            from pennylane.labs import resource_estimation as re

            compact_statevector = re.CompactState.from_state_vector(num_qubits=20, num_coeffs=2**20)

            def circ():
                re.ResourceSuperposition(compact_statevector, wires=range(20))
                return

        The resources can then be extracted as usual:

        >>> res = re.get_resources(circ)()
        >>> print(res)
        wires: 20
        gates: 75497303
        gate_types:
        {'T': 71303083, 'CNOT': 4194220}

    """

    def __init__(
        self,
        num_qubits=None,
        num_coeffs=None,
        precision=None,
        num_work_wires=None,
        num_bit_flips=None,
        positive_and_real=None,
    ):
        self.num_qubits = num_qubits
        self.num_coeffs = num_coeffs
        self.precision = precision
        self.num_work_wires = num_work_wires
        self.num_bit_flips = num_bit_flips
        self.positive_and_real = positive_and_real

    def __eq__(self, other: object) -> bool:
        return all(
            (
                (self.num_qubits == other.num_qubits),
                (self.num_coeffs == other.num_coeffs),
                (self.precision == other.precision),
                (self.num_work_wires == other.num_work_wires),
                (self.num_bit_flips == other.num_bit_flips),
                (self.positive_and_real == other.positive_and_real),
            )
        )

    @classmethod
    def from_mps(cls, num_mps_matrices, max_bond_dim):
        """Instantiate a CompactState representing an MPS.

        Args:
            num_mps_matrices (int): number of tensors in the MPS
            max_bond_dim (int): the maximum bond dimension of the MPS

        Returns:
            CompactState: the compact state representing the MPS
        """
        num_work_wires = math.ceil(math.log2(max_bond_dim))
        return cls(num_qubits=num_mps_matrices, num_work_wires=num_work_wires)

    @classmethod
    def from_bitstring(cls, num_qubits, num_bit_flips):
        r"""Instantiate a CompactState representing a computational basis state.

        Args:
            num_qubits (int): number of qubits used to represent the state
            num_bit_flips (int): The number of qubits in the :math:`|1\rangle` state for
                preparing the target basis state.

        Returns:
            CompactState: the compact state representing the bitstring
        """
        return cls(
            num_qubits=num_qubits,
            num_coeffs=1,
            num_bit_flips=num_bit_flips,
        )

    @classmethod
    def from_state_vector(
        cls,
        num_qubits,
        num_coeffs,
        precision=1e-3,
        num_work_wires=0,
        positive_and_real=False,
    ):
        r"""Instantiate a CompactState representing a statevector (dense or sparse).

        Args:
            num_qubits (int): number of qubits used to represent the state
            num_coeffs (int): The number of coefficients in the linear combination of
                computational basis states representation of the target state.
            precision (float, optional): A tolerance for approximation when preparing
                the state. Defaults to 1e-3.
            num_work_wires (int, optional): The number of additional work qubits available
                to prepare state. Defaults to 0.
            positive_and_real (bool, optional): A flag which is :code:`True` when all
                coefficients are real and positive for the target state. Defaults to False.

        Returns:
            CompactState: the compact state representing the statevector
        """
        return cls(
            num_qubits=num_qubits,
            num_coeffs=num_coeffs,
            precision=precision,
            num_work_wires=num_work_wires,
            positive_and_real=positive_and_real,
        )
