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
r"""Resource operators for state preparation templates."""
import math

import pennylane.estimator as qre
import pennylane.numpy as np
from pennylane.estimator.compact_hamiltonian import THCHamiltonian
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.wires import Wires, WiresLike

# pylint: disable= signature-differs, arguments-differ, too-many-arguments


class UniformStatePrep(ResourceOperator):
    r"""Resource class for preparing a uniform superposition.

    This operation prepares a uniform superposition over a given number of
    basis states. The uniform superposition is defined as:

    .. math::

        \frac{1}{\sqrt{l}} \sum_{i=0}^{l} |i\rangle

    where :math:`l` is the number of states.

    This operation uses ``Hadamard`` gates to create the uniform superposition when
    the number of states is a power of two. If the number of states is not a power of two,
    the amplitude amplification technique defined in
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_ is used.

    Args:
        num_states (int): the number of states in the uniform superposition
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are obtained from Figure 12 in `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.
        The circuit uses amplitude amplification to prepare a uniform superposition over :math:`l`
        basis states.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> unif_state_prep = qre.UniformStatePrep(10)
    >>> print(qre.estimate(unif_state_prep))
    --- Resources: ---
    Total wires: 5
        algorithmic wires: 4
        allocated wires: 1
        zero state: 1
        any state: 0
    Total gates : 124
    'Toffoli': 4,
    'T': 88,
    'CNOT': 4,
    'X': 12,
    'Hadamard': 16
    """

    resource_keys = {"num_states"}

    def __init__(self, num_states: int, wires: WiresLike = None):
        self.num_states = num_states
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2**k)

        self.num_wires = k
        if L != 1:
            self.num_wires += int(math.ceil(math.log2(L)))

        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_states (int): the number of states over which the uniform superposition is being prepared
        """
        return {"num_states": self.num_states}

    @classmethod
    def resource_rep(cls, num_states: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2**k)

        num_wires = k
        if L != 1:
            num_wires += int(math.ceil(math.log2(L)))
        return CompressedResourceOp(cls, num_wires, {"num_states": num_states})

    @classmethod
    def resource_decomp(cls, num_states: int) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_states (int): the number of states over which the uniform superposition is being prepared

        Resources:
            The resources are obtained from Figure 12 in `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.
            The circuit uses amplitude amplification to prepare a uniform superposition over :math:`l` basis states.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
            ``GateCount`` objects, where each object represents a specific quantum gate and the
            number of times it appears in the decomposition.
        """

        gate_lst = []
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2**k)
        if L == 1:
            gate_lst.append(GateCount(resource_rep(qre.Hadamard), k))
            return gate_lst

        logl = int(math.ceil(math.log2(L)))
        gate_lst.append(GateCount(resource_rep(qre.Hadamard), k + 3 * logl))
        gate_lst.append(
            GateCount(resource_rep(qre.IntegerComparator, {"value": L, "register_size": logl}), 1)
        )
        gate_lst.append(GateCount(resource_rep(qre.RZ), 2))
        gate_lst.append(
            GateCount(
                resource_rep(
                    qre.Adjoint,
                    {
                        "base_cmpr_op": resource_rep(
                            qre.IntegerComparator, {"value": L, "register_size": logl}
                        )
                    },
                ),
                1,
            )
        )

        return gate_lst


class AliasSampling(ResourceOperator):
    r"""Resource class for preparing a state using coherent alias sampling.

    Args:
        num_coeffs (int): the number of unique coefficients in the state
        precision (float): the precision with which the coefficients are loaded
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are obtained from Section III D in `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.
        The circuit uses coherent alias sampling to prepare a state with the given coefficients.

    **Example**

    The resources for this operation are computed using:

    >>> alias_sampling = qre.AliasSampling(num_coeffs=100)
    >>> print(qre.estimate(alias_sampling))
    --- Resources: ---
    Total wires: 133
        algorithmic wires: 7
        allocated wires: 126
        zero state: 58
        any state: 68
    Total gates : 6.505E+3
    'Toffoli': 272,
    'T': 88,
    'CNOT': 4.646E+3,
    'X': 595,
    'Hadamard': 904
    """

    resource_keys = {"num_coeffs", "precision"}

    def __init__(self, num_coeffs: int, precision: float | None = None, wires: WiresLike = None):
        self.num_coeffs = num_coeffs
        self.precision = precision
        self.num_wires = int(math.ceil(math.log2(num_coeffs)))
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_coeffs (int): the number of unique coefficients in the state
                * precision (float): the precision with which the coefficients are loaded

        """
        return {"num_coeffs": self.num_coeffs, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_coeffs: int, precision: float | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = int(math.ceil(math.log2(num_coeffs)))
        return CompressedResourceOp(
            cls, num_wires, {"num_coeffs": num_coeffs, "precision": precision}
        )

    @classmethod
    def resource_decomp(cls, num_coeffs: int, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_coeffs (int): the number of unique coefficients in the state
            precision (float): the precision with which the coefficients are loaded

        Resources:
            The resources are obtained from Section III D in `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.
            The circuit uses coherent alias sampling to prepare a state with the given coefficients.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
            ``GateCount`` objects, where each object represents a specific quantum gate and the
            number of times it appears in the decomposition.
        """

        gate_lst = []

        logl = int(math.ceil(math.log2(num_coeffs)))

        num_prec_wires = abs(math.floor(math.log2(precision)))

        gate_lst.append(Allocate(logl + 2 * num_prec_wires + 1))

        gate_lst.append(
            GateCount(resource_rep(qre.UniformStatePrep, {"num_states": num_coeffs}), 1)
        )
        gate_lst.append(GateCount(resource_rep(qre.Hadamard), num_prec_wires))
        gate_lst.append(
            GateCount(
                resource_rep(
                    qre.QROM,
                    {"num_bitstrings": num_coeffs, "size_bitstring": logl + num_prec_wires},
                ),
                1,
            )
        )
        gate_lst.append(
            GateCount(
                resource_rep(
                    qre.RegisterComparator,
                    {"first_register": num_prec_wires, "second_register": num_prec_wires},
                ),
                1,
            )
        )
        gate_lst.append(GateCount(resource_rep(qre.CSWAP), logl))

        return gate_lst


class MPSPrep(ResourceOperator):
    r"""Resource class for the MPSPrep template.

    The resource operation for preparing an initial state from a matrix product state (MPS)
    representation.

    Args:
        num_mps_matrices (int): the number of matrices in the MPS representation
        max_bond_dim (int): the bond dimension of the MPS representation
        precision (float | None): the precision used when loading the MPS matricies
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources for MPSPrep rely on a decomposition which uses the generic
        :class:`~.pennylane.estimator.QubitUnitary`. This decomposition is based on
        the routine described in `arXiv:2310.18410 <https://arxiv.org/abs/2310.18410>`_.

    .. seealso:: :class:`~.MPSPrep`

    **Example**

    The resources for this operation are computed using:

    >>> mps = qre.MPSPrep(num_mps_matrices=10, max_bond_dim=2**3)
    >>> print(qre.estimate(mps, gate_set={"CNOT", "RZ", "RY"}))
    --- Resources: ---
     Total wires: 13
        algorithmic wires: 10
        allocated wires: 3
             zero state: 3
             any state: 0
     Total gates : 1.654E+3
      'RZ': 728,
      'RY': 152,
      'CNOT': 774
    """

    resource_keys = {"num_mps_matrices", "max_bond_dim", "precision"}

    def __init__(
        self,
        num_mps_matrices: int,
        max_bond_dim: int,
        precision: float | None = None,
        wires: WiresLike = None,
    ):
        self.num_wires = num_mps_matrices
        self.precision = precision
        self.max_bond_dim = max_bond_dim
        self.num_mps_matrices = num_mps_matrices
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_mps_matrices (int): the number of matrices in the MPS representation
                * max_bond_dim (int): the bond dimension of the MPS representation
                * precision (float | None): the precision used when loading the
                  MPS matrices
        """
        return {
            "num_mps_matrices": self.num_mps_matrices,
            "max_bond_dim": self.max_bond_dim,
            "precision": self.precision,
        }

    @classmethod
    def resource_rep(
        cls, num_mps_matrices: int, max_bond_dim: int, precision: float | None = None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_mps_matrices (int): the number of matrices in the MPS representation
            max_bond_dim (int): the bond dimension of the MPS representation
            precision (float | None): the precision used when loading the MPS matrices

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "num_mps_matrices": num_mps_matrices,
            "max_bond_dim": max_bond_dim,
            "precision": precision,
        }
        num_wires = num_mps_matrices
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        num_mps_matrices: int,
        max_bond_dim: int,
        precision: float | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_mps_matrices (int): the number of matrices in the MPS representation
            max_bond_dim (int): the bond dimension of the MPS representation
            precision (float | None): the precision used when loading
                the MPS matrices

        Resources:
            The resources for MPSPrep are estimated according to the decomposition, which uses the generic
            :class:`~.pennylane.estimator.QubitUnitary`. The decomposition is based on
            the routine described in `arXiv:2310.18410 <https://arxiv.org/abs/2310.18410>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
            ``GateCount`` objects, where each object represents a specific quantum gate and the
            number of times it appears in the decomposition.
        """
        num_work_wires = min(
            math.ceil(math.log2(max_bond_dim)), math.ceil(num_mps_matrices / 2)  # truncate bond dim
        )

        gate_lst = [Allocate(num_work_wires)]

        for index in range(1, num_mps_matrices + 1):
            qubit_unitary_wires = min(index + 1, num_work_wires + 1, (num_mps_matrices - index) + 2)
            qubit_unitary = qre.QubitUnitary.resource_rep(
                num_wires=qubit_unitary_wires, precision=precision
            )
            gate_lst.append(GateCount(qubit_unitary))

        gate_lst.append(Deallocate(num_work_wires))
        return gate_lst

    @classmethod
    def tracking_name(cls, num_mps_matrices, max_bond_dim, precision) -> str:
        return f"MPSPrep({num_mps_matrices}, {max_bond_dim}, {precision})"


class QROMStatePreparation(ResourceOperator):
    r"""Resource class for the QROMStatePreparation template.

    This operation implements the state preparation method described
    in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
    :class:`~.pennylane.estimator.QROM` to dynamically load the rotation angles.

    .. note::

        This decomposition assumes an appropriately sized phase gradient state is available.
        Users should ensure the cost of constructing such a state has been accounted for.
        See also :class:`~.pennylane.pennylane.estimator.PhaseGradient`.

    Args:
        num_state_qubits (int): number of qubits required to represent the statevector
        precision (float): the precision threshold for loading in the binary representation
            of the rotation angles
        positive_and_real (bool): indicates whether or not the coefficients of the statevector are all real
            and positive
        select_swap_depths (int | Iterable(int) | None): A parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth.
            Can be ``None``, ``1`` or a positive integer power of two.
            Defaults to ``None``, which internally corresponds to the optimal depth.
        wires (WiresLike | None): The wires on which to prepare the target state. This excludes any
            additional qubits allocated during the decomposition (via select-swap).

    Resources:
        The resources for QROMStatePreparation are computed according to the decomposition described
        in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
        :class:`~.pennylane.estimator.QROM` to dynamically load the rotation angles.
        These rotations gates are implemented using an in-place controlled-adder operation
        (see figure 4. of `arXiv:2409.07332 <https://arxiv.org/abs/2409.07332>`_) to a phase gradient state.

    .. seealso:: :class:`~.QROMStatePreparation`

    **Example**

    The resources for this operation are computed using:

    >>> qrom_prep = qre.QROMStatePreparation(num_state_qubits=5, precision=1e-3)
    >>> print(qre.estimate(qrom_prep))
    --- Resources: ---
     Total wires: 28
        algorithmic wires: 5
        allocated wires: 23
             zero state: 23
             any state: 0
     Total gates : 2.756E+3
      'Toffoli': 236,
      'CNOT': 1.522E+3,
      'X': 230,
      'Z': 12,
      'S': 24,
      'Hadamard': 732

    .. details::
        :title: Usage Details

        This operation uses the :code:`QROM` subroutine to dynamically load the rotation angles.

        >>> import pennylane.estimator as qre
        >>> gate_set = {"QROM", "Hadamard", "CNOT", "T", "Adjoint(QROM)"}
        >>> qrom_prep = qre.QROMStatePreparation(
        ...     num_state_qubits = 4,
        ...     precision = 1e-2,
        ...     select_swap_depths = 1,
        ... )
        >>> res = qre.estimate(qrom_prep, gate_set)
        >>> print(res)
        --- Resources: ---
         Total wires: 21
            algorithmic wires: 4
            allocated wires: 17
                 zero state: 17
                 any state: 0
         Total gates : 2.680E+3
          'QROM': 5,
          'Adjoint(QROM)': 5,
          'T': 1.832E+3,
          'CNOT': 580,
          'Hadamard': 258

        The ``precision`` argument is used to allocate the target wires in the underlying QROM
        operations. It corresponds to the precision with which the rotation angles of the
        template are encoded. This means that the binary representation of the angle is truncated up to
        the :math:`m`-th digit, where :math:`m` is the number of precision wires allocated. See  Eq. 5
        in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_ for more details.

        The ``select_swap_depths`` parameter allows a user to configure the ``select_swap_depth`` of
        each individual :class:`~.pennylane.estimator.QROM` used. The
        ``select_swap_depths`` argument can be one of :code:`(int, None, Iterable(int, None))`.

        If an integer or :code:`None` is passed (the default value for this parameter is 1), then that
        is used as the ``select_swap_depth`` for all :code:`QROM` operations in the resource decomposition.

        >>> print(res.gate_breakdown())
        Adjoint(QROM) total: 5
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=9, params={'num_bit_flips':4, 'num_bitstrings':1, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=10, params={'num_bit_flips':9, 'num_bitstrings':2, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=11, params={'num_bit_flips':18, 'num_bitstrings':4, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=12, params={'num_bit_flips':36, 'num_bitstrings':8, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=13, params={'num_bit_flips':72, 'num_bitstrings':16, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
        QROM total: 5
            QROM {'num_bit_flips': 4, 'num_bitstrings': 1, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 9, 'num_bitstrings': 2, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 18, 'num_bitstrings': 4, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 36, 'num_bitstrings': 8, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 72, 'num_bitstrings': 16, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
        T total: 1.832E+3
        CNOT total: 580
        Hadamard total: 258

        Alternatively, we can configure each value independently by specifying a list. Note the size
        of this list should be :code:`num_state_qubits + 1` (or :code:`num_state_qubits` if the state
        is positive and real).

        >>> qrom_prep = qre.QROMStatePreparation(
        ...     num_state_qubits = 4,
        ...     precision = 1e-2,
        ...     select_swap_depths = [1, None, 1, 1, None],
        ... )
        >>> res = qre.estimate(qrom_prep, gate_set)
        >>> print(res.gate_breakdown())
        Adjoint(QROM) total: 5
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=9, params={'num_bit_flips':4, 'num_bitstrings':1, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=10, params={'num_bit_flips':9, 'num_bitstrings':2, 'restored':False, 'select_swap_depth':None, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=11, params={'num_bit_flips':18, 'num_bitstrings':4, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=12, params={'num_bit_flips':36, 'num_bitstrings':8, 'restored':False, 'select_swap_depth':1, 'size_bitstring':9})}: 1
            Adjoint(QROM) {'base_cmpr_op': CompressedResourceOp(QROM, num_wires=13, params={'num_bit_flips':72, 'num_bitstrings':16, 'restored':False, 'select_swap_depth':None, 'size_bitstring':9})}: 1
        QROM total: 5
            QROM {'num_bit_flips': 4, 'num_bitstrings': 1, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 9, 'num_bitstrings': 2, 'restored': False, 'select_swap_depth': None, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 18, 'num_bitstrings': 4, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 36, 'num_bitstrings': 8, 'restored': False, 'select_swap_depth': 1, 'size_bitstring': 9}: 1
            QROM {'num_bit_flips': 72, 'num_bitstrings': 16, 'restored': False, 'select_swap_depth': None, 'size_bitstring': 9}: 1
        T total: 1.832E+3
        CNOT total: 580
        Hadamard total: 258
    """

    resource_keys = {"num_state_qubits", "precision", "positive_and_real", "selswap_depths"}

    def __init__(
        self,
        num_state_qubits: int,
        precision: float | None = None,
        positive_and_real: bool = False,
        select_swap_depths: int = 1,
        wires: WiresLike = None,
    ):
        # Overriding the default init method to allow for CompactState as an input.
        self.num_wires = num_state_qubits
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.precision = precision
        self.positive_and_real = positive_and_real

        expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1

        if isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != expected_size:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be {expected_size}, got {len(select_swap_depths)}"
                )
        elif not (isinstance(select_swap_depths, int) or select_swap_depths is None):
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        self.selswap_depths = select_swap_depths
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_state_qubits (int): number of qubits required to represent the state-vector
                * precision (float): the precision threshold for loading in the binary representation
                  of the rotation angles
                * positive_and_real (bool): flag that the coefficients of the statevector are all real
                  and positive
                * selswap_depths (int | Iterable(int) | None): a parameter of :code:`QROM`
                  used to trade-off extra qubits for reduced circuit depth
        """

        return {
            "num_state_qubits": self.num_wires,
            "precision": self.precision,
            "positive_and_real": self.positive_and_real,
            "selswap_depths": self.selswap_depths,
        }

    @classmethod
    def resource_rep(
        cls,
        num_state_qubits: int,
        precision: float | None = None,
        positive_and_real: bool = False,
        selswap_depths=1,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            precision (float): the precision threshold for loading in the binary representation
                of the rotation angles
            positive_and_real (bool): flag that the coefficients of the statevector are all real
                and positive
            selswap_depths (int | Iterable(int) | None): a parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1
        if isinstance(selswap_depths, (list, tuple, np.ndarray)):
            if len(selswap_depths) != expected_size:
                raise ValueError(
                    f"Expected the length of `selswap_depths` to be {expected_size}, got {len(selswap_depths)}"
                )
        elif not (isinstance(selswap_depths, int) or selswap_depths is None):
            raise TypeError("`selswap_depths` must be an integer, None or iterable")

        params = {
            "num_state_qubits": num_state_qubits,
            "precision": precision,
            "positive_and_real": positive_and_real,
            "selswap_depths": selswap_depths,
        }
        num_wires = num_state_qubits
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def _decomp_selection_helper(
        cls,
        use_phase_grad_trick: bool,
        num_state_qubits: int,
        positive_and_real: bool,
        precision: float | None = None,
        selswap_depths=1,
    ) -> list[GateCount]:
        r"""A private function which implements two variants of the decomposition of QROMStatePrep,
        based on the value of the :code:`use_phase_grad_trick` argument.

        Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            use_phase_grad_trick (bool): a flag which determines if the phase gradient trick is used
                instead of controlled-RY gates and phaseshifts.
            num_state_qubits (int): number of qubits required to represent the state-vector
            positive_and_real (bool): flag that the coefficients of the statevector are all real
                and positive
            precision (float): the precision threshold for loading in the binary representation
                of the rotation angles
            select_swap_depths (int | Iterable(int) | None): a parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth

        Resources:
            The resources for QROMStatePreparation are according to the decomposition as described
            in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
            :class:`~.pennylane.estimator.QROM` to dynamically load the rotation angles.

            Controlled-RY (and phase shifts) gates are used to apply all of the rotations coherently. If
            :code:`use_phase_grad_trick == True` then these rotations gates are implmented using an
            inplace controlled semi-adder operation (see figure 4. of
            `arXiv:2409.07332 <https://arxiv.org/abs/2409.07332>`_).

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
            ``GateCount`` objects, where each object represents a specific quantum gate and the
            number of times it appears in the decomposition.
        """
        gate_counts = []

        expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1
        if isinstance(selswap_depths, int) or selswap_depths is None:
            selswap_depths = [selswap_depths] * expected_size

        num_precision_wires = math.ceil(math.log2(math.pi / precision))
        gate_counts.append(Allocate(num_precision_wires))

        for j in range(num_state_qubits):
            num_bitstrings = 2**j
            num_bit_flips = num_bitstrings * num_precision_wires // 2

            gate_counts.append(
                GateCount(
                    qre.QROM.resource_rep(
                        num_bitstrings=num_bitstrings,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=num_bit_flips,
                        restored=False,
                        select_swap_depth=selswap_depths[j],
                    )
                )
            )

            gate_counts.append(
                GateCount(
                    qre.Adjoint.resource_rep(
                        qre.resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": num_bitstrings,
                                "num_bit_flips": num_bit_flips,
                                "size_bitstring": num_precision_wires,
                                "restored": False,
                                "select_swap_depth": selswap_depths[j],
                            },
                        ),
                    )
                )
            )

        if use_phase_grad_trick:
            semi_adder = qre.SemiAdder.resource_rep(max_register_size=num_precision_wires)
            h = qre.Hadamard.resource_rep()
            s = qre.S.resource_rep()
            s_dagg = qre.Adjoint.resource_rep(base_cmpr_op=s)
            gate_counts.append(
                GateCount(
                    qre.Controlled.resource_rep(
                        base_cmpr_op=semi_adder, num_ctrl_wires=1, num_zero_ctrl=0
                    ),
                    count=num_state_qubits,
                )
            )
            gate_counts.append(GateCount(h, 2 * num_precision_wires))
            gate_counts.append(GateCount(s, num_precision_wires))
            gate_counts.append(
                GateCount(s_dagg, num_precision_wires)
            )  # map RY rotations to RZ for phase grad

        else:
            cry = qre.CRY.resource_rep()
            gate_counts.append(GateCount(cry, num_precision_wires * num_state_qubits))

        if not positive_and_real:
            gate_counts.append(
                GateCount(
                    qre.QROM.resource_rep(
                        num_bitstrings=2**num_state_qubits,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=((2**num_state_qubits) * num_precision_wires // 2),
                        restored=False,
                        select_swap_depth=selswap_depths[-1],
                    )
                )
            )

            gate_counts.append(
                GateCount(
                    qre.Adjoint.resource_rep(
                        qre.resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 2**num_state_qubits,
                                "size_bitstring": num_precision_wires,
                                "num_bit_flips": ((2**num_state_qubits) * num_precision_wires // 2),
                                "restored": False,
                                "select_swap_depth": selswap_depths[-1],
                            },
                        ),
                    )
                )
            )

            if use_phase_grad_trick:
                semi_adder = qre.SemiAdder.resource_rep(max_register_size=num_precision_wires)
                gate_counts.append(
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=semi_adder, num_ctrl_wires=1, num_zero_ctrl=0
                        ),
                    )
                )
            else:
                phase_shift = qre.PhaseShift.resource_rep()
                gate_counts.append(GateCount(phase_shift, num_precision_wires))

        gate_counts.append(Deallocate(num_precision_wires))
        return gate_counts

    @classmethod
    def controlled_ry_resource_decomp(
        cls,
        num_state_qubits: int,
        positive_and_real: bool,
        precision: float | None = None,
        selswap_depths=1,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            positive_and_real (bool): Flag that the coefficients of the statevector are all real
                and positive.
            precision (float): The precision threshold for loading in the binary representation
                of the rotation angles.
            selswap_depths (int | Iterable(int) | None): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth.

        Resources:
            The resources for QROMStatePreparation are according to the decomposition as described
            in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
            :class:`~.pennylane.estimator.QROM` to dynamically load the rotation angles.
            Controlled-RY (and phase shifts) gates are used to apply all of the rotations coherently.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
            ``GateCount`` objects, where each object represents a specific quantum gate and the
            number of times it appears in the decomposition.
        """
        return cls._decomp_selection_helper(
            use_phase_grad_trick=False,
            num_state_qubits=num_state_qubits,
            positive_and_real=positive_and_real,
            precision=precision,
            selswap_depths=selswap_depths,
        )

    @classmethod
    def resource_decomp(
        cls,
        num_state_qubits: int,
        positive_and_real: bool,
        precision: float | None = None,
        selswap_depths=1,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.pennylane.estimator.PhaseGradient`.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            positive_and_real (bool): Flag that the coefficients of the statevector are all real
                and positive.
            precision (float): The precision threshold for loading in the binary representation
                of the rotation angles.
            selswap_depths (int | Iterable(int) | None): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth.

        Resources:
            The resources for QROMStatePreparation are according to the decomposition as described
            in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
            :class:`~.pennylane.estimator.QROM` to dynamically load the rotation angles.
            These rotations gates are implmented using an inplace controlled-adder operation
            (see figure 4. of `arXiv:2409.07332 <https://arxiv.org/abs/2409.07332>`_) to phase gradient.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
            ``GateCount`` objects, where each object represents a specific quantum gate and the
            number of times it appears in the decomposition.
        """
        return cls._decomp_selection_helper(
            use_phase_grad_trick=True,
            num_state_qubits=num_state_qubits,
            positive_and_real=positive_and_real,
            precision=precision,
            selswap_depths=selswap_depths,
        )


class PrepTHC(ResourceOperator):
    r"""Resource class for preparing the state for tensor hypercontracted (THC) Hamiltonian.

    This operator customizes the Prepare circuit based on the structure of THC Hamiltonian.

    Args:
        thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a tensor hypercontracted
            Hamiltonian for which the state is being prepared
        coeff_precision (int | None): The number of bits used to represent the precision for loading
            the coefficients of Hamiltonian. If :code:`None` is provided, the default value from the
            :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
        select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
            used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        wires (WiresLike | None): the wires on which the operator acts

    Resources:
        The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> thc_ham = qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
    >>> res = qre.estimate(qre.PrepTHC(thc_ham, coeff_precision=15))
    >>> print(res)
    --- Resources: ---
     Total wires: 185
        algorithmic wires: 12
        allocated wires: 173
             zero state: 28
             any state: 145
     Total gates : 1.485E+4
      'Toffoli': 467,
      'CNOT': 1.307E+4,
      'X': 512,
      'Hadamard': 797

    """

    resource_keys = {"thc_ham", "coeff_precision", "select_swap_depth"}

    def __init__(
        self,
        thc_ham: THCHamiltonian,
        coeff_precision: int | None = None,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ):

        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        if not isinstance(coeff_precision, int) and coeff_precision is not None:
            raise TypeError(
                f"`coeff_precision` must be an integer, but type {type(coeff_precision)} was provided."
            )

        self.thc_ham = thc_ham
        self.coeff_precision = coeff_precision
        self.select_swap_depth = select_swap_depth
        tensor_rank = thc_ham.tensor_rank
        self.num_wires = 2 * int(math.ceil(math.log2(tensor_rank + 1)))
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * thc_ham (:class:`~.pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a tensor hypercontracted
                  Hamiltonian for which the state is being prepared
                * coeff_precision (int | None): The number of bits used to represent the precision for loading
                  the coefficients of Hamiltonian. If :code:`None` is provided, the default value from the
                  :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
                * select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                  used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {
            "thc_ham": self.thc_ham,
            "coeff_precision": self.coeff_precision,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls,
        thc_ham: THCHamiltonian,
        coeff_precision: int | None = None,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a tensor hypercontracted
                Hamiltonian for which the state is being prepared
            coeff_precision (int | None): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided, the default value from the
                :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        if not isinstance(coeff_precision, int) and coeff_precision is not None:
            raise TypeError(
                f"`coeff_precision` must be an integer, but type {type(coeff_precision)} was provided."
            )

        tensor_rank = thc_ham.tensor_rank
        num_wires = 2 * int(math.ceil(math.log2(tensor_rank + 1)))

        params = {
            "thc_ham": thc_ham,
            "coeff_precision": coeff_precision,
            "select_swap_depth": select_swap_depth,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        thc_ham: THCHamiltonian,
        coeff_precision: int | None = None,
        select_swap_depth: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision (int | None): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided, the default value from the
                :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        num_coeff = num_orb + tensor_rank * (tensor_rank + 1) / 2  # N+M(M+1)/2
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(math.ceil(math.log2(tensor_rank + 1)))

        gate_list = []

        # 6 auxiliary account for 2 spin registers, 1 for rotation on auxiliary, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
        gate_list.append(Allocate(coeff_register + 2 * m_register + 2 * coeff_precision + 6))

        hadamard = resource_rep(qre.Hadamard)

        gate_list.append(qre.GateCount(hadamard, 2 * m_register))

        # Figure - 3

        # Inquality tests
        toffoli = resource_rep(qre.Toffoli)
        gate_list.append(qre.GateCount(toffoli, 4 * m_register - 4))

        # Reflection on 5 registers
        ccz = resource_rep(qre.CCZ)
        gate_list.append(
            qre.GateCount(
                resource_rep(
                    qre.Controlled,
                    {"base_cmpr_op": ccz, "num_ctrl_wires": 1, "num_zero_ctrl": 0},
                ),
                1,
            )
        )
        gate_list.append(qre.GateCount(toffoli, 2))

        gate_list.append(qre.GateCount(hadamard, 2 * m_register))

        # Rotate and invert the rotation of ancilla to obtain amplitude of success
        gate_list.append(Allocate(coeff_precision))
        gate_list.append(qre.GateCount(toffoli, 2 * (coeff_precision - 3)))
        gate_list.append(Deallocate(coeff_precision))

        # Reflecting about the success amplitude
        gate_list.append(qre.GateCount(ccz, 2 * m_register - 1))

        gate_list.append(qre.GateCount(hadamard, 2 * m_register))

        # Inequality tests
        gate_list.append(qre.GateCount(toffoli, 4 * m_register - 4))

        # Checking that inequality is satisfied
        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 3, "num_zero_ctrl": 0})
        gate_list.append(qre.GateCount(mcx, 1))
        gate_list.append(qre.GateCount(toffoli, 2))

        x = resource_rep(qre.X)
        gate_list.append(qre.GateCount(x, 2))

        # Figure- 4(Subprepare Circuit)
        gate_list.append(qre.GateCount(hadamard, coeff_precision + 1))

        # Contiguous register cost Eq.29 in arXiv:2011.03494
        gate_list.append(qre.GateCount(toffoli, m_register**2 + m_register - 1))

        # QROM for keep values Eq.31 in arXiv:2011.03494
        qrom_coeff = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": num_coeff,
                "size_bitstring": 2 * m_register + 2 + coeff_precision,
                "restored": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(qre.GateCount(qrom_coeff, 1))

        # Inequality test between alt and keep registers
        comparator = resource_rep(
            qre.RegisterComparator,
            {
                "first_register": coeff_precision,
                "second_register": coeff_precision,
                "geq": False,
            },
        )
        gate_list.append(qre.GateCount(comparator))

        cz = resource_rep(qre.CZ)
        gate_list.append(qre.GateCount(cz, 2))
        gate_list.append(qre.GateCount(x, 2))

        # Swap \mu and \nu registers with alt registers
        cswap = resource_rep(qre.CSWAP)
        gate_list.append(qre.GateCount(cswap, 2 * m_register))

        # Swap \mu and \nu registers controlled on |+> state and success of inequality
        gate_list.append(qre.GateCount(cswap, m_register))
        gate_list.append(qre.GateCount(toffoli, 1))

        return gate_list

    @classmethod
    def adjoint_resource_decomp(
        cls,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the adjoint of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        thc_ham = target_resource_params["thc_ham"]
        coeff_precision = target_resource_params["coeff_precision"]
        select_swap_depth = target_resource_params.get("select_swap_depth", None)
        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        num_coeff = num_orb + tensor_rank * (tensor_rank + 1) / 2
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(math.ceil(math.log2(tensor_rank + 1)))
        gate_list = []

        hadamard = resource_rep(qre.Hadamard)
        gate_list.append(qre.GateCount(hadamard, 2 * m_register))

        # Figure - 3

        # Inquality tests from arXiv:2011.03494
        toffoli = resource_rep(qre.Toffoli)
        gate_list.append(qre.GateCount(toffoli, 4 * m_register - 4))

        # Reflection on 5 registers
        ccz = resource_rep(qre.CCZ)
        gate_list.append(
            qre.GateCount(
                resource_rep(
                    qre.Controlled,
                    {"base_cmpr_op": ccz, "num_ctrl_wires": 1, "num_zero_ctrl": 0},
                ),
                1,
            )
        )
        gate_list.append(qre.GateCount(toffoli, 2))

        gate_list.append(qre.GateCount(hadamard, 2 * m_register))

        # Rotate and invert the rotation of ancilla to obtain amplitude of success
        gate_list.append(Allocate(coeff_precision))
        gate_list.append(qre.GateCount(toffoli, 2 * (coeff_precision - 3)))
        gate_list.append(Deallocate(coeff_precision))

        # Reflecting about the success amplitude
        gate_list.append(qre.GateCount(ccz, 2 * m_register - 1))

        gate_list.append(qre.GateCount(hadamard, 2 * m_register))

        # Inequality tests
        gate_list.append(qre.GateCount(toffoli, 4 * m_register - 4))

        # Checking that inequality is satisfied
        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 3, "num_zero_ctrl": 0})
        gate_list.append(qre.GateCount(mcx, 1))
        gate_list.append(qre.GateCount(toffoli, 2))

        x = resource_rep(qre.X)
        gate_list.append(qre.GateCount(x, 2))

        # Figure- 4 (Subprepare Circuit)
        gate_list.append(qre.GateCount(hadamard, coeff_precision + 1))

        # Contiguous register cost
        gate_list.append(qre.GateCount(toffoli, m_register**2 + m_register - 1))

        # Adjoint of QROM for keep values Eq.32 in arXiv:2011.03494
        qrom_adj = resource_rep(
            qre.Adjoint,
            {
                "base_cmpr_op": resource_rep(
                    qre.QROM,
                    {
                        "num_bitstrings": num_coeff,
                        "size_bitstring": 2 * m_register + 2 + coeff_precision,
                        "restored": False,
                        "select_swap_depth": select_swap_depth,
                    },
                )
            },
        )
        gate_list.append(qre.GateCount(qrom_adj, 1))

        cz = resource_rep(qre.CZ)
        gate_list.append(qre.GateCount(cz, 2))
        gate_list.append(qre.GateCount(x, 2))

        # Swap \mu and \nu registers with alt registers
        cswap = resource_rep(qre.CSWAP)
        gate_list.append(qre.GateCount(cswap, 2 * m_register))

        # Swap \mu and \nu registers controlled on |+> state and success of inequality
        gate_list.append(qre.GateCount(cswap, m_register))
        gate_list.append(qre.GateCount(toffoli, 1))

        # Free Prepare Wires
        # 6 ancillas account for 2 spin registers, 1 for rotation on ancilla, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
        gate_list.append(Deallocate(coeff_register + 2 * m_register + 2 * coeff_precision + 6))

        return gate_list
