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
from typing import Dict

import pennylane.numpy as np
from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ, protected-access, non-parent-init-called, too-many-arguments, unused-argument


class ResourceUniformStatePrep(ResourceOperator):
    r"""Resource class for preparing a uniform superposition.

    This operation prepares a uniform superposition over a given number of
    basis states. The uniform superposition is defined as:

    .. math::

        \frac{1}{\sqrt{l}} \sum_{i=0}^{l} |i\rangle

    where :math:`l` is the number of states.

    This operation uses ``Hadamard`` gates to create the uniform superposition when
    the number of states is a power of two. If the number of states is not a power of two,
    amplitude amplification technique defined in
    `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_ is used.

    Args:
        num_states (int): the number of states in the uniform superposition
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from Figure 12 in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
        The circuit uses amplitude amplification to prepare a uniform superposition over :math:`l`
        basis states.

    **Example**

    The resources for this operation are computed using:

    >>> unif_state_prep = plre.ResourceUniformStatePrep(10)
    >>> print(plre.estimate_resources(unif_state_prep))
    --- Resources: ---
     Total qubits: 5
     Total gates : 124
     Qubit breakdown:
      clean qubits: 1, dirty qubits: 0, algorithmic qubits: 4
     Gate breakdown:
      {'Hadamard': 16, 'X': 12, 'CNOT': 4, 'Toffoli': 4, 'T': 88}
    """

    resource_keys = {"num_states"}

    def __init__(self, num_states, wires=None):
        self.num_states = num_states
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2**k)
        if L == 1:
            self.num_wires = k
        else:
            self.num_wires = k + int(math.ceil(math.log2(L)))
        super().__init__(wires=wires)

    @property
    def resource_params(self):
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
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"num_states": num_states})

    @classmethod
    def default_resource_decomp(cls, num_states, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_states (int): the number of states over which the uniform superposition is being prepared

        Resources:
            The resources are obtained from Figure 12 in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
            The circuit uses amplitude amplification to prepare a uniform superposition over :math:`l` basis states.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_lst = []
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2**k)
        if L == 1:
            gate_lst.append(GateCount(resource_rep(plre.ResourceHadamard), k))
            return gate_lst

        logl = int(math.ceil(math.log2(L)))
        gate_lst.append(GateCount(resource_rep(plre.ResourceHadamard), k + 3 * logl))
        gate_lst.append(
            GateCount(
                resource_rep(plre.ResourceIntegerComparator, {"value": L, "register_size": logl}), 1
            )
        )
        gate_lst.append(GateCount(resource_rep(plre.ResourceRZ), 2))
        gate_lst.append(
            GateCount(
                resource_rep(
                    plre.ResourceAdjoint,
                    {
                        "base_cmpr_op": resource_rep(
                            plre.ResourceIntegerComparator, {"value": L, "register_size": logl}
                        )
                    },
                ),
                1,
            )
        )

        return gate_lst


class ResourceAliasSampling(ResourceOperator):
    r"""Resource class for preparing a state using coherent alias sampling.

    Args:
        num_coeffs (int): the number of unique coefficients in the state
        precision (float): the precision with which the coefficients are loaded
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from Section III D in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
        The circuit uses coherent alias sampling to prepare a state with the given coefficients.

    **Example**

    The resources for this operation are computed using:

    >>> alias_sampling = plre.ResourceAliasSampling(num_coeffs=100)
    >>> print(plre.estimate_resources(alias_sampling))
    --- Resources: ---
     Total qubits: 81
     Total gates : 6.157E+3
     Qubit breakdown:
      clean qubits: 6, dirty qubits: 68, algorithmic qubits: 7
     Gate breakdown:
      {'Hadamard': 730, 'X': 479, 'CNOT': 4.530E+3, 'Toffoli': 330, 'T': 88}
    """

    resource_keys = {"num_coeffs", "precision"}

    def __init__(self, num_coeffs, precision=None, wires=None):
        self.num_coeffs = num_coeffs
        self.precision = precision
        self.num_wires = int(math.ceil(math.log2(num_coeffs)))
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_coeffs (int): the number of unique coefficients in the state
                * precision (float): the precision with which the coefficients are loaded

        """
        return {"num_coeffs": self.num_coeffs, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_coeffs, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"num_coeffs": num_coeffs, "precision": precision})

    @classmethod
    def default_resource_decomp(cls, num_coeffs, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_coeffs (int): the number of unique coefficients in the state
            precision (float): the precision with which the coefficients are loaded

        Resources:
            The resources are obtained from Section III D in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
            The circuit uses coherent alias sampling to prepare a state with the given coefficients.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_lst = []

        logl = int(math.ceil(math.log2(num_coeffs)))
        precision = precision or kwargs["config"]["precision_alias_sampling"]

        num_prec_wires = abs(math.floor(math.log2(precision)))

        gate_lst.append(AllocWires(logl + 2 * num_prec_wires + 1))

        gate_lst.append(
            GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": num_coeffs}), 1)
        )
        gate_lst.append(GateCount(resource_rep(plre.ResourceHadamard), num_prec_wires))
        gate_lst.append(
            GateCount(
                resource_rep(
                    plre.ResourceQROM,
                    {"num_bitstrings": num_coeffs, "size_bitstring": logl + num_prec_wires},
                ),
                1,
            )
        )
        gate_lst.append(
            GateCount(
                resource_rep(
                    plre.ResourceRegisterComparator,
                    {"first_register": num_prec_wires, "second_register": num_prec_wires},
                ),
                1,
            )
        )
        gate_lst.append(GateCount(resource_rep(plre.ResourceCSWAP), logl))

        return gate_lst


class ResourceMPSPrep(ResourceOperator):
    r"""Resource class for the MPSPrep template.

    The resource operation for preparing an initial state from a matrix product state (MPS)
    representation.

    Args:
        num_mps_matrices (int): the number of matrices in the MPS representation
        max_bond_dim (int): the bond dimension of the MPS representation
        precision (Union[None, float], optional): the precision used when loading the MPS matricies
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources for MPSPrep are according to the decomposition, which uses the generic
        :class:`~.labs.resource_estimation.ResourceQubitUnitary`. The decomposition is based on
        the routine described in `arXiv:2310.18410 <https://arxiv.org/pdf/2310.18410>`_.

    .. seealso:: :class:`~.MPSPrep`

    **Example**

    The resources for this operation are computed using:

    >>> mps = plre.ResourceMPSPrep(num_mps_matrices=10, max_bond_dim=2**3)
    >>> print(plre.estimate_resources(mps, gate_set={"CNOT", "RZ", "RY"}))
    --- Resources: ---
     Total qubits: 13
     Total gates : 1.654E+3
     Qubit breakdown:
      clean qubits: 3, dirty qubits: 0, algorithmic qubits: 10
     Gate breakdown:
      {'RZ': 728, 'CNOT': 774, 'RY': 152}
    """

    resource_keys = {"num_mps_matrices", "max_bond_dim", "precision"}

    def __init__(self, num_mps_matrices, max_bond_dim, precision=None, wires=None):
        self.num_wires = num_mps_matrices
        self.precision = precision
        self.max_bond_dim = max_bond_dim
        self.num_mps_matrices = num_mps_matrices
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_mps_matrices (int): the number of matrices in the MPS representation
                * max_bond_dim (int): the bond dimension of the MPS representation
                * precision (Union[None, float], optional): the precision used when loading the
                  MPS matrices
        """
        return {
            "num_mps_matrices": self.num_mps_matrices,
            "max_bond_dim": self.max_bond_dim,
            "precision": self.precision,
        }

    @classmethod
    def resource_rep(cls, num_mps_matrices, max_bond_dim, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_mps_matrices (int): the number of matrices in the MPS representation
            max_bond_dim (int): the bond dimension of the MPS representation
            precision (Union[None, float], optional): the precision used when loading the MPS matrices

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "num_mps_matrices": num_mps_matrices,
            "max_bond_dim": max_bond_dim,
            "precision": precision,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls,
        num_mps_matrices,
        max_bond_dim,
        precision=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_mps_matrices (int): the number of matrices in the MPS representation
            max_bond_dim (int): the bond dimension of the MPS representation
            precision (Union[None, float], optional): the precision used when loading
                the MPS matrices

        Resources:
            The resources for MPSPrep are estimated according to the decomposition, which uses the generic
            :class:`~.labs.resource_estimation.ResourceQubitUnitary`. The decomposition is based on
            the routine described in `arXiv:2310.18410 <https://arxiv.org/pdf/2310.18410>`_.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = precision or kwargs["config"]["precision_mps_prep"]
        num_work_wires = min(
            math.ceil(math.log2(max_bond_dim)), math.ceil(num_mps_matrices / 2)  # truncate bond dim
        )

        gate_lst = [AllocWires(num_work_wires)]

        for index in range(1, num_mps_matrices + 1):
            qubit_unitary_wires = min(index + 1, num_work_wires + 1, (num_mps_matrices - index) + 2)
            qubit_unitary = plre.ResourceQubitUnitary.resource_rep(
                num_wires=qubit_unitary_wires, precision=precision
            )
            gate_lst.append(GateCount(qubit_unitary))

        gate_lst.append(FreeWires(num_work_wires))
        return gate_lst

    @classmethod
    def tracking_name(cls, num_mps_matrices, max_bond_dim, precision) -> str:
        return f"MPSPrep({num_mps_matrices}, {max_bond_dim}, {precision})"


class ResourceQROMStatePreparation(ResourceOperator):
    r"""Resource class for the QROMStatePreparation template.

    This operation implements the state preparation method described
    in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
    :class:`~.labs.resource_estimation.ResourceQROM` to dynamically load the rotation angles.

    .. note::

        This decomposition assumes an appropriately sized phase gradient state is available.
        Users should ensure the cost of constructing such a state has been accounted for.
        See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

    Args:
        num_state_qubits (int): number of qubits required to represent the state-vector
        precision (float): the precision threshold for loading in the binary representation
            of the rotation angles
        positive_and_real (bool): flag that the coefficients of the statevector are all real
            and positive
        select_swap_depths (Union[None, int, Iterable(int)], optional): a parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources for QROMStatePreparation are computed according to the decomposition described
        in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
        :class:`~.labs.resource_estimation.ResourceQROM` to dynamically load the rotation angles.
        These rotations gates are implmented using an inplace controlled-adder operation
        (see figure 4. of `arXiv:2409.07332 <https://arxiv.org/pdf/2409.07332>`_) to a phase gradient.

    .. seealso:: :class:`~.QROMStatePreparation`

    **Example**

    The resources for this operation are computed using:

    >>> qrom_prep = plre.ResourceQROMStatePreparation(num_state_qubits=5, precision=1e-3)
    >>> print(plre.estimate_resources(qrom_prep, gate_set=plre.StandardGateSet))
    --- Resources: ---
     Total qubits: 19
     Total gates : 1.116E+3
     Qubit breakdown:
      clean qubits: 14, dirty qubits: 0, algorithmic qubits: 5
     Gate breakdown:
      {'X': 220, 'Toffoli': 104, 'CNOT': 370, 'Hadamard': 312, 'RY': 100, 'PhaseShift': 10}

    .. details::
        :title: Usage Details

        This operation uses the :code:`QROM` subroutine to dynamically load the rotation angles.

        >>> gate_set = {"QROM", "Hadamard", "CNOT", "T", "Adjoint(QROM)"}
        >>> qrom_prep = plre.ResourceQROMStatePreparation(
        ...     num_state_qubits = 4,
        ...     precision = 1e-2,
        ...     select_swap_depths = 2,  # default value is 1
        ... )
        >>> res = plre.estimate_resources(qrom_prep, gate_set)
        >>> print(res)
        --- Resources: ---
         Total qubits: 11
         Total gates : 1.137E+3
         Qubit breakdown:
          clean qubits: 7, dirty qubits: 0, algorithmic qubits: 4
         Gate breakdown:
          {'QROM': 5, 'Adjoint(QROM)': 5, 'CNOT': 56, 'T': 1.071E+3}

        The ``precision`` argument is used to allocate the target wires in the underlying QROM
        operations. It corresponds to the precision with which the rotation angles of the
        template are encoded. This means that the binary representation of the angle is truncated up to
        the :math:`m`-th digit, where :math:`m` is the number of precision wires allocated. See  Eq. 5
        in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_ for more details.

        The ``select_swap_depths`` parameter allows a user to configure the ``select_swap_depth`` of
        each individual :class:`~.labs.resource_estimation.ResourceQROM` used. The
        ``select_swap_depths`` argument can be one of :code:`(int, None, Iterable(int, None))`.

        If an integer or :code:`None` is passed (the default value for this parameter is 1), then that
        is used as the ``select_swap_depth`` for all :code:`QROM` operations in the resource decomposition.

        >>> for op in res.gate_types:
        ...     if op.name == "QROM":
        ...         print(op.name, op.params)
        ...
        QROM {'num_bitstrings': 1, 'num_bit_flips': 1, 'size_bitstring': 7, 'select_swap_depth': 2, 'clean': False}
        QROM {'num_bitstrings': 2, 'num_bit_flips': 1, 'size_bitstring': 7, 'select_swap_depth': 2, 'clean': False}
        QROM {'num_bitstrings': 4, 'num_bit_flips': 2, 'size_bitstring': 7, 'select_swap_depth': 2, 'clean': False}
        QROM {'num_bitstrings': 8, 'num_bit_flips': 4, 'size_bitstring': 7, 'select_swap_depth': 2, 'clean': False}
        QROM {'num_bitstrings': 16, 'num_bit_flips': 8, 'size_bitstring': 7, 'select_swap_depth': 2, 'clean': False}

        Alternatively, we can configure each value independantly by specifying a list. Note the size
        of this list should be :code:`num_state_qubits + 1` (:code:`num_state_qubits` if the state
        is positive and real).

        >>> qrom_prep = plre.ResourceQROMStatePreparation(
        ...     num_state_qubits = 4,
        ...     precision = 1e-2,
        ...     select_swap_depths = [1, None, 2, 2, None],
        ... )
        >>> res = plre.estimate_resources(qrom_prep, gate_set)
        >>> for op in res.gate_types:
        ...     if op.name == "QROM":
        ...         print(op.name, op.params)
        ...
        QROM {'num_bitstrings': 1, 'num_bit_flips': 1, 'size_bitstring': 7, 'select_swap_depth': 1, 'clean': False}
        QROM {'num_bitstrings': 2, 'num_bit_flips': 1, 'size_bitstring': 7, 'select_swap_depth': None, 'clean': False}
        QROM {'num_bitstrings': 4, 'num_bit_flips': 2, 'size_bitstring': 7, 'select_swap_depth': 2, 'clean': False}
        QROM {'num_bitstrings': 8, 'num_bit_flips': 4, 'size_bitstring': 7, 'select_swap_depth': 2, 'clean': False}
        QROM {'num_bitstrings': 16, 'num_bit_flips': 8, 'size_bitstring': 7, 'select_swap_depth': None, 'clean': False}
    """

    resource_keys = {"num_state_qubits", "precision", "positive_and_real", "selswap_depths"}

    def __init__(
        self,
        num_state_qubits,
        precision=None,
        positive_and_real=False,
        select_swap_depths=1,
        wires=None,
    ):
        # Overriding the default init method to allow for CompactState as an input.
        self.num_wires = num_state_qubits
        self.precision = precision
        self.positive_and_real = positive_and_real

        expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            pass
        elif isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != expected_size:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be {expected_size}, got {len(select_swap_depths)}"
                )
        else:
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        self.selswap_depths = select_swap_depths
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_state_qubits (int): number of qubits required to represent the state-vector
                * precision (float): The precision threshold for loading in the binary representation
                  of the rotation angles.
                * positive_and_real (bool): Flag that the coefficients of the statevector are all real
                  and positive.
                * selswap_depths (Union[None, int, Iterable(int)], optional): A parameter of :code:`QROM`
                  used to trade-off extra qubits for reduced circuit depth.
        """

        return {
            "num_state_qubits": self.num_wires,
            "precision": self.precision,
            "positive_and_real": self.positive_and_real,
            "selswap_depths": self.selswap_depths,
        }

    @classmethod
    def resource_rep(
        cls, num_state_qubits, precision=None, positive_and_real=False, selswap_depths=1
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            precision (float): The precision threshold for loading in the binary representation
                of the rotation angles.
            positive_and_real (bool): Flag that the coefficients of the statevector are all real
                and positive.
            selswap_depths (Union[None, int, Iterable(int)], optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1
        if isinstance(selswap_depths, int) or selswap_depths is None:
            pass
        elif isinstance(selswap_depths, (list, tuple, np.ndarray)):
            if len(selswap_depths) != expected_size:
                raise ValueError(
                    f"Expected the length of `selswap_depths` to be {expected_size}, got {len(selswap_depths)}"
                )
        else:
            raise TypeError("`selswap_depths` must be an integer, None or iterable")

        params = {
            "num_state_qubits": num_state_qubits,
            "precision": precision,
            "positive_and_real": positive_and_real,
            "selswap_depths": selswap_depths,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def _default_decomp_logic(
        cls,
        use_phase_grad_trick,
        num_state_qubits,
        positive_and_real,
        precision=None,
        selswap_depths=1,
        **kwargs,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            use_phase_grad_trick (bool): a flag which determines if the phase gradient trick is used
                instead of controlled-RY gates and phaseshifts.
            num_state_qubits (int): number of qubits required to represent the state-vector
            positive_and_real (bool): Flag that the coefficients of the statevector are all real
                and positive.
            precision (float): The precision threshold for loading in the binary representation
                of the rotation angles.
            select_swap_depths (Union[None, int, Iterable(int)], optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth.

        Resources:
            The resources for QROMStatePreparation are according to the decomposition as described
            in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
            :class:`~.labs.resource_estimation.ResourceQROM` to dynamically load the rotation angles.

            Controlled-RY (and phase shifts) gates are used to apply all of the rotations coherently. If
            :code:`use_phase_grad_trick == True` then these rotations gates are implmented using an
            inplace controlled semi-adder operation (see figure 4. of
            `arXiv:2409.07332 <https://arxiv.org/pdf/2409.07332>`_).

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_counts = []
        precision = precision or kwargs["config"]["precision_qrom_state_prep"]

        expected_size = num_state_qubits if positive_and_real else num_state_qubits + 1
        if isinstance(selswap_depths, int) or selswap_depths is None:
            selswap_depths = [selswap_depths] * expected_size

        num_precision_wires = math.ceil(math.log2(math.pi / precision)) + 1
        gate_counts.append(AllocWires(num_precision_wires))

        for j in range(num_state_qubits):
            num_bitstrings = 2**j
            num_bit_flips = max(2 ** (j - 1), 1)

            gate_counts.append(
                GateCount(
                    plre.ResourceQROM.resource_rep(
                        num_bitstrings=num_bitstrings,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=num_bit_flips,
                        clean=False,
                        select_swap_depth=selswap_depths[j],
                    )
                )
            )

            gate_counts.append(
                GateCount(
                    plre.ResourceAdjoint.resource_rep(
                        plre.resource_rep(
                            plre.ResourceQROM,
                            {
                                "num_bitstrings": num_bitstrings,
                                "num_bit_flips": num_bit_flips,
                                "size_bitstring": num_precision_wires,
                                "clean": False,
                                "select_swap_depth": selswap_depths[j],
                            },
                        ),
                    )
                )
            )

        if use_phase_grad_trick:
            semi_adder = plre.ResourceSemiAdder.resource_rep(max_register_size=num_precision_wires)
            h = plre.ResourceHadamard.resource_rep()
            s = plre.ResourceS.resource_rep()
            s_dagg = plre.ResourceAdjoint.resource_rep(base_cmpr_op=s)
            gate_counts.append(
                GateCount(
                    plre.ResourceControlled.resource_rep(
                        base_cmpr_op=semi_adder, num_ctrl_wires=1, num_ctrl_values=0
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
            cry = plre.ResourceCRY.resource_rep(eps=precision)
            gate_counts.append(GateCount(cry, num_precision_wires * num_state_qubits))

        if not positive_and_real:
            gate_counts.append(
                GateCount(
                    plre.ResourceQROM.resource_rep(
                        num_bitstrings=2**num_state_qubits,
                        size_bitstring=num_precision_wires,
                        num_bit_flips=2 ** (num_state_qubits - 1),
                        clean=False,
                        select_swap_depth=selswap_depths[-1],
                    )
                )
            )

            gate_counts.append(
                GateCount(
                    plre.ResourceAdjoint.resource_rep(
                        plre.resource_rep(
                            plre.ResourceQROM,
                            {
                                "num_bitstrings": 2**num_state_qubits,
                                "size_bitstring": num_precision_wires,
                                "num_bit_flips": 2 ** (num_state_qubits - 1),
                                "clean": False,
                                "select_swap_depth": selswap_depths[-1],
                            },
                        ),
                    )
                )
            )

            if use_phase_grad_trick:
                semi_adder = plre.ResourceSemiAdder.resource_rep(
                    max_register_size=num_precision_wires
                )
                gate_counts.append(
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            base_cmpr_op=semi_adder, num_ctrl_wires=1, num_ctrl_values=0
                        ),
                    )
                )
            else:
                phase_shift = plre.ResourcePhaseShift.resource_rep(eps=precision)
                gate_counts.append(GateCount(phase_shift, num_precision_wires))

        gate_counts.append(FreeWires(num_precision_wires))
        return gate_counts

    @classmethod
    def controlled_ry_resource_decomp(
        cls,
        num_state_qubits,
        positive_and_real,
        precision=None,
        selswap_depths=1,
        **kwargs,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            positive_and_real (bool): Flag that the coefficients of the statevector are all real
                and positive.
            precision (float): The precision threshold for loading in the binary representation
                of the rotation angles.
            select_swap_depths (Union[None, int, Iterable(int)], optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth.

        Resources:
            The resources for QROMStatePreparation are according to the decomposition as described
            in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
            :class:`~.labs.resource_estimation.ResourceQROM` to dynamically load the rotation angles.
            Controlled-RY (and phase shifts) gates are used to apply all of the rotations coherently.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return cls._default_decomp_logic(
            use_phase_grad_trick=False,
            num_state_qubits=num_state_qubits,
            positive_and_real=positive_and_real,
            precision=precision,
            selswap_depths=selswap_depths,
            **kwargs,
        )

    @classmethod
    def default_resource_decomp(
        cls, num_state_qubits, positive_and_real, precision=None, selswap_depths=1, **kwargs
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            positive_and_real (bool): Flag that the coefficients of the statevector are all real
                and positive.
            precision (float): The precision threshold for loading in the binary representation
                of the rotation angles.
            select_swap_depths (Union[None, int, Iterable(int)], optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth.

        Resources:
            The resources for QROMStatePreparation are according to the decomposition as described
            in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_, using
            :class:`~.labs.resource_estimation.ResourceQROM` to dynamically load the rotation angles.
            These rotations gates are implmented using an inplace controlled-adder operation
            (see figure 4. of `arXiv:2409.07332 <https://arxiv.org/pdf/2409.07332>`_) to phase gradient.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return cls._default_decomp_logic(
            use_phase_grad_trick=True,
            num_state_qubits=num_state_qubits,
            positive_and_real=positive_and_real,
            precision=precision,
            selswap_depths=selswap_depths,
            **kwargs,
        )
