# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Resource operators for PennyLane subroutine templates."""

import math

import pennylane.estimator as qre
from pennylane.allocation import AllocateState
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import WiresLike

from .wires_manager import Allocate, Deallocate

# pylint: disable=arguments-differ,too-many-arguments,unused-argument,super-init-not-called, signature-differs


class QROM(ResourceOperator):
    r"""Resource class for the Quantum Read-Only Memory (QROM) template.

    Args:
        num_bitstrings (int): the number of bitstrings that are to be encoded
        size_bitstring (int): the length of each bitstring
        num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
            :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
        restored (bool): Determine if allocated qubits should be reset after the computation
            (at the cost of higher gate counts). Defaults to :code:`True`.
        select_swap_depth (int | None): A parameter :math:`\lambda` that determines
            if data will be loaded in parallel by adding more rows following Figure 1.C of
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
            :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
            depth that minimizes T-gate count.
        wires (WiresLike | None): The wires the operation acts on (control and target), excluding
            any additional qubits allocated during the decomposition (e.g select-swap wires).

    Resources:
        The resources for QROM are derived from the following references:

        * :code:`restored=False`: Uses the Select-Swap tree decomposition from Figure 1.C of
          `Low et al. (2018) <https://arxiv.org/abs/1812.00954>`_, further optimized using the
          measurement-based uncomputation technique described in
          `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

        * :code:`restored=True`: Uses the standard QROM resource accounting from Figure 4 of
          `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

    .. seealso:: The associated PennyLane operation :class:`~.pennylane.QROM`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> qrom = qre.QROM(
    ...     num_bitstrings=10,
    ...     size_bitstring=4,
    ... )
    >>> print(qre.estimate(qrom))
    --- Resources: ---
    Total wires: 11
        algorithmic wires: 8
        allocated wires: 3
        zero state: 3
        any state: 0
    Total gates : 85
    'Toffoli': 8,
    'CNOT': 36,
    'X': 17,
    'Hadamard': 24
    """

    resource_keys = {
        "num_bitstrings",
        "size_bitstring",
        "num_bit_flips",
        "select_swap_depth",
        "restored",
    }

    @staticmethod
    def _t_optimized_select_swap_width(num_bitstrings, size_bitstring):
        opt_width_continuous = math.sqrt((2 / 3) * (num_bitstrings / size_bitstring))
        w1 = 2 ** math.floor(math.log2(opt_width_continuous))
        w2 = 2 ** math.ceil(math.log2(opt_width_continuous))

        w1 = 1 if w1 < 1 else w1
        w2 = 1 if w2 < 1 else w2  # The continuous solution could be non-physical

        def t_cost_func(w):
            return 4 * (math.ceil(num_bitstrings / w) - 2) + 6 * (w - 1) * size_bitstring

        if t_cost_func(w2) < t_cost_func(w1):
            return w2
        return w1

    def __init__(
        self,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        restored: bool = True,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ) -> None:
        self.restored = restored
        self.num_bitstrings = num_bitstrings
        self.size_bitstring = size_bitstring
        self.num_bit_flips = num_bit_flips or (num_bitstrings * size_bitstring // 2)

        self.num_control_wires = math.ceil(math.log2(num_bitstrings))
        self.num_wires = size_bitstring + self.num_control_wires

        if select_swap_depth is not None:
            if not isinstance(select_swap_depth, int):
                raise ValueError(
                    f"`select_swap_depth` must be None or an integer. Got {type(select_swap_depth)}"
                )

            exponent = int(math.log2(select_swap_depth))
            if 2**exponent != select_swap_depth:
                raise ValueError(
                    f"`select_swap_depth` must be 1 or a positive integer power of 2. Got {select_swap_depth}"
                )

        self.select_swap_depth = select_swap_depth
        super().__init__(wires=wires)

    # pylint: disable=protected-access
    @classmethod
    def resource_decomp(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        select_swap_depth: int | None = None,
        restored: bool = True,
    ) -> list[GateCount]:
        r"""Returns a list of ``GateCount`` objects representing the operator's resources.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            select_swap_depth (int | None): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
                depth that minimizes T-gate count.
            restored (bool): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code:`True`.

        Resources:
            The resources for QROM are derived from the following references:

            * :code:`restored=False`: Uses the Select-Swap tree decomposition from Figure 1.C of
              `Low et al. (2018) <https://arxiv.org/abs/1812.00954>`_, further optimized using the
              measurement-based uncomputation technique described in
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

            * :code:`restored=True`: Uses the standard QROM resource accounting from Figure 4 of
              `Berry et al. (2019) <https://arxiv.org/abs/1902.02134>`__.

            Note: we use the unary iterator trick to implement the ``Select``. This
            implementation assumes we have access to :math:`n - 1` additional
            work qubits, where :math:`n = \left\lceil \log_{2}(N) \right\rceil` and :math:`N` is
            the number of batches of unitaries to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        if select_swap_depth:
            max_depth = 2 ** math.ceil(math.log2(num_bitstrings))
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or cls._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        num_swap_alloc_wires = (W_opt - 1) * size_bitstring  # Swap registers
        if L_opt > 1:
            num_unary_alloc_wires = l - 1  # + work_wires for UI trick

        if restored:
            swap_register = Allocate(num_swap_alloc_wires, AllocateState.ANY, restored=True)
            gate_cost.append(swap_register)
            gate_cost.append(Allocate(num_unary_alloc_wires))
        else:
            gate_cost.append(Allocate(num_swap_alloc_wires + num_unary_alloc_wires))

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        hadamard = resource_rep(qre.Hadamard)

        swap_restored_prefactor = 1
        select_restored_prefactor = 1

        if restored and (W_opt > 1):
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_restored_prefactor = 4
            select_restored_prefactor = 2

        # SELECT cost:
        if L_opt > 1:
            gate_cost.append(
                GateCount(x, select_restored_prefactor * (2 * (L_opt - 2) + 1))
            )  # conjugate 0 controlled toffolis + 1 extra X gate from un-controlled unary iterator decomp
            gate_cost.append(
                GateCount(
                    cnot,
                    select_restored_prefactor * (L_opt - 2)
                    + select_restored_prefactor * num_bit_flips,
                )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, select_restored_prefactor * (L_opt - 2)))
            gate_cost.append(GateCount(r_elbow, select_restored_prefactor * (L_opt - 2)))

            gate_cost.append(Deallocate(l - 1))  # release UI trick work wires

        else:
            gate_cost.append(
                GateCount(
                    x, select_restored_prefactor * num_bit_flips
                )  # each unitary in the select is just an X gate to load the data
            )

        # SWAP cost:
        if W_opt > 1:
            ctrl_swap = resource_rep(qre.CSWAP)
            gate_cost.append(
                GateCount(ctrl_swap, swap_restored_prefactor * (W_opt - 1) * size_bitstring)
            )

            if not restored:
                gate_cost.append(GateCount(x, (W_opt - 1) * size_bitstring))  # measure and reset

        if restored:
            gate_cost.append(
                Deallocate(
                    (W_opt - 1) * size_bitstring,
                    swap_register,
                    AllocateState.ANY,
                    restored=True,
                )
            )  # release Swap registers
        else:
            gate_cost.append(Deallocate((W_opt - 1) * size_bitstring))  # release Swap registers

        return gate_cost

    @classmethod
    def single_controlled_res_decomp(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        select_swap_depth: int | None = None,
        restored: bool = True,
    ):
        r"""The resource decomposition for QROM controlled on a single wire."""
        if select_swap_depth:
            max_depth = 2 ** math.ceil(math.log2(num_bitstrings))
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        W_opt = select_swap_depth or qre.QROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        L_opt = math.ceil(num_bitstrings / W_opt)
        l = math.ceil(math.log2(L_opt))

        gate_cost = []
        num_alloc_wires = (W_opt - 1) * size_bitstring  # Swap registers
        if L_opt > 1:
            num_alloc_wires += l  # + work_wires for UI trick

        gate_cost.append(Allocate(num_alloc_wires))

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})
        hadamard = resource_rep(qre.Hadamard)

        swap_restored_prefactor = 1
        select_restored_prefactor = 1

        if restored and (W_opt > 1):
            gate_cost.append(GateCount(hadamard, 2 * size_bitstring))
            swap_restored_prefactor = 4
            select_restored_prefactor = 2

        # SELECT cost:
        if L_opt > 1:
            gate_cost.append(
                GateCount(x, select_restored_prefactor * (2 * (L_opt - 1)))
            )  # conjugate 0 controlled toffolis
            gate_cost.append(
                GateCount(
                    cnot,
                    select_restored_prefactor * (L_opt - 1)
                    + select_restored_prefactor * num_bit_flips,
                )  # num CNOTs in unary iterator trick   +   each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, select_restored_prefactor * (L_opt - 1)))
            gate_cost.append(GateCount(r_elbow, select_restored_prefactor * (L_opt - 1)))

            gate_cost.append(Deallocate(l))  # release UI trick work wires
        else:
            gate_cost.append(
                GateCount(
                    x,
                    select_restored_prefactor * num_bit_flips,
                )  #  each unitary in the select is just an X
            )

        # SWAP cost:
        if W_opt > 1:
            w = math.ceil(math.log2(W_opt))
            ctrl_swap = qre.CSWAP.resource_rep()
            gate_cost.append(Allocate(1))  # need one temporary qubit for l/r-elbow to control SWAP

            gate_cost.append(GateCount(l_elbow, w))
            gate_cost.append(
                GateCount(ctrl_swap, swap_restored_prefactor * (W_opt - 1) * size_bitstring)
            )
            gate_cost.append(GateCount(r_elbow, w))

            gate_cost.append(Deallocate(1))  # temp wires
            if restored:
                gate_cost.append(
                    Deallocate((W_opt - 1) * size_bitstring)
                )  # release Swap registers + temp wires
        return gate_cost

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources for QROM are taken from the following two papers:
            `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_ (Figure 1.C) for
            :code:`restored = False` and `Berry et al. (2019) <https://arxiv.org/pdf/1902.02134>`_
            (Figure 4) for :code:`restored = True`.

            Note: we use the single-controlled unary iterator trick to implement the ``Select``. This
            implementation assumes we have access to :math:`n` additional work qubits,
            where :math:`n = \lceil \log_{2}(N) \rceil` and :math:`N` is the number of batches of
            unitaries to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_bitstrings = target_resource_params["num_bitstrings"]
        size_bitstring = target_resource_params["size_bitstring"]
        num_bit_flips = target_resource_params.get("num_bit_flips", None)
        select_swap_depth = target_resource_params.get("select_swap_depth", None)
        restored = target_resource_params.get("restored", True)
        gate_cost = []
        if num_zero_ctrl:
            x = qre.X.resource_rep()
            gate_cost.append(GateCount(x, 2 * num_zero_ctrl))

        if num_bit_flips is None:
            num_bit_flips = (num_bitstrings * size_bitstring) // 2

        single_ctrl_cost = cls.single_controlled_res_decomp(
            num_bitstrings,
            size_bitstring,
            num_bit_flips,
            select_swap_depth,
            restored,
        )

        if num_ctrl_wires == 1:
            gate_cost.extend(single_ctrl_cost)
            return gate_cost

        gate_cost.append(Allocate(1))
        gate_cost.append(GateCount(qre.MultiControlledX.resource_rep(num_ctrl_wires, 0)))
        gate_cost.extend(single_ctrl_cost)
        gate_cost.append(GateCount(qre.MultiControlledX.resource_rep(num_ctrl_wires, 0)))
        gate_cost.append(Deallocate(1))
        return gate_cost

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_bitstrings (int): the number of bitstrings that are to be encoded
                * size_bitstring (int): the length of each bitstring
                * num_bit_flips (int | None): The total number of :math:`1`'s in the dataset.
                  Defaults to :code:`(num_bitstrings * size_bitstring) // 2`, which is half the
                  dataset.
                * restored (bool): Determine if allocated qubits should be reset after the
                  computation (at the cost of higher gate counts). Defaults to :code:`True`.
                * select_swap_depth (int | None): A parameter :math:`\lambda` that
                  determines if data will be loaded in parallel by adding more rows following
                  Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be
                  :code:`None`, :code:`1` or a positive integer power of two. Defaults to None,
                  which sets the depth that minimizes T-gate count.

        """

        return {
            "num_bitstrings": self.num_bitstrings,
            "size_bitstring": self.size_bitstring,
            "num_bit_flips": self.num_bit_flips,
            "select_swap_depth": self.select_swap_depth,
            "restored": self.restored,
        }

    @classmethod
    def _ctrl_T(cls, num_data_blocks: int, num_bit_flips: int, count: int = 1) -> list[GateCount]:
        """Constructs the control-``T`` subroutine as defined in Appendices A and B of
        `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_.

        Args:
            num_data_blocks(int): The number of data blocks formed by partitioning the total bitstrings based on select-swap depth.
            num_bit_flips (int): The total number of :math:`1`'s in the dataset.
            count (int): The number of times to repeat the subroutine.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: The resource decomposition of the control- :math:`T` subroutine.
        """

        x = resource_rep(qre.X)
        cnot = resource_rep(qre.CNOT)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        gate_cost = []

        if num_data_blocks > 1:
            gate_cost.append(
                GateCount(x, count * (2 * (num_data_blocks - 2) + 1))
            )  # conjugate 0 controlled toffolis + 1 extra X gate from un-controlled unary iterator decomp
            gate_cost.append(
                GateCount(
                    cnot,
                    count * (num_data_blocks - 2) + count * num_bit_flips,
                )  # num CNOTs in unary iterator trick + each unitary in the select is just a CNOT
            )
            gate_cost.append(GateCount(l_elbow, count * (num_data_blocks - 2)))
            gate_cost.append(GateCount(r_elbow, count * (num_data_blocks - 2)))

        else:
            gate_cost.append(
                GateCount(
                    x, count * num_bit_flips
                )  # each unitary in the select is just an X gate to load the data
            )
        return gate_cost

    @classmethod
    def _ctrl_S(cls, num_ctrl_wires: int, count: int = 1) -> list[GateCount]:
        """Constructs the control-S subroutine as defined in Figure 8 of
        `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_ excluding the initial ``X`` gate.

        Args:
            num_ctrl_wires (int): The number of control wires.
            count (int): The number of times to repeat the subroutine.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: The resource decomposition of the control- :math:`S` subroutine.
        """
        num_ctrl_swaps = 2**num_ctrl_wires - 1
        return [qre.GateCount(qre.resource_rep(qre.CSWAP), count * num_ctrl_swaps)]

    @classmethod
    def _ctrl_S_adj(cls, num_ctrl_wires: int, count: int = 1) -> list[GateCount]:
        r"""Constructs the control-S^adj subroutine as defined in Figure 10
        of `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_ excluding the terminal ``X`` gate.

        Args:
            num_ctrl_wires (int): The number of control wires.
            count (int): The number of times to repeat the subroutine.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: The resource decomposition of the control- :math:`S^{\dagger}` subroutine.

        """
        h = qre.resource_rep(qre.Hadamard)
        cz = qre.resource_rep(qre.CZ)
        cnot = qre.resource_rep(qre.CNOT)

        num_ops = 2**num_ctrl_wires - 1
        return [
            qre.GateCount(h, count * num_ops),
            qre.GateCount(cz, count * num_ops),
            qre.GateCount(cnot, count * num_ops),
        ]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources of the adjoint of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            This resources are based on Appendix C of `arXiv:1902.02134 <https://arxiv.org/abs/1902.02134>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        num_bitstrings = target_resource_params["num_bitstrings"]
        size_bitstring = target_resource_params["size_bitstring"]
        num_bit_flips = target_resource_params.get("num_bit_flips", None)
        select_swap_depth = target_resource_params.get("select_swap_depth", None)
        restored = target_resource_params.get("restored", True)

        gate_lst = []
        x = resource_rep(qre.X)
        z = resource_rep(qre.Z)
        had = qre.resource_rep(qre.Hadamard)

        # Compute the width (output + swap registers) and length (unary iter entries) of the QROM
        if select_swap_depth:
            max_depth = 2 ** math.ceil(math.log2(num_bitstrings))
            select_swap_depth = min(max_depth, select_swap_depth)  # truncate depth beyond max depth

        k = select_swap_depth or qre.QROM._t_optimized_select_swap_width(
            num_bitstrings, size_bitstring
        )
        num_qubits_l = math.ceil(math.log2(k))  # number of qubits in |l> register

        num_cols = math.ceil(num_bitstrings / k)  # number of columns of data
        num_qubits_h = math.ceil(math.log2(num_cols))  # number of qubits in |h> register

        ## Measure output register, reset qubits and construct fixup table
        gate_lst.append(qre.GateCount(had, size_bitstring))  # Figure 5.

        ## Allocate auxiliary qubits
        num_alloc_wires = k  # Swap registers
        if num_cols > 1:
            num_alloc_wires += num_qubits_h - 1  # + work_wires for UI trick

        gate_lst.append(qre.Allocate(num_alloc_wires))

        ## Cost assuming clean auxiliary qubits (Figure 6)
        if not restored:
            gate_lst.append(GateCount(x, 2))
            gate_lst.append(GateCount(had, 2 * k))

            num_bit_flips = (k * num_cols) // 2

            ctrl_S_decomp = cls._ctrl_S(num_ctrl_wires=num_qubits_l)
            ctrl_S_adj_decomp = cls._ctrl_S_adj(num_ctrl_wires=num_qubits_l)
            ctrl_T_decomp = cls._ctrl_T(num_data_blocks=num_cols, num_bit_flips=num_bit_flips)

            gate_lst.extend(ctrl_S_decomp)
            gate_lst.extend(ctrl_S_adj_decomp)
            gate_lst.extend(ctrl_T_decomp)

        ## Cost assuming dirty auxiliary qubits (Figure 7)
        else:
            gate_lst.append(GateCount(z, 2))
            gate_lst.append(GateCount(had, 2))

            num_bit_flips = (k * num_cols) // 2
            count = 1 if k == 1 else 2
            ctrl_S_decomp = cls._ctrl_S(num_ctrl_wires=num_qubits_l, count=count)
            ctrl_S_adj_decomp = cls._ctrl_S_adj(num_ctrl_wires=num_qubits_l, count=count)
            ctrl_T_decomp = cls._ctrl_T(
                num_data_blocks=num_cols, num_bit_flips=num_bit_flips, count=count
            )

            gate_lst.extend(ctrl_S_decomp)
            gate_lst.extend(ctrl_S_adj_decomp)
            gate_lst.extend(ctrl_T_decomp)

        gate_lst.append(qre.Deallocate(num_alloc_wires))

        return gate_lst

    @classmethod
    def resource_rep(
        cls,
        num_bitstrings: int,
        size_bitstring: int,
        num_bit_flips: int | None = None,
        restored: bool = True,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bitstrings (int): the number of bitstrings that are to be encoded
            size_bitstring (int): the length of each bitstring
            num_bit_flips (int | None): The total number of :math:`1`'s in the dataset. Defaults to
                :code:`(num_bitstrings * size_bitstring) // 2`, which is half the dataset.
            restored (bool): Determine if allocated qubits should be reset after the computation
                (at the cost of higher gate counts). Defaults to :code:`True`.
            select_swap_depth (int | None): A parameter :math:`\lambda` that determines
                if data will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_. Can be :code:`None`,
                :code:`1` or a positive integer power of two. Defaults to ``None``, which sets the
                depth that minimizes T-gate count.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if num_bit_flips is None:
            num_bit_flips = num_bitstrings * size_bitstring // 2

        if select_swap_depth is not None:
            if not isinstance(select_swap_depth, int):
                raise ValueError(
                    f"`select_swap_depth` must be None or an integer. Got {type(select_swap_depth)}"
                )

            exponent = int(math.log2(select_swap_depth))
            if 2**exponent != select_swap_depth:
                raise ValueError(
                    f"`select_swap_depth` must be 1 or a positive integer power of 2. Got f{select_swap_depth}"
                )

        params = {
            "num_bitstrings": num_bitstrings,
            "num_bit_flips": num_bit_flips,
            "size_bitstring": size_bitstring,
            "select_swap_depth": select_swap_depth,
            "restored": restored,
        }
        num_wires = size_bitstring + math.ceil(math.log2(num_bitstrings))
        return CompressedResourceOp(cls, num_wires, params)
