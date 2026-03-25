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

import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta.compact_hamiltonian import FirstQuantizedHamiltonian
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, ResourceOperator, resource_rep
from pennylane.labs.estimator_beta.wires_manager import Allocate, Deallocate
from pennylane.math.utils import ceil_log2
from pennylane.wires import WiresLike

class PrepFirstQuantization(ResourceOperator):
    r"""Resource class for preparing the state for first quantization algorithms.

    This operator customizes the Prepare circuit based on the structure of the first quantizated Hamiltonian.

    Args:
        fq_ham (:class:`~pennylane.labs.estimator_beta.FirstQuantizedHamiltonian`): a first quantized
            Hamiltonian for which the state is being prepared
        coeff_precision (int): The number of bits used to represent the precision for loading
            the coefficients of Hamiltonian. The default value is set to ``15`` bits.
        select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
            used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        wires (WiresLike | None): the wires on which the operator acts

    Resources:
        The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> fq_ham = qre.FirstQuantizedHamiltonian(num_orbitals, num_particles, num_nuclei, box_length)
    >>> res = qre.estimate(qre.PrepFirstQuantization(fq_ham, coeff_precision=15))
    >>> print(res)

    """

    resource_keys = {"thc_ham", "coeff_precision", "select_swap_depth"}

    def __init__(
        self,
        fq_ham: FirstQuantizedHamiltonian,
        coeff_precision: int = 15,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ):

        if not isinstance(fq_ham, FirstQuantizedHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepFirstQuantization."
                f"This method works with first quantized Hamiltonian, {type(fq_ham)} provided"
            )

        if not isinstance(coeff_precision, int):
            raise TypeError(
                f"`coeff_precision` must be an integer, but type {type(coeff_precision)} was provided."
            )

        self.fq_ham = fq_ham
        self.coeff_precision = coeff_precision
        self.select_swap_depth = select_swap_depth
        self.num_wires = 10  # This is a placeholder value. The actual number of wires will depend on the Hamiltonian and the implementation details of the state preparation algorithm.

        # if wires is not None and len(Wires(wires)) != self.num_wires:
        #     raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * fq_ham (:class:`~.pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                  Hamiltonian for which the state is being prepared
                * coeff_precision (int): The number of bits used to represent the precision for loading
                  the coefficients of Hamiltonian. The default value is set to ``15`` bits.
                * select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                  used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {
            "fq_ham": self.fq_ham,
            "coeff_precision": self.coeff_precision,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls,
        fq_ham: FirstQuantizedHamiltonian,
        coeff_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            fq_ham (:class:`~pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                Hamiltonian for which the state is being prepared
            coeff_precision (int): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if not isinstance(fq_ham, FirstQuantizedHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepFirstQuantization."
                f"This method works with first quantized Hamiltonian, {type(fq_ham)} provided"
            )

        if not isinstance(coeff_precision, int):
            raise TypeError(
                f"`coeff_precision` must be an integer, but type {type(coeff_precision)} was provided."
            )

        num_wires = 10
        params = {
            "fq_ham": fq_ham,
            "coeff_precision": coeff_precision,
            "select_swap_depth": select_swap_depth,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        fq_ham: FirstQuantizedHamiltonian,
        coeff_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            fq_ham (:class:`~pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                Hamiltonian for which the walk operator is being created
            coeff_precision (int): The number of bits used to represent the precision for loading
                the coefficients of the Hamiltonian. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """

        gate_list = []
        br = 8

        eta = fq_ham.num_electrons
        # # Step:1: Rotate the ancilla qubit for selecting between T and U+V Preparations
        # gate_list.append(GateCount(resource_rep(qre.RY), 1))

        # Step 2a: Prepare an equal superposition over i and j registers
        n_eta = ceil_log2(eta)
        ineq = resource_rep(qre.OutOfPlaceIntegerComparator, {"value": eta, "register_size": n_eta, "geq": False})
        gate_list.append(GateCount(ineq, 2)) # one for i and one for j

        toffoli = resource_rep(qre.Toffoli)
        gate_list.append(GateCount(toffoli, 2 * br - 6))  # for rotation on ancilla

        cz = resource_rep(qre.CZ)
        gate_list.append(GateCount(cz, 2))
        gate_list.append(GateCount(toffoli, 2* br - 6)) # invert the rotation
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": ineq}), 2)) # uncompute the inequality
        gate_list.append(GateCount(toffoli, 2*(n_eta - 1))) # Reflection on n_eta - 1 qubits
        gate_list.append(GateCount(ineq, 2)) # compute the inequality again

        # Step 2b and 2c:
        eq = resource_rep(qre.RegisterEquality, {"register_size": n_eta})
        gate_list.append(GateCount(eq, 1)) # compute the equality for i and j registers
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": eq}), 1)) # adjoint of equality test
        gate_list.append(GateCount(toffoli, 2)) # Extra toffolis to flag and invert the success of the equality test

        # Step 2d: Invert the superposition over i and j registers

        gate_list.append(GateCount(ineq, 2)) # one for i and one for j
        gate_list.append(GateCount(toffoli, 2 * br - 6))  # for rotation on ancilla
        gate_list.append(GateCount(cz, 2))
        gate_list.append(GateCount(toffoli, 2* br - 6)) # invert the rotation
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": ineq}), 2)) # uncompute the inequality
        gate_list.append(GateCount(toffoli, 2*(n_eta - 1))) # Reflection on n_eta - 1 qubits
        gate_list.append(GateCount(ineq, 2)) # compute the inequality again

        # Step 3: Prepare the superposition over w,r,s registers to use for T part of the select operation
        # Over w register: 3 spatial coordinates
        n_w = ceil_log2(3)
        ineq = resource_rep(qre.OutOfPlaceIntegerComparator, {"value": 3, "register_size": n_w, "geq": False})
        gate_list.append(GateCount(ineq, 1)) # one for i and one for j
        gate_list.append(GateCount(toffoli, br - 3))  # for rotation on ancilla
        gate_list.append(GateCount(cz, 1))
        gate_list.append(GateCount(toffoli, br - 3)) # invert the rotation
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": ineq}), 1)) # uncompute the inequality
        gate_list.append(GateCount(toffoli, n_w - 1)) # Reflection on n_w - 1 qubits
        gate_list.append(GateCount(ineq, 1)) # compute the inequality again

        # Over r and s registers: we need a cascade of controlled Hadamard gates


        return gate_list

