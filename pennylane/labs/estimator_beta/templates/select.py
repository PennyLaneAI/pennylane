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

import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta.compact_hamiltonian import FirstQuantizedHamiltonian
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, ResourceOperator, resource_rep
from pennylane.labs.estimator_beta.wires_manager import Allocate, Deallocate
from pennylane.math.utils import ceil_log2
from pennylane.wires import WiresLike, Wires

class SelectFirstQuantization(ResourceOperator):
    r"""Resource class for ``Select`` operator for block encoding of first quantized operators.

    This operator customizes the ``Select`` circuit based on the structure of the first quantized Hamiltonian.

    .. note::
            This decomposition assumes that a phase gradient state of size `coordinates_precision + 1`
            is available. Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.subroutines.PhaseGradient`.


    Args:
        fq_ham (:class:`~pennylane.labs.estimator_beta.FirstQuantizedHamiltonian`): a first quantized
            Hamiltonian for which the state is being prepared
        coordinates_precision (int): The number of bits used to represent the precision for loading the nuclear coordinates. The default value is set to ``35`` bits.
        wires (WiresLike | None): the wires on which the operator acts

    Resources:
        The resources are calculated based on Section II A of `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> fq_ham = qre.FirstQuantizedHamiltonian(num_plane_waves=1000, num_electrons=50, omega=3.5, charge=2)
    >>> res = qre.estimate(qre.SelectFirstQuantization(fq_ham, coordinates_precision=15))
    >>> print(res)

    """

    resource_keys = {"fq_ham", "coordinates_precision"}

    def __init__(
        self,
        fq_ham: FirstQuantizedHamiltonian,
        coordinates_precision: int = 35,
        wires: WiresLike | None = None,
    ):

        if not isinstance(fq_ham, FirstQuantizedHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepFirstQuantization."
                f"This method works with first quantized Hamiltonian, {type(fq_ham)} provided"
            )

        self.fq_ham = fq_ham
        self.coordinates_precision = coordinates_precision

        eta = fq_ham.num_electrons
        n_eta = ceil_log2(eta)
        n_p = ceil_log2(fq_ham.num_plane_waves ** (1 / 3) + 1)
        # Total number of wires is obtained based on the Appendix C of `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_.
        self.num_wires = 3*eta*n_p + 2*n_eta + 5*n_p + 16

        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * fq_ham (:class:`~.pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                  Hamiltonian for which the state is being prepared
                * coordinates_precision (int): The number of bits used to represent the precision for loading the nuclear coordinates.
        """
        return {
            "fq_ham": self.fq_ham,
            "coordinates_precision": self.coordinates_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        fq_ham: FirstQuantizedHamiltonian,
        coordinates_precision: int = 35,
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            fq_ham (:class:`~pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                Hamiltonian for which the state is being prepared
            coordinates_precision (int): The number of bits used to represent the precision for loading the nuclear coordinates. The default value is set to ``35`` bits.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if not isinstance(fq_ham, FirstQuantizedHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for PrepFirstQuantization."
                f"This method works with first quantized Hamiltonian, {type(fq_ham)} provided"
            )

        if coordinates_precision <= 0 or not isinstance(coordinates_precision, int):
            raise ValueError(f"coordinates_precision must be a positive integer, got {coordinates_precision}")

        eta = fq_ham.num_electrons
        n_p = ceil_log2(fq_ham.num_plane_waves ** (1 / 3) + 1)
        num_wires = 3*eta*n_p

        params = {
            "fq_ham": fq_ham,
            "coordinates_precision": coordinates_precision,
        }
        return CompressedResourceOp(cls, num_wires, params)


    @classmethod
    def resource_decomp(
        cls,
        fq_ham: FirstQuantizedHamiltonian,
        coordinates_precision: int = 35,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            fq_ham (:class:`~pennylane.estimator.compact_hamiltonian.FirstQuantizedHamiltonian`): a first quantized
                Hamiltonian for which the walk operator is being created
            coordinates_precision (int): The number of bits used to represent the precision for loading the nuclear coordinates. The default value is set to ``35`` bits.

        Resources:
            The resources are calculated based on Section II A of `arXiv:2105.12767 <https://arxiv.org/abs/2105.12767>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """

        gate_list = []
        eta = fq_ham.num_electrons
        num_pw = fq_ham.num_plane_waves
        n_p = ceil_log2(num_pw ** (1 / 3) + 1)


        # Step 1: Select cost for T
        # Eq 73 from arXiv:2105.12767
        # copy w
        toffoli = resource_rep(qre.Toffoli)
        gate_list.append(GateCount(toffoli, 3 * (n_p - 1)))

        # copy r and s
        gate_list.append(GateCount(toffoli, 2 * (n_p - 1)))

        # phase flip on the |+> state
        # and control qubit selecting the T operator
        gate_list.append(GateCount(toffoli, 2))

        # Select cost for U and V

        # Step 2: swap of the momentum registers p and q into ancillae
        # Eq 72 from arXiv:2105.12767
        cswap = resource_rep(qre.CSWAP)
        gate_list.append(GateCount(cswap, 12 * eta * n_p + 4 * eta - 8))

        # Step 3:
        gate_list.append(Allocate(5 * n_p + 1)) # Step 12 from Appendix C of arXiv:2105.12767
        # A: Two's complement and sign magnitude for 6 momentum components
        gate_list.append(GateCount(toffoli, 6 * (n_p - 2)))

        # B: controlled copy of \nu to ancilla
        gate_list.append(GateCount(toffoli, 6 * (n_p + 1)))

        # C: 6 additions/subtractions for momentum registers
        adder = resource_rep(qre.SemiAdder, {"max_register_size": n_p + 1})
        gate_list.append(GateCount(adder, 6))

        # D: sign magnitude to two's, numbers now have two extra bits
        gate_list.append(GateCount(toffoli, 6 * n_p))

        gate_list.append(Deallocate(5 * n_p + 1))
        # Step 4: Apply the phase factor
        # for finding the product of signs of components of \nu and R_l and controlling the phase
        gate_list.append(GateCount(resource_rep(qre.CNOT), 6))
        gate_list.append(GateCount(resource_rep(qre.CZ), 1))

        gate_list.append(Allocate(5 * coordinates_precision - 4)))
        for j in range(min(n_p, coordinates_precision)):
            register_size = n_R - j

            if (j == min(n_p, coordinates_precision) - 1) and (coordinates_precision > n_p):
                gate_list.append(GateCount(resource_rep(qre.TemporaryAND), 3 * (2*register_size - 3)))
                gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qre.TemporaryAND}), 3 * (2*register_size - 3)))

            else:
                ctrl_adder = resource_rep(qre.Controlled, {"base_op": resource_rep(qre.SemiAdder, {"max_register_size": register_size})})
                gate_list.append(GateCount(ctrl_adder, 3))

        gate_list.append(Deallocate(5 * coordinates_precision - 4))

        return gate_list
