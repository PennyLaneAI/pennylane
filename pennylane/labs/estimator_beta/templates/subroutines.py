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
r"""Resource operators for PennyLane subroutine templates."""

import math

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import CompressedResourceOp, GateCount, ResourceOperator, resource_rep
from pennylane.math import ceil_log2
from pennylane.wires import WiresLike

# pylint: disable=arguments-differ, signature-differs


class SelectPauliRot(ResourceOperator):
    r"""Resource class for the uniformly controlled rotation gate.

    Args:
        rot_axis (str): the rotation axis used in the multiplexer
        num_ctrl_wires (int): the number of control wires of the multiplexer
        precision (float | None): the precision used in the single qubit rotations
        wires (WiresLike, None): the wires the operation acts on

    Resources:
        The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
        (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
        for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
        gate and :math:`2^{n}` instances of the single qubit rotation gate (:code:`RX`,
        :code:`RY` or :code:`RZ`) depending on the :code:`rot_axis`.

    .. seealso:: The associated PennyLane operation :class:`~.pennylane.SelectPauliRot`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> mltplxr = qre.SelectPauliRot(
    ...     rot_axis = "Y",
    ...     num_ctrl_wires = 4,
    ...     precision = 1e-3,
    ... )
    >>> print(qre.estimate(mltplxr, gate_set=['RY','CNOT']))
    --- Resources: ---
    Total wires: 5
        algorithmic wires: 5
        allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 32
    'RY': 16,
    'CNOT': 16
    """

    resource_keys = {"num_ctrl_wires", "rot_axis", "precision"}

    def __init__(
        self,
        rot_axis: str,
        num_ctrl_wires: int,
        precision: float | None = None,
        wires: WiresLike = None,
    ) -> None:
        if rot_axis not in ("X", "Y", "Z"):
            raise ValueError("The `rot_axis` argument must be one of ('X', 'Y', 'Z')")

        self.num_ctrl_wires = num_ctrl_wires
        self.rot_axis = rot_axis
        self.precision = precision

        self.num_wires = num_ctrl_wires + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * rot_axis (str): the rotation axis used in the multiplexer
                * num_ctrl_wires (int): the number of control wires of the multiplexer
                * precision (float): the precision used in the single qubit rotations
        """
        return {
            "num_ctrl_wires": self.num_ctrl_wires,
            "rot_axis": self.rot_axis,
            "precision": self.precision,
        }

    @classmethod
    def resource_rep(cls, num_ctrl_wires, rot_axis, precision=None):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            rot_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float | None): the precision used in the single qubit rotations

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = num_ctrl_wires + 1
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "rot_axis": rot_axis,
                "precision": precision,
            },
        )

    @classmethod
    def resource_decomp(cls, num_ctrl_wires, rot_axis, precision):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            rot_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations

        Resources:
            The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
            (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
            for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
            gate and :math:`2^{n}` instances of the single qubit rotation gate (:code:`RX`,
            :code:`RY` or :code:`RZ`) depending on the :code:`rot_axis`.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rotation_gate_map = {
            "X": qre.RX,
            "Y": qre.RY,
            "Z": qre.RZ,
        }

        gate = resource_rep(rotation_gate_map[rot_axis], {"precision": precision})
        cnot = resource_rep(qre.CNOT)

        gate_lst = [
            GateCount(gate, 2**num_ctrl_wires),
            GateCount(cnot, 2**num_ctrl_wires),
        ]

        return gate_lst

    @classmethod
    def phase_grad_resource_decomp(cls, num_ctrl_wires, rot_axis, precision):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            rot_axis (str): the rotation axis used in the multiplexer
            num_ctrl_wires (int): the number of control wires of the multiplexer
            precision (float): the precision used in the single qubit rotations

        Resources:
            The resources are obtained from the construction scheme given in `O'Brien and Sünderhauf
            (2025), Fig 4 <https://arxiv.org/pdf/2409.07332>`_. Specifically, the resources
            use two :class:`~.pennylane.estimator.templates.subroutines.QROM`s to digitally load and unload
            the phase angles up to some precision. These are then applied using a single controlled
            :class:`~.pennylane.estimator.templates.subroutines.SemiAdder`.

            .. note::

                This method assumes a phase gradient state is prepared on an auxiliary register.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_prec_wires = ceil_log2(math.pi / precision) + 1
        gate_lst = []

        qrom = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": 2**num_ctrl_wires,
                "num_bit_flips": 2**num_ctrl_wires * num_prec_wires // 2,
                "size_bitstring": num_prec_wires,
                "restored": False,
            },
        )

        gate_lst.append(qre.Allocate(num_prec_wires))
        gate_lst.append(GateCount(qrom))
        gate_lst.append(
            GateCount(
                resource_rep(
                    qre.Controlled,
                    {
                        "base_cmpr_op": resource_rep(
                            qre.SemiAdder,
                            {"max_register_size": num_prec_wires},
                        ),
                        "num_ctrl_wires": 1,
                        "num_zero_ctrl": 0,
                    },
                )
            )
        )
        gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom})))
        gate_lst.append(qre.Deallocate(num_prec_wires))

        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

        if rot_axis == "X":
            gate_lst.append(GateCount(h, 2))
        if rot_axis == "Y":
            gate_lst.append(GateCount(h, 2))
            gate_lst.append(GateCount(s))
            gate_lst.append(GateCount(s_dagg))

        return gate_lst

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ):
        r"""Returns a list representing the resources of the controlled operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params(dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are obtained from the construction scheme given in `Möttönen and Vartiainen
            (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_. Specifically, the resources
            for an :math:`n` qubit unitary are given as :math:`2^{n}` instances of the :code:`CNOT`
            gate and :math:`2^{n}` instances of the controlled single qubit rotation gate (:code:`RX`,
            :code:`RY` or :code:`RZ`) depending on the :code:`rot_axis`.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        num_ctrl_wires_base = target_resource_params["num_ctrl_wires"]
        rot_axis = target_resource_params["rot_axis"]
        precision = target_resource_params["precision"]

        rotation_gate_map = {
            "X": qre.RX,
            "Y": qre.RY,
            "Z": qre.RZ,
        }
        gate_lst = []

        gate = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(rotation_gate_map[rot_axis], {"precision": precision}),
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        cnot = resource_rep(qre.CNOT)

        gate_lst.append(GateCount(gate, 2**num_ctrl_wires_base))
        gate_lst.append(GateCount(cnot, 2**num_ctrl_wires_base))

        return gate_lst
