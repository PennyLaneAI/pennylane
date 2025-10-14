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
r"""Mapping PL operations to their associated ResourceOperator."""
from __future__ import annotations

import math
from functools import singledispatch

import pennylane.estimator.ops as re_ops
import pennylane.estimator.templates as re_temps
import pennylane.ops as qops
import pennylane.templates as qtemps
from pennylane.operation import Operation
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

from .ops.op_math.symbolic import Prod
from .resource_operator import ResourceOperator


@singledispatch
def _map_to_resource_op(op: Operation) -> ResourceOperator:
    r"""Maps an instance of :class:`~.Operation` to its associated :class:`~.estimator.ResourceOperator`.

    Args:
        op (~.Operation): base operation to be mapped

    Return:
        (~.estimator.ResourceOperator): the resource operator equivalent of the base operator. If
        there is no resource operator equivalent, the decomposition in terms of resource operators
        is returned.

    Raises:
        TypeError: The op is not a valid operation
        NotImplementedError: Operation doesn't have a resource equivalent and doesn't define
            a decomposition.
    """

    if not isinstance(op, Operation):
        raise TypeError(f"Operator of type {type(op)} is not a valid operation.")

    if op.has_decomposition:
        decomp = op.decomposition()

        if len(decomp) == 1:
            return _map_to_resource_op(decomp[0])

        decomp_wires = Wires.all_wires([d_op.wires for d_op in decomp])
        return Prod(tuple(_map_to_resource_op(d_op) for d_op in decomp), wires=decomp_wires)

    raise NotImplementedError(
        "Operation doesn't have a resource equivalent and doesn't define a decomposition."
    )


@_map_to_resource_op.register
def _(op: qops.Identity):
    return re_ops.Identity(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.GlobalPhase):
    return re_ops.GlobalPhase(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.Hadamard):
    return re_ops.Hadamard(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.S):
    return re_ops.S(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.T):
    return re_ops.T(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.X):
    return re_ops.X(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.Y):
    return re_ops.Y(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.Z):
    return re_ops.Z(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.SWAP):
    return re_ops.SWAP(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.PhaseShift):
    return re_ops.PhaseShift(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.Rot):
    return re_ops.Rot(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.RX):
    return re_ops.RX(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.RY):
    return re_ops.RY(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.RZ):
    return re_ops.RZ(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.MultiRZ):
    return re_ops.MultiRZ(num_wires=len(op.wires), wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.PauliRot):
    return re_ops.PauliRot(
        pauli_string=op.hyperparameters["pauli_word"],
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qops.SingleExcitation):
    return re_ops.SingleExcitation(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CCZ):
    return re_ops.CCZ(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CH):
    return re_ops.CH(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CNOT):
    return re_ops.CNOT(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.ControlledPhaseShift):
    return re_ops.ControlledPhaseShift(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CRot):
    return re_ops.CRot(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CRX):
    return re_ops.CRX(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CRY):
    return re_ops.CRY(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CRZ):
    return re_ops.CRZ(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CSWAP):
    return re_ops.CSWAP(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CY):
    return re_ops.CY(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.CZ):
    return re_ops.CZ(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.MultiControlledX):
    return re_ops.MultiControlledX(
        num_ctrl_wires=len(op.wires) - 1,
        num_zero_ctrl=len(op.control_values) - sum(op.control_values),
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.TemporaryAND):
    return re_ops.TemporaryAND(wires=op.wires)


@_map_to_resource_op.register
def _(op: qops.Toffoli):
    return re_ops.Toffoli(wires=op.wires)


@_map_to_resource_op.register
def _(op: qtemps.OutMultiplier):
    return re_temps.OutMultiplier(
        a_num_wires=len(op.hyperparameters["x_wires"]),
        b_num_wires=len(op.hyperparameters["y_wires"]),
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.SemiAdder):
    return re_temps.SemiAdder(
        max_register_size=max(
            len(op.hyperparameters["x_wires"]),
            len(op.hyperparameters["y_wires"]),
        ),
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.QFT):
    return re_temps.QFT(num_wires=len(op.wires), wires=op.wires)


@_map_to_resource_op.register
def _(op: qtemps.AQFT):
    return re_temps.AQFT(
        order=op.hyperparameters["order"],
        num_wires=len(op.wires),
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.BasisRotation):
    return re_temps.BasisRotation(dim=len(op.wires), wires=op.wires)


@_map_to_resource_op.register
def _(op: qtemps.Select):
    res_ops = [_map_to_resource_op(select_op) for select_op in op.hyperparameters["ops"]]
    return re_temps.Select(ops=res_ops, wires=op.wires)


@_map_to_resource_op.register
def _(op: qtemps.QROM):
    bitstrings = op.hyperparameters["bitstrings"]
    num_bitstrings = len(bitstrings)
    size_bitstring = len(bitstrings[0]) if num_bitstrings > 0 else 0
    return re_temps.QROM(
        num_bitstrings=num_bitstrings,
        size_bitstring=size_bitstring,
        restored=op.hyperparameters["clean"],
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.SelectPauliRot):
    return re_temps.SelectPauliRot(
        rot_axis=op.hyperparameters["rot_axis"],
        num_ctrl_wires=len(op.wires) - 1,
        precision=None,
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qops.QubitUnitary):
    return re_ops.QubitUnitary(num_wires=len(op.wires), precision=None, wires=op.wires)


@_map_to_resource_op.register
def _(op: qtemps.ControlledSequence):
    res_base = _map_to_resource_op(op.hyperparameters["base"])
    num_control_wires = len(op.hyperparameters["control_wires"])
    return re_temps.ControlledSequence(
        base=res_base,
        num_control_wires=num_control_wires,
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.QuantumPhaseEstimation):
    res_base = _map_to_resource_op(op.hyperparameters["unitary"])
    num_estimation_wires = len(op.hyperparameters["estimation_wires"])
    return re_temps.QPE(
        base=res_base,
        num_estimation_wires=num_estimation_wires,
        adj_qft_op=None,
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.TrotterProduct):

    with QueuingManager.stop_recording():
        res_ops = [
            _map_to_resource_op(qops.Evolution(term))
            for term in op.hyperparameters["base"].terms()[1]
        ]

    return re_temps.TrotterProduct(
        first_order_expansion=res_ops,
        num_steps=op.hyperparameters["n"],
        order=op.hyperparameters["order"],
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.MPSPrep):
    max_bond_dim = max(data.shape[-1] for data in op.mps)
    return re_temps.MPSPrep(
        num_mps_matrices=len(op.mps),
        max_bond_dim=max_bond_dim,
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.QROMStatePreparation):
    precision = math.pi / (2 ** len(op.hyperparameters["precision_wires"]))
    return re_temps.QROMStatePreparation(
        num_state_qubits=len(op.wires),
        precision=precision,
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qops.IntegerComparator):
    return re_temps.IntegerComparator(
        value=op.hyperparameters["value"],
        register_size=len(op.wires) - 1,
        geq=op.hyperparameters["geq"],
        wires=op.wires,
    )
