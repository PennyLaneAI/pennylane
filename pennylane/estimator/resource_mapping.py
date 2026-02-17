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

import numpy as np

import pennylane.estimator.ops as re_ops
import pennylane.estimator.templates as re_temps
import pennylane.ops as qops
import pennylane.templates as qtemps
from pennylane import math as pl_math
from pennylane.operation import Operation
from pennylane.ops.functions import simplify
from pennylane.ops.op_math.adjoint import Adjoint, AdjointOperation
from pennylane.ops.op_math.controlled import Controlled, ControlledOp
from pennylane.ops.op_math.pow import Pow, PowOperation
from pennylane.ops.op_math.prod import Prod
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

from .resource_operator import ResourceOperator


def _map_term_trotter(op: Operation):
    """Exponentiate op for trotter decomposition"""
    if isinstance(op, qops.op_math.SProd):
        op = simplify(op)
    if isinstance(op, qops.op_math.Sum):
        return qtemps.TrotterProduct(op, time=1, n=1, order=1, check_hermitian=False)
    return qops.op_math.Evolution(op)


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

        return re_ops.Prod(tuple(_map_to_resource_op(d_op) for d_op in decomp), wires=op.wires)

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
    x_wires = op.hyperparameters["x_wires"]
    y_wires = op.hyperparameters["y_wires"]

    return re_temps.SemiAdder(
        max_register_size=max(len(x_wires), len(y_wires)),
        wires=Wires.all_wires([x_wires, y_wires]),
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
def _(op: qtemps.HybridQRAM):
    data = op.data[0]
    tree_wire_manager = op.hyperparameters["tree_wire_manager"]
    select_wires = op.hyperparameters["select_wires"]
    signal_wire = op.hyperparameters["signal_wire"]
    control_wires = tree_wire_manager.control_wires
    target_wires = tree_wire_manager.target_wires
    return re_temps.HybridQRAM(
        data=data,
        num_wires=len(op.wires),
        num_control_wires=len(control_wires),
        num_select_wires=len(select_wires),
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=signal_wire
        + tree_wire_manager.bus_wire
        + tree_wire_manager.dir_wires
        + tree_wire_manager.portL_wires
        + tree_wire_manager.portR_wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.SelectOnlyQRAM):
    data = op.data[0]
    control_wires = op.hyperparameters["control_wires"]
    select_wires = op.hyperparameters["select_wires"]
    target_wires = op.hyperparameters["target_wires"]
    select_value = op.hyperparameters["select_value"]
    num_control_wires = len(control_wires)
    num_select_wires = len(select_wires)
    num_wires = num_control_wires + num_select_wires + len(target_wires)

    return re_temps.SelectOnlyQRAM(
        data,
        num_wires,
        num_control_wires,
        num_select_wires,
        control_wires,
        target_wires,
        select_wires,
        select_value,
    )


@_map_to_resource_op.register
def _(op: qtemps.BBQRAM):
    bitstrings = op.data[0]
    wire_manager = op.hyperparameters["wire_manager"]
    num_bitstrings = len(bitstrings)
    size_bitstring = len(bitstrings[0]) if num_bitstrings > 0 else 0
    return re_temps.BBQRAM(
        num_bitstrings=num_bitstrings,
        size_bitstring=size_bitstring,
        num_bit_flips=pl_math.sum(bitstrings),
        num_wires=len(op.wires),
        control_wires=wire_manager.control_wires,
        target_wires=wire_manager.target_wires,
        work_wires=wire_manager.bus_wire
        + wire_manager.dir_wires
        + wire_manager.portL_wires
        + wire_manager.portR_wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.QROM):
    bitstrings = op.data[0]
    num_bitstrings = bitstrings.shape[0]
    size_bitstring = bitstrings.shape[1] if num_bitstrings > 0 else 0
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
            _map_to_resource_op(_map_term_trotter(term))
            for term in op.hyperparameters["base"].operands
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
    op_wires = op.hyperparameters["input_wires"]
    prec_wires = op.hyperparameters["precision_wires"]

    precision = math.pi / (2 ** len(prec_wires))
    phases = np.angle(op.state_vector) % (2 * math.pi)
    positive_and_real = np.allclose(phases, 0.0)

    return re_temps.QROMStatePreparation(
        num_state_qubits=len(op_wires),
        precision=precision,
        positive_and_real=positive_and_real,
        wires=op_wires,
    )


@_map_to_resource_op.register
def _(op: qops.IntegerComparator):
    return re_temps.IntegerComparator(
        value=op.hyperparameters["value"],
        register_size=len(op.wires) - 1,
        geq=op.hyperparameters["geq"],
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.Reflection):
    base = op.hyperparameters["base"]
    ref_wires = op.hyperparameters["reflection_wires"]
    return re_temps.Reflection(
        num_wires=len(ref_wires),
        U=_map_to_resource_op(base),
        alpha=op.alpha,
        wires=ref_wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.Adder):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    return re_temps.Adder(
        len(x_wires),
        mod,
        wires=x_wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.OutAdder):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    y_wires = op.hyperparameters["y_wires"]
    output_wires = op.hyperparameters["output_wires"]

    return re_temps.OutAdder(
        len(x_wires),
        len(y_wires),
        len(output_wires),
        mod=mod,
        wires=x_wires + y_wires + output_wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.Multiplier):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    return re_temps.Multiplier(
        len(x_wires),
        mod=mod,
        wires=x_wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.ModExp):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    output_wires = op.hyperparameters["output_wires"]
    return re_temps.ModExp(
        len(x_wires),
        len(output_wires),
        mod=mod,
        wires=x_wires + output_wires,
    )


@_map_to_resource_op.register
def _(op: qtemps.PhaseAdder):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    return re_temps.PhaseAdder(
        len(x_wires),
        mod=mod,
        wires=x_wires,
    )


# Symbolic Ops:
@_map_to_resource_op.register
def _(op: qops.ChangeOpBasis):
    uncompute, target, compute = op.operands
    return re_ops.ChangeOpBasis(
        _map_to_resource_op(compute),
        _map_to_resource_op(target),
        _map_to_resource_op(uncompute),
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: Prod):
    return re_ops.Prod(
        res_ops=[_map_to_resource_op(factor) for factor in op.operands],
        wires=op.wires,
    )


@_map_to_resource_op.register
def _(op: Adjoint | AdjointOperation):
    return re_ops.Adjoint(
        base_op=_map_to_resource_op(op.base),
    )


@_map_to_resource_op.register
def _(op: Pow | PowOperation):
    return re_ops.Pow(_map_to_resource_op(op.base), pow_z=op.z)


@_map_to_resource_op.register
def _(op: Controlled | ControlledOp):
    ctrl_wires = op.control_wires
    num_zero_ctrl = sum(1 if bool(val) is False else 0 for val in op.control_values)

    return re_ops.Controlled(
        base_op=_map_to_resource_op(op.base),
        num_ctrl_wires=len(ctrl_wires),
        num_zero_ctrl=num_zero_ctrl,
        wires=ctrl_wires,
    )
