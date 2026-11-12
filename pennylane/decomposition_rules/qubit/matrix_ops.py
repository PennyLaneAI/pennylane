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

"""Built-in decomposition rules for qubit.matrix_ops."""

# pylint:disable=arguments-differ

import numpy as np
import scipy as sp

import pennylane as qp
from pennylane import math
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.symbolic_decomposition import is_integer
from pennylane.decomposition_rules.op_math.decompositions.unitary_decompositions import (
    multi_qubit_decomp_rule,
    rot_decomp_rule,
    two_qubit_decomp_rule,
    xyx_decomp_rule,
    xzx_decomp_rule,
    zxz_decomp_rule,
    zyz_decomp_rule,
)
from pennylane.ops.qubit.matrix_ops import DiagonalQubitUnitary, QubitUnitary
from pennylane.typing import Bool, Complex, Float, Wire
from pennylane.wires import concatenate_wires

add_decomps(
    QubitUnitary,
    zyz_decomp_rule,
    zxz_decomp_rule,
    xzx_decomp_rule,
    xyx_decomp_rule,
    rot_decomp_rule,
    two_qubit_decomp_rule,
    multi_qubit_decomp_rule,
)


@register_resources(lambda base: {abstractify(base): 1})
def _adjoint_qubit_unitary(base):
    U = base.U
    U = U.conjugate().transpose() if sp.sparse.issparse(U) else math.moveaxis(math.conj(U), -2, -1)
    QubitUnitary(U, wires=base.wires)


add_decomps("Adjoint(QubitUnitary)", _adjoint_qubit_unitary)


def _matrix_pow(U, z):
    if sp.sparse.issparse(U):
        return sp.sparse.linalg.matrix_power(U, z)
    if is_integer(z):
        return math.linalg.matrix_power(U, z)
    eigs, vecs = math.linalg.eig(U)
    return vecs @ math.diag(eigs**z) @ math.linalg.inv(vecs)


@register_resources(lambda base, z: {abstractify(base): 1})
def _pow_qubit_unitary(base, z):
    QubitUnitary(_matrix_pow(base.U, z), wires=base.wires)


add_decomps("Pow(QubitUnitary)", _pow_qubit_unitary)


# pylint: disable-next=unused-argument
def _ctrl_qubit_unitary_resource(base, control_wires, control_values, work_wires, work_wire_type):
    num_target_wires = len(base.wires)
    num_control_wires = len(control_wires)
    return {
        qp.ControlledQubitUnitary(
            Complex[2**num_target_wires, 2**num_target_wires],
            wires=Wire[num_control_wires + num_target_wires],
            control_values=Bool[num_control_wires],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(_ctrl_qubit_unitary_resource)
def _controlled_qubit_unitary(base, control_wires, control_values, work_wires, work_wire_type):
    qp.ControlledQubitUnitary(
        base.U,
        concatenate_wires(control_wires, base.wires),
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(QubitUnitary)", _controlled_qubit_unitary)


# pylint: disable=unused-argument
def _diagonal_qu_resource(D, wires):
    num_wires = len(wires)
    if num_wires == 1:
        return {qp.RZ: 1, qp.GlobalPhase: 1}

    return {
        DiagonalQubitUnitary(Complex[2 ** (num_wires - 1)], wires=Wire[num_wires - 1]): 1,
        qp.SelectPauliRot(
            Float[2 ** (num_wires - 1)],
            control_wires=Wire[num_wires - 1],
            target_wire=Wire[1],
            rot_axis="Z",
        ): 1,
    }


@register_resources(_diagonal_qu_resource)
def _diagonal_qu_decomp(D, wires):
    angles = math.angle(D)
    diff = angles[..., 1::2] - angles[..., ::2]
    mean = (angles[..., ::2] + angles[..., 1::2]) / 2
    if len(wires) == 1:
        qp.GlobalPhase(-math.squeeze(mean, axis=-1))
        qp.RZ(math.squeeze(diff, axis=-1), wires=wires)
    else:
        qp.DiagonalQubitUnitary(math.exp(1j * mean), wires=wires[:-1])
        qp.SelectPauliRot(diff, control_wires=wires[:-1], target_wire=wires[-1])


# pylint: disable=unused-argument
def _diagonal_mux_on_aux_resources(D, wires):
    num_wires = len(wires)
    return {
        qp.SelectPauliRot(
            Float[2**num_wires],
            control_wires=Wire[num_wires],
            target_wire=Wire[1],
            rot_axis="Z",
        ): 1
    }


@register_resources(_diagonal_mux_on_aux_resources, work_wires={"zeroed": 1})
def _diagonal_mux_on_aux_decomp(D, wires, **_):
    angles = -2 * math.angle(D)
    with qp.allocate(1, "zero", restored=True) as aux_wire:
        qp.SelectPauliRot(angles, control_wires=wires, target_wire=aux_wire)


add_decomps(DiagonalQubitUnitary, _diagonal_qu_decomp, _diagonal_mux_on_aux_decomp)


@register_resources(lambda base: {abstractify(base): 1})
def _adjoint_diagonal_unitary(base):
    DiagonalQubitUnitary(math.conj(base.D), wires=base.wires)


add_decomps("Adjoint(DiagonalQubitUnitary)", _adjoint_diagonal_unitary)


# pylint: disable-next=unused-argument
@register_resources(lambda base, z: {abstractify(base): 1})
def _pow_diagonal_unitary(base, z):
    DiagonalQubitUnitary(math.cast(base.D, np.complex128) ** z, wires=base.wires)


add_decomps("Pow(DiagonalQubitUnitary)", _pow_diagonal_unitary)


def _adjoint_block_encode_resources(base):
    return {abstractify(base.adjoint()): 1}


@register_resources(_adjoint_block_encode_resources)
def _adjoint_block_encode(base):
    base.adjoint()


add_decomps("Adjoint(BlockEncode)", _adjoint_block_encode)
