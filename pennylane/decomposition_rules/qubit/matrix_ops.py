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

import numpy as np
import scipy as sp
from scipy import sparse

from pennylane import allocation, capture, compiler, math, ops, templates
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.decomposition.symbolic_decomposition import is_integer
from pennylane.ops.op_math.decompositions.unitary_decompositions import (
    _compute_num_cnots,
    _decompose_0_cnots,
    _decompose_1_cnot,
    _decompose_2_cnots,
    _decompose_3_cnots,
    _is_jax_jit,
    _qubit_unitary_multi_qubit_impl,
    _su2_rot_decomp,
    _su2_xyx_decomp,
    _su2_xzx_decomp,
    _su2_zxz_decomp,
    _su2_zyz_decomp,
)
from pennylane.ops.qubit.matrix_ops import DiagonalQubitUnitary, QubitUnitary
from pennylane.typing import Bool, Complex, Float, Wire
from pennylane.wires import concatenate_wires


def make_one_qubit_unitary_decomposition(su2_rule, su2_resource, name=""):
    """Wrapper around a naive one-qubit decomposition rule that adds a global phase."""

    def _resource_fn(*_, **__):
        return su2_resource() | {ops.GlobalPhase: 1}

    name = name or su2_rule.name

    # Resources are not exact because the global phase or rotations might be skipped
    @register_condition(lambda U, wires, **_: len(wires) == 1)
    @register_resources(_resource_fn, exact=False, name=name)
    def _impl(U, wires, **__):
        if sparse.issparse(U):
            U = U.todense()
        phase = su2_rule(U, wires=wires)
        if _is_jax_jit(U):
            ops.GlobalPhase(-phase)
        else:
            ops.cond(math.logical_not(math.allclose(phase, 0)), ops.GlobalPhase)(-phase)

    return _impl


def _su2_rot_resource():
    return {
        ops.Rot: 1,
        ops.RZ: 1,  # RZ is produced in a special case, which has to be accounted for.
    }


def _su2_zyz_resource():
    return {ops.RZ: 2, ops.RY: 1}


def _su2_xyx_resource():
    return {ops.RX: 2, ops.RY: 1}


def _su2_xzx_resource():
    return {ops.RX: 2, ops.RZ: 1}


def _su2_zxz_resource():
    return {ops.RZ: 2, ops.RX: 1}


rot_decomp_rule = make_one_qubit_unitary_decomposition(_su2_rot_decomp, _su2_rot_resource, "rot")

zyz_decomp_rule = make_one_qubit_unitary_decomposition(_su2_zyz_decomp, _su2_zyz_resource, "zyz")

xyx_decomp_rule = make_one_qubit_unitary_decomposition(_su2_xyx_decomp, _su2_xyx_resource, "xyx")

xzx_decomp_rule = make_one_qubit_unitary_decomposition(_su2_xzx_decomp, _su2_xzx_resource, "xzx")

zxz_decomp_rule = make_one_qubit_unitary_decomposition(_su2_zxz_decomp, _su2_zxz_resource, "zxz")


def _qubit_unitary_two_qubit_resources(*_, **__):
    """A worst-case over-estimate for the resources of two-qubit unitary decomposition."""
    return {
        ops.QubitUnitary(Complex[2, 2], wires=Wire[1]): 4,
        ops.CNOT: 3,
        ops.RZ: 1,
        ops.RY: 2,
        ops.GlobalPhase: 1,
    }


@register_condition(lambda U, wires, **_: len(wires) == 2)
@register_resources(_qubit_unitary_two_qubit_resources, exact=False, name="two_qubit")
def qubit_unitary_two_qubit(U, wires, **_):
    """The decomposition rule for a two-qubit unitary."""

    U, initial_phase = math.convert_to_su4(U)
    num_cnots = _compute_num_cnots(U)

    elifs = [(num_cnots == 1, _decompose_1_cnot)]

    # The 2-CNOT decomposition relies on sorting eigenvalues, which is not supported
    # with abstract tracers when capture is enabled. In that case, we fall back
    # to the 3-CNOT decomposition.
    if not capture.enabled() and not compiler.active():
        elifs.append((num_cnots == 2, _decompose_2_cnots))

    additional_phase = ops.cond(
        num_cnots == 0,
        _decompose_0_cnots,
        _decompose_3_cnots,
        elifs=elifs,
    )(U, wires, initial_phase)
    total_phase = initial_phase + additional_phase
    ops.cond(math.logical_not(math.allclose(total_phase, 0)), ops.GlobalPhase)(-total_phase)


def _qubit_unitary_multi_qubit_resources(U, wires, **_):  # pylint: disable=unused-argument
    num_wires = len(wires)
    return {
        ops.QubitUnitary(
            Complex[2 ** (num_wires - 1), 2 ** (num_wires - 1)], wires=Wire[num_wires - 1]
        ): 4,
        templates.SelectPauliRot(
            Float[2 ** (num_wires - 1)],
            control_wires=Wire[num_wires - 1],
            target_wire=Wire[1],
            rot_axis="Z",
        ): 2,
        templates.SelectPauliRot(
            Float[2 ** (num_wires - 1)],
            control_wires=Wire[num_wires - 1],
            target_wire=Wire[1],
            rot_axis="Y",
        ): 1,
    }


@register_condition(lambda U, wires, **_: len(wires) > 2)
@register_resources(_qubit_unitary_multi_qubit_resources, name="multi_qubit")
def qubit_unitary_multi_qubit(U, wires, **__):
    """The decomposition rule for a multi-qubit unitary."""
    return _qubit_unitary_multi_qubit_impl(U, wires, **__)


add_decomps(
    QubitUnitary,
    zyz_decomp_rule,
    zxz_decomp_rule,
    xzx_decomp_rule,
    xyx_decomp_rule,
    rot_decomp_rule,
    qubit_unitary_two_qubit,
    qubit_unitary_multi_qubit,
)


@register_resources(lambda base: {abstractify(base): 1}, name="conjugate_transpose")
def adjoint_qubit_unitary_standard(base):
    """Decompose ``Adjoint(QubitUnitary)`` using the ``conjugate_transpose`` strategy."""
    U = base.U
    U = U.conjugate().transpose() if sp.sparse.issparse(U) else math.moveaxis(math.conj(U), -2, -1)
    QubitUnitary(U, wires=base.wires)


add_decomps("Adjoint(QubitUnitary)", adjoint_qubit_unitary_standard)


def _matrix_pow(U, z):
    if sp.sparse.issparse(U):
        return sp.sparse.linalg.matrix_power(U, z)
    if is_integer(z):
        return math.linalg.matrix_power(U, z)
    eigs, vecs = math.linalg.eig(U)
    return vecs @ math.diag(eigs**z) @ math.linalg.inv(vecs)


@register_resources(lambda base, z: {abstractify(base): 1}, name="matrix_power")
def pow_qubit_unitary_standard(base, z):
    """Decompose ``Pow(QubitUnitary)`` using the ``matrix_power`` strategy."""
    QubitUnitary(_matrix_pow(base.U, z), wires=base.wires)


add_decomps("Pow(QubitUnitary)", pow_qubit_unitary_standard)


def _controlled_qubit_unitary_standard_resources(
    base,
    control_wires,
    control_values,  # pylint: disable=unused-argument
    work_wires,
    work_wire_type,
):
    num_target_wires = len(base.wires)
    num_control_wires = len(control_wires)
    return {
        ops.ControlledQubitUnitary(
            Complex[2**num_target_wires, 2**num_target_wires],
            wires=Wire[num_control_wires + num_target_wires],
            control_values=Bool[num_control_wires],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1
    }


@register_resources(
    _controlled_qubit_unitary_standard_resources, name="to_controlled_qubit_unitary"
)
def controlled_qubit_unitary_standard(
    base, control_wires, control_values, work_wires, work_wire_type
):
    """Decompose ``C(QubitUnitary)`` using the ``to_controlled_qubit_unitary`` strategy."""
    ops.ControlledQubitUnitary(
        base.U,
        concatenate_wires(control_wires, base.wires),
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(QubitUnitary)", controlled_qubit_unitary_standard)


def _diagonal_qubit_unitary_recursive_resources(D, wires):  # pylint: disable=unused-argument
    num_wires = len(wires)
    if num_wires == 1:
        return {ops.RZ: 1, ops.GlobalPhase: 1}

    return {
        DiagonalQubitUnitary(Complex[2 ** (num_wires - 1)], wires=Wire[num_wires - 1]): 1,
        templates.SelectPauliRot(
            Float[2 ** (num_wires - 1)],
            control_wires=Wire[num_wires - 1],
            target_wire=Wire[1],
            rot_axis="Z",
        ): 1,
    }


@register_resources(_diagonal_qubit_unitary_recursive_resources, name="recursive")
def diagonal_qubit_unitary_recursive(D, wires):
    """Decompose ``DiagonalQubitUnitary`` using the ``recursive`` strategy."""
    angles = math.angle(D)
    diff = angles[..., 1::2] - angles[..., ::2]
    mean = (angles[..., ::2] + angles[..., 1::2]) / 2
    if len(wires) == 1:
        ops.GlobalPhase(-math.squeeze(mean, axis=-1))
        ops.RZ(math.squeeze(diff, axis=-1), wires=wires)
    else:
        DiagonalQubitUnitary(math.exp(1j * mean), wires=wires[:-1])
        templates.SelectPauliRot(diff, control_wires=wires[:-1], target_wire=wires[-1])


def _diagonal_qubit_unitary_auxiliary_resources(D, wires):  # pylint: disable=unused-argument
    num_wires = len(wires)
    return {
        templates.SelectPauliRot(
            Float[2**num_wires],
            control_wires=Wire[num_wires],
            target_wire=Wire[1],
            rot_axis="Z",
        ): 1
    }


@register_resources(
    _diagonal_qubit_unitary_auxiliary_resources, work_wires={"zeroed": 1}, name="auxiliary"
)
def diagonal_qubit_unitary_auxiliary(D, wires, **_):
    """Decompose ``DiagonalQubitUnitary`` using the ``auxiliary`` strategy."""
    angles = -2 * math.angle(D)
    with allocation.allocate(1, "zero", restored=True) as aux_wire:
        templates.SelectPauliRot(angles, control_wires=wires, target_wire=aux_wire)


add_decomps(
    DiagonalQubitUnitary, diagonal_qubit_unitary_recursive, diagonal_qubit_unitary_auxiliary
)


@register_resources(lambda base: {abstractify(base): 1}, name="conjugate_diagonal")
def adjoint_diagonal_unitary_standard(base):
    """Decompose ``Adjoint(DiagonalQubitUnitary)`` using the ``conjugate_diagonal`` strategy."""
    DiagonalQubitUnitary(math.conj(base.D), wires=base.wires)


add_decomps("Adjoint(DiagonalQubitUnitary)", adjoint_diagonal_unitary_standard)


@register_resources(lambda base, z: {abstractify(base): 1}, name="power_diagonal")
def pow_diagonal_unitary_standard(base, z):
    """Decompose ``Pow(DiagonalQubitUnitary)`` using the ``power_diagonal`` strategy."""
    DiagonalQubitUnitary(math.cast(base.D, np.complex128) ** z, wires=base.wires)


add_decomps("Pow(DiagonalQubitUnitary)", pow_diagonal_unitary_standard)


def _adjoint_block_encode_standard_resources(base):
    return {abstractify(base.adjoint()): 1}


@register_resources(_adjoint_block_encode_standard_resources, name="adjoint_matrix")
def adjoint_block_encode_standard(base):
    """Decompose ``Adjoint(BlockEncode)`` using the ``adjoint_matrix`` strategy."""
    base.adjoint()


add_decomps("Adjoint(BlockEncode)", adjoint_block_encode_standard)
