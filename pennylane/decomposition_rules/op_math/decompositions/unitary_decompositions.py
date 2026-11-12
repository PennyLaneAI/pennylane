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

"""Built-in decomposition rules for op_math.decompositions.unitary_decompositions."""

from scipy import sparse

from pennylane import capture, compiler, math, ops, templates
from pennylane.decomposition.decomposition_rule import register_condition, register_resources
from pennylane.ops.op_math.decompositions.unitary_decompositions import (
    _compute_num_cnots,
    _decompose_0_cnots,
    _decompose_1_cnot,
    _decompose_2_cnots,
    _decompose_3_cnots,
    _is_jax_jit,
    _multi_qubit_decomp,
    _su2_rot_decomp,
    _su2_xyx_decomp,
    _su2_xzx_decomp,
    _su2_zxz_decomp,
    _su2_zyz_decomp,
)
from pennylane.typing import Complex, Float, Wire


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


def _two_qubit_resource(*_, **__):
    """A worst-case over-estimate for the resources of two-qubit unitary decomposition."""
    return {
        ops.QubitUnitary(Complex[2, 2], wires=Wire[1]): 4,
        ops.CNOT: 3,
        ops.RZ: 1,
        ops.RY: 2,
        ops.GlobalPhase: 1,
    }


@register_condition(lambda U, wires, **_: len(wires) == 2)  # pylint: disable=unused-argument
@register_resources(_two_qubit_resource, exact=False)
def two_qubit_decomp_rule(U, wires, **_):
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


def _multi_qubit_decomp_resource(U, wires, **_):  # pylint: disable=unused-argument
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


@register_condition(lambda U, wires, **_: len(wires) > 2)  # pylint: disable=unused-argument
@register_resources(_multi_qubit_decomp_resource)
def multi_qubit_decomp_rule(U, wires, **__):
    """The decomposition rule for a multi-qubit unitary."""

    return _multi_qubit_decomp(U, wires, **__)
