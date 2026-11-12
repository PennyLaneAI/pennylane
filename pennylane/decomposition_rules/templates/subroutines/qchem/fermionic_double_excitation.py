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

"""Built-in decomposition rules for templates.subroutines.qchem.fermionic_double_excitation."""

# pylint: disable=too-many-arguments,protected-access

import numpy as np
from jax import numpy as jnp

from pennylane import capture
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, RX, RZ, Hadamard
from pennylane.templates.subroutines.qchem.fermionic_double_excitation import (
    FermionicDoubleExcitation,
)


def _fermionic_double_excitation_resources(num_wires_1, num_wires_2):
    num_set_cnot_wires = num_wires_1 + num_wires_2 - 1

    return {Hadamard: 32, RX: 32, CNOT: 16 * num_set_cnot_wires, RZ: 8}


def _layer_qfunc1(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the first layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(1)}(\theta) = \mathrm{exp} \Big\{ \frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
          list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    RX(np.pi / 2, wires=q)
    Hadamard(wires=p)


def _layer_qfunc2(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the second layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(2)}(\theta) = \mathrm{exp} \Big\{ \frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    Hadamard(wires=r)
    RX(np.pi / 2, wires=q)
    RX(np.pi / 2, wires=p)


def _layer_qfunc3(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the third layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(3)}(\theta) = \mathrm{exp} \Big\{ \frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(-np.pi / 2, wires=r)
    RX(-np.pi / 2, wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(np.pi / 2, wires=r)
    RX(np.pi / 2, wires=q)
    RX(np.pi / 2, wires=p)


def _layer_qfunc4(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the fourth layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(4)}(\theta) = \mathrm{exp} \Big\{ \frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    RX(np.pi / 2, wires=p)


def _layer_qfunc5(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the fifth layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(5)}(\theta) = \mathrm{exp} \Big\{ -\frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{Y}_s \hat{X}_r \hat{X}_q \hat{X}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)


def _layer_qfunc6(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the sixth layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(6)}(\theta) = \mathrm{exp} \Big\{ -\frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{X}_s \hat{Y}_r \hat{X}_q \hat{X}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(np.pi / 2, wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)


def _layer_qfunc7(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the seventh layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(7)}(\theta) = \mathrm{exp} \Big\{ -\frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{Y}_s \hat{Y}_r \hat{Y}_q \hat{X}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    RX(-np.pi / 2, wires=r)
    RX(-np.pi / 2, wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    RX(np.pi / 2, wires=r)
    RX(np.pi / 2, wires=q)
    Hadamard(wires=p)


def _layer_qfunc8(weight, s, r, q, p, set_cnot_wires):
    r"""Implement the eighth layer of the circuit to exponentiate the double-excitation
    operator entering the UCCSD ansatz.

    .. math::

        \hat{U}_{pqrs}^{(8)}(\theta) = \mathrm{exp} \Big\{ -\frac{i\theta}{8}
        \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1} \hat{Z}_a
        (\hat{Y}_s \hat{Y}_r \hat{X}_q \hat{Y}_p) \Big\}

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        s (int): qubit index ``s``
        r (int): qubit index ``r``
        q (int): qubit index ``q``
        p (int): qubit index ``p``
        set_cnot_wires (list[Wires]): list of CNOT wires

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    RX(np.pi / 2, wires=r)
    Hadamard(wires=q)
    RX(np.pi / 2, wires=p)


@register_resources(_fermionic_double_excitation_resources)
def _fermionic_double_excitation_decomopsition(
    weight, wires, wires1, wires2
):  # pylint: disable=unused-argument
    s = wires1[0]
    r = wires1[-1]
    q = wires2[0]
    p = wires2[-1]

    # Sequence of the wires entering the CNOTs
    cnots_occ = [list(wires1.labels[l : l + 2]) for l in range(len(wires1) - 1)]
    cnots_unocc = [list(wires2.labels[l : l + 2]) for l in range(len(wires2) - 1)]

    set_cnot_wires = cnots_occ + [[r, q]] + cnots_unocc

    _layer_qfunc1(weight, s, r, q, p, set_cnot_wires)
    _layer_qfunc2(weight, s, r, q, p, set_cnot_wires)
    _layer_qfunc3(weight, s, r, q, p, set_cnot_wires)
    _layer_qfunc4(weight, s, r, q, p, set_cnot_wires)
    _layer_qfunc5(weight, s, r, q, p, set_cnot_wires)
    _layer_qfunc6(weight, s, r, q, p, set_cnot_wires)
    _layer_qfunc7(weight, s, r, q, p, set_cnot_wires)
    _layer_qfunc8(weight, s, r, q, p, set_cnot_wires)


add_decomps(FermionicDoubleExcitation, _fermionic_double_excitation_decomopsition)
