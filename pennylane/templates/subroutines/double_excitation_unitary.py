# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the ``DoubleExcitationUnitary`` template.
"""
import numpy as np

# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.ops import CNOT, RX, RZ, Hadamard
from pennylane.templates.decorator import template
from pennylane.templates.utils import (
    check_no_variable,
    check_shape,
    check_type,
    check_wires,
    get_shape,
)


def _layer1(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=q)
    Hadamard(wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    RX(np.pi / 2, wires=q)
    Hadamard(wires=p)


def _layer2(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    Hadamard(wires=r)
    RX(np.pi / 2, wires=q)
    RX(np.pi / 2, wires=p)


def _layer3(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(-np.pi / 2, wires=r)
    RX(-np.pi / 2, wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(np.pi / 2, wires=r)
    RX(np.pi / 2, wires=q)
    RX(np.pi / 2, wires=p)


def _layer4(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    RX(np.pi / 2, wires=p)


def _layer5(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)


def _layer6(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(np.pi / 2, wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)


def _layer7(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    RX(-np.pi / 2, wires=r)
    RX(-np.pi / 2, wires=q)
    Hadamard(wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    RX(np.pi / 2, wires=r)
    RX(np.pi / 2, wires=q)
    Hadamard(wires=p)


def _layer8(weight, s, r, q, p, set_cnot_wires):
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
        set_cnot_wires (sequence[int]): two-element sequence with the indices of the qubits
            the CNOT gates act on
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs between wires 's' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(-weight / 8, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+, U_3^+, U_4^+ acting on wires 's', 'r', 'q' and 'p'
    RX(np.pi / 2, wires=s)
    RX(np.pi / 2, wires=r)
    Hadamard(wires=q)
    RX(np.pi / 2, wires=p)


@template
def DoubleExcitationUnitary(weight, wires=None):
    r"""Circuit to exponentiate the tensor product of Pauli matrices representing the
    fermionic double-excitation operator entering the Unitary Coupled-Cluster Singles
    and Doubles (UCCSD) ansatz. UCCSD is a VQE ansatz commonly used to run quantum
    chemistry simulations.

    The CC double-excitation operator is given by

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermionic annihilation and
    creation operators and the indices :math:`r, s` and :math:`p, q` run over the occupied and
    unoccupied molecular orbitals, respectively. Using the `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ the fermionic operator defined above can be written
    in terms of Pauli matrices (for more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p + \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    The quantum circuit to exponentiate the tensor product of Pauli matrices entering
    the latter equation is shown below:

    .. figure:: ../../_static/templates/subroutines/double_excitation_unitary.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    As explained in `Seely et al. (2012) <https://arxiv.org/abs/1208.5986>`_,
    the exponential of a tensor product of Pauli-Z operators can be decomposed in terms of
    :math:`2(n-1)` CNOT gates and a single-qubit Z-rotation. If there are :math:`X` or
    :math:`Y` Pauli matrices in the product, the Hadamard (:math:`H`) or :math:`R_x` gate has
    to be applied to change to the :math:`X` or :math:`Y` basis, respectively.

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        wires (sequence[int]): four-element sequence with the qubit indices ``s, r, q, p``

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        Notice that:

        #. :math:`\hat{U}_{pqrs}(\theta)` involves eight exponentiations where
           :math:`\hat{U}_1`, :math:`\hat{U}_2`, :math:`\hat{U}_3`, :math:`\hat{U}_4` and
           :math:`\hat{U}_\theta` are defined as follows,

           .. math::

               [U_1, && U_2, U_3, U_4, U_{\theta}] = \\
               && \Bigg\{\bigg[H, H, R_x(-\frac{\pi}{2}), H, R_z(\theta/8)\bigg],
               \bigg[R_x(-\frac{\pi}{2}), H, R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               R_z(\frac{\theta}{8}) \bigg], \\
               && \bigg[H, R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               R_z(\frac{\theta}{8}) \bigg], \bigg[H, H, H, R_x(-\frac{\pi}{2}),
               R_z(\frac{\theta}{8}) \bigg], \\
               && \bigg[R_x(-\frac{\pi}{2}), H, H, H, R_z(-\frac{\theta}{8}) \bigg],
               \bigg[H, R_x(-\frac{\pi}{2}), H, H, R_z(-\frac{\theta}{8}) \bigg], \\
               && \bigg[R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               H, R_z(-\frac{\theta}{8}) \bigg], \bigg[R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               H, R_x(-\frac{\pi}{2}), R_z(-\frac{\theta}{8}) \bigg] \Bigg\}

        #. For a given quadruple ``[s, r, q, p]`` with :math:`p>q>r>s`, seventy-two single-qubit
           operations are applied. Notice also that consecutive CNOT gates act on qubits with
           indices between ``s`` and ``r`` and ``q`` and ``p`` while a single CNOT acts on wires
           ``r`` and ``q``. The operations performed across these qubits are shown in dashed lines
           in the figure above.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import DoubleExcitationUnitary

            dev = qml.device('default.qubit', wires=5)

            @qml.qnode(dev)
            def circuit(weight, pphh=None):
                DoubleExcitationUnitary(weight, wires=pphh)
                return qml.expval(qml.PauliZ(0))

            weight = 1.34817
            double_excitation = [0, 1, 3, 4]
            print(circuit(weight, pphh=double_excitation))

    """

    ##############
    # Input checks

    check_no_variable(wires, msg="'wires' cannot be differentiable")

    wires = check_wires(wires)

    expected_shape = (4,)
    check_shape(
        wires,
        expected_shape,
        msg="'wires' must be of shape {}; got {}".format(expected_shape, get_shape(wires)),
    )

    expected_shape = ()
    check_shape(
        weight,
        expected_shape,
        msg="'weight' must be of shape {}; got {}".format(expected_shape, get_shape(weight)),
    )

    check_type(wires, [list], msg="'wires' must be a list; got {}".format(wires))
    for w in wires:
        check_type(w, [int], msg="'wires' must be a list of integers; got {}".format(wires))

    if wires != sorted(wires):
        raise ValueError(
            "Elements of 'wires' must satisfy that wires_3 > wires_2 > wires_1 > wires_0.\n"
            "Got wires[3]={}, wires[2]={}, wires[1]={}, wires[0]={}".format(*wires)
        )

    if (wires[1] <= wires[0]) or (wires[3] <= wires[2]):
        raise ValueError(
            "Elements of 'wires' must satisfy that wires_1 > wires_0 and wires_3 > wires_2.\n"
            "Got wires[3]={}, wires[2]={}, wires[1]={}, wires[0]={}".format(*wires)
        )

    ###############

    s, r, q, p = wires

    # Sequence of the wires entering the CNOTs between wires 's' and 'p'
    set_cnot_wires = (
        [[l, l + 1] for l in range(s, r)] + [[r, q]] + [[l, l + 1] for l in range(q, p)]
    )

    # Apply the first layer
    _layer1(weight, s, r, q, p, set_cnot_wires)

    # Apply the second layer
    _layer2(weight, s, r, q, p, set_cnot_wires)

    # Apply the third layer
    _layer3(weight, s, r, q, p, set_cnot_wires)

    # Apply the fourth layer
    _layer4(weight, s, r, q, p, set_cnot_wires)

    # Apply the fifth layer
    _layer5(weight, s, r, q, p, set_cnot_wires)

    # Apply the sixth layer
    _layer6(weight, s, r, q, p, set_cnot_wires)

    # Apply the seventh layer
    _layer7(weight, s, r, q, p, set_cnot_wires)

    # Apply the eighth layer
    _layer8(weight, s, r, q, p, set_cnot_wires)
