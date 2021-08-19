# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the DoubleExcitationUnitary template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import RZ, RX, CNOT, Hadamard


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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(-np.pi / 2, wires=r)
    RX(-np.pi / 2, wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    Hadamard(wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    Hadamard(wires=s)
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    RX(-np.pi / 2, wires=r)
    RX(-np.pi / 2, wires=q)
    Hadamard(wires=p)

    # Applying CNOTs
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
        set_cnot_wires (list[Wires]): list of CNOT wires
    """

    # U_1, U_2, U_3, U_4 acting on wires 's', 'r', 'q' and 'p'
    RX(-np.pi / 2, wires=s)
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=q)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs
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


class DoubleExcitationUnitary(Operation):
    r"""Circuit to exponentiate the tensor product of Pauli matrices representing the
    double-excitation operator entering the Unitary Coupled-Cluster Singles
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
    the latter equation is shown below (see `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    |

    .. figure:: ../../_static/templates/subroutines/double_excitation_unitary.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    As explained in `Seely et al. (2012) <https://arxiv.org/abs/1208.5986>`_,
    the exponential of a tensor product of Pauli-Z operators can be decomposed in terms of
    :math:`2(n-1)` CNOT gates and a single-qubit Z-rotation referred to as :math:`U_\theta` in
    the figure above. If there are :math:`X` or:math:`Y` Pauli matrices in the product, the
    Hadamard (:math:`H`) or :math:`R_x` gate has to be applied to change to the :math:`X`
    or :math:`Y` basis, respectively. The latter operations are denoted as
    :math:`U_1`, :math:`U_2`, :math:`U_3` and :math:`U_4` in the figure above. See the
    Usage Details section for more details.

    Args:
        weight (float or tensor_like): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        wires1 (Iterable): Wires of the qubits representing the subset of occupied orbitals
            in the interval ``[s, r]``. The first wire is interpreted as ``s``
            and the last wire as ``r``.
            Wires in between are acted on with CNOT gates to compute the parity of the set of qubits.
        wires2 (Iterable): Wires of the qubits representing the subset of unoccupied
            orbitals in the interval ``[q, p]``. The first wire is interpreted as ``q`` and
            the last wire is interpreted as ``p``. Wires in between are acted on with CNOT gates
            to compute the parity of the set of qubits.

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
           and ``16*(len(wires1)-1 + len(wires2)-1 + 1)`` CNOT operations are applied.
           Consecutive CNOT gates act on qubits with indices between ``s`` and ``r`` and
           ``q`` and ``p`` while a single CNOT acts on wires ``r`` and ``q``. The operations
           performed across these qubits are shown in dashed lines in the figure above.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import DoubleExcitationUnitary

            dev = qml.device('default.qubit', wires=5)

            @qml.qnode(dev)
            def circuit(weight, wires1=None, wires2=None):
                DoubleExcitationUnitary(weight, wires1=wires1, wires2=wires2)
                return qml.expval(qml.PauliZ(0))

            weight = 1.34817
            print(circuit(weight, wires1=[0, 1], wires2=[2, 3, 4]))

    """

    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, weight, wires1=None, wires2=None, do_queue=True, id=None):

        if len(wires1) < 2:
            raise ValueError(
                "expected at least two wires representing the occupied orbitals; "
                "got {}".format(len(wires1))
            )
        if len(wires2) < 2:
            raise ValueError(
                "expected at least two wires representing the unoccupied orbitals; "
                "got {}".format(len(wires2))
            )

        shape = qml.math.shape(weight)
        if shape != ():
            raise ValueError(f"Weight must be a scalar; got shape {shape}.")

        self.wires1 = list(wires1)
        self.wires2 = list(wires2)
        wires = wires1 + wires2

        super().__init__(weight, wires=wires, do_queue=do_queue, id=id)

    def expand(self):

        weight = self.parameters[0]
        s = self.wires1[0]
        r = self.wires1[-1]
        q = self.wires2[0]
        p = self.wires2[-1]

        # Sequence of the wires entering the CNOTs
        cnots_occ = [self.wires1[l : l + 2] for l in range(len(self.wires1) - 1)]
        cnots_unocc = [self.wires2[l : l + 2] for l in range(len(self.wires2) - 1)]

        set_cnot_wires = cnots_occ + [[r, q]] + cnots_unocc

        with qml.tape.QuantumTape() as tape:

            _layer1(weight, s, r, q, p, set_cnot_wires)
            _layer2(weight, s, r, q, p, set_cnot_wires)
            _layer3(weight, s, r, q, p, set_cnot_wires)
            _layer4(weight, s, r, q, p, set_cnot_wires)
            _layer5(weight, s, r, q, p, set_cnot_wires)
            _layer6(weight, s, r, q, p, set_cnot_wires)
            _layer7(weight, s, r, q, p, set_cnot_wires)
            _layer8(weight, s, r, q, p, set_cnot_wires)

        return tape
