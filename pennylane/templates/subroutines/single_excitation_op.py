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
Contains the ``SingleExcitationOp`` template.
"""
from pennylane import numpy as np
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.ops import CNOT, RX, RZ, Hadamard
from pennylane.templates.decorator import template
from pennylane.templates.utils import (check_no_variable, check_shape,
                                       check_type, check_wires, get_shape)


@template
def SingleExcitationUnitary(weight, wires_rp=None):
    r"""Circuit to exponentiate the tensor product of Pauli matrices representing the
    fermionic single-excitation operator entering the Unitary Coupled-Cluster Singles
    and Doubles (UCCSD) ansatz. UCCSD is one of the VQE ansatz commonly used to run quantum
    chemistry simulations.

    The CC single-excitation operator is given by

    .. math::

        \hat{U}_{pr}^{(1)}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermionic annihilation and
    creation operators and the indices :math:`r` and :math:`p` run over the occupied and
    unoccupied molecular orbitals, respectively. Using the `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ the fermionic operator defined above can be written
    in terms of Pauli matrices (for more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_).

    .. math::

        \hat{U}_{pr}^{(1)}(\theta) = \mathrm{exp} \Big\{ \frac{i\theta}{2}
        \bigotimes_{a=r+1}^{p-1}\hat{Z}_a (\hat{Y}_r \hat{X}_p) \Big\}
        \mathrm{exp} \Big\{ -\frac{i\theta}{2}
        \bigotimes_{a=r+1}^{p-1} \hat{Z}_a (\hat{X}_r \hat{Y}_p) \Big\}.

    The quantum circuit to exponentiate the tensor product of Pauli matrices entering
    the latter equation is shown below:

    .. figure:: ../../_static/ucc_se_op.png
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
        wires_rp (sequence[int]): two-element sequence with the qubit indices ``r, p``

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        Notice that:

        #. :math:`\hat{U}_{pr}^{(1)}(\theta)` involves two exponentiations where :math:`\hat{U}_1`,
           :math:`\hat{U}_2`, and :math:`\hat{U}_\theta` are defined as follows,

           .. math::
               [U_1, U_2, U_{\theta})] = \Bigg\{\bigg[R_x(-\pi/2), H, R_z(\theta/2)\bigg],
               \bigg[H, R_x(-\frac{\pi}{2}), R_z(-\theta/2) \bigg] \Bigg\}

        #. For a given pair ``[r, p]``, ten single-qubit operations are applied. Notice also that
           CNOT gates act only on qubits with indices between ``r`` and ``p``. The operations
           performed accross these qubits are shown in dashed lines in the figure above.

    """

    ##############
    # Input checks

    check_no_variable(wires_rp, msg="'wires_rp' cannot be differentiable")

    wires_rp = check_wires(wires_rp)

    expected_shape = (2,)
    check_shape(
        wires_rp,
        expected_shape,
        msg="'wires_rp' must be of shape {}; got {}" "".format(expected_shape, get_shape(wires_rp)),
    )

    expected_shape = ()
    check_shape(
        weight,
        expected_shape,
        msg="'weight' must be of shape {}; got {}" "".format(expected_shape, get_shape(weight)),
    )

    check_type(wires_rp, [list], msg="'wires_rp' must be a list; got {}" "".format(wires_rp))
    for w in wires_rp:
        check_type(w, [int], msg="'wires_rp' must be a list of integers; got {}" "".format(wires_rp))

    if wires_rp[1] <= wires_rp[0]:
        raise ValueError(
            "wires_rp_1 must be > wires_rp_0; got wires_rp[1]={}, wires_rp[0]={}" ""
            .format(wires_rp[1], wires_rp[0])
        )

    ###############

    r, p = wires_rp

#   Sequence of the wires entering the CNOTs between wires 'r' and 'p'
    set_cnot_wires = [[l, l+1] for l in range(r, p)]

    #------------------------------------------------------------------
    # Apply the first layer

    # U_1, U_2 acting on wires 'r' and 'p'
    RX(-np.pi/2, wires=r)
    Hadamard(wires=p)

    # Applying CNOTs between wires 'r' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(weight/2, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+ acting on wires 'r' and 'p'
    RX(np.pi/2, wires=r)
    Hadamard(wires=p)

    #------------------------------------------------------------------
    # Apply the second layer

    # U_1, U_2 acting on wires 'r' and 'p'
    Hadamard(wires=r)
    RX(-np.pi/2, wires=p)

    # Applying CNOTs between wires 'r' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(-weight/2, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+ acting on wires 'r' and 'p'
    Hadamard(wires=r)
    RX(np.pi/2, wires=p)
