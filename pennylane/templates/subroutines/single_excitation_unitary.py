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
Contains the ``SingleExcitationUnitary`` template.
"""
import pennylane as qml
from pennylane import numpy as np

# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.ops import CNOT, RX, RZ, Hadamard
from pennylane.templates.decorator import template
from pennylane.templates.utils import (
    check_shape,
    get_shape,
)
from pennylane.wires import Wires


def _preprocess(weight, wires):
    """Validate and pre-process inputs as follows:

    * Check the shape of the weights tensor.
    * Check that there are at least 2 wires.

    Args:
        weight (tensor_like): trainable parameters of the template
        wires (Wires): wires that template acts on
    """
    if len(wires) < 2:
        raise ValueError("expected at least two wires; got {}".format(len(wires)))

    if qml.tape_mode_active():

        shape = qml.math.shape(weight)
        if shape != ():
            raise ValueError(f"Weight must be a scalar tensor {()}; got shape {shape}.")

    else:
        expected_shape = ()
        check_shape(
            weight,
            expected_shape,
            msg="Weight must be a scalar; got shape {}".format(expected_shape, get_shape(weight)),
        )


@template
def SingleExcitationUnitary(weight, wires=None):
    r"""Circuit to exponentiate the tensor product of Pauli matrices representing the
    single-excitation operator entering the Unitary Coupled-Cluster Singles
    and Doubles (UCCSD) ansatz. UCCSD is a VQE ansatz commonly used to run quantum
    chemistry simulations.

    The CC single-excitation operator is given by

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermionic annihilation and
    creation operators and the indices :math:`r` and :math:`p` run over the occupied and
    unoccupied molecular orbitals, respectively. Using the `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ the fermionic operator defined above can be written
    in terms of Pauli matrices (for more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_).

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \Big\{ \frac{i\theta}{2}
        \bigotimes_{a=r+1}^{p-1}\hat{Z}_a (\hat{Y}_r \hat{X}_p) \Big\}
        \mathrm{exp} \Big\{ -\frac{i\theta}{2}
        \bigotimes_{a=r+1}^{p-1} \hat{Z}_a (\hat{X}_r \hat{Y}_p) \Big\}.

    The quantum circuit to exponentiate the tensor product of Pauli matrices entering
    the latter equation is shown below (see `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    |

    .. figure:: ../../_static/templates/subroutines/single_excitation_unitary.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    As explained in `Seely et al. (2012) <https://arxiv.org/abs/1208.5986>`_,
    the exponential of a tensor product of Pauli-Z operators can be decomposed in terms of
    :math:`2(n-1)` CNOT gates and a single-qubit Z-rotation referred to as :math:`U_\theta` in
    the figure above. If there are :math:`X` or :math:`Y` Pauli matrices in the product,
    the Hadamard (:math:`H`) or :math:`R_x` gate has to be applied to change to the
    :math:`X` or :math:`Y` basis, respectively. The latter operations are denoted as
    :math:`U_1` and :math:`U_2` in the figure above. See the Usage Details section for more
    information.

    Args:
        weight (float): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        wires (Iterable or Wires): Wires that the template acts on.
            The wires represent the subset of orbitals in the interval ``[r, p]``. Must be of
            minimum length 2. The first wire is interpreted as ``r`` and the last wire as ``p``.
            Wires in between are acted on with CNOT gates to compute the parity of the set
            of qubits.

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        Notice that:

        #. :math:`\hat{U}_{pr}(\theta)` involves two exponentiations where :math:`\hat{U}_1`,
           :math:`\hat{U}_2`, and :math:`\hat{U}_\theta` are defined as follows,

           .. math::
               [U_1, U_2, U_{\theta}] = \Bigg\{\bigg[R_x(-\pi/2), H, R_z(\theta/2)\bigg],
               \bigg[H, R_x(-\frac{\pi}{2}), R_z(-\theta/2) \bigg] \Bigg\}

        #. For a given pair ``[r, p]``, ten single-qubit and ``4*(len(wires)-1)`` CNOT
           operations are applied. Notice also that CNOT gates act only on qubits
           ``wires[1]`` to ``wires[-2]``. The operations performed across these qubits
           are shown in dashed lines in the figure above.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import SingleExcitationUnitary

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(weight, wires=None):
                SingleExcitationUnitary(weight, wires=wires)
                return qml.expval(qml.PauliZ(0))

            weight = 0.56
            print(circuit(weight, wires=[0, 1, 2]))

    """

    wires = Wires(wires)
    _preprocess(weight, wires)

    # Interpret first and last wire as r and p
    r = wires[0]
    p = wires[-1]

    # Sequence of the wires entering the CNOTs between wires 'r' and 'p'
    set_cnot_wires = [wires.subset([l, l + 1]) for l in range(len(wires) - 1)]

    # ------------------------------------------------------------------
    # Apply the first layer

    # U_1, U_2 acting on wires 'r' and 'p'
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=p)

    # Applying CNOTs between wires 'r' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(weight / 2, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+ acting on wires 'r' and 'p'
    RX(np.pi / 2, wires=r)
    Hadamard(wires=p)

    # ------------------------------------------------------------------
    # Apply the second layer

    # U_1, U_2 acting on wires 'r' and 'p'
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=p)

    # Applying CNOTs between wires 'r' and 'p'
    for cnot_wires in set_cnot_wires:
        CNOT(wires=cnot_wires)

    # Z rotation acting on wire 'p'
    RZ(-weight / 2, wires=p)

    # Applying CNOTs in reverse order
    for cnot_wires in reversed(set_cnot_wires):
        CNOT(wires=cnot_wires)

    # U_1^+, U_2^+ acting on wires 'r' and 'p'
    Hadamard(wires=r)
    RX(np.pi / 2, wires=p)
