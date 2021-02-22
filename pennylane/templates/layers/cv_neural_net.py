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
Contains the ``CVNeuralNetLayers`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.ops import Squeezing, Displacement, Kerr
from pennylane.templates.subroutines import Interferometer
from pennylane.templates import broadcast
from pennylane.wires import Wires


def _preprocess(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires):
    """Validate and pre-process inputs as follows:

    * Check that the first dimensions of all weight tensors match
    * Check that the other dimensions of all weight tensors are correct for the number of qubits.

    Args:
        heta_1 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for first interferometer
        phi_1 (tensor_like): shape :math:`(L, K)` tensor of phase angles for first interferometer
        varphi_1 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after first interferometer
        r (tensor_like): shape :math:`(L, M)` tensor of squeezing amounts for :class:`~pennylane.ops.Squeezing` operations
        phi_r (tensor_like): shape :math:`(L, M)` tensor of squeezing angles for :class:`~pennylane.ops.Squeezing` operations
        theta_2 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for second interferometer
        phi_2 (tensor_like): shape :math:`(L, K)` tensor of phase angles for second interferometer
        varphi_2 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after second interferometer
        a (tensor_like): shape :math:`(L, M)` tensor of displacement magnitudes for :class:`~pennylane.ops.Displacement` operations
        phi_a (tensor_like): shape :math:`(L, M)` tensor of displacement angles for :class:`~pennylane.ops.Displacement` operations
        k (tensor_like): shape :math:`(L, M)` tensor of kerr parameters for :class:`~pennylane.ops.Kerr` operations
        wires (Wires): wires that template acts on

    Returns:
        int: number of times that the ansatz is repeated
    """

    n_wires = len(wires)
    n_if = n_wires * (n_wires - 1) // 2

    # check that first dimension is the same
    weights_list = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
    shapes = [qml.math.shape(w) for w in weights_list]

    first_dims = [s[0] for s in shapes]
    if len(set(first_dims)) > 1:
        raise ValueError(
            f"The first dimension of all parameters needs to be the same, got {first_dims}"
        )
    repeat = shapes[0][0]

    second_dims = [s[1] for s in shapes]
    expected = [
        n_if,
        n_if,
        n_wires,
        n_wires,
        n_wires,
        n_if,
        n_if,
        n_wires,
        n_wires,
        n_wires,
        n_wires,
    ]
    if not all(e == d for e, d in zip(expected, second_dims)):
        raise ValueError("Got unexpected shape for one or more parameters.")

    return repeat


@template
def CVNeuralNetLayers(
    theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires
):
    r"""A sequence of layers of a continuous-variable quantum neural network,
    as specified in `arXiv:1806.06871 <https://arxiv.org/abs/1806.06871>`_.

    The layer consists
    of interferometers, displacement and squeezing gates mimicking the linear transformation of
    a neural network in the x-basis of the quantum system, and uses a Kerr gate
    to introduce a 'quantum' nonlinearity.

    The layers act on the :math:`M` modes given in ``wires``,
    and include interferometers of :math:`K=M(M-1)/2` beamsplitters. The different weight parameters
    contain the weights for each layer. The number of layers :math:`L` is therefore derived
    from the first dimension of ``weights``.

    This example shows a 4-mode CVNeuralNet layer with squeezing gates :math:`S`, displacement gates :math:`D` and
    Kerr gates :math:`K`. The two big blocks are interferometers of type
    :mod:`pennylane.templates.layers.Interferometer`:

    .. figure:: ../../_static/layer_cvqnn.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. note::
       The CV neural network architecture includes :class:`~pennylane.ops.Kerr` operations.
       Make sure to use a suitable device, such as the :code:`strawberryfields.fock`
       device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for first interferometer
        phi_1 (tensor_like): shape :math:`(L, K)` tensor of phase angles for first interferometer
        varphi_1 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after first interferometer
        r (tensor_like): shape :math:`(L, M)` tensor of squeezing amounts for :class:`~pennylane.ops.Squeezing` operations
        phi_r (tensor_like): shape :math:`(L, M)` tensor of squeezing angles for :class:`~pennylane.ops.Squeezing` operations
        theta_2 (tensor_like): shape :math:`(L, K)` tensor of transmittivity angles for second interferometer
        phi_2 (tensor_like): shape :math:`(L, K)` tensor of phase angles for second interferometer
        varphi_2 (tensor_like): shape :math:`(L, M)` tensor of rotation angles to apply after second interferometer
        a (tensor_like): shape :math:`(L, M)` tensor of displacement magnitudes for :class:`~pennylane.ops.Displacement` operations
        phi_a (tensor_like): shape :math:`(L, M)` tensor of displacement angles for :class:`~pennylane.ops.Displacement` operations
        k (tensor_like): shape :math:`(L, M)` tensor of kerr parameters for :class:`~pennylane.ops.Kerr` operations
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
    Raises:
        ValueError: if inputs do not have the correct format
    """

    wires = Wires(wires)
    repeat = _preprocess(
        theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires
    )

    for l in range(repeat):

        Interferometer(theta=theta_1[l], phi=phi_1[l], varphi=varphi_1[l], wires=wires)

        r_and_phi_r = qml.math.stack([r[l], phi_r[l]], axis=1)
        broadcast(unitary=Squeezing, pattern="single", wires=wires, parameters=r_and_phi_r)

        Interferometer(theta=theta_2[l], phi=phi_2[l], varphi=varphi_2[l], wires=wires)

        a_and_phi_a = qml.math.stack([a[l], phi_a[l]], axis=1)
        broadcast(unitary=Displacement, pattern="single", wires=wires, parameters=a_and_phi_a)

        broadcast(unitary=Kerr, pattern="single", wires=wires, parameters=k[l])
