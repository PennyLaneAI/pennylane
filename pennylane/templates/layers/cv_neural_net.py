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
from pennylane.templates.decorator import template
from pennylane.ops import Squeezing, Displacement, Kerr
from pennylane.templates.subroutines import Interferometer
from pennylane.templates import broadcast
from pennylane.templates.utils import check_wires, check_number_of_layers, check_shapes


def cv_neural_net_layer(
    theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires
):
    r"""A single continuous-variable neural network layer.

    The layer acts on the :math:`M` wires modes specified in ``wires``, and includes interferometers
    of :math:`K=M(M-1)/2` beamsplitters.

    Args:
        theta_1 (array[float]): length :math:`(K, )` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`(K, )` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`(M, )` array of rotation angles to apply after first interferometer
        r (array[float]): length :math:`(M, )` array of squeezing amounts for
            :class:`~pennylane.ops.Squeezing` operations
        phi_r (array[float]): length :math:`(M, )` array of squeezing angles for
            :class:`~pennylane.ops.Squeezing` operations
        theta_2 (array[float]): length :math:`(K, )` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(K, )` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(M, )` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(M, )` array of displacement magnitudes for
            :class:`~pennylane.ops.Displacement` operations
        phi_a (array[float]): length :math:`(M, )` array of displacement angles for
            :class:`~pennylane.ops.Displacement` operations
        k (array[float]): length :math:`(M, )` array of kerr parameters for :class:`~pennylane.ops.Kerr` operations
        wires (Sequence[int]): sequence of mode indices that the template acts on
    """
    Interferometer(theta=theta_1, phi=phi_1, varphi=varphi_1, wires=wires)

    broadcast(unitary=Squeezing, pattern="single", wires=wires, parameters=list(zip(r, phi_r)))

    Interferometer(theta=theta_2, phi=phi_2, varphi=varphi_2, wires=wires)

    broadcast(unitary=Displacement, pattern="single", wires=wires, parameters=list(zip(a, phi_a)))

    broadcast(unitary=Kerr, pattern="single", wires=wires, parameters=k)


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
        theta_1 (array[float]): length :math:`(L, K)` array of transmittivity angles for first interferometer
        phi_1 (array[float]): length :math:`(L, K)` array of phase angles for first interferometer
        varphi_1 (array[float]): length :math:`(L, M)` array of rotation angles to apply after first interferometer
        r (array[float]): length :math:`(L, M)` array of squeezing amounts for :class:`~pennylane.ops.Squeezing` operations
        phi_r (array[float]): length :math:`(L, M)` array of squeezing angles for :class:`~pennylane.ops.Squeezing` operations
        theta_2 (array[float]): length :math:`(L, K)` array of transmittivity angles for second interferometer
        phi_2 (array[float]): length :math:`(L, K)` array of phase angles for second interferometer
        varphi_2 (array[float]): length :math:`(L, M)` array of rotation angles to apply after second interferometer
        a (array[float]): length :math:`(L, M)` array of displacement magnitudes for :class:`~pennylane.ops.Displacement` operations
        phi_a (array[float]): length :math:`(L, M)` array of displacement angles for :class:`~pennylane.ops.Displacement` operations
        k (array[float]): length :math:`(L, M)` array of kerr parameters for :class:`~pennylane.ops.Kerr` operations
        wires (Sequence[int]): sequence of mode indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks

    wires = check_wires(wires)

    n_wires = len(wires)
    n_if = n_wires * (n_wires - 1) // 2
    weights_list = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
    repeat = check_number_of_layers(weights_list)

    expected_shapes = [
        (repeat, n_if),
        (repeat, n_if),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_if),
        (repeat, n_if),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_wires),
    ]
    check_shapes(weights_list, expected_shapes, msg="wrong shape of weight input(s) detected")

    ###############

    for l in range(repeat):
        cv_neural_net_layer(
            theta_1=theta_1[l],
            phi_1=phi_1[l],
            varphi_1=varphi_1[l],
            r=r[l],
            phi_r=phi_r[l],
            theta_2=theta_2[l],
            phi_2=phi_2[l],
            varphi_2=varphi_2[l],
            a=a[l],
            phi_a=phi_a[l],
            k=k[l],
            wires=wires,
        )
