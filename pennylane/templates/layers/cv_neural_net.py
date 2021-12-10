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
Contains the CVNeuralNetLayers template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class CVNeuralNetLayers(Operation):
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
    :mod:`pennylane.Interferometer`:

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
        wires (Iterable): wires that the template acts on

    .. UsageDetails:

        **Parameter shapes**

        A list of shapes for the 11 input parameter tensors can be computed by the static method
        :meth:`~.CVNeuralNetLayers.shape` and used when creating randomly
        initialised weights:

        .. code-block:: python

            shapes = CVNeuralNetLayers.shape(n_layers=2, n_wires=2)
            weights = [np.random.random(shape) for shape in shapes]

            def circuit():
              CVNeuralNetLayers(*weights, wires=[0, 1])
              return qml.expval(qml.X(0))

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self,
        theta_1,
        phi_1,
        varphi_1,
        r,
        phi_r,
        theta_2,
        phi_2,
        varphi_2,
        a,
        phi_a,
        k,
        wires,
        do_queue=True,
        id=None,
    ):

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

        # check second dimensions
        second_dims = [s[1] for s in shapes]
        expected = [n_if] * 2 + [n_wires] * 3 + [n_if] * 2 + [n_wires] * 4
        if not all(e == d for e, d in zip(expected, second_dims)):
            raise ValueError("Got unexpected shape for one or more parameters.")

        self.n_layers = shapes[0][0]

        super().__init__(
            theta_1,
            phi_1,
            varphi_1,
            r,
            phi_r,
            theta_2,
            phi_2,
            varphi_2,
            a,
            phi_a,
            k,
            wires=wires,
            do_queue=do_queue,
            id=id,
        )

    @property
    def num_params(self):
        return 11

    def expand(self):

        with qml.tape.QuantumTape() as tape:

            for l in range(self.n_layers):

                qml.Interferometer(
                    theta=self.parameters[0][l],
                    phi=self.parameters[1][l],
                    varphi=self.parameters[2][l],
                    wires=self.wires,
                )

                for i in range(len(self.wires)):
                    qml.Squeezing(
                        self.parameters[3][l, i], self.parameters[4][l, i], wires=self.wires[i]
                    )

                qml.Interferometer(
                    theta=self.parameters[5][l],
                    phi=self.parameters[6][l],
                    varphi=self.parameters[7][l],
                    wires=self.wires,
                )

                for i in range(len(self.wires)):
                    qml.Displacement(
                        self.parameters[8][l, i], self.parameters[9][l, i], wires=self.wires[i]
                    )

                for i in range(len(self.wires)):
                    qml.Kerr(self.parameters[10][l, i], wires=self.wires[i])

        if self.inverse:
            tape.inv()

        return tape

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns a list of shapes for the 11 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        """
        n_if = n_wires * (n_wires - 1) // 2

        shapes = (
            [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 3
            + [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 4
        )

        return shapes

    def adjoint(self):
        adjoint_op = CVNeuralNetLayers(*self.parameters, wires=self.wires)
        adjoint_op.inverse = not self.inverse
        return adjoint_op
