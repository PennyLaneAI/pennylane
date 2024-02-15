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
Contains the SimplifiedTwoDesign template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class SimplifiedTwoDesign(Operation):
    r"""
    Layers consisting of a simplified 2-design architecture of Pauli-Y rotations and controlled-Z entanglers
    proposed in `Cerezo et al. (2021) <https://doi.org/10.1038/s41467-021-21728-w>`_.

    A 2-design is an ensemble of unitaries whose statistical properties are the same as sampling random unitaries
    with respect to the Haar measure up to the first 2 moments.

    The template is not a strict 2-design, since
    it does not consist of universal 2-qubit gates as building blocks, but has been shown in
    `Cerezo et al. (2021) <https://doi.org/10.1038/s41467-021-21728-w>`_ to exhibit important properties to study "barren plateaus"
    in quantum optimization landscapes.

    The template starts with an initial layer of single qubit Pauli-Y rotations, before the main
    :math:`L` layers are applied. The basic building block of the main layers are controlled-Z entanglers
    followed by a pair of Pauli-Y rotation gates (one for each wire).
    Each layer consists of an "even" part whose entanglers start with the first qubit,
    and an "odd" part that starts with the second qubit.

    This is an example of two layers, including the initial layer:

    .. figure:: ../../_static/templates/layers/simplified_two_design.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    |

    The argument ``initial_layer_weights`` contains the rotation angles of the initial layer of Pauli-Y rotations,
    while ``weights`` contains the pairs of Pauli-Y rotation angles of the respective layers. Each layer takes
    :math:`\lfloor M/2 \rfloor + \lfloor (M-1)/2 \rfloor = M-1` pairs of angles, where :math:`M` is the number of wires.
    The number of layers :math:`L` is derived from the first dimension of ``weights``.

    Args:
        initial_layer_weights (tensor_like): weight tensor for the initial rotation block, shape ``(M,)``
        weights (tensor_like): tensor of rotation angles for the layers, shape ``(L, M-1, 2)``
        wires (Iterable): wires that the template acts on


    .. details::
        :title: Usage Details

        template - here shown for two layers - is used inside a :class:`QNode <pennylane.QNode>`:

        .. code-block:: python

            import pennylane as qml
            from math import pi

            n_wires = 3
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(init_weights, weights):
                qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))
                return [qml.expval(qml.Z(i)) for i in range(n_wires)]

            init_weights = [pi, pi, pi]
            weights_layer1 = [[0., pi],
                              [0., pi]]
            weights_layer2 = [[pi, 0.],
                              [pi, 0.]]
            weights = [weights_layer1, weights_layer2]

            >>> circuit(init_weights, weights)
            [1., -1., 1.]

        **Parameter shapes**

        A list of shapes for the two weights arguments can be computed with the static method
        :meth:`~.qml.SimplifiedTwoDesign.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shapes = qml.SimplifiedTwoDesign.shape(n_layers=2, n_wires=2)
            weights = [np.random.random(size=shape) for shape in shapes]

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, initial_layer_weights, weights, wires, id=None):
        shape = qml.math.shape(weights)

        if len(shape) > 1:
            if shape[1] != len(wires) - 1:
                raise ValueError(
                    f"Weights tensor must have second dimension of length {len(wires) - 1}; got {shape[1]}"
                )

            if shape[2] != 2:
                raise ValueError(
                    f"Weights tensor must have third dimension of length 2; got {shape[2]}"
                )

        shape2 = qml.math.shape(initial_layer_weights)
        if shape2 != (len(wires),):
            raise ValueError(
                f"Initial layer weights must be of shape {(len(wires),)}; got {shape2}"
            )

        self.n_layers = shape[0]

        super().__init__(initial_layer_weights, weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @staticmethod
    def compute_decomposition(
        initial_layer_weights, weights, wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.SimplifiedTwoDesign.decomposition`.

        Args:
            initial_layer_weights (tensor_like): weight tensor for the initial rotation block
            weights (tensor_like): tensor of rotation angles for the layers
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> qml.SimplifiedTwoDesign.compute_decomposition(initial_layer_weights, weights, wires=["a", "b", "c"])
        [RY(tensor(3.1416), wires=['a']), RY(tensor(3.1416), wires=['b']), RY(tensor(3.1416), wires=['c']),
        CZ(wires=['a', 'b']),
        RY(tensor(0.), wires=['a']), RY(tensor(3.1416), wires=['b']),
        CZ(wires=['b', 'c']),
        RY(tensor(0.), wires=['b']), RY(tensor(3.1416), wires=['c']),
        CZ(wires=['a', 'b']),
        RY(tensor(3.1416), wires=['a']), RY(tensor(0.), wires=['b']),
        CZ(wires=['b', 'c']),
        RY(tensor(3.1416), wires=['b']), RY(tensor(0.), wires=['c'])]
        """

        n_layers = qml.math.shape(weights)[0]
        op_list = []

        # initial rotations
        for i in range(len(wires)):  # pylint: disable=consider-using-enumerate
            op_list.append(qml.RY(initial_layer_weights[i], wires=wires[i]))

        for layer in range(n_layers):
            # even layer of entanglers
            even_wires = [wires[i : i + 2] for i in range(0, len(wires) - 1, 2)]
            for i, wire_pair in enumerate(even_wires):
                op_list.append(qml.CZ(wires=wire_pair))
                op_list.append(qml.RY(weights[layer, i, 0], wires=wire_pair[0]))
                op_list.append(qml.RY(weights[layer, i, 1], wires=wire_pair[1]))

            # odd layer of entanglers
            odd_wires = [wires[i : i + 2] for i in range(1, len(wires) - 1, 2)]
            for i, wire_pair in enumerate(odd_wires):
                op_list.append(qml.CZ(wires=wire_pair))
                op_list.append(qml.RY(weights[layer, len(wires) // 2 + i, 0], wires=wire_pair[0]))
                op_list.append(qml.RY(weights[layer, len(wires) // 2 + i, 1], wires=wire_pair[1]))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns a list of shapes for the 2 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        """

        if n_wires == 1:
            return [(n_wires,), (n_layers,)]

        return [(n_wires,), (n_layers, n_wires - 1, 2)]
