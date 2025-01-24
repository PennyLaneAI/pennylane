# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the MPSPrep template.
"""

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires


class MPSPrep(Operation):
    r"""Prepares an initial state from a matrix product state (MPS) representation.

    .. note::

        Currently, this operator can only be used with ``qml.device(“lightning.tensor”)``.


    Args:
        mps (List[Array]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state as a
            product of site matrices. See the usage details section for more information.

        wires (Sequence[int]): wires that the template acts on

    **Example**

    .. code-block::

        mps = [
            np.array([[0.0, 0.107], [0.994, 0.0]]),
            np.array(
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ]
            ),
            np.array([[-1.0, -0.0], [-0.0, -1.0]]),
        ]

        dev = qml.device("lightning.tensor", wires=3)
        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires = [0,1,2])
            return qml.state()

    .. code-block:: pycon

        >>> print(circuit())
        [ 0.        +0.j -0.10705513+0.j  0.        +0.j  0.        +0.j
        0.        +0.j  0.        +0.j -0.99451217+0.j  0.        +0.j]

    .. details::
        :title: Usage Details

        The input MPS must be a list of :math:`n` tensors :math:`[A^{(0)}, ..., A^{(n-1)}]`
        with shapes :math:`d_0, ..., d_{n-1}`, respectively. The first and last tensors have rank :math:`2`
        while the intermediate tensors have rank :math:`3`.

        The first tensor must have the shape :math:`d_0 = (d_{0,0}, d_{0,1})` where :math:`d_{0,0}`
        and :math:`d_{0,1}`  correspond to the physical dimension of the site and an auxiliary bond
        dimension connecting it to the next tensor, respectively.

        The last tensor must have the shape :math:`d_{n-1} = (d_{n-1,0}, d_{n-1,1})` where :math:`d_{n-1,0}`
        and :math:`d_{n-1,1}` represent the auxiliary dimension from the previous site and the physical
        dimension of the site, respectively.

        The intermediate tensors must have the shape :math:`d_j = (d_{j,0}, d_{j,1}, d_{j,2})`, where:

        - :math:`d_{j,0}` is the bond dimension connecting to the previous tensor
        - :math:`d_{j,1}` is the physical dimension for the site
        - :math:`d_{j,2}` is the bond dimension connecting to the next tensor

        Note that the bond dimensions must match between adjacent tensors such that :math:`d_{j-1,2} = d_{j,0}`.

        Additionally, the physical dimension of the site should always be fixed at :math:`2`
        (since the dimension of a qubit is :math:`2`), while the other dimensions must be powers of two.

        The following example shows a valid MPS input containing four tensors with
        dimensions :math:`[(2,2), (2,2,4), (4,2,2), (2,2)]` which satisfy the criteria described above.

        .. code-block::

            mps = [
                np.array([[0.0, 0.107], [0.994, 0.0]]),
                np.array(
                    [
                        [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                        [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                    ]
                ),
                np.array(
                    [
                        [[-1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 1.0]],
                        [[0.0, -1.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 0.0]],
                    ]
                ),
                np.array([[-1.0, -0.0], [-0.0, -1.0]]),
            ]
    """

    def __init__(self, mps, wires, id=None):

        # Validate the shape and dimensions of the first tensor
        assert qml.math.isclose(
            len(qml.math.shape(mps[0])), 2
        ), "The first tensor must have exactly 2 dimensions."
        dj0, dj2 = qml.math.shape(mps[0])
        assert qml.math.isclose(
            dj0, 2
        ), "The first dimension of the first tensor must be exactly 2."
        assert qml.math.log2(
            dj2
        ).is_integer(), "The second dimension of the first tensor must be a power of 2."

        # Validate the shapes of the intermediate tensors
        for i, array in enumerate(mps[1:-1], start=1):
            shape = qml.math.shape(array)
            assert qml.math.isclose(len(shape), 3), f"Tensor {i} must have exactly 3 dimensions."
            new_dj0, new_dj1, new_dj2 = shape
            assert qml.math.isclose(
                new_dj1, 2
            ), f"The second dimension of tensor {i} must be exactly 2."
            assert qml.math.log2(
                new_dj0
            ).is_integer(), f"The first dimension of tensor {i} must be a power of 2."
            assert qml.math.isclose(
                new_dj1, 2
            ), f"The second dimension of tensor {i} must be exactly 2."
            assert qml.math.log2(
                new_dj2
            ).is_integer(), f"The third dimension of tensor {i} must be a power of 2."
            assert qml.math.isclose(
                new_dj0, dj2
            ), f"Dimension mismatch: tensor {i}'s first dimension does not match the previous third dimension."
            dj2 = new_dj2

        # Validate the shape and dimensions of the last tensor
        assert qml.math.isclose(
            len(qml.math.shape(mps[-1])), 2
        ), "The last tensor must have exactly 2 dimensions."
        new_dj0, new_dj1 = qml.math.shape(mps[-1])
        assert new_dj1 == 2, "The second dimension of the last tensor must be exactly 2."
        assert qml.math.log2(
            new_dj0
        ).is_integer(), "The first dimension of the last tensor must be a power of 2."
        assert qml.math.isclose(
            new_dj0, dj2
        ), "Dimension mismatch: the last tensor's first dimension does not match the previous third dimension."
        super().__init__(*mps, wires=wires, id=id)

    @property
    def mps(self):
        """list representing the MPS input"""
        return self.data

    def _flatten(self):
        hyperparameters = (("wires", self.wires),)
        return self.mps, hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(data, **hyperparams_dict)

    def map_wires(self, wire_map):
        new_wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        return MPSPrep(self.mps, new_wires)

    @classmethod
    def _primitive_bind_call(cls, mps, wires, id=None):
        # pylint: disable=arguments-differ
        if cls._primitive is None:
            # guard against this being called when primitive is not defined.
            return type.__call__(cls, mps=mps, wires=wires, id=id)  # pragma: no cover
        return cls._primitive.bind(*mps, wires=wires, id=id)


if MPSPrep._primitive is not None:  # pylint: disable=protected-access

    @MPSPrep._primitive.def_impl  # pylint: disable=protected-access
    def _(*args, **kwargs):
        return type.__call__(MPSPrep, args, **kwargs)
