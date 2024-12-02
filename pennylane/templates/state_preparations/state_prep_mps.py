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

from pennylane.operation import Operation
from pennylane.wires import Wires


class MPSPrep(Operation):
    r"""Prepares an initial state using a matrix product state (MPS) representation.

    .. note::

        Currently, this operator can only be used with ``qml.device(“lightning.tensor”)``.


    Args:
        mps (List[arrays]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state as a product of MPS
            site matrices. More information on dimensions below in Usage Deatils

        wires (Sequence[int]): wires that the template acts on

    **Example**

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

        dev = qml.device("lightning.tensor", wires = 3)
        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires = [0,1,2])
            return qml.state()

    .. details::
        :title: Usage Details

        The matrix product state, is a list of :math:`n` tensors of the following form:

        1. **First tensor**:
           - The first tensor :math:`A^{(1)}` has **rank 2** (i.e., it's a matrix).
           - Its dimensions are :math:`d_{1,0}` and :math:`d_{1,1}`.
           - These correspond to the physical dimension of the site and an auxiliary bond dimension connecting it to the next tensor.

        2. **Last tensor**:
           - The last tensor :math:`A^{(n)}` also has **rank 2** (another matrix).
           - Its dimensions are :math:`d_{n-1,0}` and :math:`d_{n-1,1}`, representing the auxiliary dimension from the previous site and the physical dimension of the site.

        3. **Intermediate tensors**:
           - All tensors in the middle (i.e., :math:`A^{(j)}` for :math:`2 \\leq j \\leq n-1`) have **rank 3** (three-dimensional tensors).
           - Their dimensions are :math:`d_{j,0}, d_{j,1},` and :math:`d_{j,2}`, where:
             - :math:`d_{j,0}`: Bond dimension connecting to the previous tensor.
             - :math:`d_{j,1}`: Physical dimension for the site.
             - :math:`d_{j,2}`: Bond dimension connecting to the next tensor.

           - Importantly, the bond dimensions match between adjacent tensors:

             .. math::

                d_{j-1,2} = d_{j,0}.

        In addition, all dimensions must be powers of two.
        The following input is valid:

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

        The dimensions of ``mps`` are: :math:`[(2,2), (2,2,4), (4,2,2), (2,2)]`, that satisfy the criteria described above.




    """

    def __init__(self, mps, wires, id=None):
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
