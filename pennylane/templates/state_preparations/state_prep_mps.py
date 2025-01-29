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

import numpy as np

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires


def _complete_unitary(columns):
    """
    Completes a unitary matrix given a list of orthonormal columns.

    Args:
        columns (List[Array]): List of initial orthonormal columns of dimension d.

    Returns:
        Array: Completed unitary matrix of dimension :math:`(d, d)`.
    """

    columns = qml.math.stack(columns).T
    d = columns.shape[0]
    k = columns.shape[1]

    unitary = qml.math.zeros_like(columns @ columns.T)

    if qml.math.get_interface(columns) == "jax":
        unitary = unitary.at[:, :k].set(columns)

    else:
        unitary[:, :k] = columns

    # Complete the remaining columns using Gram-Schmidt
    rng = np.random.default_rng(42)
    for j in range(k, d):
        random_vec = qml.math.array(rng.random(d))
        for i in range(j):
            random_vec -= qml.math.dot(qml.math.conj(unitary[:, i]), random_vec) * unitary[:, i]

        random_vec /= qml.math.linalg.norm(random_vec)

        if qml.math.get_interface(columns) == "jax":
            unitary = unitary.at[:, j].set(random_vec)
        else:
            unitary[:, j] = random_vec

    return unitary


class MPSPrep(Operation):
    r"""Prepares an initial state from a matrix product state (MPS) representation.

    .. note::

        Tensor simulators are designed to run MPS structures more efficiently.
        However, it may be useful to use state vector simulator if you are looking for a gate decomposition that
        prepares the mps in a quantum circuit. For the gate decomposition is required to introduce work qubits
        for the decomposition.



    Args:
        mps (List[Array]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state as a
            product of site matrices. See the usage details section for more information.

        wires (Sequence[int]): wires that the template acts on
        work_wires (Sequence[int]): list of extra qubits needed in the decomposition. The bond dimension of the mps
                        is defined as ``2^len(work_wires)``. Default is ``None``

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

    def __init__(self, mps, wires, work_wires=None, id=None):

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

        self.hyperparameters["input_wires"] = qml.wires.Wires(wires)

        if work_wires:
            self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires)
            all_wires = self.hyperparameters["input_wires"] + self.hyperparameters["work_wires"]
        else:
            self.hyperparameters["work_wires"] = None
            all_wires = self.hyperparameters["input_wires"]

        super().__init__(*mps, wires=all_wires, id=id)

    @property
    def mps(self):
        """list representing the MPS input"""
        return self.data

    def _flatten(self):
        hyperparameters = (
            ("wires", self.hyperparameters["input_wires"]),
            ("work_wires", self.hyperparameters["work_wires"]),
        )
        return self.mps, hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(data, **hyperparams_dict)

    def map_wires(self, wire_map):
        new_wires = Wires(
            [wire_map.get(wire, wire) for wire in self.hyperparameters["input_wires"]]
        )
        new_work_wires = Wires(
            [wire_map.get(wire, wire) for wire in self.hyperparameters["work_wires"]]
        )

        return MPSPrep(self.mps, new_wires, new_work_wires)

    @classmethod
    def _primitive_bind_call(cls, mps, wires, id=None):
        # pylint: disable=arguments-differ
        if cls._primitive is None:
            # guard against this being called when primitive is not defined.
            return type.__call__(cls, mps=mps, wires=wires, id=id)  # pragma: no cover
        return cls._primitive.bind(*mps, wires=wires, id=id)

    def decomposition(self):  # pylint: disable=arguments-differ
        filtered_hyperparameters = {
            key: value for key, value in self.hyperparameters.items() if key != "input_wires"
        }
        return self.compute_decomposition(
            self.parameters, wires=self.hyperparameters["input_wires"], **filtered_hyperparameters
        )

    @staticmethod
    def compute_decomposition(mps, wires, work_wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            mps (List[Array]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state as a
                product of site matrices. See the usage details section for more information.

            wires (Sequence[int]): wires that the template acts on
            work_wires (Sequence[int]): list of extra qubits needed. The bond dimension of the mps
                is defined as ``2^len(work_wires)``

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        if work_wires is None:
            raise AssertionError("To decompose MPSPrep you must specify `work_wires`.")

        ops = []
        n_wires = len(work_wires) + 1

        for i, Ai in enumerate(mps):
            vectors = []

            # encodes the tensor Ai in a unitary matrix following Eq.23 in https://arxiv.org/pdf/2310.18410
            if i == 0:
                Ai = Ai.reshape((1, *Ai.shape))
            elif i == len(mps) - 1:
                Ai = Ai.reshape((*Ai.shape, 1))

            for column in Ai:

                vector = qml.math.zeros(2**n_wires, like=mps[0])

                if qml.math.get_interface(mps[0]) == "jax":
                    vector = vector.at[: len(column[0])].set(column[0])
                    vector = vector.at[
                        2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])
                    ].set(column[1])

                else:
                    vector[: len(column[0])] = column[0]
                    vector[2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])] = column[1]

                vectors.append(vector)

            matrix = _complete_unitary(vectors)
            ops.append(qml.QubitUnitary(matrix, wires=[wires[i]] + work_wires))

        return ops


if MPSPrep._primitive is not None:  # pylint: disable=protected-access

    @MPSPrep._primitive.def_impl  # pylint: disable=protected-access
    def _(*args, **kwargs):
        return type.__call__(MPSPrep, args, **kwargs)
