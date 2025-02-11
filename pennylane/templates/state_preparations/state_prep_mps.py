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


def right_canonicalize_mps(mps):
    """
    Transform an MPS into a right-canonical MPS.

    Args:
      mps (list[Array]): List of tensors representing the MPS.

    Returns:
      A list of tensors representing the MPS in right-canonical form.
    """

    L = len(mps)
    output_mps = [None] * L

    is_right_canonical = True
    for i in range(1, L - 1):
        tensor = mps[i]
        # Right-canonical definition
        M = np.tensordot(tensor, tensor.conj(), axes=([1, 2], [1, 2]))
        if not np.allclose(M, np.eye(tensor.shape[0])):
            is_right_canonical = False
            break

    if is_right_canonical:
        return mps

    max_bond_dim = 0
    for tensor in mps[1:-1]:
        D_left = tensor.shape[0]
        D_right = tensor.shape[2]
        max_bond_dim = max(max_bond_dim, D_left, D_right)

    # Procedure analogous to the left-canonical conversion but starting from the right and storing the Vd
    for i in range(L - 1, 0, -1):
        chi_left, d, chi_right = mps[i].shape
        M = mps[i].reshape(chi_left, d * chi_right)
        U, S, Vd = qml.math.linalg.svd(M, full_matrices=False)

        # Truncate SVD components if needed
        chi_new = min(int(max_bond_dim), len(S))
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vd = Vd[:chi_new, :]

        output_mps[i] = Vd.reshape(chi_new, d, chi_right)

        US = U @ qml.math.diag(S)
        mps[i - 1] = qml.math.tensordot(mps[i - 1], US, axes=([2], [0]))

    output_mps[0] = mps[0]
    return output_mps


class MPSPrep(Operation):
    r"""Prepares an initial state from a matrix product state (MPS) representation.

    .. note::

        This operator is natively supported on the ``lightning.tensor`` device, designed to run MPS structures
        efficiently. For other devices, implementing this operation uses a gate-based decomposition which requires
        auxiliary qubits (via ``work_wires``) to prepare the state vector represented by the MPS in a quantum circuit.



    Args:
        mps (list[Array]):  list of arrays of rank-3 and rank-2 tensors representing a right-canonized MPS state
            as a product of site matrices. See the usage details section for more information.

        wires (Sequence[int]): wires that the template acts on
        work_wires (Sequence[int]): list of extra qubits needed in the decomposition. The maximum permissible bond
            dimension of the provided MPS is defined as ``2^len(work_wires)``. Default is ``None``.


    The decomposition follows Eq. (23) in `[arXiv:2310.18410] <https://arxiv.org/pdf/2310.18410>`_.

    .. seealso:: :func:`~.right_canonicalize_mps`.

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

        A right-canonized MPS is a matrix product state where each tensor :math:`A^{(j)}` satisfies
        the following orthonormality condition:

        .. math::

            \sum_{\alpha_j} A^{(j)}_{\alpha_{j-1}, s, \alpha_j} \left( A^{(j)}_{\alpha'_{j-1}, s, \alpha_j} \right)^* = \delta_{\alpha_{j-1}, \alpha'_{j-1}}

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

    def __init__(self, mps, wires, work_wires=None, right_canonicalize=False, id=None):

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
        self.hyperparameters["right_canonicalize"] = right_canonicalize

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
            ("right_canonicalize", self.hyperparameters["right_canonicalize"]),
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

        return MPSPrep(
            self.mps, new_wires, new_work_wires, self.hyperparameters["right_canonicalize"]
        )

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
    def compute_decomposition(
        mps, wires, work_wires, right_canonicalize=False
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        The decomposition follows Eq. (23) in `[arXiv:2310.18410] <https://arxiv.org/pdf/2310.18410>`_.

        Args:
            mps (list[Array]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state as a
                product of site matrices.

            wires (Sequence[int]): wires that the template acts on
            work_wires (Sequence[int]): list of extra qubits needed in the decomposition. The maximum permissible bond
                dimension of the provided MPS is defined as ``2^len(work_wires)``. Default is ``None``.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        bond_dimensions = []

        for i in range(len(mps) - 1):
            bond_dim = mps[i].shape[-1]
            bond_dimensions.append(bond_dim)

        max_bond_dimension = max(bond_dimensions)

        if work_wires is None:
            raise ValueError("The qml.MPSPrep decomposition requires `work_wires` to be specified.")

        if 2 ** len(work_wires) < max_bond_dimension:
            raise ValueError("The bond dimension cannot exceed `2**len(work_wires)`.")

        ops = []
        n_wires = len(work_wires) + 1

        mps[0] = mps[0].reshape((1, *mps[0].shape))
        mps[-1] = mps[-1].reshape((*mps[-1].shape, 1))

        # We transform the mps to ensure that the generated matrix is unitary
        if right_canonicalize:
            mps = right_canonicalize_mps(mps)

        for i, Ai in enumerate(mps):

            # encodes the tensor Ai in a unitary matrix following Eq.23 in https://arxiv.org/pdf/2310.18410

            vectors = []
            for column in Ai:

                interface, dtype = qml.math.get_interface(mps[0]), mps[0].dtype
                vector = qml.math.zeros(2**n_wires, like=interface, dtype=dtype)

                if interface == "jax":
                    vector = vector.at[: len(column[0])].set(column[0])
                    vector = vector.at[
                        2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])
                    ].set(column[1])

                else:
                    vector[: len(column[0])] = column[0]
                    vector[2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])] = column[1]

                vectors.append(vector)

            vectors = qml.math.stack(vectors).T
            d = vectors.shape[0]
            k = vectors.shape[1]

            # The unitary is completed using QR decomposition
            rng = np.random.default_rng(42)
            new_columns = qml.math.array(rng.random((d, d - k)))

            matrix, R = qml.math.linalg.qr(qml.math.hstack([vectors, new_columns]))
            matrix *= qml.math.sign(qml.math.diag(R))  # enforces uniqueness for QR decomposition

            ops.append(qml.QubitUnitary(matrix, wires=[wires[i]] + work_wires))

        return ops


if MPSPrep._primitive is not None:  # pylint: disable=protected-access

    @MPSPrep._primitive.def_impl  # pylint: disable=protected-access
    def _(*args, **kwargs):
        return type.__call__(MPSPrep, args, **kwargs)
