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
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.wires import Wires


def _validate_mps_shape(mps):
    r"""Validate that the MPS dimensions are correct.

    Args:
        mps (list[TensorLike]): List of tensors representing the MPS.
    """

    # Validate the shape and dimensions of the first tensor
    assert qml.math.isclose(
        len(qml.math.shape(mps[0])), 2
    ), "The first tensor must have exactly 2 dimensions."
    dj0, dj2 = qml.math.shape(mps[0])
    assert qml.math.isclose(dj0, 2), "The first dimension of the first tensor must be exactly 2."
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


def right_canonicalize_mps(mps):
    r"""Transform a matrix product state (MPS) into its right-canonical form.

    A right-canonicalized MPS is a matrix product state in which the constituent tensors, :math:`A^{(j)}`, satisfy
    the following orthonormality condition [Eq. (21) of `arXiv:2310.18410 <https://arxiv.org/pdf/2310.18410>`_]:

    .. math::

        \sum_{d_{j,1}, d_{j,2}} A^{(j)}_{d_{j, 0}, d_{j, 1}, d_{j, 2}} \left( A^{(j)}_{d'_{j, 0}, d_{j, 1}, d_{j, 2}} \right)^* = \delta_{d_{j, 0}, d'_{j, 0}},

    where :math:`d_{i,j}` denotes the :math:`j` dimension of the :math:`i` tensor and :math:`\delta` is the Kronecker delta.


    Args:
        mps (list[TensorLike]): List of tensors representing the MPS.

    Returns:
        List of tensors representing the MPS in right-canonical form with the same dimensions as the initial MPS.

    .. seealso:: :class:`~.MPSPrep`.

    **Example**

    .. code-block::

        n_sites = 4

        import numpy as np

        mps = ([np.ones((2, 4))] +
               [np.ones((4, 2, 4)) for _ in range(1, n_sites - 1)] +
               [np.ones((4, 2))])

        mps_rc = qml.right_canonicalize_mps(mps)

        # Check that the right-canonical definition is fulfilled
        for i in range(1, n_sites - 1):
            tensor = mps_rc[i]
            contraction_matrix = np.tensordot(tensor, tensor.conj(), axes=([1, 2], [1, 2]))
            assert np.allclose(contraction_matrix, np.eye(tensor.shape[0]))

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
        - :math:`d_{j,1}` is the physical dimension of the site
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

    _validate_mps_shape(mps)

    mps = mps.copy()
    mps[0] = mps[0].reshape((1, *mps[0].shape))
    mps[-1] = mps[-1].reshape((*mps[-1].shape, 1))

    if not qml.math.is_abstract(mps[0]):
        is_right_canonical = True
        for tensor in mps[1:-1]:
            # Right-canonical definition
            input_matrix = qml.math.tensordot(tensor, tensor.conj(), axes=([1, 2], [1, 2]))
            if not qml.math.allclose(input_matrix, qml.math.eye(tensor.shape[0])):
                is_right_canonical = False
                break

        if is_right_canonical:
            mps[0] = mps[0][0]
            mps[-1] = mps[-1][:, :, 0]
            return mps

    d_shapes = []
    for tensor in mps[1:-1]:
        d_shapes += tensor.shape

    max_bond_dim = qml.math.max(d_shapes)

    n_sites = len(mps)
    output_mps = [None] * n_sites

    # Procedure analogous to the left-canonical conversion but starting from the right and storing the Vd,
    # where Vd is the right matrix in the Singular Value Decomposition (SVD)
    for i in range(n_sites - 1, 0, -1):
        chi_left, d, chi_right = mps[i].shape
        input_matrix = mps[i].reshape(chi_left, d * chi_right)

        u_matrix, s_diag, vd_matrix = qml.math.linalg.svd(input_matrix, full_matrices=False)

        # Truncate SVD components if needed
        chi_new = min(int(max_bond_dim), len(s_diag))
        u_matrix = u_matrix[:, :chi_new]
        s_diag = s_diag[:chi_new]
        vd_matrix = vd_matrix[:chi_new, :]

        # Store Vd reshaped as an MPS tensor in the output MPS
        output_mps[i] = vd_matrix.reshape(chi_new, d, chi_right)

        # Contract U with diag(S) and merge it with the preceding MPS tensor, preserving the canonical structure
        mps[i - 1] = qml.math.tensordot(
            mps[i - 1], u_matrix @ qml.math.diag(s_diag), axes=([2], [0])
        )

    output_mps[0] = mps[0][0]
    output_mps[-1] = output_mps[-1][:, :, 0]

    return output_mps


class MPSPrep(Operation):
    r"""Prepares an initial state from a matrix product state (MPS) representation.

    .. note::

        This operator is natively supported on the ``lightning.tensor`` device, which is designed to run MPS
        structures efficiently. For other devices, this operation prepares the state vector represented by the
        MPS using a gate-based decomposition from Eq. (23) in `arXiv:2310.18410
        <https://arxiv.org/pdf/2310.18410>`_, which requires the right canonicalization of the MPS using
        the :func:`~.right_canonicalize_mps` function and defining auxiliary qubits with ``work_wires``.

    Args:
        mps (list[TensorLike]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state
            as a product of site matrices. See the usage details section for more information.

        wires (Sequence[int]): wires that the template acts on. It should match the number of MPS tensors.
        work_wires (Sequence[int]): list of extra qubits needed in the decomposition. If the maximum dimension
            of the MPS tensors is :math:`2^k`, then :math:`k` ``work_wires`` will be needed. If no ``work_wires`` are given,
            this operator can only be executed on the ``lightning.tensor`` device. Default is ``None``.

        right_canonicalize (bool): indicates whether a conversion to right-canonical form should be performed to the MPS.
            Default is ``False``.


    .. seealso:: :func:`~.right_canonicalize_mps`.

    **Example**

    Example using the ``lightning.tensor`` device:

    .. code-block:: python

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

    .. code-block::

        dev = qml.device("lightning.tensor", wires=3)
        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires = [0,1,2])
            return qml.state()

    >>> print(circuit()) # doctest: +SKIP
    [ 0.        +0.j -0.10705513+0.j  0.        +0.j  0.        +0.j
    0.        +0.j  0.        +0.j -0.99451217+0.j  0.        +0.j]

    Example using the ``default.qubit`` device:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=4)
        @qml.qnode(dev)
        def circuit():
            qml.MPSPrep(mps, wires = [1,2,3], work_wires = [0])
            return qml.state()

    >>> print(circuit()[:8]) # doctest: +SKIP
    [ 0.        +0.j -0.10702756+0.j  0.        +0.j  0.        +0.j
      0.        +0.j  0.        +0.j -0.99425605+0.j  0.        +0.j]

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
        - :math:`d_{j,1}` is the physical dimension of the site
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

    resource_keys = {"bond_dimensions", "num_sites", "num_work_wires"}

    def __init__(
        self, mps, wires, work_wires=None, right_canonicalize=False, id=None
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        _validate_mps_shape(mps)

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

    @property
    def resource_params(self) -> dict:
        return {
            "bond_dimensions": [data.shape[-1] for data in self.data],
            "num_sites": len(self.data),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
        }

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

    # pylint: disable=arguments-differ, too-many-arguments
    @classmethod
    def _primitive_bind_call(cls, mps, wires, work_wires=None, id=None, right_canonicalize=False):
        return super()._primitive_bind_call(
            *mps, wires=wires, work_wires=work_wires, id=id, right_canonicalize=right_canonicalize
        )

    def decomposition(self):
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
        The decomposition follows Eq. (23) in `arXiv:2310.18410 <https://arxiv.org/pdf/2310.18410>`_.

        Args:
            mps (list[Array]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state as a
                product of site matrices.

            wires (Sequence[int]): wires that the template acts on. It should match the number of MPS tensors.
            work_wires (Sequence[int]): list of extra qubits needed in the decomposition. If the maximum dimension
                of the MPS tensors is ``2^k``, then k ``work_wires`` will be needed. If no ``work_wires`` are given,
                this operator can only be executed on the ``lightning.tensor`` device. Default is ``None``.

            right_canonicalize (bool): Indicates whether a conversion to right-canonical form should be performed
                to the mps. Default is ``False``.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        if work_wires is None:
            raise ValueError("The qml.MPSPrep decomposition requires `work_wires` to be specified.")

        max_bond_dimension = 0
        for i in range(len(mps) - 1):
            bond_dim = mps[i].shape[-1]
            max_bond_dimension = max(max_bond_dimension, bond_dim)

        if max_bond_dimension > 2 ** len(work_wires):
            raise ValueError(
                f"Incorrect number of `work_wires`. At least {int(qml.math.ceil(qml.math.log2(max_bond_dimension)))} `work_wires` must be provided."
            )

        ops = []
        n_wires = len(work_wires) + 1

        mps = mps.copy()

        # Transform the MPS to ensure that the generated matrix is unitary
        if right_canonicalize:
            mps = right_canonicalize_mps(mps)

        mps[0] = mps[0].reshape((1, *mps[0].shape))
        mps[-1] = mps[-1].reshape((*mps[-1].shape, 1))

        interface, dtype = qml.math.get_interface(mps[0]), mps[0].dtype

        for i, Ai in enumerate(mps):

            # Encode the tensor Ai in a unitary matrix following Eq.23 in https://arxiv.org/pdf/2310.18410
            vectors = []
            for column in Ai:
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
            # The unitary is completed using QR decomposition
            d, k = vectors.shape
            new_columns = qml.math.array(np.random.RandomState(42).random((d, d - k)))
            unitary_matrix, R = qml.math.linalg.qr(qml.math.hstack([vectors, new_columns]))
            unitary_matrix *= qml.math.sign(
                qml.math.diag(R)
            )  # Enforce uniqueness for QR decomposition

            ops.append(qml.QubitUnitary(unitary_matrix, wires=[wires[i]] + work_wires))

        return ops


if MPSPrep._primitive is not None:  # pylint: disable=protected-access

    @MPSPrep._primitive.def_impl  # pylint: disable=protected-access
    def _(*args, n_wires, **kwargs):
        mps, wires = args[:-n_wires], args[-n_wires:]
        return type.__call__(MPSPrep, mps, wires=wires, **kwargs)


def _mps_prep_decomposition_resources(
    bond_dimensions, num_sites, num_work_wires
):  # pylint: disable=unused-argument
    return {resource_rep(qml.QubitUnitary, num_wires=1 + num_work_wires): num_sites}


def _work_wires_bond_dimension_condition(
    bond_dimensions, num_sites, num_work_wires
):  # pylint: disable=unused-argument
    max_bond_dimension = max(bond_dimensions[:-1])

    return (
        num_work_wires is not None
        and num_work_wires > 0
        and 2**num_work_wires >= max_bond_dimension
    )


@qml.register_condition(_work_wires_bond_dimension_condition)
@register_resources(_mps_prep_decomposition_resources)
def _mps_prep_decomposition(*mps, **kwargs):
    wires = kwargs["wires"]
    work_wires = kwargs["work_wires"]
    right_canonicalize = kwargs["right_canonicalize"]
    mps = list(mps)

    n_wires = len(work_wires) + 1

    mps = mps.copy()

    # Transform the MPS to ensure that the generated matrix is unitary
    if right_canonicalize:
        mps = right_canonicalize_mps(mps)

    #  NOTE: tensor legs assignment convention is (vL, p, vR)
    mps[0] = mps[0].reshape((1, *mps[0].shape))
    mps[-1] = mps[-1].reshape((*mps[-1].shape, 1))

    interface, dtype = qml.math.get_interface(mps[0]), mps[0].dtype

    for i, Ai in enumerate(mps):

        vectors = []
        for column in Ai:
            vector = qml.math.zeros(2**n_wires, like=interface, dtype=dtype)
            if interface == "jax":
                vector = vector.at[: len(column[0])].set(column[0])
                vector = vector.at[2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])].set(
                    column[1]
                )
            else:
                vector[: len(column[0])] = column[0]
                vector[2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])] = column[1]
            vectors.append(vector)
        vectors = qml.math.stack(vectors).T
        # The unitary is completed using QR decomposition
        d, k = vectors.shape
        assert d == 2**n_wires, "The first dimension of the vectors must match 2**n_wires."
        assert (
            k <= d
        ), "The second dimension of the vectors must be less than or equal to 2**(n_wires-1)."
        new_columns = qml.math.array(np.random.RandomState(42).random((d, d - k)))
        unitary_matrix, R = qml.math.linalg.qr(qml.math.hstack([vectors, new_columns]))
        unitary_matrix *= qml.math.sign(qml.math.diag(R))  # Enforce uniqueness for QR decomposition

        qml.QubitUnitary(unitary_matrix, wires=[wires[i]] + work_wires)


add_decomps(MPSPrep, _mps_prep_decomposition)
