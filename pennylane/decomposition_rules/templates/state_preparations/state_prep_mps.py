# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Built-in decomposition rules for templates.state_preparations.state_prep_mps."""

import numpy as np

import pennylane as qp
from pennylane.decomposition import add_decomps, register_resources
from pennylane.templates.state_preparations.state_prep_mps import MPSPrep, right_canonicalize_mps
from pennylane.typing import Complex, Wire


def _mps_prep_decomposition_resources(
    mps, wires, work_wires=None, right_canonicalize=False
):  # pylint: disable=unused-argument
    num_work_wires = len(work_wires)
    num_sites = len(mps)
    return {
        qp.QubitUnitary(
            Complex[2 ** (1 + num_work_wires), 2 ** (1 + num_work_wires)],
            wires=Wire[1 + num_work_wires],
        ): num_sites
    }


def _work_wires_bond_dimension_condition(
    mps, wires, work_wires=None, right_canonicalize=False
):  # pylint: disable=unused-argument
    bond_dimensions = [data.shape[-1] for data in mps]
    max_bond_dimension = max(bond_dimensions[:-1])

    return (
        work_wires is not None
        and len(work_wires) > 0
        and 2 ** len(work_wires) >= max_bond_dimension
    )


@qp.register_condition(_work_wires_bond_dimension_condition)
@register_resources(_mps_prep_decomposition_resources)
def _mps_prep_decomposition(mps, wires, work_wires=None, right_canonicalize=False):
    mps = list(mps)

    n_wires = len(work_wires) + 1

    mps = mps.copy()

    # Transform the MPS to ensure that the generated matrix is unitary
    if right_canonicalize:
        mps = right_canonicalize_mps(mps)

    #  NOTE: tensor legs assignment convention is (vL, p, vR)
    mps[0] = mps[0].reshape((1, *mps[0].shape))
    mps[-1] = mps[-1].reshape((*mps[-1].shape, 1))

    interface, dtype = qp.math.get_interface(mps[0]), mps[0].dtype

    for i, Ai in enumerate(mps):

        vectors = []
        for column in Ai:
            vector = qp.math.zeros(2**n_wires, like=interface, dtype=dtype)
            if interface == "jax":
                vector = vector.at[: len(column[0])].set(column[0])
                vector = vector.at[2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])].set(
                    column[1]
                )
            else:
                vector[: len(column[0])] = column[0]
                vector[2 ** (n_wires - 1) : 2 ** (n_wires - 1) + len(column[1])] = column[1]
            vectors.append(vector)
        vectors = qp.math.stack(vectors).T
        # The unitary is completed using QR decomposition
        d, k = vectors.shape
        assert d == 2**n_wires, "The first dimension of the vectors must match 2**n_wires."
        assert (
            k <= d
        ), "The second dimension of the vectors must be less than or equal to 2**(n_wires-1)."
        new_columns = qp.math.array(np.random.RandomState(42).random((d, d - k)))
        unitary_matrix, R = qp.math.linalg.qr(qp.math.hstack([vectors, new_columns]))
        unitary_matrix *= qp.math.sign(qp.math.diag(R))  # Enforce uniqueness for QR decomposition

        qp.QubitUnitary(unitary_matrix, wires=[wires[i], *work_wires])


add_decomps(MPSPrep, _mps_prep_decomposition)
