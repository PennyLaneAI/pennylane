import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator


@pytest.mark.parametrize(
    ("matrix_elements", "init_term", "mapping", "terms_exp"),
    [
        (
            [
                np.array([[0.0, 0.0, 0.5], [1.0, 1.0, -0.5], [1.0, 0.0, 0.0]]),
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.5],
                        [0.0, 1.0, 1.0, 0.0, -0.5],
                        [1.0, 0.0, 0.0, 1.0, -1.0],
                    ]
                ),
            ],
            1 / 4,
            "bravyi_KITAEV",
            {
                (): (0.0625 + 0j),
                ((0, "Z"),): (-0.0625 + 0j),
                ((0, "Z"), (1, "Z")): (0.4375 + 0j),
                ((1, "Z"),): (-0.1875 + 0j),
            },
        ),
        (
            [
                np.array([[0.0, 0.0, 0.5], [1.0, 1.0, -0.5], [1.0, 0.0, 0.0]]),
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.5],
                        [0.0, 1.0, 1.0, 0.0, -0.5],
                        [1.0, 0.0, 0.0, 1.0, -1.0],
                    ]
                ),
            ],
            1 / 4,
            "JORDAN_wigner",
            {
                (): (0.0625 + 0j),
                ((0, "Z"),): (-0.0625 + 0j),
                ((1, "Z"),): (0.4375 + 0j),
                ((0, "Z"), (1, "Z")): (-0.1875 + 0j),
            },
        ),
        (
            [np.array([[0.0, 0.0, 0.5], [1.0, 1.0, -0.5], [1.0, 0.0, 0.0]])],
            1 / 2,
            "JORDAN_wigner",
            {(): (0.5 + 0j), ((0, "Z"),): (-0.25 + 0j), ((1, "Z"),): (0.25 + 0j)},
        ),
        (
            [np.array([[0.0, 0.0, 0.5], [1.0, 1.0, -0.5], [1.0, 0.0, 0.0]])],
            0,
            "JORDAN_wigner",
            {((0, "Z"),): (-0.25 + 0j), ((1, "Z"),): (0.25 + 0j)},
        ),
        (
            [np.array([[0.0, 0.0, 0.2], [1.0, 1.0, -0.5], [1.0, 0.0, 0.0]])],
            1 / 2,
            "bravyi_KITAEV",
            {(): (0.35 + 0j), ((0, "Z"),): (-0.1 + 0j), ((0, "Z"), (1, "Z")): (0.25 + 0j),},
        ),
        (
            [
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.5],
                        [0.0, 1.0, 1.0, 0.0, -0.5],
                        [0.0, 2.0, 2.0, 0.0, 0.5],
                        [0.0, 3.0, 3.0, 0.0, -0.5],
                        [1.0, 0.0, 0.0, 1.0, -0.5],
                        [2.0, 0.0, 0.0, 2.0, 0.5],
                    ]
                )
            ],
            1 / 2,
            "JORDAN_wigner",
            {
                (): (0.4375 + 0j),
                ((1, "Z"),): (0.125 + 0j),
                ((0, "Z"), (1, "Z")): (-0.125 + 0j),
                ((2, "Z"),): (-0.125 + 0j),
                ((0, "Z"), (2, "Z")): (0.125 + 0j),
                ((0, "Z"),): (0.0625 + 0j),
                ((3, "Z"),): (0.0625 + 0j),
                ((0, "Z"), (3, "Z")): (-0.0625 + 0j),
            },
        ),
        (
            [
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0, -0.5],
                        [0.0, 2.0, 2.0, 0.0, 1.0],
                        [0.0, 3.0, 3.0, 0.0, -0.5],
                        [1.0, 0.0, 0.0, 1.0, -0.5],
                        [2.0, 0.0, 0.0, 2.0, -0.5],
                    ]
                )
            ],
            1 / 4,
            "bravyi_KITAEV",
            {
                (): (0.125 + 0j),
                ((0, "Z"), (1, "Z")): (0.125 + 0j),
                ((1, "Z"),): (-0.125 + 0j),
                ((2, "Z"),): (-0.0625 + 0j),
                ((0, "Z"), (2, "Z")): (0.0625 + 0j),
                ((1, "Z"), (2, "Z"), (3, "Z")): (0.0625 + 0j),
                ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")): (-0.0625 + 0j),
                ((0, "Z"),): (0.125 + 0j),
            },
        ),
    ],
)
def test_observable(matrix_elements, init_term, mapping, terms_exp, custom_wires, monkeypatch):
    r"""Tests the correctness of the 'observable' function used to build many-body observables.

    The parametrized inputs `terms_exp` are `.terms` attribute of the corresponding
    `QubitOperator. The equality checking is implemented in the `qchem` module itself
    as it could be something useful to the users as well.
    """

    res_obs = qchem.observable(
        matrix_elements, init_term=init_term, mapping=mapping, wires=custom_wires
    )

    qubit_op = QubitOperator()
    monkeypatch.setattr(qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(qubit_op, res_obs, wires=custom_wires)


msg1 = "Expected entries of matrix element tables to be of shape"
msg2 = "Expected dimension for arrays in 'matrix_elements' is 2"


@pytest.mark.parametrize(
    ("matrix_elements", "msg_match"),
    [
        ([np.array([[0.0, 0.0, 1.0, 0.5]])], msg1),
        ([np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 0.5]])], msg1),
        ([np.array([[0.0, 0.0, 0.5]]), np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.25]])], msg1),
        ([np.array([[0.0, 0.0, 0.5, 3]]), np.array([[0.0, 0.0, 0.0, 0.0, 0.25]])], msg1),
        ([np.array([0.0, 0.0, 1.0, 0.5])], msg2),
        ([np.array([[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.5]])], msg2),
    ],
)
def test_exceptions_observable(matrix_elements, msg_match):
    """Test that the 'observable' function throws an exception if the
    array containing the matrix elements has incorrect shapes."""

    with pytest.raises(ValueError, match=msg_match):
        qchem.observable(matrix_elements)


def test_mapping_observable(msg_match="transformation is not available"):
    """Test that the 'observable' function throws an exception if the
    fermionic-to-qubit mapping is not properly defined."""

    matrix_elements = [np.array([[0.0, 0.0, 0.5], [1.0, 1.0, -0.5]])]

    with pytest.raises(TypeError, match=msg_match):
        qchem.observable(matrix_elements, mapping="no_valid_transformation")
