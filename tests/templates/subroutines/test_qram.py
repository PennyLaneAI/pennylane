# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the :func:`pennylane.template.subroutines.qram` class.
"""

import re

import numpy as np
import pytest

from pennylane import device, measure, qnode, registers, workflow
from pennylane.decomposition import list_decomps
from pennylane.measurements import probs
from pennylane.ops import CH, CNOT, CSWAP, CZ, SWAP, Controlled, MultiControlledX, Toffoli, X
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid
from pennylane.tape import QuantumScript
from pennylane.templates import BasisEmbedding
from pennylane.templates.subroutines.qram import BBQRAM, FFQRAM, HybridQRAM, SelectOnlyQRAM

has_jax = True
try:
    from jax import numpy as jnp
except ImportError:
    has_jax = False


dev = device("default.qubit")


@qnode(dev)
def bb_quantum(data, control_wires, target_wires, work_wires, address):
    BasisEmbedding(address, wires=control_wires)
    BBQRAM(
        data,
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=work_wires,
    )
    return probs(wires=target_wires)


@pytest.mark.jax
@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize(
    (
        "data",
        "control_wires",
        "target_wires",
        "work_wires",
        "address",
        "probabilities",
    ),
    [
        (
            [
                "010",
                "111",
                "110",
                "000",
            ],
            [0, 1],
            [2, 3, 4],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            2,  # addressed from the left
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # |110>
        ),
        (
            np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
            np.array([0, 1]),
            np.array([2, 3, 4]),
            np.array([5, 11, 10, 9, 6, 7, 8, 12, 13, 14]),
            1,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # |111>
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            [5, 6, 7, 8, 12, 13, 14, 9, 10, 11],
            0,
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |010>
        ),
    ],
)
def test_bb_quantum(
    data,
    control_wires,
    target_wires,
    work_wires,
    address,
    probabilities,
):  # pylint: disable=too-many-arguments

    if has_jax and not isinstance(data[0], str) and not isinstance(data, np.ndarray):
        data, control_wires, target_wires, work_wires = (
            jnp.array(data),
            jnp.array(control_wires),
            jnp.array(target_wires),
            jnp.array(work_wires),
        )

    assert np.allclose(
        probabilities,
        bb_quantum(
            data,
            control_wires,
            target_wires,
            work_wires,
            address,
        ),
    )


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        (
            (
                [],
                [0, 1],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            ),
            ValueError,
            "data' cannot be empty.",
        ),
        (
            (
                [[0, 0, 0], [1, 1, 1]],
                [0, 1],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            ),
            ValueError,
            "data.shape[0] must be 2^(len(control_wires)).",
        ),
        (
            (
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                [0, 1],
                [2, 3],
                [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            ),
            ValueError,
            "len(target_wires) must equal bitstring length.",
        ),
        (
            (
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                [0, 1],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13],
            ),
            ValueError,
            "work_wires must have length 10.",
        ),
    ],
)
def test_raises(params, error, match):
    with pytest.raises(error, match=re.escape(match)):
        BBQRAM(*params)


@pytest.mark.parametrize(
    (
        "data",
        "control_wires",
        "target_wires",
        "bus",
        "dir_wires",
        "portL_wires",
        "portR_wires",
    ),
    [
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            5,
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            5,
            [11, 10, 9],
            [6, 7, 8],
            [12, 13, 14],
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            5,
            [6, 7, 8],
            [12, 13, 14],
            [9, 10, 11],
        ),
    ],
)
def test_bbqram_decomposition_new(
    data,
    control_wires,
    target_wires,
    bus,
    dir_wires,
    portL_wires,
    portR_wires,
):  # pylint: disable=too-many-arguments
    op = BBQRAM(
        data,
        control_wires,
        target_wires,
        [bus] + dir_wires + portL_wires + portR_wires,
    )

    for rule in list_decomps(BBQRAM):
        _test_decomposition_rule(op, rule)


@qnode(dev)
def hybrid_quantum(
    data, control_wires, target_wires, work_wires, k, address
):  # pylint: disable=too-many-arguments
    BasisEmbedding(address, wires=control_wires)
    HybridQRAM(
        data,
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=work_wires,
        k=k,
    )
    return probs(wires=target_wires)


@pytest.mark.jax
@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize(
    (
        "data",
        "control_wires",
        "target_wires",
        "work_wires",
        "k",
        "address",
        "probabilities",
        "expected_circuit",
    ),
    [
        (
            np.array(
                [
                    "010",
                    "111",
                    "110",
                    "000",
                ]
            ),
            np.array([0, 1]),
            np.array([2, 3, 4]),
            np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            0,
            2,  # addressed from the left
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # |110>
            [
                BasisEmbedding([1, 0], wires=[0, 1]),
                X(5),
                CSWAP(wires=[5, 0, 6]),
                CSWAP(wires=[5, 6, 7]),
                CSWAP(wires=[5, 1, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                CSWAP(wires=[5, 10, 8]),
                CSWAP(wires=[5, 13, 9]),
                CH(wires=[5, 2]),
                CSWAP(wires=[5, 2, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[10, 14]), control_wires=[5, 8]),
                Controlled(
                    SWAP(wires=[10, 11]), control_wires=[5, 8], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[13, 15]), control_wires=[5, 9]),
                Controlled(
                    SWAP(wires=[13, 12]), control_wires=[5, 9], control_values=[True, False]
                ),
                CZ(wires=[5, 14]),
                CZ(wires=[5, 12]),
                Controlled(
                    SWAP(wires=[13, 12]), control_wires=[5, 9], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[13, 15]), control_wires=[5, 9]),
                Controlled(
                    SWAP(wires=[10, 11]), control_wires=[5, 8], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[10, 14]), control_wires=[5, 8]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 2, 6]),
                CH(wires=[5, 2]),
                CH(wires=[5, 3]),
                CSWAP(wires=[5, 3, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[10, 14]), control_wires=[5, 8]),
                Controlled(
                    SWAP(wires=[10, 11]), control_wires=[5, 8], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[13, 15]), control_wires=[5, 9]),
                Controlled(
                    SWAP(wires=[13, 12]), control_wires=[5, 9], control_values=[True, False]
                ),
                CZ(wires=[5, 11]),
                CZ(wires=[5, 14]),
                CZ(wires=[5, 12]),
                Controlled(
                    SWAP(wires=[13, 12]), control_wires=[5, 9], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[13, 15]), control_wires=[5, 9]),
                Controlled(
                    SWAP(wires=[10, 11]), control_wires=[5, 8], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[10, 14]), control_wires=[5, 8]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 3, 6]),
                CH(wires=[5, 3]),
                CH(wires=[5, 4]),
                CSWAP(wires=[5, 4, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[10, 14]), control_wires=[5, 8]),
                Controlled(
                    SWAP(wires=[10, 11]), control_wires=[5, 8], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[13, 15]), control_wires=[5, 9]),
                Controlled(
                    SWAP(wires=[13, 12]), control_wires=[5, 9], control_values=[True, False]
                ),
                CZ(wires=[5, 14]),
                Controlled(
                    SWAP(wires=[13, 12]), control_wires=[5, 9], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[13, 15]), control_wires=[5, 9]),
                Controlled(
                    SWAP(wires=[10, 11]), control_wires=[5, 8], control_values=[True, False]
                ),
                Controlled(SWAP(wires=[10, 14]), control_wires=[5, 8]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 4, 6]),
                CH(wires=[5, 4]),
                CSWAP(wires=[5, 13, 9]),
                CSWAP(wires=[5, 10, 8]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 1, 6]),
                CSWAP(wires=[5, 6, 7]),
                CSWAP(wires=[5, 0, 6]),
                X(5),
            ],
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            [5, 6, 7, 10, 13],
            1,
            0,  # addressed from the left
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |010>
            [
                BasisEmbedding([0, 0], wires=[0, 1]),
                MultiControlledX(wires=[0, 5], control_values=[False]),
                CSWAP(wires=[5, 1, 6]),
                CSWAP(wires=[5, 6, 7]),
                CH(wires=[5, 2]),
                CSWAP(wires=[5, 2, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                CZ(wires=[5, 13]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 2, 6]),
                CH(wires=[5, 2]),
                CH(wires=[5, 3]),
                CSWAP(wires=[5, 3, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                CZ(wires=[5, 10]),
                CZ(wires=[5, 13]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 3, 6]),
                CH(wires=[5, 3]),
                CH(wires=[5, 4]),
                CSWAP(wires=[5, 4, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                CZ(wires=[5, 13]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 4, 6]),
                CH(wires=[5, 4]),
                CSWAP(wires=[5, 6, 7]),
                CSWAP(wires=[5, 1, 6]),
                MultiControlledX(wires=[0, 5], control_values=[False]),
                CNOT(wires=[0, 5]),
                CSWAP(wires=[5, 1, 6]),
                CSWAP(wires=[5, 6, 7]),
                CH(wires=[5, 2]),
                CSWAP(wires=[5, 2, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                CZ(wires=[5, 10]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 2, 6]),
                CH(wires=[5, 2]),
                CH(wires=[5, 3]),
                CSWAP(wires=[5, 3, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                CZ(wires=[5, 10]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 3, 6]),
                CH(wires=[5, 3]),
                CH(wires=[5, 4]),
                CSWAP(wires=[5, 4, 6]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 10]), control_wires=[5, 7], control_values=[True, False]),
                Controlled(SWAP(wires=[6, 13]), control_wires=[5, 7]),
                CSWAP(wires=[5, 4, 6]),
                CH(wires=[5, 4]),
                CSWAP(wires=[5, 6, 7]),
                CSWAP(wires=[5, 1, 6]),
                CNOT(wires=[0, 5]),
            ],
        ),
    ],
)
def test_hybrid_quantum(
    data,
    control_wires,
    target_wires,
    work_wires,
    k,
    address,
    probabilities,
    expected_circuit,
):  # pylint: disable=too-many-arguments

    if has_jax and not isinstance(data[0], str) and not isinstance(data, np.ndarray):
        data, control_wires, target_wires, work_wires = (
            jnp.array(data),
            jnp.array(control_wires),
            jnp.array(target_wires),
            jnp.array(work_wires),
        )

    real_probs = hybrid_quantum(
        data,
        control_wires,
        target_wires,
        work_wires,
        k,
        address,
    )
    assert np.allclose(probabilities, real_probs)
    tape = workflow.construct_tape(hybrid_quantum, level="device")(
        data,
        control_wires,
        target_wires,
        work_wires,
        k,
        address,
    )
    assert tape.operations == expected_circuit


@pytest.mark.parametrize(
    (
        "data",
        "control_wires",
        "target_wires",
        "signal",
        "bus",
        "dir_wires",
        "portL_wires",
        "portR_wires",
        "k",
    ),
    [
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            5,
            6,
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            0,
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            5,
            6,
            [7],
            [10],
            [13],
            1,
        ),
    ],
)
def test_hybrid_decomposition_new(
    data,
    control_wires,
    target_wires,
    signal,
    bus,
    dir_wires,
    portL_wires,
    portR_wires,
    k,
):  # pylint: disable=too-many-arguments
    op = HybridQRAM(
        data,
        control_wires=control_wires,
        target_wires=target_wires,
        work_wires=[signal] + [bus] + dir_wires + portL_wires + portR_wires,
        k=k,
    )
    for rule in list_decomps(HybridQRAM):
        _test_decomposition_rule(op, rule)


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        (
            ([], [0, 1], [2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0),
            ValueError,
            "data' cannot be empty.",
        ),
        (
            (
                [[0, 0, 0], [1, 1, 1]],
                [0, 1],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                0,
            ),
            ValueError,
            "data.shape[0] must be 2^(len(control_wires)).",
        ),
        (
            (
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                [0, 1],
                [2, 3],
                [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                1,
            ),
            ValueError,
            "len(target_wires) must equal bitstring length.",
        ),
        (
            (
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                [0, 1],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13],
                0,
            ),
            ValueError,
            "work_wires must have length 11",
        ),
        (
            (
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                [0, 1],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                3,
            ),
            ValueError,
            "k must satisfy 0 <= k < len(control_wires).",
        ),
        (
            (
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                [],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                0,
            ),
            ValueError,
            "len(control_wires) must be > 0",
        ),
    ],
)
def test_hybrid_raises(params, error, match):
    with pytest.raises(error, match=re.escape(match)):
        HybridQRAM(*params)


@qnode(dev)
def select_only_quantum(
    data, control_wires, target_wires, select_wires, select_value, address
):  # pylint: disable=too-many-arguments
    BasisEmbedding(address, wires=control_wires)
    SelectOnlyQRAM(
        data,
        control_wires=control_wires,
        target_wires=target_wires,
        select_wires=select_wires,
        select_value=select_value,
    )
    return probs(wires=target_wires)


@pytest.mark.jax
@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize(
    (
        "data",
        "control_wires",
        "target_wires",
        "select_wires",
        "select_value",
        "address",
        "probabilities",
        "expected_circuit",
    ),
    [
        (
            [
                "010",
                "111",
                "110",
                "000",
                "010",
                "111",
                "110",
                "000",
                "010",
                "111",
                "110",
                "000",
                "010",
                "111",
                "110",
                "000",
            ],
            [0, 1],
            [2, 3, 4],
            [5, 6],
            0,
            3,  # addressed from the left
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |000>
            [
                BasisEmbedding([1, 1], wires=[0, 1]),
                X(5),
                X(6),
                X(0),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(0),
                X(1),
                X(5),
                X(6),
                X(0),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 4], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(0),
                X(5),
                X(6),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(1),
                X(5),
                X(6),
                X(5),
                X(6),
            ],
        ),
        (
            np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
            np.array([0, 1]),
            np.array([2, 3, 4]),
            np.array([5, 6]),
            0,  # Note: if this were set to 1, the test would not pass... due to the select.
            2,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # |110>
            [
                BasisEmbedding([1, 0], wires=[0, 1]),
                X(5),
                X(6),
                X(0),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(0),
                X(1),
                X(5),
                X(6),
                X(0),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 4], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(0),
                X(5),
                X(6),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(1),
                X(5),
                X(6),
                X(5),
                X(6),
            ],
        ),
        (
            np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ]
            ),
            np.array([0, 1]),
            np.array([2, 3, 4]),
            np.array([5, 6]),
            None,
            1,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # |111>
            [
                BasisEmbedding([0, 1], wires=[0, 1]),
                X(5),
                X(6),
                X(0),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(0),
                X(1),
                X(5),
                X(6),
                X(0),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 4], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(0),
                X(5),
                X(6),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(6),
                X(1),
                X(5),
                X(6),
                X(5),
                X(6),
                X(5),
                X(0),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(0),
                X(1),
                X(5),
                X(0),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 4], control_values=[True, True, True, True]),
                X(5),
                X(0),
                X(5),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(5),
                X(1),
                X(5),
                X(5),
                X(6),
                X(0),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(6),
                X(0),
                X(1),
                X(6),
                X(0),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 4], control_values=[True, True, True, True]),
                X(6),
                X(0),
                X(6),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(6),
                X(1),
                X(6),
                X(6),
                X(0),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(0),
                X(1),
                X(0),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 4], control_values=[True, True, True, True]),
                X(0),
                X(1),
                MultiControlledX(wires=[5, 6, 0, 1, 2], control_values=[True, True, True, True]),
                MultiControlledX(wires=[5, 6, 0, 1, 3], control_values=[True, True, True, True]),
                X(1),
            ],
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            [],
            None,
            0,
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |010>
            [
                BasisEmbedding([0, 0], wires=[0, 1]),
                X(0),
                X(1),
                Toffoli(wires=[0, 1, 3]),
                X(0),
                X(1),
                X(0),
                Toffoli(wires=[0, 1, 2]),
                Toffoli(wires=[0, 1, 3]),
                Toffoli(wires=[0, 1, 4]),
                X(0),
                X(1),
                Toffoli(wires=[0, 1, 2]),
                Toffoli(wires=[0, 1, 3]),
                X(1),
            ],
        ),
    ],
)
def test_select_only_quantum(
    data,
    control_wires,
    target_wires,
    select_wires,
    select_value,
    address,
    probabilities,
    expected_circuit,
):  # pylint: disable=too-many-arguments

    if has_jax and not isinstance(data[0], str) and not isinstance(data, np.ndarray):
        data, control_wires, target_wires, select_wires = (
            jnp.array(data),
            jnp.array(control_wires),
            jnp.array(target_wires),
            jnp.array(select_wires),
        )

    real_probs = select_only_quantum(
        data,
        control_wires,
        target_wires,
        select_wires,
        select_value,
        address,
    )
    assert np.allclose(probabilities, real_probs)
    tape = workflow.construct_tape(select_only_quantum, level="device")(
        data,
        control_wires,
        target_wires,
        select_wires,
        select_value,
        address,
    )
    assert tape.operations == expected_circuit


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        (
            (
                [[0, 0, 0], [1, 1, 1]],
                [0, 1],
                [2, 3, 4],
                [5, 6],
                2,
            ),
            ValueError,
            "data.shape[0] must be 2^(len(select_wires)+len(control_wires)).",
        ),
        (
            (
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                ],
                [0, 1],
                [2, 3, 4],
                [],
                1,
            ),
            ValueError,
            "select_value cannot be used when len(select_wires) == 0.",
        ),
        (
            (
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                ],
                [0, 1],
                [2, 3, 4],
                [15],
                4,
            ),
            ValueError,
            "select_value must be an integer in [0, 1].",
        ),
        (
            (
                [],
                [0, 1],
                [2, 3, 4],
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            ),
            ValueError,
            "data' cannot be empty.",
        ),
        (
            (
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                [0, 1],
                [2, 3],
                [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            ),
            ValueError,
            "len(target_wires) must equal bitstring length.",
        ),
    ],
)
def test_select_only_raises(params, error, match):
    with pytest.raises(error, match=re.escape(match)):
        SelectOnlyQRAM(*params)


@pytest.mark.parametrize(
    (
        "data",
        "control_wires",
        "target_wires",
        "select_wires",
        "select_value",
    ),
    [
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            [5, 6],
            0,
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            [5, 6],
            1,
        ),
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [0, 1],
            [2, 3, 4],
            [],
            None,
        ),
    ],
)
def test_select_decomposition_new(
    data, control_wires, target_wires, select_wires, select_value
):  # pylint: disable=too-many-arguments
    op = SelectOnlyQRAM(
        data,
        control_wires,
        target_wires,
        select_wires,
        select_value,
    )

    for rule in list_decomps(SelectOnlyQRAM):
        _test_decomposition_rule(op, rule)


def test_ffqram_standard_validity():
    """Check the operation using the assert_valid function."""
    op = FFQRAM([np.sqrt(0.3), np.sqrt(0.7)], wires=[0, 1, 2, 3], address=["000", "001"])
    assert_valid(op)


def test_ffqram_postselected_probabilities():
    """Post-selected (conditional) distribution matches the encoded amplitudes."""
    amplitudes = [np.sqrt(0.3), np.sqrt(0.7)]
    address = ["000", "001"]

    wires = registers({"address": 3, "register": 1})

    @qnode(dev)
    def circuit():
        FFQRAM(
            amplitudes=amplitudes,
            wires=wires["address"] + wires["register"],
            address=address,
        )
        # Post-select the register qubit in |1>
        measure(wires["register"], postselect=1)
        return probs(wires=wires["address"])

    # addresses "000" -> index 0, "001" -> index 1; everything else should vanish
    expected = np.zeros(2 ** len(wires["address"]))
    expected[0] = 0.3
    expected[1] = 0.7

    assert np.allclose(circuit(), expected)


def test_ffqram_success_probability():
    """The post-selection succeeds with probability 1 / 2^m."""
    amplitudes = [np.sqrt(0.3), np.sqrt(0.7)]
    address = ["000", "001"]

    wires = registers({"address": 3, "register": 1})

    @qnode(dev)
    def circuit():
        FFQRAM(
            amplitudes=amplitudes,
            wires=wires["address"] + wires["register"],
            address=address,
        )
        # No post-selection here: read P(register = 0/1) directly.
        return probs(wires=wires["register"])

    success_prob = circuit()[1]  # P(register = 1)
    assert np.allclose(success_prob, 0.125)  # 1 / 2**3 for 3 address wires


class TestFFQRAMDecomposition:
    """Tests that FFQRAM defines the correct decomposition."""

    def test_decomposition_contents(self):
        """Checks the decomposition for a standard FF-QRAM example."""
        amplitudes = [np.sqrt(0.3), np.sqrt(0.7)]
        op = FFQRAM(amplitudes, wires=[0, 1, 2, 3], address=["000", "001"])
        tape = QuantumScript(op.decomposition())

        expected_names = [
            "Hadamard",
            "Hadamard",
            "Hadamard",
            "PauliX",
            "PauliX",
            "PauliX",
            "C(RY)",
            "PauliX",
            "PauliX",
            "PauliX",
            "PauliX",
            "PauliX",
            "C(RY)",
            "PauliX",
            "PauliX",
        ]
        expected_wires = [
            [0],
            [1],
            [2],
            [0],
            [1],
            [2],
            [0, 1, 2, 3],
            [0],
            [1],
            [2],
            [0],
            [1],
            [0, 1, 2, 3],
            [0],
            [1],
        ]

        assert [gate.name for gate in tape.operations] == expected_names
        assert [gate.wires.tolist() for gate in tape.operations] == expected_wires

        expected_angles = 2 * np.arcsin(amplitudes)
        assert np.allclose(tape.operations[6].parameters, [expected_angles[0]])
        assert np.allclose(tape.operations[12].parameters, [expected_angles[1]])

    @pytest.mark.capture
    def test_decomposition_new(self):
        """Tests the decomposition rule implemented with the new system."""
        op = FFQRAM([np.sqrt(0.3), np.sqrt(0.7)], wires=[0, 1, 2, 3], address=["000", "001"])

        for rule in list_decomps(FFQRAM):
            _test_decomposition_rule(op, rule)

    def test_decomposition_broadcasted(self):
        """Checks the decomposition for broadcasted amplitudes."""
        amplitudes = np.array(
            [
                [np.sqrt(0.3), np.sqrt(0.7)],
                [np.sqrt(0.2), np.sqrt(0.8)],
            ]
        )

        op = FFQRAM(amplitudes, wires=[0, 1, 2, 3], address=["000", "001"])
        assert op.batch_size == 2

        tape = QuantumScript(op.decomposition())
        assert tape.operations[6].name == "C(RY)"
        assert tape.operations[12].name == "C(RY)"
        assert tape.operations[6].batch_size == 2
        assert tape.operations[12].batch_size == 2

        expected_angles = 2 * np.arcsin(amplitudes)
        assert np.allclose(tape.operations[6].parameters[0], expected_angles[:, 0])
        assert np.allclose(tape.operations[12].parameters[0], expected_angles[:, 1])
