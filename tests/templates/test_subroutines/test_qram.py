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

from pennylane import device, qnode, queuing
from pennylane.decomposition import list_decomps
from pennylane.measurements import probs
from pennylane.ops import CH, CNOT, CSWAP, CZ, SWAP, Controlled, MultiControlledX, Toffoli, X
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates import BasisEmbedding
from pennylane.templates.subroutines.qram import BBQRAM, HybridQRAM, SelectOnlyQRAM

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


@pytest.mark.parametrize(
    (
        "data",
        "control_wires",
        "target_wires",
        "bus",
        "dir_wires",
        "portL_wires",
        "portR_wires",
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
            5,
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            2,  # addressed from the left
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # |110>
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
            5,
            [6, 7, 8],
            [12, 13, 14],
            [9, 10, 11],
            0,
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |010>
        ),
    ],
)
def test_bb_quantum(
    data,
    control_wires,
    target_wires,
    bus,
    dir_wires,
    portL_wires,
    portR_wires,
    address,
    probabilities,
):  # pylint: disable=too-many-arguments
    assert np.allclose(
        probabilities,
        bb_quantum(
            data,
            control_wires,
            target_wires,
            [bus] + dir_wires + portL_wires + portR_wires,
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
            ],
            [0, 1],
            [2, 3, 4],
            5,
            6,
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            0,
            2,  # addressed from the left
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # |110>
            [
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
            5,
            6,
            [7],
            [10],
            [13],
            1,
            0,  # addressed from the left
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # |010>
            [
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
    signal,
    bus,
    dir_wires,
    portL_wires,
    portR_wires,
    k,
    address,
    probabilities,
    expected_circuit,
):  # pylint: disable=too-many-arguments
    with queuing.AnnotatedQueue() as q:
        real_probs = hybrid_quantum(
            data,
            control_wires,
            target_wires,
            [signal] + [bus] + dir_wires + portL_wires + portR_wires,
            k,
            address,
        )
    assert np.allclose(probabilities, real_probs)
    assert q.queue == expected_circuit


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
                BasisEmbedding([0, 0], wires=[5, 6]),
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
            0,  # Note: if this were set to 1, the test would not pass... due to the select.
            2,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # |110>
            [
                BasisEmbedding([0, 0], wires=[5, 6]),
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
            None,
            1,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # |111>
            [
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
    with queuing.AnnotatedQueue() as q:
        real_probs = select_only_quantum(
            data,
            control_wires,
            target_wires,
            select_wires,
            select_value,
            address,
        )
    assert np.allclose(probabilities, real_probs)
    assert q.queue == expected_circuit


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
