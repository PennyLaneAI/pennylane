import numpy as np
import pytest

from pennylane import device, qnode
from pennylane.measurements import probs
from pennylane.templates import BasisEmbedding
from pennylane.templates.subroutines.qram import BBQRAM

dev = device("default.qubit")


@qnode(dev)
def bb_quantum(bitstrings, qram_wires, target_wires, bus, dir_wires, portL_wires, portR_wires, address):
    BasisEmbedding(address, wires=qram_wires)

    BBQRAM(
        bitstrings,
        qram_wires=qram_wires,
        target_wires=target_wires,
        work_wires=[bus] + dir_wires + portL_wires + portR_wires,
    )
    return probs(wires=target_wires)


@pytest.mark.parametrize(
    (
        "bitstrings",
        "qram_wires",
        "target_wires",
        "bus",
        "dir_wires",
        "portL_wires",
        "portR_wires",
        "address",
        "probs",
    ),
    [
        (
            ["010", "111", "110", "000"],
            [0, 1],
            [2, 3, 4],
            5,
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            2,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
(
            ["010", "111", "110", "000"],
            [0, 1],
            [2, 3, 4],
            5,
            [11, 10, 9],
            [6, 7, 8],
            [12, 13, 14],
            1,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        (
            ["010", "111", "110", "000"],
            [0, 1],
            [2, 3, 4],
            5,
            [6, 7, 8],
            [12, 13, 14],
            [9, 10, 11],
            0,
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ],
)
def test_bb_quantum(
    bitstrings, qram_wires, target_wires, bus, dir_wires, portL_wires, portR_wires, address, probs
):
    assert np.allclose(
        probs,
        bb_quantum(bitstrings, qram_wires, target_wires, bus, dir_wires, portL_wires, portR_wires, address),
    )
