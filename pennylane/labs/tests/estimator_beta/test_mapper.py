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
"""
Test the mapping functionality overrides operators as exected.
"""

import pytest

import pennylane as qp
import pennylane.estimator as pl_qre
import pennylane.labs.estimator_beta as qre
from pennylane.templates.subroutines.qrom import QROM
from pennylane.templates.subroutines.qrom_load import QROMLoad

# pylint: disable=protected-access


@pytest.mark.parametrize("mod, work_wires", [(7, [3, 4]), (8, [])])
def test_adder_mapping(mod, work_wires):
    """Test that the Adder operator gets mapped correctly"""
    adder = qp.Adder(3, x_wires=[0, 1, 2], mod=mod, work_wires=work_wires)

    mapped = pl_qre.resource_mapping._map_to_resource_op(adder)
    assert mapped == qre.Adder(num_x_wires=3, mod=mod, wires=[0, 1, 2])
    assert mapped.wires == qp.wires.Wires([0, 1, 2])


def test_qrom_mapping():
    """Test that the QROM operator gets mapped correctly"""
    qrom = QROM(
        bitstrings=[[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]],
        control_wires=[0, 1],
        target_wires=[2, 3, 4],
        work_wires=[5, 6, 7],
    )

    mapped = pl_qre.resource_mapping._map_to_resource_op(qrom)
    assert isinstance(mapped, qre.LabsQROM)
    assert mapped.select_swap_depth == 1


@pytest.mark.parametrize("clean", [True, False])
def test_qrom_load_mapping(clean):
    """Test that the QROMLoad operator gets mapped correctly"""
    qrom = QROMLoad(
        bitstrings=[[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]],
        control_wires=[0, 1],
        target_wires=[2, 3, 4],
        work_wires=[5, 6, 7],
        clean=clean,
    )

    mapped = pl_qre.resource_mapping._map_to_resource_op(qrom)
    assert isinstance(mapped, qre.LabsQROM)
    assert mapped.borrow_qubits is not clean
