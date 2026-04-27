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

import pennylane.estimator as pl_qre
import pennylane.labs.estimator_beta as qre
from pennylane.templates.subroutines.qrom import QROM


@pytest.mark.parametrize("clean", [True, False])
def test_qrom_mapping(clean):
    """Test that the qrom operator gets mapped correctly"""
    qrom = QROM(
        data=[[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]],
        control_wires=[0, 1],
        target_wires=[2, 3, 4],
        work_wires=[5, 6, 7],
        clean=clean,
    )

    assert isinstance(pl_qre.resource_mapping._map_to_resource_op(qrom), qre.LabsQROM)
