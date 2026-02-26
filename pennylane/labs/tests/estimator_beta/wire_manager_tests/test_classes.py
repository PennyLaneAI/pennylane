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
r"""Tests for the base classes used when tracking qubits for resource estimation."""

import pytest

import pennylane.estimator as qre
import pennylane.labs.estimator_beta as qre_exp
from pennylane.allocation import AllocateState
from pennylane.estimator import (
    GateCount,
)
from pennylane.labs.estimator_beta.wires_manager import (
    Allocate,
    Deallocate,
    MarkClean,
    _estimate_auxiliary_wires,
    _process_circuit_lst,
    estimate_wires_from_circuit,
    estimate_wires_from_resources,
)


class TestAllocate:
    pass


class TestDeallocate:
    pass


class TestMarkClean:
    pass
