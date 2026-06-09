# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the specs transform with Catalyst QJIT objects."""

from functools import partial

import pytest

import pennylane as qp

pytest.importorskip("catalyst")

pytestmark = pytest.mark.external


class TestFunctoolsPartial:
    """Tests that qp.specs supports functools.partial-wrapped Catalyst QJIT objects."""

    @staticmethod
    def _example_qjit():
        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def rx_circuit(x, n_iter):
            qp.RX(x, 0)
            qp.RX(x, 0)
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        return rx_circuit

    def test_specs_qjit_partial_device_level(self):
        """Partial-wrapped QJIT objects can be inspected at the device level."""
        qjit = self._example_qjit()
        fixed_qjit = partial(qjit, n_iter=3)
        result = qp.specs(fixed_qjit)(0.5)
        assert result.resources.gate_types["RX"] == 3

    def test_specs_qjit_partial_intermediate_level(self):
        """Partial-wrapped QJIT objects can be inspected at an intermediate level."""
        qjit = self._example_qjit()
        fixed_qjit = partial(qjit, n_iter=3)
        expected = qp.specs(qjit, level=0)(0.5, 3)
        result = qp.specs(fixed_qjit, level=0)(0.5)
        assert result.resources.gate_types == expected.resources.gate_types
