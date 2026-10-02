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
Unit tests for ``qp.flatten``. Most of the functionality is tested in Catalyst.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pennylane as qp
from pennylane.exceptions import CompileError
from pennylane.tape import QuantumScript

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst


def test_flatten_returns_compiled_tape():
    """qp.flatten dispatches to the compiler and returns a tape with the compile pipeline
    applied, the measurements and the shots."""

    @qp.qjit(capture=True, collect_decomp_rules=False)
    @qp.transforms.merge_rotations
    @qp.transforms.cancel_inverses
    @qp.qnode(qp.device("lightning.qubit", wires=2), shots=5)
    def f(x):
        qp.H(0)
        qp.H(0)

        @qp.for_loop(0, 2)
        def loop(i):
            qp.RX(x, 1)

        loop()
        return qp.sample(wires=[1])

    tape = qp.flatten(f)(0.25)
    assert isinstance(tape, QuantumScript)
    assert len(tape.operations) == 1
    assert qp.equal(tape.operations[0], qp.RX(0.5, 1))
    assert tape.measurements == [qp.sample(wires=[1])]
    assert tape.shots == qp.measurements.Shots(5)


def test_compiler_without_flatten():
    """An informative error is raised if the compiler does not support flatten."""
    entry_points = {"catalyst": {"ops": SimpleNamespace(load=lambda: object())}}
    with patch.object(qp.compiler.compiler.AvailableCompilers, "names_entrypoints", entry_points):
        with pytest.raises(CompileError, match="does not support flatten"):
            qp.flatten(lambda: None)
