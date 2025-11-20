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
Unit tests for the decompositions transforms ``match_relative_phase_toffoli`` and ``match_controlled_iX_gate``.
"""
from functools import partial
from itertools import permutations

import pytest

import pennylane as qml
from pennylane.transforms.optimization.relative_phases import (  # pylint: disable=no-name-in-module
    match_controlled_iX_gate,
    match_relative_phase_toffoli,
)


class TestMultiControlledPhaseXGate:

    def test_no_controls(self):
        def qfunc():
            qml.S(wires=[0])
            qml.PauliX(wires=[0])
            return qml.expval(qml.Z(0))

        with pytest.raises(ValueError, match="There must be at least one control wire"):
            transformed_qfunc = match_controlled_iX_gate(qfunc, 0)
            qml.tape.make_qscript(transformed_qfunc)()

    def test_multiple_matches(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            # first instance
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
            # second instance
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
        ]

    def test_surrounded(self):
        def qfunc():
            qml.PauliZ(wires=0)
            qml.PauliY(wires=2)
            qml.PauliX(wires=3)
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.PauliZ(wires=5)
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.PauliZ(wires=0)
            qml.Hadamard(wires=2)
            qml.PauliX(wires=3)
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 17
        assert tape.operations == [
            qml.PauliY(wires=2),
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
            qml.PauliZ(wires=0),
            qml.PauliX(wires=3),
            qml.PauliZ(wires=5),
            qml.PauliZ(wires=0),
            qml.Hadamard(wires=2),
            qml.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qml.H(3),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[2, 3]),
            qml.T(3),
            qml.Toffoli(wires=[0, 1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[2, 3]),
            qml.T(3),
            qml.Toffoli(wires=[0, 1, 3]),
            qml.H(3),
            qml.adjoint(qml.ctrl(qml.S(2), control=[0, 1])),
        ]

    def test_wire_permutations(self):
        for first, second, third, fourth in permutations([0, 1, 2, 3]):

            def qfunc(one, two, three, four):
                qml.ctrl(qml.S(wires=[three]), control=[one, two])
                qml.MultiControlledX(wires=[one, two, three, four])
                return qml.expval(qml.Z(one))

            func = partial(qfunc, first, second, third, fourth)

            transformed_qfunc = match_controlled_iX_gate(func, 2)

            tape = qml.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 10
            assert tape.operations == [
                qml.Hadamard(wires=fourth),
                qml.adjoint(qml.T(wires=fourth)),
                qml.CNOT(wires=[third] + [fourth]),
                qml.T(wires=fourth),
                qml.Toffoli(wires=[first, second] + [fourth]),
                qml.adjoint(qml.T(wires=fourth)),
                qml.CNOT(wires=[third] + [fourth]),
                qml.T(wires=fourth),
                qml.Toffoli(wires=[first, second] + [fourth]),
                qml.Hadamard(wires=fourth),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.PauliX(4)  # change it to five and it breaks! A bug?
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
            qml.PauliX(wires=4),
        ]

    def test_basic_transform(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 10
        assert tape.operations == [
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.CNOT(wires=[2] + [3]),
            qml.T(wires=3),
            qml.Toffoli(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
        ]


class TestPhaseXGate:

    def test_multiple_matches(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.Toffoli(wires=[0, 1, 2])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            # first instance
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
            # second instance
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
        ]

    def test_surrounded(self):
        def qfunc():
            qml.PauliZ(wires=0)
            qml.PauliY(wires=2)
            qml.PauliX(wires=3)
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliZ(wires=5)
            qml.Toffoli(wires=[0, 1, 2])
            qml.PauliZ(wires=0)
            qml.Hadamard(wires=2)  # cancels with H in the replacement
            qml.PauliX(wires=3)
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 15
        assert tape.operations == [
            qml.PauliY(wires=2),
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.PauliZ(wires=0),
            qml.PauliX(wires=3),
            qml.PauliZ(wires=5),
            qml.PauliZ(wires=0),
            qml.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qml.H(2),
            qml.adjoint(qml.T(2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(2),
            qml.CNOT(wires=[0, 2]),
            qml.H(2),
            qml.adjoint(qml.ctrl(qml.S(1), control=[0])),
        ]

    def test_wire_permutations(self):
        for first, second, third in permutations([0, 1, 2]):

            def qfunc(one, two, three):
                qml.ctrl(qml.S(wires=[two]), control=[one])
                qml.Toffoli(wires=[one, two, three])
                return qml.expval(qml.Z(one))

            func = partial(qfunc, one=first, two=second, three=third)

            transformed_qfunc = match_controlled_iX_gate(func, 1)

            tape = qml.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 10
            assert tape.operations == [
                qml.Hadamard(wires=third),
                qml.adjoint(qml.T(wires=third)),
                qml.CNOT(wires=[second, third]),
                qml.T(wires=third),
                qml.CNOT(wires=[first, third]),
                qml.adjoint(qml.T(wires=third)),
                qml.CNOT(wires=[second, third]),
                qml.T(wires=third),
                qml.CNOT(wires=[first, third]),
                qml.Hadamard(wires=third),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliX(3)
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
            qml.PauliX(wires=3),
        ]

    def test_basic_transform(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 10
        assert tape.operations == [
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
        ]


class TestRelativePhaseToffoli:

    def test_repeated(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 18 * 2
        assert tape.operations == [
            # first instance
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            # second instance
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
        ]

    def test_surrounded(self):
        def qfunc():
            qml.PauliZ(wires=0)
            qml.PauliY(wires=2)
            qml.PauliX(wires=3)
            qml.CCZ(wires=[0, 1, 3])
            qml.PauliX(wires=4)
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliY(wires=5)
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.PauliZ(wires=0)
            qml.Hadamard(wires=2)
            qml.PauliX(wires=3)
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 26
        assert tape.operations == [
            qml.PauliY(wires=2),
            qml.PauliX(wires=3),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.PauliZ(wires=0),
            qml.PauliX(wires=4),
            qml.PauliY(wires=5),
            qml.PauliZ(wires=0),
            qml.Hadamard(wires=2),
            qml.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert tape.operations == [
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.adjoint(qml.ctrl(qml.S(1), control=[0])),
        ]

    def test_wire_permutations(self):
        for first, second, third, fourth in permutations([0, 1, 2, 3]):

            def qfunc(one, two, three, four):
                qml.CCZ(wires=[one, two, four])
                qml.ctrl(qml.S(wires=[two]), control=[one])
                qml.ctrl(qml.S(wires=[three]), control=[one, two])
                qml.MultiControlledX(wires=[one, two, three, four])
                return qml.expval(qml.Z(one))

            func = partial(qfunc, first, second, third, fourth)

            transformed_qfunc = match_relative_phase_toffoli(func)

            tape = qml.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 18
            assert tape.operations == [
                qml.H(fourth),
                qml.T(fourth),
                qml.CNOT(wires=[third, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.H(fourth),
                qml.CNOT(wires=[first, fourth]),
                qml.T(fourth),
                qml.CNOT(wires=[second, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.CNOT(wires=[first, fourth]),
                qml.T(fourth),
                qml.CNOT(wires=[second, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.H(fourth),
                qml.T(fourth),
                qml.CNOT(wires=[third, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.H(fourth),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.PauliX(wires=4)
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliY(wires=5)
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.PauliX(wires=4),
            qml.PauliY(wires=5),
        ]

    def test_basic_transform(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 18
        assert tape.operations == [
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
        ]
