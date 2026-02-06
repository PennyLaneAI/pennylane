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

import pennylane as qp
from pennylane.transforms.optimization.relative_phases import (  # pylint: disable=no-name-in-module
    match_controlled_iX_gate,
    match_relative_phase_toffoli,
)


class TestMultiControlledPhaseXGate:

    def test_no_controls(self):
        def qfunc():
            qp.S(wires=[0])
            qp.PauliX(wires=[0])
            return qp.expval(qp.Z(0))

        with pytest.raises(ValueError, match="There must be at least one control wire"):
            transformed_qfunc = match_controlled_iX_gate(qfunc, 0)
            qp.tape.make_qscript(transformed_qfunc)()

    def test_multiple_matches(self):
        def qfunc():
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            # first instance
            qp.Hadamard(wires=3),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.Hadamard(wires=3),
            # second instance
            qp.Hadamard(wires=3),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.Hadamard(wires=3),
        ]

    def test_surrounded(self):
        def qfunc():
            qp.PauliZ(wires=0)
            qp.PauliY(wires=2)
            qp.PauliX(wires=3)
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.PauliZ(wires=5)
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            qp.PauliZ(wires=0)
            qp.Hadamard(wires=2)
            qp.PauliX(wires=3)
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 17
        assert tape.operations == [
            qp.PauliY(wires=2),
            qp.Hadamard(wires=3),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.Hadamard(wires=3),
            qp.PauliZ(wires=0),
            qp.PauliX(wires=3),
            qp.PauliZ(wires=5),
            qp.PauliZ(wires=0),
            qp.Hadamard(wires=2),
            qp.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qp.H(3),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[2, 3]),
            qp.T(3),
            qp.Toffoli(wires=[0, 1, 3]),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[2, 3]),
            qp.T(3),
            qp.Toffoli(wires=[0, 1, 3]),
            qp.H(3),
            qp.adjoint(qp.ctrl(qp.S(2), control=[0, 1])),
        ]

    def test_wire_permutations(self):
        for first, second, third, fourth in permutations([0, 1, 2, 3]):

            def qfunc(one, two, three, four):
                qp.ctrl(qp.S(wires=[three]), control=[one, two])
                qp.MultiControlledX(wires=[one, two, three, four])
                return qp.expval(qp.Z(one))

            func = partial(qfunc, first, second, third, fourth)

            transformed_qfunc = match_controlled_iX_gate(func, 2)

            tape = qp.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 10
            assert tape.operations == [
                qp.Hadamard(wires=fourth),
                qp.adjoint(qp.T(wires=fourth)),
                qp.CNOT(wires=[third] + [fourth]),
                qp.T(wires=fourth),
                qp.Toffoli(wires=[first, second] + [fourth]),
                qp.adjoint(qp.T(wires=fourth)),
                qp.CNOT(wires=[third] + [fourth]),
                qp.T(wires=fourth),
                qp.Toffoli(wires=[first, second] + [fourth]),
                qp.Hadamard(wires=fourth),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.PauliX(4)  # change it to five and it breaks! A bug?
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qp.Hadamard(wires=3),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.Hadamard(wires=3),
            qp.PauliX(wires=4),
        ]

    def test_basic_transform(self):
        def qfunc():
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 2)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 10
        assert tape.operations == [
            qp.Hadamard(wires=3),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.adjoint(qp.T(wires=3)),
            qp.CNOT(wires=[2] + [3]),
            qp.T(wires=3),
            qp.Toffoli(wires=[0, 1] + [3]),
            qp.Hadamard(wires=3),
        ]


class TestPhaseXGate:

    def test_multiple_matches(self):
        def qfunc():
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.Toffoli(wires=[0, 1, 2])
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.Toffoli(wires=[0, 1, 2])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            # first instance
            qp.Hadamard(wires=2),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.Hadamard(wires=2),
            # second instance
            qp.Hadamard(wires=2),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.Hadamard(wires=2),
        ]

    def test_surrounded(self):
        def qfunc():
            qp.PauliZ(wires=0)
            qp.PauliY(wires=2)
            qp.PauliX(wires=3)
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.PauliZ(wires=5)
            qp.Toffoli(wires=[0, 1, 2])
            qp.PauliZ(wires=0)
            qp.Hadamard(wires=2)  # cancels with H in the replacement
            qp.PauliX(wires=3)
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 15
        assert tape.operations == [
            qp.PauliY(wires=2),
            qp.Hadamard(wires=2),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.PauliZ(wires=0),
            qp.PauliX(wires=3),
            qp.PauliZ(wires=5),
            qp.PauliZ(wires=0),
            qp.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qp.Toffoli(wires=[0, 1, 2])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qp.H(2),
            qp.adjoint(qp.T(2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(2),
            qp.CNOT(wires=[0, 2]),
            qp.adjoint(qp.T(2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(2),
            qp.CNOT(wires=[0, 2]),
            qp.H(2),
            qp.adjoint(qp.ctrl(qp.S(1), control=[0])),
        ]

    def test_wire_permutations(self):
        for first, second, third in permutations([0, 1, 2]):

            def qfunc(one, two, three):
                qp.ctrl(qp.S(wires=[two]), control=[one])
                qp.Toffoli(wires=[one, two, three])
                return qp.expval(qp.Z(one))

            func = partial(qfunc, one=first, two=second, three=third)

            transformed_qfunc = match_controlled_iX_gate(func, 1)

            tape = qp.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 10
            assert tape.operations == [
                qp.Hadamard(wires=third),
                qp.adjoint(qp.T(wires=third)),
                qp.CNOT(wires=[second, third]),
                qp.T(wires=third),
                qp.CNOT(wires=[first, third]),
                qp.adjoint(qp.T(wires=third)),
                qp.CNOT(wires=[second, third]),
                qp.T(wires=third),
                qp.CNOT(wires=[first, third]),
                qp.Hadamard(wires=third),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.PauliX(3)
            qp.Toffoli(wires=[0, 1, 2])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qp.Hadamard(wires=2),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.Hadamard(wires=2),
            qp.PauliX(wires=3),
        ]

    def test_basic_transform(self):
        def qfunc():
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.Toffoli(wires=[0, 1, 2])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_controlled_iX_gate(qfunc, 1)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 10
        assert tape.operations == [
            qp.Hadamard(wires=2),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.adjoint(qp.T(wires=2)),
            qp.CNOT(wires=[1, 2]),
            qp.T(wires=2),
            qp.CNOT(wires=[0, 2]),
            qp.Hadamard(wires=2),
        ]


class TestRelativePhaseToffoli:

    def test_repeated(self):
        def qfunc():
            qp.CCZ(wires=[0, 1, 3])
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            qp.CCZ(wires=[0, 1, 3])
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 18 * 2
        assert tape.operations == [
            # first instance
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            # second instance
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
        ]

    def test_surrounded(self):
        def qfunc():
            qp.PauliZ(wires=0)
            qp.PauliY(wires=2)
            qp.PauliX(wires=3)
            qp.CCZ(wires=[0, 1, 3])
            qp.PauliX(wires=4)
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.PauliY(wires=5)
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            qp.PauliZ(wires=0)
            qp.Hadamard(wires=2)
            qp.PauliX(wires=3)
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 26
        assert tape.operations == [
            qp.PauliY(wires=2),
            qp.PauliX(wires=3),
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.PauliZ(wires=0),
            qp.PauliX(wires=4),
            qp.PauliY(wires=5),
            qp.PauliZ(wires=0),
            qp.Hadamard(wires=2),
            qp.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qp.CCZ(wires=[0, 1, 3])
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert tape.operations == [
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.adjoint(qp.ctrl(qp.S(1), control=[0])),
        ]

    def test_wire_permutations(self):
        for first, second, third, fourth in permutations([0, 1, 2, 3]):

            def qfunc(one, two, three, four):
                qp.CCZ(wires=[one, two, four])
                qp.ctrl(qp.S(wires=[two]), control=[one])
                qp.ctrl(qp.S(wires=[three]), control=[one, two])
                qp.MultiControlledX(wires=[one, two, three, four])
                return qp.expval(qp.Z(one))

            func = partial(qfunc, first, second, third, fourth)

            transformed_qfunc = match_relative_phase_toffoli(func)

            tape = qp.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 18
            assert tape.operations == [
                qp.H(fourth),
                qp.T(fourth),
                qp.CNOT(wires=[third, fourth]),
                qp.adjoint(qp.T(fourth)),
                qp.H(fourth),
                qp.CNOT(wires=[first, fourth]),
                qp.T(fourth),
                qp.CNOT(wires=[second, fourth]),
                qp.adjoint(qp.T(fourth)),
                qp.CNOT(wires=[first, fourth]),
                qp.T(fourth),
                qp.CNOT(wires=[second, fourth]),
                qp.adjoint(qp.T(fourth)),
                qp.H(fourth),
                qp.T(fourth),
                qp.CNOT(wires=[third, fourth]),
                qp.adjoint(qp.T(fourth)),
                qp.H(fourth),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qp.CCZ(wires=[0, 1, 3])
            qp.PauliX(wires=4)
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.PauliY(wires=5)
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.PauliX(wires=4),
            qp.PauliY(wires=5),
        ]

    def test_basic_transform(self):
        def qfunc():
            qp.CCZ(wires=[0, 1, 3])
            qp.ctrl(qp.S(wires=[1]), control=[0])
            qp.ctrl(qp.S(wires=[2]), control=[0, 1])
            qp.MultiControlledX(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

        transformed_qfunc = match_relative_phase_toffoli(qfunc)

        tape = qp.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 18
        assert tape.operations == [
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.CNOT(wires=[0, 3]),
            qp.T(3),
            qp.CNOT(wires=[1, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
            qp.T(3),
            qp.CNOT(wires=[2, 3]),
            qp.adjoint(qp.T(3)),
            qp.H(3),
        ]
