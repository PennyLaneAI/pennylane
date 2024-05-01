# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains unit tests for the ``sign_expand`` transform."""
import functools

import numpy as np
import pytest

import pennylane as qml
import pennylane.tape

# Defines the device used for all tests
dev = qml.device("default.qubit", wires=[0, 1, 2, 3, "Hadamard", "Target"])

# Defines circuits to be used in queueing/output tests
with pennylane.tape.QuantumTape() as tape1:
    qml.PauliX(0)
    H1 = qml.Hamiltonian([1.5], [qml.PauliZ(0) @ qml.PauliZ(1)])
    qml.expval(H1)

with pennylane.tape.QuantumTape() as tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    H2 = qml.Hamiltonian(
        [1, 1],
        [
            qml.PauliX(0) @ qml.PauliZ(2),
            qml.PauliZ(1),
        ],
    )
    qml.expval(H2)

H3 = qml.Hamiltonian([1.5, 0.3], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(2)])

with qml.tape.QuantumTape() as tape3:
    qml.PauliX(0)
    qml.expval(H3)


H4 = qml.Hamiltonian(
    [1, 3, -2, 1, 1],
    [qml.PauliX(0) @ qml.PauliZ(2), qml.PauliZ(2), qml.PauliX(0), qml.PauliZ(2), qml.PauliZ(2)],
)

with qml.tape.QuantumTape() as tape4:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    qml.expval(H4)

TAPES = [tape1, tape2, tape3, tape4]
OUTPUTS = [-1.5, -1, -1.5, -7]

with pennylane.tape.QuantumTape() as tape1_var:
    qml.PauliX(0)
    H1 = qml.Hamiltonian([1.5], [qml.PauliZ(0) @ qml.PauliZ(1)])
    qml.var(H1)

with pennylane.tape.QuantumTape() as tape2_var:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    H2 = qml.Hamiltonian(
        [1, 1],
        [
            qml.PauliX(0) @ qml.PauliZ(2),
            qml.PauliZ(1),
        ],
    )
    qml.var(H2)

TAPES_var = [tape1_var, tape2_var]
OUTPUTS_var = [0, 2]


class TestSignExpand:
    """Tests for the sign_expand transform"""

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians(self, tape, output):
        """Tests that the sign_expand transform returns the correct value"""

        tapes, fn = qml.transforms.sign_expand(tape)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians_qnode(self, tape, output):
        """Tests that the sign_expand transform returns the correct value as a transform program"""

        @qml.transforms.sign_expand
        @qml.qnode(dev)
        def qnode():
            for op in tape.operations:
                qml.apply(op)

            return qml.apply(tape.measurements[0])

        expval = qnode()
        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians_circuit_impl(self, tape, output):
        """Tests that the sign_expand transform returns the correct value
        if we do not calculate analytical expectation values of groups but rely on their circuit approximations
        """
        tapes, fn = qml.transforms.sign_expand(tape, circuit=True)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval, 1e-2)
        # as these are approximations, these are only correct up to finite precision

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians_circuit_impl_qnode(self, tape, output):
        """Tests that the sign_expand transform returns the correct value as a transform program
        if we do not calculate analytical expectation values of groups but rely on their circuit approximations
        """

        @functools.partial(qml.transforms.sign_expand, circuit=True)
        @qml.qnode(dev)
        def qnode():
            for op in tape.operations:
                qml.apply(op)

            return qml.apply(tape.measurements[0])

        expval = qnode()
        assert np.isclose(output, expval, 1e-2)
        # as these are approximations, these are only correct up to finite precision

    @pytest.mark.parametrize("shots", [None, 100])
    @pytest.mark.parametrize("circuit", [True, False])
    def test_shots_attribute(self, shots, circuit):
        """Tests that the shots attribute is copied to the new tapes"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(0)
            qml.expval(H1)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        new_tapes, _ = qml.transforms.sign_expand(tape, circuit=circuit)

        assert all(new_tape.shots == tape.shots for new_tape in new_tapes)

    def test_hamiltonian_error(self):
        """Tests if wrong observables get caught in the test"""

        with pennylane.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=r"Passed tape must end in"):
            qml.transforms.sign_expand(tape)

    def test_hamiltonian_error_not_jointly_measurable(self):
        """Test if hamiltonians that are not jointly measurable throw an error"""

        with pennylane.tape.QuantumTape() as tape:
            H_mult = qml.Hamiltonian([1.5, 2, 0.3], [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0)])
            qml.expval(H_mult)

        with pytest.raises(ValueError, match=r"Passed hamiltonian"):
            qml.transforms.sign_expand(tape)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES_var, OUTPUTS_var))
    def test_hamiltonians_vars(self, tape, output):
        """Tests that the sign_expand transform returns the correct value"""

        tapes, fn = qml.transforms.sign_expand(tape)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES_var, OUTPUTS_var))
    def test_hamiltonians_vars_circuit_impl(self, tape, output):
        """Tests that the sign_expand transform returns the correct value
        if we do not calculate analytical expectation values of groups but rely on their circuit approximations
        """

        tapes, fn = qml.transforms.sign_expand(tape, circuit=True)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval, 1e-1)
