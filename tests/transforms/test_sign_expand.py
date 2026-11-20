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

import numpy as np
import pytest

import pennylane as qp

# Defines the device used for all tests
dev = qp.device("default.qubit", wires=[0, 1, 2, 3, "Hadamard", "Target"])

# Defines circuits to be used in queueing/output tests
with qp.queuing.AnnotatedQueue() as tape1_queue:
    qp.PauliX(0)
    H1 = qp.Hamiltonian([1.5], [qp.PauliZ(0) @ qp.PauliZ(1)])
    qp.expval(H1)

tape1 = qp.core.qscript.QuantumScript.from_queue(tape1_queue)

with qp.queuing.AnnotatedQueue() as tape2_queue:
    qp.Hadamard(0)
    qp.Hadamard(1)
    qp.PauliZ(1)
    qp.PauliX(2)
    H2 = qp.Hamiltonian(
        [1, 1],
        [
            qp.PauliX(0) @ qp.PauliZ(2),
            qp.PauliZ(1),
        ],
    )
    qp.expval(H2)

tape2 = qp.core.qscript.QuantumScript.from_queue(tape2_queue)

H3 = qp.Hamiltonian([1.5, 0.3], [qp.PauliZ(0) @ qp.PauliZ(1), qp.PauliX(2)])

with qp.queuing.AnnotatedQueue() as tape3_queue:
    qp.PauliX(0)
    qp.expval(H3)

tape3 = qp.core.qscript.QuantumScript.from_queue(tape3_queue)


H4 = qp.Hamiltonian(
    [1, 3, -2, 1, 1],
    [qp.PauliX(0) @ qp.PauliZ(2), qp.PauliZ(2), qp.PauliX(0), qp.PauliZ(2), qp.PauliZ(2)],
)

with qp.queuing.AnnotatedQueue() as tape4_queue:
    qp.Hadamard(0)
    qp.Hadamard(1)
    qp.PauliZ(1)
    qp.PauliX(2)

    qp.expval(H4)

tape4 = qp.core.qscript.QuantumScript.from_queue(tape4_queue)

TAPES = [tape1, tape2, tape3, tape4]
OUTPUTS = [-1.5, -1, -1.5, -7]

with qp.queuing.AnnotatedQueue() as tape1_var_queue:
    qp.PauliX(0)
    H1 = qp.Hamiltonian([1.5], [qp.PauliZ(0) @ qp.PauliZ(1)])
    qp.var(H1)

tape1_var = qp.core.qscript.QuantumScript.from_queue(tape1_var_queue)

with qp.queuing.AnnotatedQueue() as tape2_var_queue:
    qp.Hadamard(0)
    qp.Hadamard(1)
    qp.PauliZ(1)
    qp.PauliX(2)
    H2 = qp.Hamiltonian(
        [1, 1],
        [
            qp.PauliX(0) @ qp.PauliZ(2),
            qp.PauliZ(1),
        ],
    )
    qp.var(H2)

tape2_var = qp.core.qscript.QuantumScript.from_queue(tape2_var_queue)

TAPES_var = [tape1_var, tape2_var]
OUTPUTS_var = [0, 2]


class TestSignExpand:
    """Tests for the sign_expand transform"""

    @pytest.mark.parametrize(("tape", "output"), list(zip(TAPES, OUTPUTS)))
    def test_hamiltonians(self, tape, output):
        """Tests that the sign_expand transform returns the correct value"""

        tapes, fn = qp.transforms.sign_expand(tape)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), list(zip(TAPES, OUTPUTS)))
    def test_hamiltonians_qnode(self, tape, output):
        """Tests that the sign_expand transform returns the correct value as a transform program"""

        @qp.transforms.sign_expand
        @qp.qnode(dev)
        def qnode():
            for op in tape.operations:
                qp.apply(op)

            return qp.apply(tape.measurements[0])

        expval = qnode()
        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), list(zip(TAPES, OUTPUTS)))
    def test_hamiltonians_circuit_impl(self, tape, output):
        """Tests that the sign_expand transform returns the correct value
        if we do not calculate analytical expectation values of groups but rely on their circuit approximations
        """
        tapes, fn = qp.transforms.sign_expand(tape, circuit=True)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval, 1e-2)
        # as these are approximations, these are only correct up to finite precision

    @pytest.mark.parametrize(("tape", "output"), list(zip(TAPES, OUTPUTS)))
    def test_hamiltonians_circuit_impl_qnode(self, tape, output):
        """Tests that the sign_expand transform returns the correct value as a transform program
        if we do not calculate analytical expectation values of groups but rely on their circuit approximations
        """

        @qp.transforms.sign_expand(circuit=True)
        @qp.qnode(dev)
        def qnode():
            for op in tape.operations:
                qp.apply(op)

            return qp.apply(tape.measurements[0])

        expval = qnode()
        assert np.isclose(output, expval, 1e-2)
        # as these are approximations, these are only correct up to finite precision

    @pytest.mark.parametrize("shots", [None, 100])
    @pytest.mark.parametrize("circuit", [True, False])
    def test_shots_attribute(self, shots, circuit):
        """Tests that the shots attribute is copied to the new tapes"""
        with qp.queuing.AnnotatedQueue() as q:
            qp.PauliX(0)
            qp.expval(H1)

        tape = qp.tape.QuantumScript.from_queue(q, shots=shots)
        new_tapes, _ = qp.transforms.sign_expand(tape, circuit=circuit)

        assert all(new_tape.shots == tape.shots for new_tape in new_tapes)

    def test_hamiltonian_error(self):
        """Tests if wrong observables get caught in the test"""

        with qp.queuing.AnnotatedQueue() as tape_queue:
            qp.expval(qp.PauliZ(0))

        tape = qp.core.qscript.QuantumScript.from_queue(tape_queue)

        with pytest.raises(ValueError, match=r"Passed tape must end in"):
            qp.transforms.sign_expand(tape)

    def test_hamiltonian_error_not_jointly_measurable(self):
        """Test if hamiltonians that are not jointly measurable throw an error"""

        with qp.queuing.AnnotatedQueue() as tape_queue:
            H_mult = qp.Hamiltonian([1.5, 2, 0.3], [qp.PauliZ(0), qp.PauliZ(1), qp.PauliX(0)])
            qp.expval(H_mult)

        tape = qp.core.qscript.QuantumScript.from_queue(tape_queue)

        with pytest.raises(ValueError, match=r"Passed hamiltonian"):
            qp.transforms.sign_expand(tape)

    @pytest.mark.parametrize(("tape", "output"), list(zip(TAPES_var, OUTPUTS_var)))
    def test_hamiltonians_vars(self, tape, output):
        """Tests that the sign_expand transform returns the correct value"""

        tapes, fn = qp.transforms.sign_expand(tape)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), list(zip(TAPES_var, OUTPUTS_var)))
    def test_hamiltonians_vars_circuit_impl(self, tape, output):
        """Tests that the sign_expand transform returns the correct value
        if we do not calculate analytical expectation values of groups but rely on their circuit approximations
        """

        tapes, fn = qp.transforms.sign_expand(tape, circuit=True)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval, 1e-1)
