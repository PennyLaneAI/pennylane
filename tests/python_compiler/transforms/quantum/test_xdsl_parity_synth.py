# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test module for the ParitySynth transform"""
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import ParitySynthPass, parity_synth_pass
from pennylane.compiler.python_compiler.transforms.quantum.parity_synth import (
    _parity_network_synth,
)
from pennylane.transforms.intermediate_reps import phase_polynomial


def assert_binary_matrix(matrix: np.ndarray):
    """Check that the input matrix is two-dimensional, ``np.int64``-dtyped and
    only contains zeros and ones.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected the matrix to be two-dimensional, but got {matrix.ndim} dimensions."
        )
    if matrix.dtype != np.int64:
        raise ValueError(
            f"Expected the data type of the matrix to be np.int64, but got {matrix.dtype}."
        )
    if not set(matrix.flat).issubset({0, 1}):
        raise ValueError(
            f"Expected the entries of the matrix to be from {{0, 1}} but got {set(matrix.flat)}."
        )


class TestParityNetworkSynth:
    """Tests for the synthesizing of a parity network with ``_parity_network_synth``."""

    @staticmethod
    def validate_circuit_entry(entry, exp_len=None):
        """Validate that an object is a three-tuple consisting of an two integers and a list
        of two-tuples with integers in them, like ``(1, 4, [(0, 2), (1, 0)])``. This constitutes
        the format for circuit entries in the output of ``_parity_network_synth``."""
        assert isinstance(entry, tuple) and len(entry) == 3
        parity_idx, qubit_idx, cnot_circuit = entry
        assert isinstance(parity_idx, np.int64)
        assert isinstance(qubit_idx, int)
        assert isinstance(cnot_circuit, list)
        if exp_len is not None:
            assert len(cnot_circuit) == exp_len
        assert all(isinstance(_cnot, tuple) and len(_cnot) == 2 for _cnot in cnot_circuit)

    def test_empty_parity_table(self):
        """Test that an empty parity table results in an empty circuit."""
        P = np.ones(shape=(10, 0), dtype=int)
        circuit, inv_synth_matrix = _parity_network_synth(P)
        assert not circuit
        assert inv_synth_matrix is None

    @pytest.mark.parametrize("n, idx", [(1, 0), (2, 0), (2, 1), (3, 2), (10, 5)])
    def test_single_unit_vector_parity(self, n, idx):
        """Test that a single unit vector-parity is synthesized into no CNOTs and a single RZ."""
        I = np.eye(n, dtype=int)
        P = I[:, idx : idx + 1]
        circuit, inv_synth_matrix = _parity_network_synth(P)
        assert isinstance(circuit, list) and len(circuit) == 1
        self.validate_circuit_entry(circuit[0], exp_len=0)
        assert_binary_matrix(inv_synth_matrix)
        assert_equal(I, inv_synth_matrix)

    @pytest.mark.parametrize(
        "n, ids",
        [
            (1, [0]),
            (2, [1, 0]),
            (2, [0, 1]),
            (3, [2, 0]),
            (3, [0, 1]),
            (5, [0, 4]),
            (10, [5, 4, 9, 0, 2, 3]),
        ],
    )
    def test_multiple_unit_vector_parities(self, n, ids):
        """Test that multiple unit vector-parities are synthesized into no CNOTs and a
        series of RZ gates."""
        I = np.eye(n, dtype=int)
        P = np.concatenate([I[:, idx : idx + 1] for idx in ids], axis=1)
        circuit, inv_synth_matrix = _parity_network_synth(P)
        assert isinstance(circuit, list) and len(circuit) == len(ids)
        for entry in circuit:
            self.validate_circuit_entry(entry, exp_len=0)
        assert_binary_matrix(inv_synth_matrix)
        assert_equal(I, inv_synth_matrix)

    @pytest.mark.parametrize(
        "parity",
        [
            [1, 0, 0, 1, 0, 1],
            [1, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ],
    )
    def test_single_non_unit_parity(self, parity):
        """Test that a single non-unit vector-parity ``p`` is synthesized into
        ``|p|-1`` CNOTs and a single RZ."""
        P = np.array([parity]).T
        circuit, inv_synth_matrix = _parity_network_synth(P)
        assert isinstance(circuit, list) and len(circuit) == 1
        self.validate_circuit_entry(circuit[0], exp_len=np.sum(P) - 1)
        I = np.eye(len(parity), dtype=int)
        assert I.shape == inv_synth_matrix.shape
        assert set(inv_synth_matrix.flat).issubset({0, 1})
        assert not np.allclose(I, inv_synth_matrix)

    @pytest.mark.parametrize(
        "parities, exp_lens",
        [
            ([[1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0]], (1, 0, 2)),
            ([[1, 1, 1], [0, 1, 1], [1, 1, 1]], (1, 1, 0)),
            ([[0, 1, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 1]], (4, 0)),
            ([[1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [0, 1]], (0, 0, 0, 0, 1, 0)),
            (
                [[1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1]],
                (1, 0, 0),
            ),
        ],
    )
    def test_with_repeated_parities(self, parities, exp_lens):
        """Test that repeated (non-unit vector-)parities are synthesized in sequence and
        require no CNOT between their RZ gates."""
        P = np.array(parities).T
        circuit, inv_synth_matrix = _parity_network_synth(P)
        assert isinstance(circuit, list) and len(circuit) == len(parities)
        for entry, exp_len in zip(circuit, exp_lens, strict=True):
            self.validate_circuit_entry(entry, exp_len=exp_len)
        I = np.eye(len(parities[0]), dtype=int)
        assert I.shape == inv_synth_matrix.shape
        assert set(inv_synth_matrix.flat).issubset({0, 1})
        assert not np.allclose(I, inv_synth_matrix)

    @pytest.mark.parametrize("n, seed", [(2, 851), (3, 231), (4, 8241), (5, 214)])
    @pytest.mark.parametrize("num_parities", (1, 2, 3, 10, 20))
    def test_roundtrip(self, num_parities, n, seed):
        """Test that the parity table of a randomly sampled CNOT+RZ circuit is synthesized
        into a new CNOT+RZ circuit with the same parities, and that the inverse of the
        parity matrix of the new circuit is reported correctly."""
        # pylint: disable=unbalanced-tuple-unpacking

        np.random.seed(seed)  # todo: proper seeding
        # Make all cnot ops
        all_cnots = [qml.CNOT((i, j)) for i, j in product(range(n), repeat=2) if i != j]
        # Sample random CNOTs (by index into above list) and rotation angles
        cnots = [
            np.random.choice(len(all_cnots), size=n, replace=True) for _ in range(num_parities)
        ]
        thetas = np.random.random(num_parities)
        # Make PL circuit
        circuit = sum(
            [
                [all_cnots[i] for i in sub_circuit] + [qml.RZ(x, j % n)]
                for j, (sub_circuit, x) in enumerate(zip(cnots, thetas, strict=True))
            ],
            start=[],
        )
        # Compute IR
        _, P, angles = phase_polynomial(qml.tape.QuantumScript(circuit), wire_order=range(n))

        angles_ = list(angles)
        # Synthesize parity network and compute new PL circuit from it
        new_circuit, inv_parity_matrix = _parity_network_synth(P)
        new_circuit = sum(
            [
                [qml.CNOT(_cnot) for _cnot in sub_circuit]
                + [qml.RZ(angles_.pop(angle_idx), qubit_idx)]
                for angle_idx, qubit_idx, sub_circuit in new_circuit
            ],
            start=[],
        )
        # Compute IR of new PL circuit
        new_parity_matrix, new_P, new_angles = phase_polynomial(
            qml.tape.QuantumScript(new_circuit), wire_order=range(n)
        )
        # Compare phase parities and make sure that the inv_parity_matrix is valid
        assert_allclose(new_P @ new_angles, P @ angles)
        assert_binary_matrix(inv_parity_matrix)
        assert_equal((new_parity_matrix @ inv_parity_matrix) % 2, np.eye(n, dtype=int))


def translate_program_to_xdsl(program):
    """Translate an almost-xDSL-program into an xDSL program by replacing some shorthand notations."""
    new_lines = []
    for line in program.split("\n"):
        if "INIT_QUBIT" in line:
            i = int(line.strip().split(" ")[0][1:])
            new_lines.extend(
                [
                    f'// CHECK: [[q{i}:%.+]] = "test.op"() : () -> !quantum.bit',
                    f'%{i} = "test.op"() : () -> !quantum.bit',
                ]
            )
        elif "_CNOT" in line:
            bits = line.strip().split(" ")
            new_bits = (
                bits[:3] + ['quantum.custom "CNOT"()'] + bits[4:] + [": !quantum.bit, !quantum.bit"]
            )
            new_lines.append(" ".join(new_bits))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


class TestParitySynthPass:
    """Unit tests for ParitySynthPass."""

    pipeline = (ParitySynthPass(),)

    def test_no_phase_polynomial_ops(self, run_filecheck):
        """Test that nothing changes when there are no phase polynomial gates."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                // CHECK: [[q1:%.+]] = quantum.custom "Hadamard"() [[q0]] : !quantum.bit
                // CHECK: quantum.custom "RX"(%arg0) [[q1]] : !quantum.bit
                %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
                %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
                return
            }
        """

        run_filecheck(translate_program_to_xdsl(program), self.pipeline)

    def test_composable_cnots(self, run_filecheck):
        """Test that two out of three CNOT gates are merged."""
        program = """
            func.func @test_func() {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                // CHECK: quantum.custom "CNOT"() [[q0]], [[q1]] : !quantum.bit, !quantum.bit
                %2, %3 = _CNOT %0, %1
                %4, %5 = _CNOT %2, %3
                %6, %7 = _CNOT %4, %5
                // CHECK-NOT: "quantum.custom"
                return
            }
        """

        run_filecheck(translate_program_to_xdsl(program), self.pipeline)

    def test_two_cnots_single_rotation_no_merge(self, run_filecheck):
        """Test that a phase polynomial of two CNOTs separated by a rotation on the target
        is maintained."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                // In the following check, q1 and q0 are exchanged. This is a symmetry
                // of the test case and ParitySynth chooses to flip the CNOTs.
                // CHECK: [[q2:%.+]], [[q3:%.+]] = quantum.custom "CNOT"() [[q1]], [[q0]] : !quantum.bit, !quantum.bit
                %2, %3 = _CNOT %0, %1
                // CHECK: [[q4:%.+]] = quantum.custom "RZ"(%arg0) [[q3]] : !quantum.bit
                %4 = quantum.custom "RZ"(%arg0) %3 : !quantum.bit
                // CHECK: quantum.custom "CNOT"() [[q2]], [[q4]] : !quantum.bit, !quantum.bit
                %5, %6 = _CNOT %2, %4
                // CHECK-NOT: "quantum.custom"
                return
            }
        """

        run_filecheck(translate_program_to_xdsl(program), self.pipeline)

    def test_two_cnots_single_rotation_with_merge(self, run_filecheck):
        """Test that a phase polynomial of two CNOTs separated by a rotation on the control
        is reduced."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2, %3 = _CNOT %0, %1
                // CHECK: quantum.custom "RZ"(%arg0) [[q0]] : !quantum.bit
                %4 = quantum.custom "RZ"(%arg0) %2 : !quantum.bit
                %5, %6 = _CNOT %4, %3
                // CHECK-NOT: "quantum.custom"
                return
            }
        """

        run_filecheck(translate_program_to_xdsl(program), self.pipeline)

    def test_two_phase_polynomials_first_merge(self, run_filecheck):
        """Test that two phase polynomials separated by a non-phase-polynomial operation is
        compiled correctly if the former polynomial can be reduced."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64, %arg2: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2 = INIT_QUBIT

                %3, %4 = _CNOT %0, %2
                // CHECK: [[q3:%.+]] = quantum.custom "RZ"(%arg2) [[q0]] : !quantum.bit
                %5 = quantum.custom "RZ"(%arg2) %3 : !quantum.bit
                %6, %7 = _CNOT %5, %4

                // CHECK: [[q4:%.+]] = quantum.custom "RX"(%arg1) [[q3]] : !quantum.bit
                %8 = quantum.custom "RX"(%arg1) %6 : !quantum.bit

                // CHECK: [[q5:%.+]], [[q6:%.+]] = quantum.custom "CNOT"() [[q1]], [[q4]] : !quantum.bit, !quantum.bit
                %9, %10 = _CNOT %8, %1
                // CHECK: [[q7:%.+]] = quantum.custom "RZ"(%arg0) [[q6]] : !quantum.bit
                %11 = quantum.custom "RZ"(%arg0) %10 : !quantum.bit
                // CHECK: quantum.custom "CNOT"() [[q5]], [[q7]] : !quantum.bit, !quantum.bit
                %12, %13 = _CNOT %9, %11
                // CHECK-NOT: "quantum.custom"
                return
            }
        """
        run_filecheck(translate_program_to_xdsl(program), self.pipeline)

    def test_two_phase_polynomials_second_merge(self, run_filecheck):
        """Test that two phase polynomials separated by a non-phase-polynomial operation is
        compiled correctly if the latter polynomial can be reduced."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64, %arg2: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2 = INIT_QUBIT

                // CHECK: [[q3:%.+]], [[q4:%.+]] = quantum.custom "CNOT"() [[q1]], [[q0]] : !quantum.bit, !quantum.bit
                %3, %4 = _CNOT %0, %1
                // CHECK: [[q5:%.+]] = quantum.custom "RZ"(%arg0) [[q4]] : !quantum.bit
                %5 = quantum.custom "RZ"(%arg0) %4 : !quantum.bit
                // CHECK: [[q6:%.+]], [[q7:%.+]] = quantum.custom "CNOT"() [[q3]], [[q5]] : !quantum.bit, !quantum.bit
                %6, %7 = _CNOT %3, %5

                // CHECK: [[q8:%.+]] = quantum.custom "RX"(%arg1) [[q6]] : !quantum.bit
                %8 = quantum.custom "RX"(%arg1) %7 : !quantum.bit

                %9, %10 = _CNOT %8, %2
                // CHECK: quantum.custom "RZ"(%arg2) [[q8]] : !quantum.bit
                %11 = quantum.custom "RZ"(%arg2) %9 : !quantum.bit
                %12, %13 = _CNOT %11, %10
                // CHECK-NOT: "quantum.custom"
                return
            }
        """
        run_filecheck(translate_program_to_xdsl(program), self.pipeline)

    def test_large_phase_polynomial(self, run_filecheck):
        """Test that a larger phase polynomial block is handled without an error."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2 = INIT_QUBIT
                %3 = INIT_QUBIT

                %4, %5 = _CNOT %0, %2
                %6, %7 = _CNOT %1, %4
                %8 = quantum.custom "RZ"(%arg0) %7 : !quantum.bit
                %9, %10 = _CNOT %8, %6
                %11 = quantum.custom "RZ"(%arg1) %9 : !quantum.bit
                %12, %13 = _CNOT %3, %10
                %14, %15 = _CNOT %12, %11
                %16 = quantum.custom "RZ"(%arg2) %14 : !quantum.bit
                %17, %18 = _CNOT %5, %15
                %19, %20 = _CNOT %13, %18
                %21 = quantum.custom "RZ"(%arg3) %20 : !quantum.bit
                %22, %23 = _CNOT %16, %17
                %24, %25 = _CNOT %22, %19
                %26, %27 = _CNOT %21, %23
                %28, %29 = _CNOT %25, %27
                %30, %31 = _CNOT %28, %26
                %32, %33 = _CNOT %29, %31
                %34, %35 = _CNOT %32, %30
                %36, %37 = _CNOT %34, %33
                %38 = quantum.custom "RZ"(%arg4) %35 : !quantum.bit

                // CHECK: [[q4:%.+]] = quantum.custom "RZ"(%arg2) [[q3]] : !quantum.bit
                // CHECK: [[q5:%.+]], [[q6:%.+]] = quantum.custom "CNOT"() [[q1]], [[q0]] : !quantum.bit, !quantum.bit
                // CHECK: [[q7:%.+]] = quantum.custom "RZ"(%arg0) [[q6]] : !quantum.bit
                // CHECK: [[q8:%.+]] = quantum.custom "RZ"(%arg1) [[q7]] : !quantum.bit
                // CHECK: [[q9:%.+]], [[q10:%.+]] = quantum.custom "CNOT"() [[q2]], [[q8]] : !quantum.bit, !quantum.bit
                // CHECK: [[q11:%.+]] = quantum.custom "RZ"(%arg3) [[q10]] : !quantum.bit
                // CHECK: [[q12:%.+]], [[q13:%.+]] = quantum.custom "CNOT"() [[q4]], [[q5]] : !quantum.bit, !quantum.bit
                // CHECK: [[q14:%.+]] = quantum.custom "RZ"(%arg4) [[q13]] : !quantum.bit

                // CHECK: [[q15:%.+]], [[q16:%.+]] = quantum.custom "CNOT"() [[q12]], [[q9]] : !quantum.bit, !quantum.bit
                // CHECK: [[q17:%.+]], [[q18:%.+]] = quantum.custom "CNOT"() [[q14]], [[q16]] : !quantum.bit, !quantum.bit
                // CHECK: [[q19:%.+]], [[q20:%.+]] = quantum.custom "CNOT"() [[q17]], [[q11]] : !quantum.bit, !quantum.bit
                // CHECK: [[q21:%.+]], [[q22:%.+]] = quantum.custom "CNOT"() [[q20]], [[q18]] : !quantum.bit, !quantum.bit
                // CHECK: [[q23:%.+]], [[q24:%.+]] = quantum.custom "CNOT"() [[q22]], [[q21]] : !quantum.bit, !quantum.bit
                // CHECK-NOT: "quantum.custom"
                return
            }
        """
        run_filecheck(translate_program_to_xdsl(program), self.pipeline)

    def test_phase_polynomial_with_adjoint(self, run_filecheck):
        """Test that adjoint is handled correctly."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2 = INIT_QUBIT

                // CHECK: [[new_angle:%.+]] = arith.negf %arg0
                // CHECK: [[q3:%.+]], [[q4:%.+]] = quantum.custom "CNOT"() [[q1]], [[q0]] : !quantum.bit, !quantum.bit
                %3, %4 = _CNOT %0, %1
                // CHECK: [[q5:%.+]] = quantum.custom "RZ"([[new_angle]]) [[q4]] : !quantum.bit
                %5 = quantum.custom "RZ"(%arg0) %4 adj : !quantum.bit
                // CHECK: [[q6:%.+]], [[q7:%.+]] = quantum.custom "CNOT"() [[q3]], [[q5]] : !quantum.bit, !quantum.bit
                %6, %7 = _CNOT %3, %5
                // CHECK-NOT: "quantum.custom"
                return
            }
        """
        run_filecheck(translate_program_to_xdsl(program), self.pipeline)


# pylint: disable=too-few-public-methods
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestParitySynthIntegration:
    """Integration tests for the ParitySynthPass."""

    def test_qjit(self, run_filecheck_qjit):
        """Test that the ParitySynthPass works correctly with qjit."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir", pass_plugins=[getXDSLPluginAbsolutePath()])
        @parity_synth_pass
        @qml.qnode(dev)
        def circuit(x: float, y: float, z: float):
            # CHECK: [[phi:%.+]] = tensor.extract %arg0
            # CHECK: quantum.custom "CNOT"()
            # CHECK: quantum.custom "RZ"([[phi]])
            # CHECK: quantum.custom "CNOT"()
            # CHECK: [[omega:%.+]] = tensor.extract %arg1
            # CHECK: quantum.custom "RX"([[omega]])
            # CHECK: [[theta:%.+]] = tensor.extract %arg2
            # CHECK: quantum.custom "RZ"([[theta]])
            # CHECK-NOT: quantum.custom
            qml.CNOT((0, 1))
            qml.RZ(x, 1)
            qml.CNOT((0, 1))
            qml.RX(y, 1)
            qml.CNOT((1, 0))
            qml.RZ(z, 1)
            qml.CNOT((1, 0))
            return qml.state()

        run_filecheck_qjit(circuit)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
