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
import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import ParitySynthPass, parity_synth_pass


def translate_program_to_xdsl(program):

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
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
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
                // CHECK-NOT: "quantum.gphase"
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
                // CHECK-NOT: "quantum.gphase"
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
                // CHECK-NOT: "quantum.gphase"
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
                // CHECK-NOT: "quantum.gphase"
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
                // CHECK-NOT: "quantum.gphase"
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
                // CHECK-NOT: "quantum.gphase"
                return
            }
        """
        run_filecheck(translate_program_to_xdsl(program), self.pipeline)


class TestAdditionalOps:

    pipeline = (ParitySynthPass(),)

    def test_phase_shift(self, run_filecheck):
        """Test that ``PhaseShift`` is handled correctly."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                // CHECK: [[half:%.+]] = arith.constant -5.000000e-01 : f64
                // CHECK: [[gphase:%.+]] = arith.mulf %arg0, [[half]] : f64
                // CHECK: [[q2:%.+]] = quantum.custom "RZ"(%arg0) [[q0]] : !quantum.bit
                %2 = quantum.custom "PhaseShift"(%arg0) %0 : !quantum.bit
                // CHECK: [[q3:%.+]], [[q4:%.+]] = quantum.custom "CNOT"() [[q2]], [[q1]] : !quantum.bit, !quantum.bit
                %3, %4 = _CNOT %2, %1
                // CHECK: quantum.gphase([[gphase]]) :
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }
        """

        _program = translate_program_to_xdsl(program)
        run_filecheck(_program, self.pipeline)

    @pytest.mark.parametrize("name", ["IsingZZ", "MultiRZ"])
    def test_isingzz_multirz_two_qubits(self, name, run_filecheck):
        """Test that ``MultiRZ`` and ``IsingZZ`` are handled correctly."""
        program = f"""
            func.func @test_func(%arg0: f64, %arg1: f64) {{
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2 = INIT_QUBIT
                // CHECK: [[q3:%.+]], [[q4:%.+]] = quantum.custom "CNOT"() [[q0]], [[q1]] : !quantum.bit
                // CHECK: [[q5:%.+]] = quantum.custom "RZ"(%arg0) [[q4]] : !quantum.bit
                %3, %4 = quantum.custom "{name}"(%arg0) %0, %1 : !quantum.bit, !quantum.bit
                %5, %6 = _CNOT %2, %4
                %7, %8 = quantum.custom "{name}"(%arg1) %3, %5 : !quantum.bit, !quantum.bit
                // CHECK: [[q6:%.+]], [[q7:%.+]] = quantum.custom "CNOT"() [[q2]], [[q3]] : !quantum.bit
                // CHECK: [[q8:%.+]] = quantum.custom "RZ"(%arg1) [[q7]] : !quantum.bit
                // CHECK: [[q9:%.+]], [[q10:%.+]] = quantum.custom "CNOT"() [[q6]], [[q5]] : !quantum.bit, !quantum.bit
                // CHECK: [[q11:%.+]], [[q12:%.+]] = quantum.custom "CNOT"() [[q9]], [[q8]] : !quantum.bit, !quantum.bit
                // CHECK: [[q13:%.+]], [[q14:%.+]] = quantum.custom "CNOT"() [[q12]], [[q10]] : !quantum.bit, !quantum.bit
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }}
        """

        _program = translate_program_to_xdsl(program)
        run_filecheck(_program, self.pipeline)

    def test_multirz_with_cnot_ladder(self, run_filecheck):
        """Test that ``MultiRZ`` is handled correctly on more qubits."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {{
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2 = INIT_QUBIT
                %3 = INIT_QUBIT
                // CHECK: [[q4:%.+]] = quantum.custom "RZ"(%arg1) [[q1]] : !quantum.bit
                // CHECK: [[q5:%.+]], [[q6:%.+]] = quantum.custom "CNOT"() [[q4]], [[q3]] : !quantum.bit
                // CHECK: [[q7:%.+]], [[q8:%.+]] = quantum.custom "CNOT"() [[q6]], [[q2]] : !quantum.bit
                // CHECK: [[q9:%.+]], [[q10:%.+]] = quantum.custom "CNOT"() [[q8]], [[q0]] : !quantum.bit
                // CHECK: [[q11:%.+]] = quantum.custom "RZ"(%arg0) [[q10]] : !quantum.bit
                // CHECK: [[q12:%.+]], [[q13:%.+]] = quantum.custom "CNOT"() [[q7]], [[q9]] : !quantum.bit
                // CHECK: [[q14:%.+]], [[q15:%.+]] = quantum.custom "CNOT"() [[q5]], [[q12]] : !quantum.bit
                // CHECK: [[q16:%.+]], [[q17:%.+]] = quantum.custom "CNOT"() [[q15]], [[q11]] : !quantum.bit
                // CHECK: [[q18:%.+]], [[q19:%.+]] = quantum.custom "CNOT"() [[q13]], [[q17]] : !quantum.bit
                // CHECK: [[q20:%.+]], [[q21:%.+]] = quantum.custom "CNOT"() [[q14]], [[q19]] : !quantum.bit
                %4, %5, %6, %7 = quantum.custom "MultiRZ"(%arg0) %0, %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
                %10 = quantum.custom "RZ"(%arg1) %5 : !quantum.bit
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }}
        """
        _program = translate_program_to_xdsl(program)
        run_filecheck(_program, self.pipeline)

    def test_z_gate(self, run_filecheck):
        """Test that ``Z`` is handled correctly."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2, %3 = _CNOT %0, %1
                // CHECK: [[pi:%.+]] = arith.constant 3.1415926535897931 : f64
                // CHECK: [[half:%.+]] = arith.constant -5.000000e-01 : f64
                // CHECK: [[gphase:%.+]] = arith.mulf [[pi]], [[half]] : f64
                // CHECK: [[q2:%.+]] = quantum.custom "RZ"([[pi]]) [[q0]] : !quantum.bit
                %4 = quantum.custom "PauliZ"() %2 : !quantum.bit
                %5, %6 = _CNOT %4, %3
                // CHECK: quantum.gphase([[gphase]]) :
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }
        """

        _program = translate_program_to_xdsl(program)
        run_filecheck(_program, self.pipeline)

    def test_s_gate(self, run_filecheck):
        """Test that ``S`` is handled correctly."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2, %3 = _CNOT %0, %1
                // CHECK: [[pi_2:%.+]] = arith.constant 1.5707963267948966 : f64
                // CHECK: [[half:%.+]] = arith.constant -5.000000e-01 : f64
                // CHECK: [[gphase:%.+]] = arith.mulf [[pi_2]], [[half]] : f64
                // CHECK: [[q2:%.+]] = quantum.custom "RZ"([[pi_2]]) [[q0]] : !quantum.bit
                %4 = quantum.custom "S"() %2 : !quantum.bit
                %5, %6 = _CNOT %4, %3
                // CHECK: quantum.gphase([[gphase]]) :
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }
        """

        _program = translate_program_to_xdsl(program)
        run_filecheck(_program, self.pipeline)

    def test_t_gate(self, run_filecheck):
        """Test that ``T`` is handled correctly."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                %2, %3 = _CNOT %0, %1
                // CHECK: [[pi_4:%.+]] = arith.constant 0.78539816339744828 : f64
                // CHECK: [[half:%.+]] = arith.constant -5.000000e-01 : f64
                // CHECK: [[gphase:%.+]] = arith.mulf [[pi_4]], [[half]] : f64
                // CHECK: [[q2:%.+]] = quantum.custom "RZ"([[pi_4]]) [[q0]] : !quantum.bit
                %4 = quantum.custom "T"() %2 : !quantum.bit
                %5, %6 = _CNOT %4, %3
                // CHECK: quantum.gphase([[gphase]]) :
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }
        """

        _program = translate_program_to_xdsl(program)
        run_filecheck(_program, self.pipeline)

    def test_global_phase(self, run_filecheck):
        """Test that ``GlobalPhase`` together with a phase poly op is handled correctly."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                // CHECK: quantum.custom "CNOT"() [[q1]], [[q0]] : !quantum.bit, !quantum.bit
                // CHECK: quantum.gphase(%arg0) :
                quantum.gphase(%arg0) :
                %2, %3 = _CNOT %1, %0
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }
        """

        _program = translate_program_to_xdsl(program)
        run_filecheck(_program, self.pipeline)

    def test_global_phase_alone_raises(self, run_filecheck):
        """Test that ``GlobalPhase`` together with a phase poly op is handled correctly."""
        program = """
            func.func @test_func(%arg0: f64) {
                %0 = INIT_QUBIT
                %1 = INIT_QUBIT
                // CHECK: quantum.gphase(%arg0) :
                quantum.gphase(%arg0) :
                // CHECK-NOT: "quantum.custom"
                // CHECK-NOT: "quantum.gphase"
                return
            }
        """
        _program = translate_program_to_xdsl(program)
        with pytest.raises(NotImplementedError, match="Can't optimize a circuit that only"):
            run_filecheck(_program, self.pipeline)


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

    def test_qjit_with_additional_ops(self, run_filecheck_qjit):
        """Test that the ParitySynthPass works correctly with qjit when
        additional operation types causing global phases are present."""
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
            # CHECK: quantum.custom "RZ"
            # CHECK: quantum.custom "RZ"
            # CHECK: quantum.gphase
            # CHECK-NOT: quantum.custom
            qml.CNOT((0, 1))
            qml.PhaseShift(x, 1)
            qml.CNOT((0, 1))
            qml.RX(y, 1)
            qml.CNOT((1, 0))
            qml.RZ(z, 1)
            qml.CNOT((1, 0))
            qml.Z(0)
            qml.S(1)
            return qml.state()

        run_filecheck_qjit(circuit)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
