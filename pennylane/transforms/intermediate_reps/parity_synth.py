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
"""Alias for the ``parity_synth`` pass from Catalyst's passes module."""

from functools import partial

from pennylane.transforms.core import transform


@partial(transform, pass_name="parity-synth")
def parity_synth(tape):
    r"""Pass for applying ParitySynth to phase polynomials in a circuit.

    ParitySynth has been proposed by Vandaele et al. in `arXiv:2104.00934
    <https://arxiv.org/abs/2104.00934>`__ as a technique to synthesize
    `phase polynomials
    <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
    into elementary quantum gates, namely ``CNOT`` and ``RZ``. For this, it synthesizes the
    `parity table <https://pennylane.ai/compilation/parity-table>`__ of the phase polynomial,
    and defers the remaining `parity matrix <https://pennylane.ai/compilation/parity-matrix>`__
    synthesis to `RowCol <https://pennylane.ai/compilation/rowcol-algorithm>`__.

    This pass walks over the input circuit and aggregates all ``CNOT`` and ``RZ`` operators
    into a subcircuit that describes a phase polyonomial. Other gates form the boundaries of
    these subcircuits, and whenever one is encountered the phase polynomial of the aggregated
    subcircuit is resynthesized with the ParitySynth algorithm. This implies that while this
    pass works on circuits containing any operations, it is recommended to maximize the
    subcircuits that represent phase polynomials (i.e. consist of ``CNOT`` and ``RZ`` gates) to
    enhance the effectiveness of the pass. This might be possible through decomposition or
    re-ordering of commuting gates.
    Note that higher-level program structures, such as nested functions and control flow, are
    synthesized independently, i.e., boundaries of such structures are always treated as boundaries
    of phase polynomial subcircuits as well. Similarly, dynamic wires create boundaries around the
    operations using them, potentially causing the separation of consecutive phase polynomial
    operations into multiple subcircuits.

    **Example**

    In the following, we apply the pass to a simple quantum circuit that has optimization
    potential in terms of commuting gates that can be interchanged to unlock a cancellation of
    a self-inverse gate (``CNOT``) with itself. Concretely, the circuit is:

    .. code-block:: python

        import pennylane as qml

        qml.capture.enable()
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x: float, y: float, z: float):
            qml.CNOT((0, 1))
            qml.RZ(x, 1)
            qml.CNOT((0, 1))
            qml.RX(y, 1)
            qml.CNOT((1, 0))
            qml.RZ(z, 1)
            qml.CNOT((1, 0))
            return qml.state()

    We can draw the circuit and observe the last ``RZ`` gate being wrapped in a pair of ``CNOT``
    gates that commute with it:

    >>> print(qml.draw(circuit)(0.52, 0.12, 0.2))
    0: ─╭●───────────╭●───────────╭X───────────╭X─┤  State
    1: ─╰X──RZ(0.52)─╰X──RX(0.12)─╰●──RZ(0.20)─╰●─┤  State

    Now we apply the ``parity_synth_pass`` to the circuit and quantum just-in-time (qjit) compile
    the circuit into a reduced MLIR module:

    .. code-block:: python

        circuit_qjit = qml.qjit(parity_synth_pass(circuit), autograph=True, target="mlir")
        compiler = Compiler()
        mlir_module = compiler.run(circuit_qjit.mlir_module)

    TODO: Port the following output an ddiscussion to PL-friendly version
    Looking at the compiled module below, we find only five gates left in the program (note that
    we reduced the output for the purpose of this example); the ``CNOT``\ s
    have been cancelled successfully. Note that for this circuit, ParitySynth is run twice; once
    for the first three gates and once for the last three gates. This is because ``RX`` is not
    a phase polynomial operation, so that it forms a boundary for the phase polynomial subcircuits
    that are re-synthesized by the pass.

    >>> print(mlir_module) # The following output has manually been reduced for readability
    module @circuit {
      func.func public @jit_circuit([...]) -> tensor<4xcomplex<f64>> {
        %0 = "catalyst.launch_kernel"(%arg0, %arg1, %arg2) <[...]> :
            (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<4xcomplex<f64>>
        return %0 : tensor<4xcomplex<f64>>
      }
      module @module_circuit {
        func.func public @circuit(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) ->
            tensor<4xcomplex<f64>> attributes [...] {
          %c = stablehlo.constant dense<0> : tensor<i64>
          %0 = "tensor.extract"(%c) : (tensor<i64>) -> i64
          "quantum.device"(%0) <[...]> : (i64) -> ()
          %c_0 = stablehlo.constant dense<2> : tensor<i64>
          %1 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
          %2 = "tensor.extract"(%c) : (tensor<i64>) -> i64
          %3 = "quantum.extract"(%1, %2) : (!quantum.reg, i64) -> !quantum.bit
          %c_1 = stablehlo.constant dense<1> : tensor<i64>
          %4 = "tensor.extract"(%c_1) : (tensor<i64>) -> i64
          %5 = "quantum.extract"(%1, %4) : (!quantum.reg, i64) -> !quantum.bit
          %6 = "tensor.extract"(%arg0) : (tensor<f64>) -> f64
          %7:2 = "quantum.custom"(%5, %3) <{gate_name = "CNOT", [...]> :[...]
          %8 = "quantum.custom"(%6, %7#1) <{gate_name = "RZ", [...]> :[...]
          %9:2 = "quantum.custom"(%7#0, %8) <{gate_name = "CNOT", [...]> : [...]
          %10 = "tensor.extract"(%arg1) : (tensor<f64>) -> f64
          %11 = "quantum.custom"(%10, %9#0) <{gate_name = "RX", [...]> : [...]
          %12 = "tensor.extract"(%arg2) : (tensor<f64>) -> f64
          %13 = "quantum.custom"(%12, %11) <{gate_name = "RZ", [...]> : [...]
          %14 = "tensor.extract"(%c) : (tensor<i64>) -> i64
          %15 = "quantum.insert"(%1, %14, %9#1) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
          %16 = "tensor.extract"(%c_1) : (tensor<i64>) -> i64
          %17 = "quantum.insert"(%15, %16, %13) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
          %18 = "quantum.compbasis"(%17) <[...]> : (!quantum.reg) -> !quantum.obs
          %19 = "quantum.state"(%18) <[...]> : (!quantum.obs) -> tensor<4xcomplex<f64>>
          "quantum.dealloc"(%17) : (!quantum.reg) -> ()
          "quantum.device_release"() : () -> ()
          return %19 : tensor<4xcomplex<f64>>
        }
      }
      func.func @setup() {
        "quantum.init"() : () -> ()
        return
      }
      func.func @teardown() {
        "quantum.finalize"() : () -> ()
        return
      }
    }

    """
    raise NotImplementedError(
        "The parity_synth compilation pass has no tape implementation, and can only be applied "
        "when decorating the entire worfklow with @qml.qjit and when it is placed after all "
        "transforms that only have a tape implementation."
    )
