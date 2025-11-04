.. code-block:: python

    from dataclasses import dataclass

    import pennylane as qml
    from pennylane.compiler.python_compiler.conversion import xdsl_from_qjit
    from pennylane.compiler.python_compiler.dialects.quantum import CustomOp, QubitType

    from xdsl import context, passes, pattern_rewriter
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, func
    from xdsl.ir import Block, Region

Convert into xDSL module
========================

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=5)

    @xdsl_from_qjit
    @qml.qjit(target="mlir")
    @qml.qnode(dev)
    def circuit(x):
        qml.H(0)
        return qml.expval(qml.Z(0))


>>> qjit_mod = circuit(1.5)
>>> print(qjit_mod)
builtin.module @circuit {
  func.func public @jit_circuit(%arg2 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg2) : (tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
  builtin.module @module_circuit {
    builtin.module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1 : !transform.op<"builtin.module">) {
        transform.yield
      }
    }
    func.func public @circuit(%arg0 : tensor<f64>) -> (tensor<f64>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %1 = tensor.extract %0[] : tensor<i64>
      quantum.device shots(%1) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %2 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %3 = quantum.alloc(5) : !quantum.reg
      %4 = tensor.extract %0[] : tensor<i64>
      %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
      %6 = quantum.custom "Hadamard"() %5 : !quantum.bit
      %7 = quantum.namedobs %6[PauliZ] : !quantum.obs
      %8 = quantum.expval %7 : f64
      %9 = tensor.from_elements %8 : tensor<f64>
      %10 = tensor.extract %0[] : tensor<i64>
      %11 = quantum.insert %3[%10], %6 : !quantum.reg, !quantum.bit
      quantum.dealloc %11 : !quantum.reg
      quantum.device_release
      func.return %9 : tensor<f64>
    }
  }
  func.func @setup() {
    quantum.init
    func.return
  }
  func.func @teardown() {
    quantum.finalize
    func.return
  }
}


Let’s create a quantum subroutine
=================================

This subroutine’s purpose is to replace Hadamard gates with a blackbox
(which will be empty for now), so it must take a single qubit as its
argument. Additionally, we are assuming that the qubits on which the
subroutine will act are not just the ones that the subroutine takes as
input, so we must also provide the quantum register as input, and also
give it as the output.

``FuncOp``\ s have a single region with a single block that contains the
body of the function.

We need to build the function’s body by populating its inner ``Block``
with operations. We can do so using the ``xdsl.builder.ImplicitBuilder``
class. This class can be used as a context manager that takes a
``Block`` as input, and any operations created within the context of the
builder get added to the block. Let’s try it out.

Here, we create a subroutine that we will use to replace
``Hadamard``\ s. This subroutine applies a gate provided by the user,
and returns the ``out_qubit`` of the gate.

.. code-block:: python

    def create_hadamard_replacement_subroutine(gate_name):
        input_types = (QubitType(),)
        output_types = (QubitType(),)
        block = Block(arg_types=input_types)

        with ImplicitBuilder(block):
            in_qubits = [block.args[0]]
            op1 = CustomOp(in_qubits=in_qubits, gate_name=gate_name)
            func.ReturnOp(op1.out_qubits[0])

        region = Region([block])
        funcOp = func.FuncOp("replace_hadamard", (input_types, output_types), region=region)
        return funcOp


>>> funcOp = create_hadamard_replacement_subroutine("S")
>>> print(funcOp)
func.func @replace_hadamard(%0 : !quantum.bit) -> !quantum.bit {
  %1 = quantum.custom "S"() %0 : !quantum.bit
  func.return %1 : !quantum.bit
}


Now, we write a pass to do the substitution
===========================================

.. code-block:: python

    class ReplaceHadamardPattern(pattern_rewriter.RewritePattern):

        def __init__(self, subroutine: func.FuncOp):
            self.subroutine = subroutine

        @pattern_rewriter.op_type_rewrite_pattern
        def match_and_rewrite(self, customOp: CustomOp, rewriter: pattern_rewriter.PatternRewriter):
            if customOp.gate_name.data != "Hadamard":
                return

            callOp = func.CallOp(
                builtin.SymbolRefAttr("replace_hadamard"),
                [customOp.in_qubits[0]],
                self.subroutine.function_type.outputs.data,
            )
            rewriter.insert_op_after_matched_op(callOp)
            rewriter.replace_all_uses_with(customOp.out_qubits[0], callOp.results[0])
            rewriter.erase_op(customOp)


    @dataclass(frozen=True)
    class ReplaceHadamardPass(passes.ModulePass):
        name = "replace-hadamard"
        gate_name: str

        def apply(self, ctx: context.Context, module: builtin.ModuleOp):
            funcOp = create_hadamard_replacement_subroutine(self.gate_name)
            module.regions[0].blocks.first.add_op(funcOp)

            pattern_rewriter.PatternRewriteWalker(
                pattern_rewriter.GreedyRewritePatternApplier([ReplaceHadamardPattern(funcOp)])
            ).rewrite_module(module)

Let’s see it in action
======================

Here, we will replace all ``Hadamard``\ s with ``S``\ s

.. code-block:: python

    ctx = context.Context()

    pipeline = passes.PassPipeline((ReplaceHadamardPass("S"),))
    pipeline.apply(ctx, qjit_mod)

Great! We can see below that ``Hadamard`` was replaced by a call to
``replace_hadamard``, which applies a single ``S`` gate.

>>> print(qjit_mod)
builtin.module @circuit {
  func.func public @jit_circuit(%arg2 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg2) : (tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
  builtin.module @module_circuit {
    builtin.module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1 : !transform.op<"builtin.module">) {
        transform.yield
      }
    }
    func.func public @circuit(%arg0 : tensor<f64>) -> (tensor<f64>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %1 = tensor.extract %0[] : tensor<i64>
      quantum.device shots(%1) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %2 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %3 = quantum.alloc(5) : !quantum.reg
      %4 = tensor.extract %0[] : tensor<i64>
      %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
      %6 = func.call @replace_hadamard(%5) : (!quantum.bit) -> !quantum.bit
      %7 = quantum.namedobs %6[PauliZ] : !quantum.obs
      %8 = quantum.expval %7 : f64
      %9 = tensor.from_elements %8 : tensor<f64>
      %10 = tensor.extract %0[] : tensor<i64>
      %11 = quantum.insert %3[%10], %6 : !quantum.reg, !quantum.bit
      quantum.dealloc %11 : !quantum.reg
      quantum.device_release
      func.return %9 : tensor<f64>
    }
  }
  func.func @setup() {
    quantum.init
    func.return
  }
  func.func @teardown() {
    quantum.finalize
    func.return
  }
  func.func @replace_hadamard(%0 : !quantum.bit) -> !quantum.bit {
    %1 = quantum.custom "S"() %0 : !quantum.bit
    func.return %1 : !quantum.bit
  }
}
