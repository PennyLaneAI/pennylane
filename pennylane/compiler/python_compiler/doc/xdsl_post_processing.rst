Simple tutorial for injecting functions into xDSL modules
=========================================================

.. code-block:: python

    from dataclasses import dataclass
    import jax

    import pennylane as qml
    from pennylane.compiler.python_compiler.conversion import inline_module, xdsl_from_qjit, xdsl_module

    from xdsl import context, passes, pattern_rewriter
    from xdsl.dialects import builtin, func
    from xdsl.traits import SymbolTable
    from xdsl.rewriter import InsertPoint

Create workflow and convert to xDSL module
==========================================

.. code-block:: python

    @xdsl_from_qjit
    @qml.qjit(target="mlir")
    def workflow(x, y):
        dev = qml.device("lightning.qubit", wires=5)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        res = circuit(x)
        return res - y


>>> xmod = workflow(3.5, 4.5)
>>> print(xmod)
builtin.module @workflow {
  func.func public @jit_workflow(%arg2 : tensor<f64>, %arg3 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg2) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.convert"(%arg3) : (tensor<f64>) -> tensor<f64>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %2 : tensor<f64>
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
      %6 = tensor.extract %arg0[] : tensor<f64>
      %7 = quantum.custom "RX"(%6) %5 : !quantum.bit
      %8 = quantum.namedobs %7[PauliZ] : !quantum.obs
      %9 = quantum.expval %8 : f64
      %10 = tensor.from_elements %9 : tensor<f64>
      %11 = tensor.extract %0[] : tensor<i64>
      %12 = quantum.insert %3[%11], %7 : !quantum.reg, !quantum.bit
      quantum.dealloc %12 : !quantum.reg
      quantum.device_release
      func.return %10 : tensor<f64>
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


Now, let’s try creating a pass that squares the output of the qnode
===================================================================

To do so, we can use the ``inline_module`` utility to easily add our
post-processing function into the module we’re transforming. First, we
create the function that squares the input value and turn it into an
xDSL module.

.. code-block:: python

    @jax.jit
    def square(x):
        return x * x


>>> square_mod = xdsl_module(square)(1.5)
>>> print(square_mod)
builtin.module @jit_square attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0 : tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
}


.. code-block:: python

    def is_kernel_launch(op):
        return op.name == "catalyst.launch_kernel"


    class SquarePattern(pattern_rewriter.RewritePattern):

        @pattern_rewriter.op_type_rewrite_pattern
        def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
            # We only rewrite the function that calls the qnode, and the caller of the qnode will
            # always have catalyst.launch_kernel present. Additionally, we only rewrite the caller
            # if it hasn't already been rewritten. We can put a UnitAttr() inside the caller's
            # attributes to indicate whether it has been rewritten or not.
            if funcOp.attributes.get("transformed") == builtin.UnitAttr() or not any(
                is_kernel_launch(op) for op in funcOp.body.ops
            ):
                return

            # Update funcOp to inidicate that it has been rewritten
            funcOp.attributes["transformed"] = builtin.UnitAttr()

            # Insert square into the module
            mod = funcOp.parent_op()
            inline_module(square_mod, mod, change_main_to="square")
            square_fn = SymbolTable.lookup_symbol(mod, "square")

            # Call square_fn and use its results instead of the qnode's results for
            # the rest of the function
            for op in funcOp.body.walk():
                if is_kernel_launch(op):
                    callOp = func.CallOp(
                        builtin.SymbolRefAttr(square_fn.sym_name),
                        op.results,
                        square_fn.function_type.outputs.data,
                    )
                    rewriter.insert_op(callOp, InsertPoint.after(op))

                    # We have inserted a CallOp that takes the output of the qnode as input. Let's call
                    # the qnode output %0, and the CallOp output %1. The following replaces all uses of
                    # %0 with %1 EXCEPT for the case where %0 is an input to callOp
                    op.results[0].replace_by_if(callOp.results[0], lambda use: use.operation != callOp)
                    rewriter.notify_op_modified(funcOp)


    @dataclass(frozen=True)
    class SquarePass(passes.ModulePass):
        name = "square"

        def apply(self, ctx: context.Context, module: builtin.ModuleOp):
            pattern_rewriter.PatternRewriteWalker(
                pattern_rewriter.GreedyRewritePatternApplier([SquarePattern()])
            ).rewrite_module(module)

Let’s apply the pass to our workflow
====================================

.. code-block:: python

    ctx = context.Context()

    pipeline = passes.PassPipeline((SquarePass(),))
    pipeline.apply(ctx, xmod)

Great! Let’s see what the transformed module looks like
=======================================================

As you can see below, the ``square_xdsl`` function is the first function
in the module, and it gets called by ``jit_workflow``, and its
inputs/outputs are consistent with the behaviour we wanted.

>>> print(xmod)
builtin.module @workflow {
  func.func public @jit_workflow(%arg2 : tensor<f64>, %arg3 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface, transformed} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg2) : (tensor<f64>) -> tensor<f64>
    %1 = func.call @square(%0) : (tensor<f64>) -> tensor<f64>
    %2 = "stablehlo.convert"(%arg3) : (tensor<f64>) -> tensor<f64>
    %3 = "stablehlo.subtract"(%1, %2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %3 : tensor<f64>
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
      %6 = tensor.extract %arg0[] : tensor<f64>
      %7 = quantum.custom "RX"(%6) %5 : !quantum.bit
      %8 = quantum.namedobs %7[PauliZ] : !quantum.obs
      %9 = quantum.expval %8 : f64
      %10 = tensor.from_elements %9 : tensor<f64>
      %11 = tensor.extract %0[] : tensor<i64>
      %12 = quantum.insert %3[%11], %7 : !quantum.reg, !quantum.bit
      quantum.dealloc %12 : !quantum.reg
      quantum.device_release
      func.return %10 : tensor<f64>
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
  func.func public @square(%arg0 : tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
}
