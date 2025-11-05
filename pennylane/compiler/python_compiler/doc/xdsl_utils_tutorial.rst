Python compiler utilities
=========================

All utilities we care about are in the
``pennylane.compiler.python_compiler.conversion`` submodule.

.. code-block:: python

    import pennylane as qml

    from pennylane.compiler.python_compiler.conversion import (
        inline_jit_to_module,
        inline_module,
        xdsl_from_qjit,
        xdsl_module,
    )

``xdsl_module``
===============

This function takes a ``jax.jit``-ed function as input, and returns a
wrapper. This wrapper can be called to return an xDSL module. Note that
this function is intended to be used to covert purely classical
functions into xDSL modules. Let’s take a look at a very simple example:

.. code-block:: python

    import jax


    @jax.jit
    def inner(x):
        return x**2


    @jax.jit
    def outer(x, y):
        return inner(x) - y


>>> wrapped_outer = xdsl_module(outer)
>>> jit_mod = wrapped_outer(1.5, 2.5)
>>> print(jit_mod)
builtin.module @jit_outer attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg1 : tensor<f64>, %arg2 : tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = func.call @inner(%arg1) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.subtract"(%0, %arg2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %1 : tensor<f64>
  }
  func.func private @inner(%arg0 : tensor<f64>) -> (tensor<f64>) {
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
}


Nice! Key points to note: \* The module has the same name as the
decorated function (``outer``) with the ``jit_`` prefix. \* The entry
point function is aptly named ``main``. \* Any jitted functions called
within the entry point have their own function inside the module, and a
corresponding ``func.call`` operation where it gets called. \* If
``inner`` was not decorated with ``jax.jit``, its body would have been
inlined into ``outer``:

.. code-block:: python

    def inner2(x):
        return x**2


    @xdsl_module
    @jax.jit
    def outer2(x, y):
        return inner2(x) - y


>>> mod2 = outer2(1.5, 2.5)
>>> print(mod2)
builtin.module @jit_outer2 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0 : tensor<f64>, %arg1 : tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.subtract"(%0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %1 : tensor<f64>
  }
}


``xdsl_from_qjit``
==================

Since ``xdsl_module`` is for purely classical code, there is another
function that can lower hybrid quantum-classical code written in Python
to an xDSL module. ``xdsl_from_qjit`` takes a ``QJIT``-ed function as
input, and converts it into an xDSL module with the same structure as a
Catalyst program. Let’s check it out:

.. code-block:: python

    @xdsl_from_qjit
    @qml.qjit
    def workflow(x, y):
        dev = qml.device("lightning.qubit", wires=5)

        @qml.qnode(dev)
        def qnode(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        return qnode(x) ** 2 - y


>>> qjit_mod = workflow(2.5, 3.5)
>>> print(qjit_mod)
builtin.module @workflow {
  func.func public @jit_workflow(%arg2 : tensor<f64>, %arg3 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_qnode::@qnode(%arg2) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.multiply"(%0, %0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %2 = "stablehlo.convert"(%arg3) : (tensor<f64>) -> tensor<f64>
    %3 = "stablehlo.subtract"(%1, %2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %3 : tensor<f64>
  }
  builtin.module @module_qnode {
    builtin.module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1 : !transform.op<"builtin.module">) {
        transform.yield
      }
    }
    func.func public @qnode(%arg0 : tensor<f64>) -> (tensor<f64>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
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


Nice! The usefulness of having this utility is that all dialects that we
would commonly find being used at the MLIR layer are already loaded, so
users don’t need to worry about loading dialects manually.

``inline_jit_to_module``
========================

This utility takes ``xdsl_module`` a step beyond. It takes a
``jax.jit``-ed function ``func`` and an xDSL module ``mod`` as input. It
lowers ``func`` to an xDSL module, and appends the lowered module’s body
to the end of ``mod``\ ’s body. An additional step that this function
takes is that it renames the ``main`` function in the lowered module to
the same name as ``func`` so that it’s easier to find.

Let’s try inlining the previous ``outer`` function into ``qjit_mod``.
``inline_jit_to_module`` will modify ``qjit_mod`` in-place:

>>> inline_jit_to_module(outer, qjit_mod)(1.5, 3.5)
>>> print(qjit_mod)
builtin.module @workflow {
  func.func public @jit_workflow(%arg2 : tensor<f64>, %arg3 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_qnode::@qnode(%arg2) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.multiply"(%0, %0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %2 = "stablehlo.convert"(%arg3) : (tensor<f64>) -> tensor<f64>
    %3 = "stablehlo.subtract"(%1, %2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %3 : tensor<f64>
  }
  builtin.module @module_qnode {
    builtin.module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1 : !transform.op<"builtin.module">) {
        transform.yield
      }
    }
    func.func public @qnode(%arg0 : tensor<f64>) -> (tensor<f64>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
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
  func.func public @outer(%arg1 : tensor<f64>, %arg2 : tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = func.call @inner(%arg1) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.subtract"(%0, %arg2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %1 : tensor<f64>
  }
  func.func private @inner(%arg0 : tensor<f64>) -> (tensor<f64>) {
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
}


Nice! We can see that two new functions have been added to the bottom of
``qjit_mod``, corresponding to ``outer`` and ``inner``. This allows us
to make function calls to both functions, which can be useful for
embedding post-processing logic into a module, say, when applying a
compiler pass.

``inline_module``
=================

This utility takes two modules as input, and inlines the body of the
first module into the body of the second module. Additionally,
recognizing that modules created from ``jax.jit``-ed functions may
contain a ``FuncOp`` called ``main``, the function also takes a string
as an optional input to rename the ``main`` function to something else.

Let’s try revisiting the above modules and inlining them using
``inline_module`` instead of ``inline_jit_to_module``:

.. code-block:: python

    @xdsl_from_qjit
    @qml.qjit
    def workflow2(x, y):
        dev = qml.device("lightning.qubit", wires=5)

        @qml.qnode(dev)
        def qnode(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        return qnode(x) ** 2 - y


>>> qjit_mod2 = workflow2(2.5, 3.5)
>>> print(qjit_mod2)
builtin.module @workflow2 {
  func.func public @jit_workflow2(%arg2 : tensor<f64>, %arg3 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_qnode::@qnode(%arg2) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.multiply"(%0, %0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %2 = "stablehlo.convert"(%arg3) : (tensor<f64>) -> tensor<f64>
    %3 = "stablehlo.subtract"(%1, %2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %3 : tensor<f64>
  }
  builtin.module @module_qnode {
    builtin.module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1 : !transform.op<"builtin.module">) {
        transform.yield
      }
    }
    func.func public @qnode(%arg0 : tensor<f64>) -> (tensor<f64>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
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


.. code-block:: python

    @jax.jit
    def inner3(x):
        return x**2


    @jax.jit
    def outer3(x, y):
        return inner3(x) - y


>>> wrapped_outer3 = xdsl_module(outer3)
>>> jit_mod3 = wrapped_outer3(1.5, 2.5)
>>> print(jit_mod3)
builtin.module @jit_outer3 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg1 : tensor<f64>, %arg2 : tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = func.call @inner3(%arg1) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.subtract"(%0, %arg2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %1 : tensor<f64>
  }
  func.func private @inner3(%arg0 : tensor<f64>) -> (tensor<f64>) {
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
}


Now, we will inline the contents of ``jit_mod3`` into ``qjit_mod2``, and
rename ``main`` to ``outer3``. As seen below, ``outer3`` and ``inner3``
have been inlined into ``qjit_mod2``, just like we wanted.

One might wonder why we need both ``inline_jit_to_module`` and
``inline_module``. At this stage, the intention is just to enable as
much functionality for users of the Python compiler as possible. There
may be use cases where users may want to create their own module
manually instead of having one be created automatically by JAX or
Catalyst.

>>> inline_module(jit_mod3, qjit_mod2, change_main_to="outer3")
>>> print(qjit_mod2)
builtin.module @workflow2 {
  func.func public @jit_workflow2(%arg2 : tensor<f64>, %arg3 : tensor<f64>) -> (tensor<f64>) attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_qnode::@qnode(%arg2) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.multiply"(%0, %0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %2 = "stablehlo.convert"(%arg3) : (tensor<f64>) -> tensor<f64>
    %3 = "stablehlo.subtract"(%1, %2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %3 : tensor<f64>
  }
  builtin.module @module_qnode {
    builtin.module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1 : !transform.op<"builtin.module">) {
        transform.yield
      }
    }
    func.func public @qnode(%arg0 : tensor<f64>) -> (tensor<f64>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
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
  func.func public @outer3(%arg1 : tensor<f64>, %arg2 : tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = func.call @inner3(%arg1) : (tensor<f64>) -> tensor<f64>
    %1 = "stablehlo.subtract"(%0, %arg2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %1 : tensor<f64>
  }
  func.func private @inner3(%arg0 : tensor<f64>) -> (tensor<f64>) {
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
  }
}
