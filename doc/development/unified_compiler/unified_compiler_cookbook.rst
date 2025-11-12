Unified Compiler Cookbook
=========================

**Note:** The cookbook is developed with the following package versions,
on Python 3.12.11:

.. code-block:: bash

    jax==0.6.2
    jaxlib==0.6.2
    numpy==2.3.1
    pennylane==0.44.0-dev19
    pennylane-lightning==0.43.0
    pennylane-catalyst==0.14.0-dev15
    xdsl==0.53.0
    xdsl-jax==git+https://github.com/xdslproject/xdsl-jax.git@895f7c13e8d0f02bbe99d7fb9ebcaafea4ea629f#egg=xdsl_jax

Note that ``xdsl-jax`` does not currently have a release published on
PyPI, so it needs to be installed from GitHub by running the following:

.. code-block:: bash

    pip install git+https://github.com/xdslproject/xdsl-jax.git

Motivation
==========

As we approach FTQC, quantum compilation becomes a more and more
important research area. Catalyst uses MLIR as its intermediate
representation, which is also the layer in which a majority of the
optimizations happen. However, quantum compilation researchers are not
likely to be accustomed with MLIR or C++.

So, the motivation of the “Unified Compiler” is to provide a Python
layer in which compilation passes can be implemented and applied.
Additionally, we also want to enable researchers to use abstractions
that they are familiar with. We’re aiming to do this using xDSL, which
is a reimplementation of MLIR in Python.

This document is meant to be a quickstart for users that are interested
in developing compiler passes for Catalyst, but are not familiar with
MLIR or xDSL.

MLIR basics
===========

If readers are already familiar to some degree with MLIR, this section
can be skipped.

SSA
---

“In compiler design, static single assignment form (often abbreviated as
SSA form or simply SSA) is a type of intermediate representation (IR)
where each variable is assigned exactly once” [`1 <#references>`__].

SSA is powerful because it allows us to define chains of uses and
definitions, i.e, we can keep track of which operation created a
variable (since each variable only gets created once), and all
operations that use that variable. These chains of uses and definitions,
if used well, can make transformations quite performant, and in some
cases, very simple to implement, but they can also make the IR harder to
parse.

IR structure
------------

MLIR represents programs using a hierarchical, graph-like structure. In
this structure, nodes are called *operations*, and edges are called
*values*. Each value is the result of exactly one operation or argument
for a block (more on that later), and has a type that is defined by the
type system (more on that later also) [`2 <#references>`__].

The IR is recursively nested - operations may contain regions, which
contain blocks, which contain operations. More concretely, an operation
may contain zero or more regions, each of which must contain one or more
blocks, each of which holds a list of arguments and an ordered list of
operations that may use those arguments [`3 <#references>`__].

Operations
~~~~~~~~~~

Operations are basic units of execution. Operations are fully
extensible, i.e., there is no fixed list of operations. Operations can
return zero or more results, take in zero or more operands, declare
properties and attributes, and can have zero or more successors and
regions [`2 <#references>`__].

Regions
~~~~~~~

A region is an ordered list of blocks. Its semantics are defined by the
operation that contains them [`2 <#references>`__]. For example, an
``scf.IfOp``, which represents a conditional operation, contains two
regions, one for the true branch, and another for the false branch, and
the way we interpret these regions is dependent on the fact that they
belong to the ``scf.IfOp``.

Blocks
~~~~~~

Blocks are lists of operations. The operations inside blocks are
executed in order. Blocks take a list of block arguments, annotated in a
function-like way. The first block in a region is special, and is called
the “entry block”. The block arguments of the entry block are also
arguments to its outer region [`2 <#references>`__].

Values
~~~~~~

In MLIR, there are computable values with a type, a single defining
operation, and zero or more uses [`4 <#references>`__]. These values are
either the results of operations or block arguments, and adhere to SSA
semantics.

Dialects
--------

MLIR uses dialects to allow developers to define a set of high level
operations, attributes, and types, which can be used in the intermediate
representation to represent custom subroutines, etc. and can be
converted into lower level representations using interpretation rules
that can be defined. For example, the MLIR ``arith`` dialect defines
operations for arithmetic computations, the ``linalg`` dialect defines
linear algebra operations, etc. We use dialects to define quantum
instructions such as gates, measurements, and attributes such as qubits,
quantum registers.

Def-use and use-def chains
--------------------------

Frequently, we might have a value and we want to determine which
instructions use this value. The list of all users of a value is the
*def-use chain*. Conversely, we might have an instruction and we want to
determine which values it uses. The list of values used by an
instruction is the *use-def chain* [`5 <#references>`__]. These chains
allow us to iterate through operations topologically, which can be very
powerful when implementing passes.

xDSL API
========

Now that we’re familiar with the high-level constructs that MLIR uses,
let’s go over what their xDSL actual implementation looks like.

``SSAValue``
------------

``SSAValue`` is the class used to represent variables. SSA values may
used by operations as operands, and returned as results. The three key
properties that this class provides are listed below:

- ``type``: The type of the value. Since SSA values are variables, their
  value is not known at compile time, but their type is.
- ``uses``: A set of all operations that use a given ``SSAValue`` as an
  operand. The operations are wrapped around a convenience class called
  ``Use``, and the corresponding operation can be accessed using
  ``Use.operation``.
- ``owner``: The operation or block that defined a given ``SSAValue``
  (more on that later)

``SSAValue`` has two subclasses, which will be seen a lot when
inspecting xDSL modules. These are:

- ``OpResult``: This subclass represents SSA values that are defined by
  an operation

  .. code-block::

      c = add a b

  In the above pseudocode, ``c`` is defined by the operation ``add``,
  and in xDSL, will be represented as an ``OpResult`` instance.

- ``BlockArgument``: This subclass represents SSA values that are block
  arguments

``Attribute``
-------------

An attribute defines a compile-time value. These can be used to define
types, and can also be used to define properties of operations that are
known at compile time. For example:

- ``CustomOp``, which is an operation from the ``Quantum`` dialect (more
  on that later) has a ``gate_name`` that must be a string, represented
  by the ``StringAttr`` attribute. ``StringAttr`` has a reference to the
  concrete string representing the gate name, and we can access this
  concrete value at compile-time using ``CustomOp.gate_name.data``.
- ``CustomOp`` also takes qubits as inputs and outputs, which are of
  type ``QubitType``. The ``QubitType`` class inherits ``Attribute``,
  but we use it to declare a type that can be used to define SSA values.

Below is the definition of ``QubitType`` to illustrate:

.. code-block:: python

    # TypeAttribute just means that this attribute can be used to represent
    # the types of the operands and results of an operation, but not its
    # properties.
    # ParametrizedAttribute means that this attribute can take parameters as
    # input (although QubitType doesn't have any parameters).
    @irdl_attr_definition
    class QubitType(ParametrizedAttribute, TypeAttribute):
        """A value-semantic qubit (state)."""

        name = "quantum.bit"

``Operation``
-------------

The ``Operation`` class is used to represent operations, which are basic
units of execution. All instructions in a module are operations. In
fact, the modules themselves are operations. Operations contain several
fields used to define their form and function:

- Operands: operands are runtime values that the operation consumes.
  Note that when defining an operation, we only declare the *types* of
  the operands, not the actual operands. Only when we construct an
  instance of an operation do we provide actual ``SSAValue``\ s as the
  operands, and these must adhere to the type system.
- Properties: properties are compile-time values used to define the
  semantics of an operation. For example, the ``CustomOp`` operation
  that is used to define quantum gates in Catalyst has a property called
  ``gate_name``, which is a string specifier of the gate’s name. This
  name directly impacts how the operation should be interpreted, and its
  value is known at compile-time.
- Attributes: Operation attributes are stored as a dictionary,
  containing more compile-time values. Generally, these don’t get used
  much in xDSL, but they serve a purpose very similar to properties.
- Result types: operations may return values, and if so, the types of
  the return values must be defined.

Below is the definition of ``CustomOp`` from the ``Quantum`` dialect,
which represents general quantum gates to illustrate what defining
operations looks like:

.. code-block:: python

    @irdl_op_definition
    class CustomOp(IRDLOperation):
        """A generic quantum gate on n qubits with m floating point parameters."""

        name = "quantum.custom"

        # assembly_format defines what the operation should look like
        # when pretty-printed in the IR
        assembly_format = """
            $gate_name `(` $params `)` $in_qubits
            (`adj` $adjoint^)?
            attr-dict
            ( `ctrls` `(` $in_ctrl_qubits^ `)` )?
            ( `ctrlvals` `(` $in_ctrl_values^ `)` )?
            `:` type($out_qubits) (`ctrls` type($out_ctrl_qubits)^ )?
        """

        # These options are used because we have operands whose lengths aren't
        # known (eg. different instances of CustomOp may have different number
        # of qubits depending on the gate they represent (2 for CNOT, 1 for
        # RX, etc.). These options basically say, "when the operation instance
        # is initialized, create 2 properties that store the length of each of
        # the different groups of operands and results.
        irdl_options = [
            AttrSizedOperandSegments(as_property=True),
            AttrSizedResultSegments(as_property=True),
        ]

        # var_operand_def means that the length of this operand
        # can vary.
        params = var_operand_def(EqAttrConstraint(Float64Type()))

        in_qubits = var_operand_def(BaseAttr(QubitType))

        # prop_def means that gate_name is a required property
        gate_name = prop_def(BaseAttr(StringAttr))

        # opt_prop_def means adjoint is an optional property. Additionally,
        # it's type is a UnitAttr(), which essentially means that a given
        # instance of CustomOp is an adjoint gate iff it has an adjoint property.
        # The value of the property is irrelevant; it only gets meaning from its
        # existance.
        adjoint = opt_prop_def(EqAttrConstraint(UnitAttr()))

        in_ctrl_qubits = var_operand_def(BaseAttr(QubitType))

        in_ctrl_values = var_operand_def(EqAttrConstraint(IntegerType(1)))

        # var_result_def means that the length of out_qubits can vary
        out_qubits = var_result_def(BaseAttr(QubitType))

        out_ctrl_qubits = var_result_def(BaseAttr(QubitType))

.. _dialects-1:

Dialects
--------

The ``Dialect`` class in xDSL is used as a container around a list of
operations, types, and attributes, and it also declares a name for the
dialect. At the time of writing this, there are currently 4 custom
dialects available in the xDSL layer of Catalyst:

- ``Quantum``: this dialect contains operations and attributes necessary
  for general qubit-level operations, such as gates, measurements,
  qubits, etc.
- ``Catalyst``: this dialect contains operations and attributes for
  classical computing features unavailable out of the box with xDSL/MLIR
- ``QEC``: This dialect contains operations and attributes useful for
  QEC, such as PPRs/PPMs.
- ``MBQC``: This dialect contains operations and attributes for
  representing MBQC formalism.

Pass API
--------

xDSL provides an API for defining and applying transformations on
programs (or modules), which is described below:

``ModulePass``
~~~~~~~~~~~~~~

``ModulePass`` is used to create rewrite passes over an IR module. It is
the parent class used to define compiler passes (or transforms).
``ModulePass`` has two key fields that must be implemented

- ``name``: This is the name that is used to reference the pass.
- ``apply``: This method takes a ``ModuleOp`` as input and applies the
  rewrite pattern of the pass to the module. Note that this mutates the
  input module *in-place* rather than returning a new, transformed
  module.

``RewritePattern``
~~~~~~~~~~~~~~~~~~

``RewritePattern`` is the class that provides the API for pattern
matching. The most important method of this class is
``match_and_rewrite``. The first argument to this method is an
operation, and it must be type-hinted using the specific operation we’re
trying to match. This type hint gets used by xDSL to match the operation
we’re trying to rewrite. The second argument is a ``PatternRewriter``
instance. I will cover this class in detail below, but it is essentially
the class that provides the API for rewriting the operations that we’re
matching.

For example, if I wanted to match all Hadamard gates, I would use
``CustomOp`` from the ``Quantum`` dialect in the type hint for the first
argument (since there is no ``Hadamard`` operation in the ``Quantum``
dialect), and check in the body of the method if the op is a
``Hadamard``:

.. code-block:: python

    from xdsl import pattern_rewriter
    from pennylane.compiler.python_compiler.quantum_dialect import CustomOp

    class MyPattern(pattern_rewriter.RewritePattern):
        """Dummy class for example."""

        # This decorator is what xDSL uses to match operations
        # based on the type hint.
        @pattern_rewriter.op_type_rewrite_pattern
        def match_and_rewrite(
            self, op: CustomOp, rewriter: pattern_rewriter.PatternRewriter
        ):
            if op.gate_name.data != "Hadamard":
                # If not Hadamard, we do nothing
                return

            # Do whatever we need

``PatternRewriter``
~~~~~~~~~~~~~~~~~~~

``PatternRewriter`` is the class that provides the API for rewriting the
IR. It includes several methods for replacing/removing/updating
operations, replacing values, replacing uses of a value with another
value, etc. In most cases, any rewriting that users want to do must be
done through this API rather than manually, as it includes state
management for keeping track of whether any changes were made, which is
necessary for the worklist algorithm (more on that later).

Some key methods are:

- ``replace_op``: Replaces one operation with another.
- ``replace_all_uses_with``: Replaces all uses of one value with
  another.
- ``erase_op``: Erases an operation. If this operation returns any
  values, all uses of these values must be updated accordingly before
  the erasure.
- ``notify_op_modified``: Method to notify the rewriter that a change
  was made to an operation manually.

The example below shows us implementing a ``RewritePattern`` that
updates all ``Hadamard``\ s with ``PauliX``\ s:

.. code-block:: python

    from xdsl import pattern_rewriter
    from xdsl.dialects import builtin
    from pennylane.compiler.python_compiler.dialects.quantum import CustomOp

    class HToXPattern(pattern_rewriter.RewritePattern):
        """Dummy class for example."""

        @pattern_rewriter.op_type_rewrite_pattern
        def match_and_rewrite(
            self, op: CustomOp, rewriter: pattern_rewriter.PatternRewriter
        ):
            if op.gate_name.data != "Hadamard":
                # If not Hadamard, we do nothing
                return

            # Update the gate name to PauliX and notify the rewriter that
            # the op was manually updated
            op.gate_name = builtin.StringAttr("PauliX")
            rewriter.notify_op_modified(op)

            # Alternatively, we could also create a new CustomOp for
            # the PauliX from scratch, and replace the Hadamard with
            # the new op:
            # new_op = CustomOp(
            #     gate_name="PauliX",
            #     in_qubits=op.in_qubits,
            # )
            # rewriter.replace_op(op, new_op)

``PatternRewriteWalker``
~~~~~~~~~~~~~~~~~~~~~~~~

``PatternRewriteWalker`` walks over the IR in depth-first order, and
applies a provided ``RewritePattern`` to it. By default, it implements a
worklist algorithm that keeps iterating over the operations and matching
and rewriting until a steady state is reached (i.e. no new changes are
detected; this is why ``PatternRewriter`` needs to keep track of whether
any changes were made).

Putting everything together, we can create a ``ModulePass`` that
replaces all ``Hadamard``\ s with ``PauliX``\ s

.. code-block:: python

    from xdsl import passes, pattern_rewriter
    from xdsl.dialects import builtin
    from pennylane.compiler.python_compiler.dialects.quantum import CustomOp
    from pennylane.compiler.python_compiler import compiler_transform

    class HToXPattern(pattern_rewriter.RewritePattern):
        """Dummy class for example."""

        @pattern_rewriter.op_type_rewrite_pattern
        def match_and_rewrite(
            self, op: CustomOp, rewriter: pattern_rewriter.PatternRewriter
        ):
            if op.gate_name.data != "Hadamard":
                # If not Hadamard, we do nothing
                return

            # Update the gate name to PauliX and notify the rewriter that
            # the op was manually updated
            op.gate_name = builtin.StringAttr("PauliX")
            rewriter.notify_op_modified(op)

    class HToXPass(passes.ModulePass):
        """Pass that replaces Hadamards with PauliXs"""

        name = "h-to-x"

        def apply(self, ctx, module):
            """Apply the iterative pass."""
            walker = pattern_rewriter.PatternRewriteWalker(
                pattern=HToXPattern()
            )
            walker.rewrite_module(module)

    # We will cover this later
    h_to_x_pass = compiler_transform(HToXPass)

``PassPipeline``
~~~~~~~~~~~~~~~~

``PassPipeline`` is a meta-pass that takes a sequence of
``ModulePass``\ es as input, and applies them to the input module. The
following example shows how a sequence of ``ModulePass``\ s can be
applied to a module ``mod``:

.. code-block:: python

    from xdsl import passes

    pipeline = passes.PassPipeline((Pass1(), Pass2(), Pass3()))
    pipeline.apply(xdsl.context.Context(), mod)

To complete the example we’ve been building in this section, let’s put
it all together and implement a ``PassPipeline`` to apply the
``HToXPass`` to an xDSL module.

Let’s first create the module to which we want to apply the pass. For
this, we will use the ``xdsl_from_qjit`` utility, which is described in
the “PennyLane integration” section below.

- **Creating the module**

  .. code-block:: python

      import pennylane as qml
      from pennylane.compiler.python_compiler.conversion import xdsl_from_qjit

      dev = qml.device("lightning.qubit", wires=3)

      @xdsl_from_qjit
      @qml.qjit(target="mlir")
      @qml.qnode(dev)
      def circuit():
          qml.Hadamard(0)
          qml.Hadamard(1)
          qml.Hadamard(2)
          return qml.state()

  >>> mod = circuit()
  >>> print(mod)
  builtin.module @circuit {
    func.func public @jit_circuit() -> (tensor<8xcomplex<f64>>) attributes {llvm.emit_c_interface} {
      %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<8xcomplex<f64>>
      func.return %0 : tensor<8xcomplex<f64>>
    }
    builtin.module @module_circuit {
      builtin.module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
          transform.yield
        }
      }
      func.func public @circuit() -> (tensor<8xcomplex<f64>>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
        %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %1 = tensor.extract %0[] : tensor<i64>
        quantum.device shots(%1) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
        %2 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
        %3 = quantum.alloc(3) : !quantum.reg
        %4 = tensor.extract %0[] : tensor<i64>
        %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
        %6 = quantum.custom "Hadamard"() %5 : !quantum.bit
        %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
        %8 = tensor.extract %7[] : tensor<i64>
        %9 = quantum.extract %3[%8] : !quantum.reg -> !quantum.bit
        %10 = quantum.custom "Hadamard"() %9 : !quantum.bit
        %11 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
        %12 = tensor.extract %11[] : tensor<i64>
        %13 = quantum.extract %3[%12] : !quantum.reg -> !quantum.bit
        %14 = quantum.custom "Hadamard"() %13 : !quantum.bit
        %15 = tensor.extract %0[] : tensor<i64>
        %16 = quantum.insert %3[%15], %6 : !quantum.reg, !quantum.bit
        %17 = tensor.extract %7[] : tensor<i64>
        %18 = quantum.insert %16[%17], %10 : !quantum.reg, !quantum.bit
        %19 = tensor.extract %11[] : tensor<i64>
        %20 = quantum.insert %18[%19], %14 : !quantum.reg, !quantum.bit
        %21 = quantum.compbasis qreg %20 : !quantum.obs
        %22 = quantum.state %21 : tensor<8xcomplex<f64>>
        quantum.dealloc %20 : !quantum.reg
        quantum.device_release
        func.return %22 : tensor<8xcomplex<f64>>
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

- **Transforming the module**

  In the above module, there are 3 ``CustomOp``\ s, each with gate name
  ``Hadamard``. Let’s try applying our pass to it. Bear in mind that
  passes update modules in-place:

  .. code-block:: python

      from xdsl import passes

      pipeline = passes.PassPipeline((HToXPass(),))
      pipeline.apply(xdsl.context.Context(), mod)

  >>> print(mod)
  builtin.module @circuit {
    func.func public @jit_circuit() -> (tensor<8xcomplex<f64>>) attributes {llvm.emit_c_interface} {
      %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<8xcomplex<f64>>
      func.return %0 : tensor<8xcomplex<f64>>
    }
    builtin.module @module_circuit {
      builtin.module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
          transform.yield
        }
      }
      func.func public @circuit() -> (tensor<8xcomplex<f64>>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
        %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %1 = tensor.extract %0[] : tensor<i64>
        quantum.device shots(%1) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
        %2 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
        %3 = quantum.alloc(3) : !quantum.reg
        %4 = tensor.extract %0[] : tensor<i64>
        %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
        %6 = quantum.custom "PauliX"() %5 : !quantum.bit
        %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
        %8 = tensor.extract %7[] : tensor<i64>
        %9 = quantum.extract %3[%8] : !quantum.reg -> !quantum.bit
        %10 = quantum.custom "PauliX"() %9 : !quantum.bit
        %11 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
        %12 = tensor.extract %11[] : tensor<i64>
        %13 = quantum.extract %3[%12] : !quantum.reg -> !quantum.bit
        %14 = quantum.custom "PauliX"() %13 : !quantum.bit
        %15 = tensor.extract %0[] : tensor<i64>
        %16 = quantum.insert %3[%15], %6 : !quantum.reg, !quantum.bit
        %17 = tensor.extract %7[] : tensor<i64>
        %18 = quantum.insert %16[%17], %10 : !quantum.reg, !quantum.bit
        %19 = tensor.extract %11[] : tensor<i64>
        %20 = quantum.insert %18[%19], %14 : !quantum.reg, !quantum.bit
        %21 = quantum.compbasis qreg %20 : !quantum.obs
        %22 = quantum.state %21 : tensor<8xcomplex<f64>>
        quantum.dealloc %20 : !quantum.reg
        quantum.device_release
        func.return %22 : tensor<8xcomplex<f64>>
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

  Great! We can see that all the ``Hadamard``\ s have been replaced with
  ``PauliX``\ s, just how we wanted.

PennyLane integration
=====================

This section will cover the API in the ``qml.compiler.python_compiler``
submodule.

Lowering to MLIR
----------------

Catalyst compiles programs using the following workflow, when program
capture is enabled:

- Trace user function to create ``ClosedJaxpr`` (plxpr)
- Convert plxpr to value-semantic jaxpr
- Lower value-semantic jaxpr to MLIR
- Apply passes defined in a static pipeline to the MLIR

  - These passes include optimizations, further lowering to more
    elementary dialects, and lowering to LLVM

- Generate machine code

The integration with the xDSL layer happens after we lower to MLIR. We
currently rely on JAX’s API to lower to MLIR. This has the special
effect of lowering to a specific dialect called StableHLO, which is used
to represent all arithmetic operations present in the program.

Once lowered to MLIR, if the original ``qjit`` decorator specified the
xDSL pass plugin, we pass control over to the xDSL layer, which applies
all transforms that were requested by the user. We can request the use
of the xDSL plugin like so:

.. code-block:: python

    from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

    @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
    ...

.. _ir-structure-1:

IR structure
~~~~~~~~~~~~

The lowered MLIR has a lot of structure to aid developers, which is
described below using the following example:

.. code-block:: python

    import pennylane as qml

    qml.capture.enable()
    dev = qml.device("lightning.qubit", wires=1)

    @qml.qjit
    @qml.transforms.cancel_inverses
    @qml.transforms.merge_rotations
    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return qml.state()

>>> print(circuit.mlir)
module @circuit {
  func.func public @jit_circuit() -> tensor<2xcomplex<f64>> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<2xcomplex<f64>>
    return %0 : tensor<2xcomplex<f64>>
  }
  module @module_circuit {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "merge-rotations" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %1 = transform.apply_registered_pass "remove-chained-self-inverse" to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield
      }
    }
    func.func public @circuit() -> tensor<2xcomplex<f64>> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c0_i64 = arith.constant 0 : i64
      quantum.device shots(%c0_i64) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %0 = quantum.alloc( 1) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
      %2 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
      %3 = quantum.compbasis qreg %2 : !quantum.obs
      %4 = quantum.state %3 : tensor<2xcomplex<f64>>
      quantum.dealloc %2 : !quantum.reg
      quantum.device_release
      return %4 : tensor<2xcomplex<f64>>
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}

- The program is represented as a module with a function inside it. This
  function contains non-QNode user code (which there is none in our
  example for simplicity), and calls to QNodes using the
  ``catalyst.launch_kernel`` operation.
- QNodes are represented as modules as well—each QNode has its own
  module. These modules have 4 key components:

  - The ``module attributes {transform.with_named_sequence}`` contains
    the transforms that the user requested for the QNode. In our
    example, those are the ``merge_rotations`` and ``cancel_inverses``
    transforms.
  - The ``@circuit`` function represents the body of the QNode. Inside
    this body, we initialize a device using the specified shots,
    allocate a quantum register, and then apply both quantum and
    classical instructions. Note that quantum instructions can *only* be
    present inside this function. Note that this function will contain
    an attribute called ``qnode``.

SSA in the ``Quantum`` dialect
------------------------------

In the ``Quantum`` dialect, as mentioned earlier, gates are represented
using the ``CustomOp`` operation. This operation accepts ``in_qubits``
and ``in_ctrl_qubits``, which are variable length sequences of
``QubitType`` attributes, which correspond to wire indices.
``CustomOp``\ s also return ``out_qubits`` and ``out_ctrl_qubits`` of
the same type.

Before a ``QubitType`` can be used by any gates, it must be extracted
from the quantum register, or a ``QregType``. Quantum registers can be
thought of as a sequence containing all valid wire indices that are
available to be used by gates/measurements. Qubits can be extracted from
and inserted into a quantum register using the ``ExtractOp`` and
``InsertOp`` operations. An ``AllocOp`` is used to allocate a quantum
register with the user-provided number of device wires.

Let’s take a look at a very simple example. Below, I have a very simple
circuit with two gates applied to the same wire. Let’s take a look at
its MLIR representation:

- **Example**

  .. code-block:: python

      dev = qml.device("lightning.qubit", wires=3)

      @qml.qjit(target="mlir")
      @qml.qnode(dev)
      def circuit():
          qml.X(0)
          qml.H(0)
          return qml.state()

  >>> print(circuit.mlir)
  module @circuit {
    func.func public @jit_circuit() -> tensor<8xcomplex<f64>> attributes {llvm.emit_c_interface} {
      %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<8xcomplex<f64>>
      return %0 : tensor<8xcomplex<f64>>
    }
    module @module_circuit {
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
          transform.yield
        }
      }
      func.func public @circuit() -> tensor<8xcomplex<f64>> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
        %c0_i64 = arith.constant 0 : i64
        quantum.device shots(%c0_i64) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
        %0 = quantum.alloc( 3) : !quantum.reg
        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
        %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
        %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
        %3 = quantum.compbasis qreg %2 : !quantum.obs
        %4 = quantum.state %3 : tensor<8xcomplex<f64>>
        quantum.dealloc %2 : !quantum.reg
        quantum.device_release
        return %4 : tensor<8xcomplex<f64>>
      }
    }
    func.func @setup() {
      quantum.init
      return
    }
    func.func @teardown() {
      quantum.finalize
      return
    }
  }

**Notes**

- ``quantum.alloc`` initializes a quantum register (``%0``) with 3
  wires, which is the number of wires used to create the PennyLane
  device.
- ``quantum.extract`` extracts a qubit corresponding to wire index 0
  (``%1``)

  - ``%1`` is used as input by the ``X`` gate.

- The ``X`` gate returns a qubit (``%out_qubits``), which is used by the
  ``quantum.custom`` corresponding to the ``H`` gate.

  - Note that this is different from how the circuit is defined in
    Python. Instead of just using ``%1`` again, the ``H`` gate uses
    ``%out_qubits``.

- The ``H`` gate returns a new qubit (``%out_qubits_0``). This qubit is
  consumed by ``quantum.insert``, which inserts updates the quantum
  register to essentially say that ``%out_qubits_0`` should be the new
  qubit that corresponds to wire index 0.
- In this example, note that ``%1``, ``%out_qubits``, and
  ``%out_qubits_0`` all correspond to wire index 0. This has cool
  implications, that I will discuss below.

Dynamic wires
~~~~~~~~~~~~~

The lowering rules for Catalyst automatically handle dynamic wires. When
a new dynamic wire, ``w``, is used, all wires used before it are first
inserted into the quantum register using ``InsertOp``. Only then does
the ``QubitType`` corresponding to ``w`` get extracted from the quantum
register using ``ExtractOp``. Consider the following example:

- **Example**

  .. code-block:: python

      import pennylane as qml

      qml.capture.enable()

      dev = qml.device("lightning.qubit", wires=3)

      @qml.qjit(target="mlir")
      @qml.qnode(dev)
      def circuit(w1: int, w2: int):
          qml.X(0)
          qml.Y(w1)
          qml.Z(w1)
          qml.S(w2)
          qml.T(w1)
          qml.H(0)
          return qml.state()

  >>> print(circuit.mlir)
  module @circuit {
    func.func public @jit_circuit(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<8xcomplex<f64>> attributes {llvm.emit_c_interface} {
      %0 = catalyst.launch_kernel @module_circuit::@circuit(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<8xcomplex<f64>>
      return %0 : tensor<8xcomplex<f64>>
    }
    module @module_circuit {
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
          transform.yield
        }
      }
      func.func public @circuit(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<8xcomplex<f64>> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
        %c0_i64 = arith.constant 0 : i64
        quantum.device shots(%c0_i64) ["/Users/mudit.pandey/.pyenv/versions/pennylane-xdsl/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
        %0 = quantum.alloc( 3) : !quantum.reg
        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
        %2 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
        %extracted = tensor.extract %arg0[] : tensor<i64>
        %3 = quantum.extract %2[%extracted] : !quantum.reg -> !quantum.bit
        %out_qubits_0 = quantum.custom "PauliY"() %3 : !quantum.bit
        %out_qubits_1 = quantum.custom "PauliZ"() %out_qubits_0 : !quantum.bit
        %extracted_2 = tensor.extract %arg0[] : tensor<i64>
        %4 = quantum.insert %2[%extracted_2], %out_qubits_1 : !quantum.reg, !quantum.bit
        %extracted_3 = tensor.extract %arg1[] : tensor<i64>
        %5 = quantum.extract %4[%extracted_3] : !quantum.reg -> !quantum.bit
        %out_qubits_4 = quantum.custom "S"() %5 : !quantum.bit
        %extracted_5 = tensor.extract %arg1[] : tensor<i64>
        %6 = quantum.insert %4[%extracted_5], %out_qubits_4 : !quantum.reg, !quantum.bit
        %extracted_6 = tensor.extract %arg0[] : tensor<i64>
        %7 = quantum.extract %6[%extracted_6] : !quantum.reg -> !quantum.bit
        %out_qubits_7 = quantum.custom "T"() %7 : !quantum.bit
        %extracted_8 = tensor.extract %arg0[] : tensor<i64>
        %8 = quantum.insert %6[%extracted_8], %out_qubits_7 : !quantum.reg, !quantum.bit
        %9 = quantum.extract %8[ 0] : !quantum.reg -> !quantum.bit
        %out_qubits_9 = quantum.custom "Hadamard"() %9 : !quantum.bit
        %10 = quantum.insert %8[ 0], %out_qubits_9 : !quantum.reg, !quantum.bit
        %11 = quantum.compbasis qreg %10 : !quantum.obs
        %12 = quantum.state %11 : tensor<8xcomplex<f64>>
        quantum.dealloc %10 : !quantum.reg
        quantum.device_release
        return %12 : tensor<8xcomplex<f64>>
      }
    }
    func.func @setup() {
      quantum.init
      return
    }
    func.func @teardown() {
      quantum.finalize
      return
    }
  }

  **Notes**

  - If a qubit is inserted into the quantum register, it is not reused
    again, and a new qubit corresponding to the same wire label must be
    extracted from the register.
  - When a dynamic wire is going to be used for the first time, all
    qubits that have been used previously without being re-inserted into
    the quantum register must be inserted.
  - If a dynamic wire is reused before any other wires, then it does not
    need to be inserted to and extracted from the quantum register
    again.
  - To use static wires after dynamic wires, the dynamic wires are again
    re-inserted into the quantum register.
  - All of the above make it such that dynamic wires essentially create
    barriers that break the qubit def-use chains. This causes some
    functionality to be lost, but makes sure that we’re using qubits
    safely.

Implications/notes
~~~~~~~~~~~~~~~~~~

- Operations on the same wires can be tracked quickly using the chain of
  definitions and uses of the qubits (def-use chains).
- Operations on dynamic wires (i.e wires whose values are only known at
  run-time) are handled automatically and we don’t need to worry about
  managing how we work around them. For context, this is an issue that
  we found in the plxpr variant of ``cancel_inverses``, where two
  operations on the same wire that were separated by another operation
  on a dynamic wire would get cancelled, which is incorrect
  (`source <https://github.com/PennyLaneAI/pennylane/issues/7349>`__).
- One thing to keep in mind is that qubits (``QubitType``) and quantum
  registers (``QregType``) are there so that we conform to SSA semantics
  in a way that works well for our purposes. They get meaning from how
  we choose to interpret them. We could just as easily have defined the
  ``Quantum`` dialect and Catalyst’s lowering rules to use wire indices
  the same way we do in Python, but we may lose capabilities that MLIR
  and xDSL enable through the SSA form.

``compiler_transform``
----------------------

``compiler_transform`` is the function used to register xDSL
``ModulePass``\ es to be used with ``qjit``-ed workflows. It is
currently accessible as
``qml.compiler.python_compiler.compiler_transform``.

.. code-block:: python

    from pennylane.compiler.python_compiler import compiler_transform

    class MyPass(xdsl.passes.ModulePass):
        """MyPass that does something"""

        name = "my-pass"

        def apply(self, ctx, module):
            # Apply the pass to module
            return

    my_pass = compiler_transform(MyPass)

    # Program capture must be enabled to use the compiler transform
    # as a decorator
    qml.capture.enable()
    dev = qml.device("lightning.qubit", wires=1)

    @qml.qjit(
        pass_plugins=[catalyst.passes.xdsl_plugin.getXDSLPluginAbsolutePath()]
    )
    @my_pass
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    circuit(1.5)

The ``compiler_transform`` function returns an object that gives easy
access to the underlying ``ModulePass``, as well as its name as seen by
the compiler.

>>> my_pass.module_pass
__main__.MyPass
>>> my_pass.name
'my-pass'

Additionally, we don’t need to manually apply passes using
``PassPipeline`` when decorating QNodes with registered compiler
transforms. Those transforms get applied automatically when the workflow
is compiled!

Conversion utilities
--------------------

The ``python_compiler.conversion`` submodule provides several utilities
for creating xDSL modules from Python functions. There are many more
utilities in the submodule, but I will focus on the most important ones,
and provide examples of how to use them.

- ``xdsl_module(jitted_fn: Callable) -> Callable[Any, [xdsl.dialects.builtin.ModuleOp]]``:
  Create a wrapper around a ``jax.jit``-ed function that returns an xDSL
  module
- ``xdsl_from_qjit(qjitted_fn: QJIT) -> Callable[..., xbuiltin.ModuleOp]``:
  Create a wrapper around a ``qjit``-ed function that returns an xDSL
  module. This is currently not merged to ``master``
- ``inline_module(from_mod: xbuiltin.ModuleOp, to_mod: xbuiltin.ModuleOp, change_main_to: str = None) -> None``:
  This function takes two modules as input, and inlines the whole body
  of the first module into the second. Additionally, if
  ``change_main_to`` is provided, it looks for a function named
  ``main``, and updates its name to ``change_main_to``
- ``inline_jit_to_module(func: JaxJittedFunction, mod: xbuiltin.ModuleOp, *args, **kwargs) -> None``:
  This function takes a ``jax.jit``-ed function, converts it into an
  xDSL module, and then inlines the contents of the xDSL module into
  ``mod``. Note that this function does not return anything; instead, it
  modifies ``mod`` in-place.

Check out the :doc:`xDSL conversion utilities tutorial <./xdsl_utils_tutorial>` to see examples of how each of
the utilities can be used.

Useful patterns
===============

Now that we have gone over compilers, xDSL, and how it’s being used in
PennyLane, let’s take a look at some common patterns that might be
useful.

Post-processing functions
-------------------------

Post-processing functions are purely classical, so we can leverage the
``xdsl_module`` utility function to create xDSL modules from Python
code, and inject it into the modules we are rewriting as needed. The
:doc:`xDSL post-processing tutorial <./xdsl_post_processing>` shows an
example where we perform very simple post-processing on a QNode that
returns an expectation value by squaring the expectation value.

Splitting tapes
---------------

When implementing passes that may transform our module such that
multiple device executions are required (akin to tape transforms that
return multiple tapes), there are various strategies that can be used.
Because there is no guarantee of shared structure between the tapes,
there is no one perfect strategy that can be used for all transforms.
I’ll use tapes to provide details below about some common cases:

- If all the tapes are identical (eg. ``dynamic_one_shot``), then the
  entire execution can be put inside a ``for`` loop, with
  post-processing on the execution outputs done how I showed in the
  above notebook.
- If all gates are identical, but measurements are different (eg.
  ``split_non_commuting``), we can capture all gates in a single
  function, and then use a ``for`` loop that iterates over the number of
  tapes. Within this ``for`` loop, we would call the aforementioned
  function, which would evolve the state, and apply a different
  measurement inside each iteration of the loop. Post-processing can be
  handled same as above.
- If there is little/no shared structure between the tapes, we would
  need separate functions for each of the transformed tapes. We would
  need to call each function one by one, and then use the results for
  post-processing.

All of the above are very non-trivial. I will leave out code examples
for now, as that may be unnecessarily time consuming. If we get to the
stage where we need to write a transform that splits into multiple
tapes, we can revisit this section and the Python compiler/compilation
team can assist in developing such transforms.

Note
~~~~

Currently, we don’t have a consistent API for implementing transforms
that split QNodes into multiple device executions (eg.
``split_non_commuting``, etc.). Thus, it is *very* hard to implement
transforms that do that. We cannot guarantee that one transform that
does split QNodes into multiple device executions will work well when
there are other transforms in the pipeline.

Writing tests
=============

**Note to readers**: this section is written based on how the testing
infrastructure for the Python compiler exists in PennyLane. However, the
Python compiler may be getting moved to Catalyst, in which case, the
infrastructure would likely change.

FileCheck
---------

`FileCheck <https://llvm.org/docs/CommandGuide/FileCheck.html>`__ is a
pattern matching file verifier developed by the LLVM project and is
widely used to test MLIR. Since xDSL and MLIR are syntactically
identical, we can use the string representation of xDSL modules for
testing with FileCheck.

Below is an example of how an MLIR/xDSL string may look when populated
with directives that FileCheck can use for testing. In this example, we
have a void function ``test_func`` that takes no arguments, creates two
qubits (corresponding to different wires), and applies a ``PauliX`` to
each of these wires. We can see that there are 4 comments starting with
``CHECK`` - these comments are what FileCheck uses to match the expected
string against the actual string.

The number of ``CHECK`` comments does not need to be the same as the
number of operations in the program - we can see below that there is no
``CHECK`` for ``func.func`` and ``return``.

``CHECK`` statements can assign parameters that can be reused by later
``CHECK``\ s. Below, the first two ``CHECK``\ s create ``q0`` and
``q1``, which are matched using the regular expression ``%.+``, which
matches any expressions that start with ``%`` and contain at least one
character after it. The latter 2 ``CHECK``\ s then use ``q0`` and ``q1``
as the input qubits for the assertion.

``CHECK`` statements can contain partial statements. Below, the last two
``CHECK``\ s don’t include the outputs of the ``quantum.custom``
operations.

.. code-block:: python

    program = """
        func.func @test_func() {
            // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
            // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
            %0 = "test.op"() : () -> !quantum.bit
            %1 = "test.op"() : () -> !quantum.bit
            // CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
            // CHECK: quantum.custom "PauliX"() [[q1]] : !quantum.bit
            %2 = quantum.custom "PauliX"() %0 : !quantum.bit
            %3 = quantum.custom "PauliX"() %0 : !quantum.bit
            return
        }
    """

FileCheck essentially uses the MLIR generated by a program and asserts
it against directives that can be specified by the user. Commonly used
directives are:

- ``CHECK``: This is used to check if a specified pattern is found. Its
  syntax is very simple: they are fixed strings that must occur in
  order, and horizontal whitespace is ignored by default.
- ``CHECK-NOT``: This directive is used to check that the provided
  string does not occur between two matches, before the first match, or
  after the last match.
- ``CHECK-DAG``: This directive may be used when it’s necessary to match
  strings that don’t have to occur in the same order, but it is able to
  match valid topological orderings of the program DAG, with edges from
  the definition to the uses of a variable.
- ``CHECK-NEXT``: This directive is used to check that the matched line
  occurs right after the previously matched line with no other lines in
  between them.
- ``CHECK-SAME``: This directive is used when we want to match lines and
  would like to verify that matches happen on the same line as the
  previous match.

To find more details about the above directives, or to learn about other
available directives, please refer to the `FileCheck
documentation <https://llvm.org/docs/CommandGuide/FileCheck.html>`__.

Test dialect
------------

xDSL provides a ``Test`` dialect
(`source <https://github.com/xdslproject/xdsl/blob/main/xdsl/dialects/test.py>`__),
which contains many operations that are useful for unit testing. In our
testing, we found the ``TestOp`` operation to be the most useful. This
operation can produce arbitrary results, which we can use to limit
artificial dependencies on other dialects.

For example, if I just need to assert that a specific gate is present,
without ``TestOp``, I would need my module to contain an ``AllocOp``
that creates a quantum register, an ``ExtractOp`` that extracts a qubit
from the register, and only then I can use the qubit for the gate I’m
trying to match. Instead, I can just insert a ``TestOp`` that returns a
qubit and use that.

This is very powerful for unit testing, as it makes writing tests much
simpler, while also limiting the scope of the test as one would expect
for unit tests.

In the code block in the previous section, ``TestOp`` has been used in
exactly the described way - we use it to create 2 qubits that are then
used as input for the 2 ``PauliX`` gates.

.. _pennylane-integration-1:

PennyLane integration
---------------------

To use FileCheck with ``pytest``, we use the ```filecheck`` Python
package <https://pypi.org/project/filecheck/>`__, which allows us to use
assertions for testing in a way that ``pytest`` can understand. All of
the ``filecheck`` API has been captured inside two fixtures available
within the ``tests/python_compiler`` folder:

- ``run_filecheck``: This fixture is for unit testing. One can specify a
  program along with filecheck directives as a multi-line string.
- ``run_filecheck_qjit``: This fixture is for integration testing. One
  can create a normal ``qml.qjit``-ed workflow and include filecheck
  directives as in-line comments.

Let’s write tests for the ``HToXPass`` that was implemented in the
`“Pass API” <#pass-api>`__ sub-section to illustrate. The dev comments
will explain what is going on.

``run_filecheck`` example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_h_to_x_pass(run_filecheck):
        """Test that Hadamard gets converted into PauliX."""
        # The original program creates a qubit, and gives it to a
        # CustomOp that is a Hadamard. The CHECK directives check that
        # the transformed program has a CustomOp that is a PauliX applied
        # to the same qubit, and no CustomOp that is a Hadamard.

        # Below we also see how we can use `test.op`, which we use to
        # create a qubit to give to the Hadamard.
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
                // CHECK-NOT: quantum.custom "Hadamard"
                %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
                return
            }
        """

        # First, we must create the pass pipeline that we want to apply
        # to our program.
        pipeline = (HToXPass(),)

        # Next, we use run_filecheck to run filecheck testing on the
        # program. The fixture will create an xDSL module from the program
        # string, call the filecheck API, and assert correctness.
        run_filecheck(program, pipeline)

        # Optionally, there are two keyword arguments that can modify the
        # behaviour of run_filecheck. Both the roundtrip and verify
        # arguments are false by default.
        run_filecheck(program, pipeline, roundtrip=True, verify=True)

        # roundtrip=True makes it so that after parsing the program string
        # to an xDSL module, we print it as a string again, and then parse
        # it back into an xDSL module. This is useful when writing tests
        # for dialects, so that we can check that the dialects can be printed
        # parsed correctly.

        # verify=True simply runs `module.verify()`, which iteratively uses
        # xDSL's verifiers to verify all operations and attributes in the
        # module.

``run_filecheck_qjit`` example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

    def test_h_to_x_pass_integration(run_filecheck_qjit):
        """Test that Hadamard gets converted into PauliX."""
        # The original program simply applies a Hadamard to a circuit
        # Remember that we need to specify the pass plugin. Additionally,
        # with qjit, we can use the decorator created using
        # `compiler_transform`. To make sure that the xDSL API works
        # correctly, program capture must be enabled.
        # qml.capture.enable()
        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath])
        @h_to_x_pass
        def circuit():
            # CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
            # CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
            # CHECK-NOT: quantum.custom "Hadamard"
            qml.Hadamard(0)
            return qml.state()

        # Finally, we use the run_filecheck_qjit fixture. We pass it our
        # original qjitted workflow. It extracts the filecheck directives
        # from the workflow, creates an MLIR program, parses it to xDSL,
        # applies the specified transforms, and uses the filecheck API to
        # assert correctness.
        run_filecheck_qjit(circuit)

        # run_filecheck_qjit also accepts a boolean `verify` argument
        # that is false by default, which works exactly the same way as
        # the `verify` argument of `run_filecheck`.
        run_filecheck_qjit(circuit, verify=True)

Key blockers
============

There are several blockers that are currently disabling developers from
taking full advantage of the ``python_compiler`` submodule. These
include:

* Lack of support for quantum subroutines. This impacts pattern
  matching passes that need to substitute the matched operation(s) with
  subroutines containing quantum instructions.

Strategies to circumvent blockers
---------------------------------

* We can use dummy subroutines for now. We know what the inputs and
  outputs of these subroutines should be, so we can create our own
  ``FuncOp``\ s that adhere to the input/output spec and just have
  their body be empty for now. To see an example where we create a
  dummy quantum subroutine and use it to develop a pass, check out the
  :doc:`xDSL subroutines tutorial <./xdsl_dummy_quantum_subroutines>`.

Suggested reading
=================

Useful dialects
---------------

- ``scf``: Structured control flow
- ``func``: Functions
- ``builtin``: Core types and attributes
- ``arith``: arithmetic operations
- ``stablehlo``: Advanced match dialect

References
==========

#. Wikimedia Foundation. (2025, August 11). *Static single-assignment
   form*. Wikipedia.
   https://en.wikipedia.org/wiki/Static_single-assignment_form
#. *MLIR Language Reference*. MLIR. (n.d.).
   https://mlir.llvm.org/docs/LangRef/
#. *Understanding the IR Structure*. MLIR. (n.d.-b).
   https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
#. *Mlir::Value class reference*. MLIR. (n.d.-b).
   https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
#. *LLVM Programmer’s Manual*. LLVM. (n.d.).
   https://llvm.org/docs/ProgrammersManual.html
