.. role:: html(raw)
   :format: html

Program capture sharp bits
==========================

.. warning:: 

    Program capture is an experimental feature under active development.
    Bugs and unexpected behaviour may occur, and breaking changes are possible in future releases.
    Execution without Catalyst is no longer being developed or maintained; please use
    program capture with Catalyst present only, which can be done with `qml.qjit(capture=True)`.

Program capture is a new feature of PennyLane that allows for compactly expressing 
details about hybrid workflows, including quantum operations, classical processing, 
structured control flow, and dynamicism.

This new feature unlocks better performance, more functional and dynamic workflows, 
and smoother integration with just-in-time compilation frameworks like 
`Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ 
(via the :func:`~.pennylane.qjit` decorator) and JAX-jit.

Internally, program capture is supported by representing hybrid programs via a new 
intermediate representation (IR) called ``plxpr``, rather than a quantum tape. The 
``plxpr`` IR is an adaptation of JAX's ``jaxpr`` IR.

Our vision with ``plxpr`` is for it to be a vessel for unifying Catalyst with PennyLane, 
and to support the versatility required for hybrid quantum-classical compilation 
and dynamic programs.

There are some **quirks and restrictions to be aware of while we strive towards 
that ideal**. Additionally, we've added backward compatibility features that make 
the transition from tape-based code to program capture smooth. In this 
document, we provide an overview of the constraints, "gotchas" to be aware of, and
features that will help get your existing tape-based code working with program capture.

.. note::

    Using program capture requires that JAX be installed. Please consult the 
    JAX documentation for `installation instructions <https://docs.jax.dev/en/latest/installation.html>`__, 
    and ensure that the version of JAX that is installed corresponds to the version
    in the "Interface dependencies" section in :doc:`/development/guide/installation`.

Device compatibility
--------------------

Currently, ``default.qubit``, ``lightning.qubit``, ``lightning.kokkos``, and ``lightning.gpu`` are the only 
devices compatible with program capture.

Device wires 
~~~~~~~~~~~~

With program capture enabled, all devices that Catalyst supports require 
that ``wires`` be specified at device instantiation.

.. code-block:: python 

    import pennylane as qml

    @qml.qjit(capture=True)
    @qml.qnode(qml.device('lightning.qubit'))
    def circuit():
        qml.Hadamard(0)
        return qml.state()

>>> circuit()
NotImplementedError: devices must specify wires for integration with program capture.

.. code-block:: python 

    import pennylane as qml

    @qml.qjit(capture=True)
    @qml.qnode(qml.device('lightning.qubit', wires=1))
    def circuit():
        qml.Hadamard(0)
        return qml.state()

>>> circuit()
Array([0.70710677+0.j, 0.70710677+0.j], dtype=complex64)

Gradients
---------

Currently the devices ``lightning.qubit``, ``lightning.kokkos``, and ``lightning.gpu`` 
are the only devices that support gradients with program capture enabled. ``lightning.qubit``, 
``lightning.kokkos``, and ``lightning.gpu`` currently only support ``adjoint``. 
The ``parameter_shift`` method is not yet supported with program capture enabled, 
and will raise an error if used. 

.. code-block:: python

    import pennylane as qml
    import jax 

    qml.capture.enable() 
    # TODO
    dev = qml.device('default.qubit', wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

    bp_qn = circuit.update(diff_method="backprop")
    adj_qn = circuit.update(diff_method="adjoint")

>>> x = jax.numpy.array(jax.numpy.pi / 4)
>>> jax.jacobian(bp_qn)(x)
Array(-0.70710677, dtype=float32)
>>> jax.jacobian(adj_qn)(x)
Array(-0.70710677, dtype=float32)

However, there are some limitations to be aware of 
when using ``adjoint`` with ``default.qubit``.

Control flow and diff_method="adjoint"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control flow like ``for``, ``while`` and ``if/else`` 
are not currently supported when using ``"adjoint"`` with ``default.qubit``.
For example, the following code will raise an error:

.. code-block:: python

    import jax

    qml.capture.enable()

    dev = qml.device("default.qubit",wires=2)

    @qml.qnode(dev, diff_method="adjoint")
    def f(x):
        for i in range(2):
            qml.RX(x, wires=i)
        return qml.expval(qml.Z(0))

>>> x = jax.numpy.array(jax.numpy.pi / 4)
>>> jax.jacobian(f)(x)
NotImplementedError: Primitive for_loop does not have a jvp rule and is not supported.

Higher-order primitives and diff_method="adjoint"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Higher-order primitives like ``qml.ctrl`` and ``qml.adjoint`` are not currently supported
when using ``"adjoint"`` with ``default.qubit``. For example, the following code will raise an error:

.. code-block:: python

    import jax

    qml.capture.enable()

    dev = qml.device("default.qubit",wires=2)

    @qml.qnode(dev, diff_method="adjoint")
    def f(x):
        qml.ctrl(qml.RX, control=0)(x, 1)
        return qml.expval(qml.Z(0))

>>> x = jax.numpy.array(jax.numpy.pi / 4)
>>> jax.jacobian(f)(x)
NotImplementedError: Primitive ctrl_transform does not have a jvp rule and is not supported.

Gradients with lightning devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When executing a QNode on ``lightning.qubit``, ``lightning.kokkos`` or ``lightning.gpu`` with capture enabled,
calculating the gradient, jacobian, JVP, or VJP with JAX currently requires that we convert 
the plxpr representation of the program back to a tape for the calculation of the 
gradient, jacobian, JVP, or VJP. 

This conversion, in turn, requires that PennyLane make the assumption that each 
of the QNode's arguments are trainable, which can lead to a host of unique errors.

For instance, calculating the jacobian of this circuit with ``lightning.qubit`` 
raises an error due to a discrepancy in the ordering of the positional arguments 
when tape conversion happens.

.. code-block:: python 

    import pennylane as qml 
    import jax 

    qml.capture.enable() 

    @qml.qnode(device=qml.device("lightning.qubit", wires=1)) 
    def circuit(x, y): 
        qml.RY(y, 0) 
        qml.RX(x, 0) 
        return qml.expval(qml.Z(0)) 

>>> args = (0.1, 0.2) 
>>> jax.jacobian(circuit)(*args)
NotImplementedError: The provided arguments do not match the parameters of the jaxpr converted to quantum tape.

Valid JAX data types 
--------------------

Because of the nature of creating and executing plxpr, it is **best practice to 
use JAX-compatible types whenever possible**, in particular for arguments to quantum 
functions and QNodes, and positional arguments in PennyLane gate operations. 

Examples of JAX-compatible types are ``jax.numpy`` arrays, regular NumPy arrays, 
and standard Python ``int``\ s and ``float``\ s. Functions can accept any valid 
`Pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`__ of Jax-compatible leaves.

For example ``range``\ s or strings are not valid JAX types for the ``wires`` keyword 
argument in :class:`~.pennylane.MultiRZ`, and will result in an error:

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(jnp.array([0.1, 0.2]), wires=range(2))
        return qml.expval(qml.X(0))

>>> circuit()
...
TypeError: Argument '<pennylane.capture.autograph.ag_primitives.PRange object at 0x161b6bbd0>' of type '<class 'pennylane.capture.autograph.ag_primitives.PRange'>' is not a valid JAX type

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(jnp.array([0.1, 0.2]), wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
Array([0., 0.], dtype=float32)

lists
~~~~~

Python ``lists`` are valid Pytrees, but there are cases with program capture enabled
where they can lead to errors, and we recommend using ``jax.numpy`` arrays in place 
of Python lists wherever possible.

For example, the positional argument in ``qml.MultiRZ`` can't be a list:

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)
    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ([0.1, 0.2], wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
...
TypeError: Value [0.1, 0.2] with type <class 'list'> is not a valid JAX type

But a list can be passed to ``qml.MultiRZ`` as a keyword argument:

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)
    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(theta=[0.1, 0.2], wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
Array([0., 0.], dtype=float32)

Using a ``jax.numpy.array`` as the positional argument gives expected behaviour:

.. code-block:: python

    import pennylane as qml 

    import jax.numpy as jnp

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(jnp.array([0.1, 0.2]), wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
Array([0., 0.], dtype=float32)

Keyword arguments
~~~~~~~~~~~~~~~~~

JAX-incompatible types, like Python ``range``\ s, are acceptable as **keyword arguments**
to QNodes and quantum functions:

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()
    
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(x, range_of_wires=None):
        for w in range_of_wires:
            qml.RZ(x[0], wires=w)
            qml.RX(x[1], wires=w)

        return qml.expval(qml.X(0))

>>> circuit([0.1, 0.2], range_of_wires=range(2))
Array(0., dtype=float32)

But, again, using JAX-compatible types wherever possible is recommended.

Positional arguments
~~~~~~~~~~~~~~~~~~~~

Positional arguments in PennyLane are flexible in that their variable names can 
instead be employed as keyword arguments (e.g., ``RZ(0.1, wires=0)`` versus 
``RZ(phi=0.1, wires=0)``). However, to ensure compatibility with program capture 
enabled, such arguments must be kept as positional, regardless of whether they're 
provided as an acceptable JAX type. 

For instance, consider this example with ``RZ``:

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(angle):
        qml.RX(phi=angle, wires=0)
        return qml.expval(qml.Z(0))

>>> angle = jnp.array(0.1)
>>> circuit(angle)
...
UnexpectedTracerError: Encountered an unexpected tracer. A function transformed by JAX had a side effect, allowing for a reference to an intermediate value with type float32[] wrapped in a DynamicJaxprTracer to escape the scope of the transformation.
...

Even though the value for ``phi`` in ``RZ`` is given as a valid JAX type, the 
fact that it was provided as a keyword argument results in an error.

But, when the angle is passed as a positional argument, the circuit executes as 
expected:

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    @qml.qnode(dev)
    def circuit(angle):
        qml.RX(angle, wires=0)
        return qml.expval(qml.Z(0))

>>> angle = jnp.array(0.1)
>>> circuit(angle)
Array(0.9950042, dtype=float32)

Using program capture with Catalyst
-----------------------------------

To use the program capture feature with Catalyst, the ``qml.capture.enable()`` toggle
is also required.

.. code-block:: python

    import pennylane as qml

    qml.capture.enable()

    dev = qml.device('lightning.qubit', wires=1)

    @qml.qjit
    @qml.qnode(dev)
    def circuit():
        qml.RX(0.1, wires=0)
        return qml.state()

>>> circuit()
Array([0.99875026+0.j        , 0.        -0.04997917j], dtype=complex128)

Transforms
----------

One of the core features of PennyLane is modularity, which has allowed users to 
transform QNodes in a NumPy-like way and to create their own transforms with ease. 
Your favourite transforms will still work with program capture enabled (including
custom transforms), but there are a few caveats to be aware of.

Some transforms in the :doc:`/code/qml_transforms` module have natively support 
program capture:

* :func:`~.pennylane.transforms.merge_rotations`
* :func:`~.pennylane.transforms.single_qubit_fusion`
* :func:`~.pennylane.transforms.unitary_to_rot`
* :func:`~.pennylane.transforms.merge_amplitude_embedding`
* :func:`~.pennylane.transforms.commute_controlled`
* :func:`~.pennylane.transforms.decompose`
* :func:`~.pennylane.map_wires`
* :func:`~.pennylane.transforms.cancel_inverses`

For transforms that do not natively work with program capture, they can continue 
to be used with certain limitations:

* Transforms that return multiple tapes are not supported.
* Transforms that return non-trivial post-processing functions are not supported.
* Tape transforms may give incorrect results if the circuit has dynamic wires (i.e. there are operators
  in the circuit whose wires are dynamic parameters).
* Tape transforms will fail to execute if the transformed quantum function or QNode contains:

   * ``qml.cond`` with dynamic parameters as predicates.
   * ``qml.for_loop`` with dynamic parameters for ``start``, ``stop``, or ``step``.
   * ``qml.while_loop``.

Here is an example a toy transform called ``shift_rx_to_end``, which just moves 
``RX`` gates to the end of the circuit.

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    @qml.transform
    def shift_rx_to_end(tape):
        """Transform that moves all RX gates to the end of the operations list."""
        new_ops, rxs = [], []

        for op in tape.operations:
            if isinstance(op, qml.RX):
                rxs.append(op)
            else:
                new_ops.append(op)
        
        operations = new_ops + rxs
        new_tape = tape.copy(operations=operations)
        return [new_tape], lambda res: res[0]

When used in a workflow that contains a dynamic parameter that affects the transform's
action, an error will be raised. Consider this QNode that has a dynamic argument 
corresponding to ``stop`` in ``qml.for_loop``.

.. code-block:: python

    import pennylane as qml 

    @shift_rx_to_end
    @qml.qnode(qml.device("default.qubit", wires=4))
    def circuit(stop):

        @qml.for_loop(0, stop, 1)
        def loop(i):
            qml.RX(0.1, wires=i)
            qml.H(wires=i)
        
        loop(stop)

        return qml.state()

>>> circuit(4)
TracerIntegerConversionError: The __index__() method was called on traced array with shape int32[].
The error occurred while tracing the function wrapper at <path to environment>/site-packages/pennylane/transforms/core/transform_dispatcher.py:41 for make_jaxpr. This concrete value was not available in Python because it depends on the value of the argument inner_args[0].
See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerIntegerConversionError

Higher-order primitives and transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transforms do not apply "through" higher-order primitives like mid-circuit measurements,
gradients, and control flow when capture is enabled. An example is best to demonstrate 
this behaviour:

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=1)

    @qml.transforms.merge_rotations
    @qml.qnode(dev)
    def circuit():
        qml.RX(0.1, wires=0)

        for _ in range(4):
            qml.RX(0.1, wires=0)
            qml.RX(0.1, wires=0)

        qml.RX(0.1, wires=0)

        return qml.state()

The above example should result in a single ``RX`` gate with an angle of ``1.0``, 
but transforms are unable to transfer through the circuit in its entirety.

To illustrate what is actually happening internally, consider the plxpr representation 
of this program: 

>>> print(qml.capture.make_plxpr(circuit)())
{ ...
    qfunc_jaxpr={ lambda ; . let
        _:AbstractOperator() = RX[n_wires=1] 0.1 0
        for_loop[
          abstract_shapes_slice=slice(0, 0, None)
          args_slice=slice(0, None, None)
          consts_slice=slice(0, 0, None)
          jaxpr_body_fn={ lambda ; b:i32[]. let
              _:AbstractOperator() = RX[n_wires=1] 0.2 0
            in () }
        ] 0 4 1
        _:AbstractOperator() = RX[n_wires=1] 0.1 0
    ...
}

As one can see, the outer ``RX`` gates do not merge with those in the ``for`` loop, 
nor does the transform merge all 4 iterations from the ``for`` loop. Generally speaking, 
transform application is partitioned into "blocks" that are delimited by higher-order 
primitives.

Drawing circuits
----------------

Using :func:`~.pennylane.draw` or :func:`~.pennylane.draw_mpl` with program capture 
will generally produce inconsistent or incorrect results. Consider the following 
example:

.. code-block:: python
    
    import pennylane as qml

    qml.capture.enable()

    @qml.transforms.merge_rotations
    @qml.qnode(qml.device("default.qubit", wires=2))
    def f(x):
        qml.X(0)
        qml.X(0)
        qml.RX(x, 0)
        qml.RX(x, 0)

>>> print(qml.draw(f)(1.5))
0: ──RX(Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>)─┤  

The output does not show the two ``X`` gates, and the ``RX`` gate's value is inconsistent 
with typical behaviour (it shows a JAX tracer).

Autograph and Pythonic control flow
-----------------------------------

Autograph is a feature that allows for users to use standard Pythonic control flow
like ``for``, ``while``, etc., instead of :func:`~.pennylane.for_loop` and :func:`~.pennylane.while_loop` 
and still have compatibility with program capture. This feature is enabled by default, 
but can be switched off with the ``autograph`` keyword argument.

.. code-block:: python

    import pennylane as qml

    @qml.qnode(qml.device("default.qubit", wires=2), autograph=False)
    def circuit():
        for _ in range(10):
            qml.RX(0.1, 0)

        return qml.state()

>>> circuit()
array([0.87758256+0.j        , 0.        +0.j        ,
       0.        -0.47942554j, 0.        +0.j        ])

Note that this will unroll Pythonic control flow in your program.

Dynamic shapes
--------------

A dynamically shaped array is an array whose shape depends on an abstract value 
(e.g., a function argument). Creating and manipulating dynamically shaped objects 
within a quantum function or QNode when capture is enabled is supported with 
`JAX's experimental dynamic shapes <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes>`__. 
Given the experimental nature of this feature, PennyLane's dynamic shapes support 
is at best a subset of what is possible with purely classical programs using JAX. 

To use JAX's experimental dynamic shapes support, you must add the following toggle 
to the top level of your program: 

.. code-block:: python

    import jax

    jax.config.update("jax_dynamic_shapes", True)

Parameter broadcasting and vmap
-------------------------------

Parameter broadcasting is generally not compatible with program capture. There are 
cases that magically work, but one shouldn't extrapolate beyond those particular 
cases.

Instead, it is best practice to `use jax.vmap <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>`__:

.. code-block:: python

    import pennylane as qml 
    import jax

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

>>> x = jnp.array([0.1, 0.2, 0.3])
>>> vmap_circuit = jax.vmap(circuit)
>>> vmap_circuit(x)
Array([0.9950042 , 0.9800666 , 0.95533645], dtype=float32)

More information for using ``jax.vmap`` can be found in the 
`JAX documentation <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap>`__.

Decompositions
--------------

With program capture enabled, operators used in circuits may raise an error when 
the :func:`~.pennylane.transforms.decompose` transform is applied. This can happen 
if the operator

* defines a ``compute_decomposition`` method that contains control flow (e.g., ``if`` statements),
* does not define a ``compute_qfunc_decomposition`` method, and/or
* receives a traced argument as part of the control flow condition.

For example, the :class:`~.pennylane.RandomLayers` template does not implement a 
``compute_qfunc_decomposition`` method, and its ``compute_decomposition`` method 
includes an ``if`` statement where the condition depends on its ``ratio_imprim`` 
argument. If ``ratio_imprim`` is passed as a traced value, an error occurs:

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=2)

    @qml.transforms.decompose
    @qml.qnode(dev)
    def circuit(weights, arg):
        qml.RandomLayers(weights, wires=[0, 1], ratio_imprim=arg)
        return qml.expval(qml.Z(0))

>>> weights = jnp.array([[0.1, -2.1, 1.4]])
>>> arg = 0.5
>>> circuit(weights, arg)
...
The error occurred while tracing the function eval at pennylane/transforms/decompose.py:243 for jit. This value became a tracer due to JAX operations on these lines:
  operation a:bool[] = lt b c
    from line pennylane/templates/layers/random.py:245:19 (RandomLayers.compute_decomposition)
See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError

As a workaround, we can pass ``ratio_imprim`` as a regular (non-traced) constant:

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=2)

    @qml.transforms.decompose
    @qml.qnode(dev)
    def circuit(weights):
        qml.RandomLayers(weights, wires=[0, 1], ratio_imprim=0.5)
        return qml.expval(qml.Z(0))

>>> circuit(jnp.array([[0.1, -2.1, 1.4]]))
Array(0.99500424, dtype=float32)

Currently, the operators that define a ``compute_qfunc_decomposition`` are:

* :class:`~.StronglyEntanglingLayers`
* :class:`~.GroverOperator`
* :class:`~.QFT`

qml.cond
--------

When using :func:`~.cond`, if the ``True`` branch of a condition returns something, 
then a ``False`` branch much be provided that returns the same generic type:

.. code-block:: python

    import pennylane as qml

    qml.capture.enable()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit():

        def true_branch(x):
            return qml.X(0)

        m0 = qml.measure(0)
        qml.cond(m0, true_branch)(4)

        return qml.expval(qml.X(0))

>>> circuit()
ValueError: The false branch must be provided if the true branch returns any variables

In this particular example, to acheive the desired behaviour to "do nothing" when 
the condition is ``False``, a ``false_fn`` must be provided:

.. code-block:: python

    import pennylane as qml

    qml.capture.enable()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit():

        def true_branch(x):
            return qml.X(0)

        def false_branch(x):
            return qml.Identity(0)

        m0 = qml.measure(0)
        qml.cond(m0, true_fn=true_branch, false_fn=false_branch)(4)

        return qml.expval(qml.X(0))

>>> circuit()
Array(0., dtype=float32)

Or the ``true_fn`` itself can be an operator type itself:

.. code-block:: python

    import pennylane as qml

    qml.capture.enable()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit():
        m0 = qml.measure(0)
        qml.cond(m0, true_fn=qml.X)(0)

        return qml.expval(qml.X(0))

>>> circuit()
Array(0., dtype=float32)

while loops 
-----------

While loops written with :func:`~.pennylane.while_loop` cannot accept a ``lambda``
function:

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():

        @qml.while_loop(lambda a: a > 3)
        def loop(a):
            a += 1
            return a

        a = 0
        loop(a)

        qml.RX(0, wires=0)
        return qml.state()

>>> circuit()
...
AutoGraph currently does not support lambda functions as a loop condition for `qml.while_loop`. Please define the condition using a named function rather than a lambda function.

As a workaround, set the ``lambda`` to a callable variable,

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():

        func = lambda x: x > 3

        @qml.while_loop(func)
        def loop(a):
            a += 1
            return a

        a = 0
        loop(a)
        
        qml.RX(0, wires=0)
        return qml.state()

>>> circuit()
Array([1.+0.j, 0.+0.j], dtype=complex64)

or use a regular Python function,

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    def func(x):
        return x > 3

    @qml.qnode(dev)
    def circuit():

        @qml.while_loop(func)
        def loop(a):
            a += 1
            return a

        a = 0
        loop(a)
        
        qml.RX(0, wires=0)
        return qml.state()

>>> circuit()
Array([1.+0.j, 0.+0.j], dtype=complex64)

Calculating operator matrices in QNodes
---------------------------------------

The matrix of an operator cannot be computed with :func:`~.pennylane.matrix` within
a QNode, and will raise an error:

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        mat = qml.matrix(qml.X(0))
        return qml.state()

>>> circuit()
...
TransformError: Input is not an Operator, tape, QNode, or quantum function

.. code-block:: python

    import pennylane as qml 

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        mat = qml.matrix(qml.X)(0)
        return qml.state()

>>> circuit()
...
NotImplementedError: 
