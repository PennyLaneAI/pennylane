.. role:: html(raw)
   :format: html

🔪 Program capture sharp bits
=============================

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

Our vision with ``plxpr`` is to supplant the quantum tape as our default program 
representation, and to support *more* than just the core features of PennyLane.

There are some **quirks and restrictions to be aware of while we strive towards 
that ideal**. Additionally, we've added backward compatibility features that make 
the transition from tape-based code to program capture smooth. In this 
document, we provide an overview of the constraints, "gotchas" to be aware of, and
features that will help get your existing tape-based code working with program capture.

.. note::

    Using program capture requires that JAX be installed. Please consult the 
    JAX documentation for `installation instructions <https://docs.jax.dev/en/latest/installation.html>`__.

Device compatibility
--------------------

Currently, ``default.qubit`` and ``lightning.qubit`` are the only devices compatible 
with program capture.

Device wires 
~~~~~~~~~~~~

With program capture enabled, both ``lightning.qubit`` and ``default.qubit`` require 
that ``wires`` be specified at device instantiation (this is in contrast to when 
program capture is disabled, where automatic qubit management takes place internally
with ``default.qubit``).

.. code-block:: python 

    qml.capture.enable()

    @qml.qnode(qml.device('default.qubit'))
    def circuit():
        qml.Hadamard(0)
        return qml.state()

>>> circuit()
NotImplementedError: devices must specify wires for integration with program capture.
>>> circuit = circuit.update(device = qml.device('default.qubit', wires=1)) 
>>> circuit()
Array([0.70710677+0.j, 0.70710677+0.j], dtype=complex64)

This also affects mid-circuit measurements (MCMs) with the deferred measurements
method:

.. code-block:: python

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        return qml.state()

>>> circuit(0.1)
...
TransformError: Too many mid-circuit measurements for the specified number of wires with 'defer_measurements'.

Recall that the deferred measurements MCM method adds a temporary wire and represents 
the physical MCM as a controlled operation, deferring the measurement until the 
end of the circuit. By adding an additional wire to the device, the above circuit
executes as expected: 

>>> circuit = circuit.update(device = qml.device('default.qubit', wires=2))
>>> circuit(0.1)
Array([0.99875027+0.j        , 0.        +0.j        ,
       0.        +0.j        , 0.        -0.04997917j], dtype=complex64)

Valid JAX data types 
--------------------

Because of the nature of creating and executing plxpr, it is **best practice to 
use JAX-compatible types whenever possible**, in particular for arguments to quantum 
functions and QNodes, and positional arguments in PennyLane gate operations. Examples of 
JAX-compatible types are ``jax.numpy`` arrays, regular NumPy arrays, dictionaries, standard
Python ``int``\ s and ``float``\ s, and anything else with a valid `Pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`__
representation.

For example, ``list``\ s, ``range``\ s, and strings are not valid JAX types for 
the positional argument in :class:`~.pennylane.MultiRZ`, and will result in an error:

.. code-block:: python 

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ([0.1, 0.2], wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
TypeError: Value [0.1, 0.2] with type <class 'list'> is not a valid JAX type

.. code-block:: python 

    import jax.numpy as jnp

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(jnp.array([0.1, 0.2]), wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
Array([0., 0.], dtype=float32)

Providing a ``list`` as input to a quantum function or QNode is accepted in cases 
where the ``list`` is being indexed into, thereby retrieving a valid JAX type:

.. code-block:: python 

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RZ(x[0], wires=0)
        qml.RX(x[1], wires=1)
        return qml.expval(qml.X(0))

>>> circuit([0.1, 0.2])
Array(0., dtype=float32)

Keyword arguments
~~~~~~~~~~~~~~~~~

JAX-incompatible types, like Python ``range``\ s, are acceptable as **keyword arguments**:

.. code-block:: python 

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
instead be employed as keyword arguments (e.g., ``qml.RZ(0.1, wires=0)`` versus 
``qml.RZ(phi=0.1, wires=0)``). However, to ensure differentiability and, in general,
compatibility with program capture enabled, such arguments must be kept as positional, 
regardless of whether they're provided as an acceptable JAX type. 

For instance, consider this example with ``qml.RZ``:

.. code-block:: python 

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

Even though the value for ``phi`` in ``qml.RZ`` is given as a valid JAX type, the 
fact that it was provided as a keyword argument results in an error.

But, when the angle is passed as a positional argument, the circuit executes as 
expected:

.. code-block:: python 

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
is not required. Instead, when decorating a workflow with :func:`~.pennylane.qjit`, 
add the ``experimental_capture=True`` flag:

.. code-block:: python

    dev = qml.device('lightning.qubit', wires=1)

    @qml.qjit(experimental_capture=True)
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
custom transforms), but decorating QNodes with just ``@transform_name`` **will not 
work** and will give a vague error. Additionally, decorating QNodes with the experimental 
:func:`~.pennylane.capture.expand_plxpr_transforms` decorator is required.

Consider the following toy example, which shows a tape-based transform that shifts 
all :class:`~.pennylane.RX` gates to the end of a circuit.

.. code-block:: python 

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

Decorating with just ``@shift_rx_to_end`` will not work, and will give a vague error:

.. code-block:: python 

    qml.capture.enable()

    @shift_rx_to_end
    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit():
        qml.RX(0.1, wires=0)
        qml.H(wires=0)
        return qml.state()

>>> print(qml.draw(circuit)())
...
NotImplementedError: 

A requirement for transforms to be compatible with program capture is to further 
decorate QNodes with the experimental :func:`~.pennylane.capture.expand_plxpr_transforms` 
decorator:

.. code-block:: python 

    qml.capture.enable()

    @qml.capture.expand_plxpr_transforms
    @shift_rx_to_end
    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit():
        qml.RX(0.1, wires=0)
        qml.H(wires=0)
        return qml.state()

>>> print(qml.draw(circuit)())
0: ──H──RX(0.10)─┤  State

Higher-order primitives and transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transforms do not apply "through" higher-order primitives like mid-circuit measurements,
gradients, and control flow when capture is enabled. An example is best to demonstrate 
this behaviour:

.. code-block:: python 

    qml.capture.enable()

    dev = qml.device('default.qubit', wires=1)

    @qml.capture.expand_plxpr_transforms
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
but transforms are unable to transfer through the circuit in its entirety. Drawing
this circuit will result in an inaccurate circuit:

>>> print(qml.draw(circuit)())
0: ──RX(0.20)─┤  State

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

Dynamic variables and transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some transforms in the :doc:`/code/qml_transforms` module have natively support program capture:

#. :func:`~.pennylane.transforms.merge_rotations`
#. :func:`~.pennylane.transforms.single_qubit_fusion`
#. :func:`~.pennylane.transforms.unitary_to_rot`
#. :func:`~.pennylane.transforms.merge_amplitude_embedding`
#. :func:`~.pennylane.transforms.commute_controlled`
#. :func:`~.pennylane.transforms.decompose`
#. :func:`~.pennylane.map_wires`
#. :func:`~.pennylane.transforms.cancel_inverses`

For transforms that do not natively work with program capture, they can continue to be used with certain limitations:

#. Transforms that return multiple tapes are not supported.
#. Transforms that return non-trivial post-processing functions are not supported.
#. Tape transforms will fail to execute if the transformed quantum function or QNode contains:

   #. ``qml.cond`` with dynamic parameters as predicates.
   #. ``qml.for_loop`` with dynamic parameters for ``start``, ``stop``, or ``step``.
   #. ``qml.while_loop``.

Here is an example with our toy ``shift_rx_to_end`` transform and a dynamic parameter
for ``stop`` in ``qml.for_loop``.

.. code-block:: python 

    qml.capture.enable()

    @qml.capture.expand_plxpr_transforms
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

Autograph and Pythonic control flow
-----------------------------------

Autograph is a feature that allows for users to use standard Pythonic control flow
like ``for``, ``while``, etc., instead of :func:`~.pennylane.for_loop` and :func:`~.pennylane.while_loop` 
and still have compatibility with program capture. This feature is enabled by default, 
but can be switched off with the ``autograph`` keyword argument.

.. code-block:: python

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

    jax.config.update("jax_dynamic_shapes", True)

Parameter broadcasting and vmap
-------------------------------

Parameter-broadcasting is generally not compatible with program capture. There are 
cases that magically work, but one shouldn't extrapolate beyond those particular 
cases.

Instead, it is best practice to `use jax.vmap <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>`__:

.. code-block:: python 

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

while loops 
-----------

While loops written with :func:`~.pennylane.while_loop` cannot accept a ``lambda``
function:

.. code-block:: python 

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
KeyError: <gast.gast.Lambda object at 0x136ff82b0>

As a workaround, set the ``lambda`` to a callable variable,

.. code-block:: python 

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

    qml.capture.enable()

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        mat = qml.matrix(qml.X)(0)
        return qml.state()

>>> circuit()
...
NotImplementedError: 

Section title 
-------------

blah blah blah

.. code-block:: python 

    qml.capture.enable()

    # nice code block!!!!!!!!!

>>> print("hello plxpr")
hello plxpr

blah blah blah