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
and ``parameter-shift`` differentation methods.

.. code-block:: python

    import pennylane as qml
    import jax.numpy as jnp 

    dev = qml.device('lightning.qubit', wires=1)

    @qml.qjit(capture=True)
    def workflow(x):

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(_x):
            qml.RX(_x, wires=0)
            return qml.expval(qml.Z(0))
        
        return qml.grad(circuit)(x)

>>> x = jnp.array(jnp.pi / 4)
>>> workflow(x)
Array(-0.70710678, dtype=float64)

Valid JAX data types 
--------------------

Because of the nature of creating and executing plxpr, it is **best practice to 
use JAX-compatible types whenever possible**, in particular for arguments to quantum 
functions and QNodes, and positional arguments in PennyLane gate operations. 

Examples of JAX-compatible types are ``jax.numpy`` arrays, regular NumPy arrays, 
and standard Python ``int``\ s and ``float``\ s. Functions can accept any valid 
`Pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`__ of Jax-compatible leaves.

For example, strings are not valid JAX types for the ``wires`` keyword argument 
in quantum operations, and will result in an error:

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    dev = qml.device('lightning.qubit', wires=["a", "b", "c"])

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(jnp.array(0.1), wires=["a", "b"])
        return qml.expval(qml.X(0))

>>> circuit()
...
TypeError: Argument 'a' of type <class 'str'> is not a valid JAX type

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    dev = qml.device('lightning.qubit', wires=[0, 1, 2])

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ(jnp.array(0.1), wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
Array(0., dtype=float64)

lists
~~~~~

Python ``lists`` are valid Pytrees, but there are cases with program capture enabled
where they can lead to errors, and we recommend using ``jax.numpy`` arrays in place 
of Python lists wherever possible.

For example, the positional argument in ``qml.QubitUnitary`` can't be a ``list``:

.. code-block:: python

    import pennylane as qml 

    my_unitary = [[1, 0], [0, 1]]

    dev = qml.device('lightning.qubit', wires=2)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit():
        qml.QubitUnitary(my_unitary, wires=0)
        return qml.state()

>>> circuit()
...
TypeError: Argument '[[1, 0], [0, 1]]' of type '<class 'list'>' is not a valid JAX type

But a ``list`` can be passed to ``qml.QubitUnitary`` as a keyword argument:

.. code-block:: python

    import pennylane as qml 

    my_unitary = [[1, 0], [0, 1]]

    dev = qml.device('lightning.qubit', wires=2)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit():
        qml.QubitUnitary(U=my_unitary, wires=0)
        return qml.state()

>>> circuit()
Array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex128)

Using a ``jax.numpy.array`` as the positional argument gives expected behaviour:

.. code-block:: python

    import pennylane as qml 
    import jax

    my_unitary = jax.numpy.array([[1, 0], [0, 1]])

    dev = qml.device('lightning.qubit', wires=2)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit():
        qml.QubitUnitary(my_unitary, wires=0)
        return qml.state()

>>> circuit()
Array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dtype=complex128)

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

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit(angle):
        qml.RX(phi=angle, wires=0)
        return qml.expval(qml.Z(0))

>>> angle = jnp.array(0.1)
>>> circuit(angle)
...
InvalidInputException: Argument 'JitTracer<~float64[]>' of type <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> is not a valid JAX type.

Even though the value for ``phi`` in ``RZ`` is given as a valid JAX type, the 
fact that it was provided as a keyword argument results in an error.

But, when the angle is passed as a positional argument, the circuit executes as 
expected:

.. code-block:: python

    import pennylane as qml 
    import jax.numpy as jnp

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit(angle):
        qml.RX(phi=angle, wires=0)
        return qml.expval(qml.Z(0))

>>> angle = jnp.array(0.1)
>>> circuit(angle)
Array(0.9950042, dtype=float32)

Transforms
----------

Without program capture enabled, your program is represented as a tape, which is
essentially a straight-line list of instructions that define your program. With 
program capture enabled, the representation of the program is fundamentally different, 
which will impact how you can manipulate and optimize circuits via transforms.

In general, if a transform that is available in PennyLane only has a tape-compatible 
definition, we do not recommend using it with ``@qjit(capture=True)``, though no 
errors may occur. Currently, this includes:

* :func:`~.pennylane.transforms.combine_global_phases`
* :func:`~.pennylane.transforms.commute_controlled`
* :func:`~.pennylane.transforms.merge_amplitude_embedding`
* :func:`~.pennylane.transforms.pattern_matching_optimization`
* :func:`~.pennylane.transforms.match_relative_phase_toffoli`
* :func:`~.pennylane.transforms.match_controlled_iX_gate`
* :func:`~.pennylane.transforms.remove_barrier`
* :func:`~.pennylane.transforms.single_qubit_fusion`
* :func:`~.pennylane.transforms.transpile`
* :func:`~.pennylane.transforms.unitary_to_rot`
* :func:`~.pennylane.transforms.transpile`
* :func:`~.pennylane.transforms.rz_phase_gradient`
* :func:`~.pennylane.transforms.zx.optimize_t_count`
* :func:`~.pennylane.transforms.zx.push_hadamards`
* :func:`~.pennylane.transforms.zx.reduce_non_clifford`
* :func:`~.pennylane.transforms.zx.todd`
* :func:`~.pennylane.add_noise`
* :func:`~.pennylane.transforms.undo_swaps`
* :func:`~.pennylane.cut_circuit_mc`
* :func:`~.pennylane.cut_circuit`
* :func:`~.pennylane.transforms.rowcol`
* :func:`~.pennylane.map_wires`
* Custom tape transforms created with :func:`~.pennylane.transform`.

Drawing circuits
----------------

Using :func:`~.pennylane.draw` or :func:`~.pennylane.draw_mpl` with program capture 
and Catalyst may not produce correct results due to the dynamic nature of compiled
programs. Instead, it is recommended to use :func:`catalyst.draw_graph`, which 
will accurately depict the dynamicism of programs compiled with ``@qjit(capture=True)``.

Autograph and Pythonic control flow
-----------------------------------

Autograph is a feature that allows for users to use standard Pythonic control flow
like ``for``, ``while``, etc., instead of :func:`~.pennylane.for_loop` and :func:`~.pennylane.while_loop` 
and still have program structure preserved. This can be accessed with ``qjit(capture=True, autograph=True)``:

.. code-block:: python

    import pennylane as qml

    @qml.qjit(capture=True, autograph=True)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit():
        for _ in range(1000000):
            qml.RX(0.1, 0)

        return qml.state()

>>> circuit()
Array([-0.01787726+0.j        ,  0.        +0.j        ,
        0.        +0.99984019j,  0.        +0.j        ],      dtype=complex128)

``autograph=False`` by default. If ``autograph=False`` in the above example, the 
circuit's ``for`` loop would be completely unrolled, resulting in an extremely 
expensive calculation given the depth of the circuit.

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
    import jax.numpy as jnp

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

>>> x = jnp.array([0.1, 0.2, 0.3])
>>> vmap_circuit = jax.vmap(circuit)
>>> vmap_circuit(x)
Array([0.99500417, 0.98006658, 0.95533649], dtype=float64)

More information for using ``jax.vmap`` can be found in the 
`JAX documentation <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap>`__.

qml.cond
--------

When using :func:`~.cond`, if the ``True`` branch of a condition returns something, 
then a ``False`` branch much be provided that returns the same generic type:

.. code-block:: python

    import pennylane as qml

    @qml.qjit(capture=True)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit():

        def true_branch(x):
            return qml.X(0)

        m0 = qml.measure(0)
        qml.cond(m0, true_branch)(4)

        return qml.expval(qml.X(0))

>>> circuit()
ValueError: The false branch must be provided if the true branch returns any variables

In this particular example, to acheive the desired behaviour when the condition 
is ``False``, a ``false_fn`` must be provided:

.. code-block:: python

    import pennylane as qml

    @qml.qjit(capture=True)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit():

        def true_branch(x):
            return qml.X(0)

        def false_branch(x):
            return qml.Identity(0)

        m0 = qml.measure(0)
        qml.cond(m0, true_fn=true_branch, false_fn=false_branch)(4)

        return qml.expval(qml.X(0))

>>> circuit()
Array(0., dtype=float64)

That, or the ``true_fn`` itself can be an operator type:

.. code-block:: python

    import pennylane as qml

    @qml.qjit(capture=True)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit():

        m0 = qml.measure(0)
        qml.cond(m0, true_fn=qml.X)(0)

        return qml.expval(qml.X(0))

>>> circuit()
Array(0., dtype=float64)

Calculating operator matrices in QNodes
---------------------------------------

The matrix of an operator cannot be computed with :func:`~.pennylane.matrix` within
a QNode, and will raise an error:

.. code-block:: python

    import pennylane as qml 

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit():
        mat = qml.matrix(qml.X(0))
        return qml.state()

>>> circuit()
...
TransformError: Input is not an Operator, tape, QNode, or quantum function

Instead, ``qml.matrix`` must be invoked on an operator *type* (e.g., ``qml.X`` instead of ``qml.X(0)``):

.. code-block:: python

    import pennylane as qml 

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qjit(capture=True)
    @qml.qnode(dev)
    def circuit():
        mat = qml.matrix(qml.X)
        return qml.state()

>>> circuit()
Array([1.+0.j, 0.+0.j], dtype=complex128)
