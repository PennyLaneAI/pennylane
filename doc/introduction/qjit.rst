.. role:: html(raw)
   :format: html

.. _intro_ref_qjit_compilation:

Quantum just-in-time compilation
================================

The PennyLane2 framework is powered by the
`Catalyst compiler <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`_,
which enables just-in-time compilation of quantum-classical hybrid programs via
the :func:`~pennylane.qjit` decorator.

The :doc:`compiling_circuits` page provides an overview of quantum circuit
transformations. Here, "compilation" means *just-in-time compilation* of the
entire quantum-classical program, which translates Python code into an optimized
executable, rather than transforming the quantum circuit itself.

Benefits of qjit
----------------

With traditional Python program execution, instructions are executed one at a
time with little knowledge of program structure (e.g., control flow) or context.
Quick inspiration-to-implementation feedback loops in research and development
are attainable in that framework, but poorer runtime performance and suboptimal
programs being sent to the device executing the program plague industrial-scale
utility.

With :func:`~pennylane.qjit`, you get software that enables seamless inspiration
to implementation, is fast where it matters, scalable where it counts, and
hardware-ready.

In practice, this means PennyLane2 enables many key features, including:

* Compile and profile the quantum resources of workflows at every stage of compilation while preserving its structure.

* Write workflows that use heuristic information to get fast resource estimates that enable inspiration and further implementation.

* Advanced control flow that supports both quantum and classical instructions.

* Significant performance and scalability improvements over standard Python execution.

* Compile an entire quantum-classical workflow, including any optimization loops.

* Support for PennyLane-Lightning high performance simulators, and Amazon Braket devices.

Using qjit in practice
----------------------

It's as simple as using :func:`~pennylane.qjit` in-line or as a decorator to
compile your hybrid workflows:

.. code-block:: python

    from jax import numpy as jnp

    dev = qp.device("lightning.qubit", wires=2)

    @qp.set_shots(shots=1000)
    @qp.qnode(dev)
    def circuit(params):
        qp.Hadamard(0)
        qp.RX(jnp.sin(params[0]) ** 2, wires=1)
        qp.CRY(params[0], wires=[0, 1])
        qp.RX(jnp.sqrt(params[1]), wires=1)
        return qp.expval(qp.Z(1))

>>> params = jnp.array([0.1, 0.2])
>>> qp.qjit(circuit)(params)
Array(0.884, dtype=float64)

.. note::

    The ``jax.numpy`` interface is recommended for expressing numeric arrays
    in PennyLane2, though there can be tradeoffs. See :doc:`sharp_bits` for more
    details.

The :func:`~.qjit` decorator can also be used on hybrid functions -- that is,
functions that include both quantum and classical processing.

.. code-block:: python

    @qp.qjit
    def hybrid_function(params, x):
        grad = qp.grad(circuit)(params)
        return jnp.abs(grad - x) ** 2

>>> x = 0.2
>>> hybrid_function(params, x)
Array([0.11068767, 0.47929955], dtype=float32)

Performance when compiling with :func:`~pennylane.qjit` is especially enhanced
with repeated executions of the same function, which is a common usage pattern
in quantum machine learning workflows.

.. note::

    Optimization of :func:`~pennylane.qjit`'d workflows is supported with the
    ``optax`` package, which does not come with PennyLane2 and must be installed
    separately. See the
    `optax installation instructions <https://optax.readthedocs.io/en/latest/#installation>`_
    for details.

.. code-block:: python

    import optax

    @qp.qjit(autograph=True)
    def optimization():
        # initial parameter
        params = jnp.array([0.54, 0.3154])

        opt = optax.adam(learning_rate=0.3)
        opt_state = opt.init(params)

        def update_step(params, opt_state):
            grads = qp.grad(circuit, argnums=0)(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        for _ in range(10_000):
            params, _ = update_step(params, opt_state)

        return params

>>> optimization()
Array([1.73811307, 3.91381022], dtype=float64)

Static and dynamic arguments
----------------------------

By default, the arguments of a :func:`~pennylane.qjit`'d function are
*dynamic*: the function is traced with placeholder values that only carry a
shape and type, compiled once, and the compiled program is reused for any new
argument values. Since Python code like ``print`` only runs during tracing, it
shows when compilation happens:

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit
    @qp.qnode(dev)
    def circuit(x):
        print("Compiling...")
        qp.RX(x, wires=0)
        return qp.expval(qp.Z(0))

    circuit(0.1)  # prints "Compiling..."
    circuit(0.2)  # new value: compiled program is reused, nothing is printed

Calling the function with an argument of a different shape or type (e.g., an
array of a different length) triggers recompilation.

Arguments listed in ``static_argnums`` (or ``static_argnames``) are instead
*static*: their Python values are fixed in the compiled program, so they can
drive Python control flow such as the number of layers. Static arguments must
be hashable, and the function is recompiled whenever their value changes:

.. code-block:: python

    @qp.qjit(static_argnums=1)
    @qp.qnode(dev)
    def circuit(x, n_layers):
        print(f"Compiling for n_layers={n_layers}")
        for _ in range(n_layers):
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
        return qp.expval(qp.Z(1))

    circuit(0.1, 2)  # prints "Compiling for n_layers=2"
    circuit(0.5, 2)  # only the dynamic argument changed: no recompilation
    circuit(0.1, 3)  # prints "Compiling for n_layers=3"

A Python ``if`` or ``for`` that depends on a *dynamic* argument cannot be
evaluated during tracing; use ``@qp.qjit(autograph=True)`` to convert it into
compiled control flow. See the Catalyst
:doc:`sharp bits <catalyst:dev/sharp_bits>` page for more details.
