.. role:: html(raw)
   :format: html

.. _intro_ref_qjit_compilation:

Quantum just-in-time compilation
================================

The PennyLane2 framework is powered by the
`Catalyst compiler <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`_,
which enables just-in-time compilation of quantum-classical hybrid programs via
the :func:`~pennylane.qjit` decorator.

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
>>> qp.qjit(circuit, capture=True)(params)
Array(0.884, dtype=float64)

.. note::

    The ``jax.numpy`` interface is recommended for expressing numeric arrays
    in PennyLane2, though there can be tradeoffs. See :doc:`sharp_bits` for more
    details.

The :func:`~.qjit` decorator can also be used on hybrid functions -- that is,
functions that include both quantum and classical processing.

.. code-block:: python

    @qp.qjit(capture=True)
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

    Differentiation of :func:`~pennylane.qjit`'d workflows is supported with the
    ``optax`` package, which comes out-of-the-box with PennyLane2.

.. code-block:: python

    import optax

    @qp.qjit(capture=True)
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

        @qp.for_loop(0, 10_000, 1)
        def loop(_, params, opt_state):
            params, _ = update_step(params, opt_state)
            return params, opt_state

        params, _ = loop(params, opt_state)

        return params

>>> optimization()
Array([1.73811307, 3.91381022], dtype=float64)
