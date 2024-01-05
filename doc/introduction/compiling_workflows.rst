.. role:: html(raw)
   :format: html

.. _intro_ref_compile_worklfows:

Compiling workflows
===================

In addition to :doc:`quantum circuit transformations </introduction/compiling_circuits>`, PennyLane also supports full
hybrid just-in-time (JIT) compilation via the :func:`~.qjit` decorator and
the `Catalyst hybrid compiler <https://github.com/pennylaneai/catalyst>`__.
Catalyst allows you to compile the entire quantum-classical workflow,
including any optimization loops, which allows for optimized performance, and
the ability to run the entire workflow on accelerator devices as
appropriate.

Installing compilers
--------------------

Currently, Catalyst must be installed separately, and only supports the JAX
interface and select devices such as ``lightning.qubit``,
``lightning.kokkos``, ``braket.local.qubit`` and ``braket.aws.qubit``. It
does **not** support ``default.qubit``.

On MacOS and Linux, Catalyst can be installed with ``pip``:

.. code-block:: console

    pip install pennylane-catalyst

Check out the Catalyst documentation for
:doc:`installation instructions <catalyst:dev/installation>`.

Just-in-time compilation
------------------------

Using Catalyst with PennyLane is a simple as using the :func:`@qjit <.qjit>` decorator to
compile your hybrid workflows:

.. code-block:: python

    from jax import numpy as jnp

    dev = qml.device("lightning.qubit", wires=2, shots=1000)

    @qml.qjit
    @qml.qnode(dev)
    def circuit(params):
        qml.Hadamard(0)
        qml.RX(jnp.sin(params[0]) ** 2, wires=1)
        qml.CRY(params[0], wires=[0, 1])
        qml.RX(jnp.sqrt(params[1]), wires=1)
        return qml.expval(qml.PauliZ(1))

The :func:`~.qjit` decorator can also be used on hybrid functions --
that is, functions that include both QNodes and classical processing.

.. code-block:: python

    @qml.qjit
    def hybrid_function(params, x):
        grad = qml.grad(circuit)(params)[0]
        return jnp.abs(grad - x) ** 2

In addition, functions that are compiled with ``@jax.jit`` can contain calls
to qjit-compiled functions. For example, below we compile a full optimization loop,
using ``@jax.jit``:

.. code-block:: python

    import jaxopt

    @jax.jit
    def optimization():
        # initial parameter
        params = jnp.array([0.54, 0.3154])

        # define the optimizer
        opt = jaxopt.GradientDescent(circuit, stepsize=0.4)
        update = lambda i, args: tuple(opt.update(*args))

        # perform optimization loop
        state = opt.init_state(params)
        (params, _) = jax.lax.fori_loop(0, 100, update, (params, state))

        return params

Compiling the entire hybrid workflow using ``@qml.qjit`` however will lead to better
performance. For more details, please see
`the Catalyst documentation <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/sharp_bits.html#try-and-compile-the-full-workflow>`__.

Control flow
------------

The Catalyst compiler also supports capturing imperative Python control flow
in compiled programs, resulting in control flow being interpreted at runtime
rather than in Python at compile time. You can enable this feature via the
``autograph=True`` keyword argument.

.. code-block:: python

    @qml.qjit(autograph=True)
    @qml.qnode(dev)
    def circuit(x: int):

        if x < 5:
            qml.Hadamard(wires=0)
        else:
            qml.T(wires=0)

        return qml.expval(qml.PauliZ(0))

>>> circuit(3)
array(0.)
>>> circuit(5)
array(1.)

Note that AutoGraph results in additional restrictions, in particular whenever
global state is involved.
Please refer to the :doc:`AutoGraph guide<catalyst:dev/autograph>` for a
complete discussion of the supported and unsupported use-cases.

Additional resources
--------------------

For more details on using the :func:`~.qjit` decorator and Catalyst
with PennyLane, please refer to the Catalyst
:doc:`quickstart guide <catalyst:dev/quick_start>`, as well as the :doc:`sharp
bits and debugging tips <catalyst:dev/sharp_bits>` page for an overview of
the differences between Catalyst and PennyLane, and how to best structure
your workflows to improve performance when using Catalyst.

To make your own compiler compatible with PennyLane, please see
the :mod:`~.compiler` module documentation.
