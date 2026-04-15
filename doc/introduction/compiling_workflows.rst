.. role:: html(raw)
   :format: html

.. _intro_ref_compile_worklfows:

Compiling workflows
===================

In addition to :doc:`quantum circuit transformations </introduction/compiling_circuits>`, PennyLane also supports full
hybrid just-in-time (JIT) compilation via the :func:`~.qjit` decorator and various
hybrid compilers, which can be installed separately.

The best supported and default compiler is the `Catalyst hybrid compiler
<https://github.com/pennylaneai/catalyst>`__. Catalyst allows you to compile the entire
quantum-classical workflow, including any optimization loops. This maximizes
performance and enables running the entire workflow on accelerator devices.

In addition, PennyLane also supports compiling restricted programs via CUDA Quantum; see the CUDA Quantum section below for more details.

Installing compilers
--------------------

Currently, Catalyst must be installed separately, and only supports the JAX
interface and select devices. Supported backend devices for Catalyst include
``lightning.qubit``, ``lightning.kokkos``, ``lightning.gpu``, and ``braket.aws.qubit``,
but **not** ``default.qubit``.
For a full list of supported devices, please see :doc:`catalyst:dev/devices`.

On MacOS and Linux, Catalyst can be installed with ``pip``:

.. code-block:: console

    pip install pennylane-catalyst

Check out the Catalyst documentation for
:doc:`installation instructions <catalyst:dev/installation>`.

Just-in-time compilation
------------------------

Using Catalyst with PennyLane is as simple as using the :func:`@qjit <.qjit>` decorator to
compile your hybrid workflows:

.. code-block:: python

    from jax import numpy as jnp

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit
    @qp.set_shots(shots=1000)
    @qp.qnode(dev)
    def circuit(params):
        qp.Hadamard(0)
        qp.RX(jnp.sin(params[0]) ** 2, wires=1)
        qp.CRY(params[0], wires=[0, 1])
        qp.RX(jnp.sqrt(params[1]), wires=1)
        return qp.expval(qp.Z(1))

The :func:`~.qjit` decorator can also be used on hybrid functions --
that is, functions that include both QNodes and classical processing.

.. code-block:: python

    @qp.qjit
    def hybrid_function(params, x):
        grad = qp.grad(circuit)(params)
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

        # define the optimizer using a qjit-decorated function
        opt = jaxopt.GradientDescent(circuit, stepsize=0.4)
        update = lambda i, args: tuple(opt.update(*args))

        # perform optimization loop
        state = opt.init_state(params)
        (params, _) = jax.lax.fori_loop(0, 100, update, (params, state))

        return params

Compiling the entire hybrid workflow using ``@qp.qjit`` however will lead to better
performance. For more details, please see
`the Catalyst documentation <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/sharp_bits.html#try-and-compile-the-full-workflow>`__.

Control flow
------------

The Catalyst compiler also supports capturing imperative Python control flow
in compiled programs, resulting in control flow being interpreted at runtime
rather than in Python at compile time. You can enable this feature via the
``autograph=True`` keyword argument.

.. code-block:: python

    @qp.qjit(autograph=True)
    @qp.qnode(dev)
    def circuit(x: int):

        if x < 5:
            qp.Hadamard(wires=0)
        else:
            qp.T(wires=0)

        return qp.expval(qp.Z(0))

>>> circuit(3)
array(0.)
>>> circuit(5)
array(1.)

Note that AutoGraph results in additional restrictions, in particular whenever
global state is involved.
Please refer to the :doc:`AutoGraph guide<catalyst:dev/autograph>` for a
complete discussion of the supported and unsupported use-cases.

CUDA Quantum
------------

The PennyLane :func:`.qjit` decorator  can also be used to compile programs
using `CUDA Quantum <https://pennylane.ai/qml/glossary/what-is-cuda-quantum/>`__,
a hybrid compiler toolchain by NVIDIA.

First, Catalyst and CUDA Quantum need to be installed:

.. code-block:: bash

    pip install pennylane-catalyst cuda_quantum

Then, simply specify ``compiler="cuda_quantum"`` in the ``@qjit``
decorator:

.. code-block:: python

    dev = qp.device("softwareq.qpp", wires=2)

    @qp.qjit(compiler="cuda_quantum")
    @qp.qnode(dev)
    def circuit(x):
        qp.RX(x[0], wires=0)
        qp.RY(x[1], wires=1)
        qp.CNOT(wires=[0, 1])
        return qp.expval(qp.Y(0))

>>> circuit(jnp.array([0.5, 1.4]))
-0.47244976756708373

The following devices are available when compiling with CUDA Quantum:

* ``softwareq.qpp``: a modern C++ statevector simulator
* ``nvidia.custatevec``: The NVIDIA CuStateVec GPU simulator (with support for multi-gpu)
* ``nvidia.cutensornet``: The NVIDIA CuTensorNet GPU simulator (with support for matrix product state)

Note that CUDA Quantum compilation currently does not have feature parity with Catalyst compilation;
in particular, AutoGraph, control flow, differentiation, and various measurement statistics (such as
probabilities and variance) are not yet supported.

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
