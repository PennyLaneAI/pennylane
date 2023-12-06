.. role:: html(raw)
   :format: html

.. _intro_ref_compile:

Compiling circuits
==================

PennyLane offers multiple tools for compiling circuits. We use the term "compilation"
here in a loose sense as the process of transforming one circuit 
into one or more differing circuits. A circuit could be either a quantum function or a sequence of operators. For example, such a transformation could
replace a gate type with another, fuse gates, exploit mathematical relations that simplify an observable,
or replace a large circuit by a number of smaller circuits.

Compilation functionality is mostly designed as **transforms**, which you can read up on in the
section on :doc:`inspecting circuits </introduction/inspecting_circuits>`.

In addition to quantum circuit transforms, PennyLane also
supports experimental just-in-time compilation, via the :func:`~.qjit` decorator and
`Catalyst <https://github.com/pennylaneai/catalyst>`__. This is more general, and
supports full hybrid compilation --- compiling both the classical and quantum components
of your workflow into a binary that can be run close to the accelerators.
that you are using.

Simplifying Operators
----------------------

PennyLane provides the :func:`~.pennylane.simplify` function to simplify single operators, quantum
functions, QNodes and tapes. This function has several purposes:

* Reducing the arithmetic depth of the given operators to its minimum.
* Grouping like terms in sums and products.
* Resolving products of Pauli operators.
* Combining identical rotation gates by summing its angles.

Here are some simple simplification routines:

>>> qml.simplify(qml.RX(4*np.pi+0.1, 0 ))
RX(0.09999999999999964, wires=[0])
>>> qml.simplify(qml.adjoint(qml.RX(1.23, 0)))
RX(11.336370614359172, wires=[0])
>>> qml.simplify(qml.ops.Pow(qml.RX(1, 0), 3))
RX(3.0, wires=[0])
>>> qml.simplify(qml.sum(qml.PauliY(3), qml.PauliY(3)))
2*(PauliY(wires=[3]))
>>> qml.simplify(qml.RX(1, 0) @ qml.RX(1, 0))
RX(2.0, wires=[0])
>>> qml.simplify(qml.prod(qml.PauliX(0), qml.PauliZ(0)))
-1j*(PauliY(wires=[0]))

Now lets simplify a nested operator:

>>> sum_op = qml.RX(1, 0) + qml.PauliX(0)
>>> prod1 = qml.PauliX(0) @ sum_op
>>> nested_op = prod1 @ qml.RX(1, 0)
>>> qml.simplify(nested_op)
(PauliX(wires=[0]) @ RX(2.0, wires=[0])) + RX(1.0, wires=[0])

Several simplifications steps are happening here. First of all, the nested products are removed:

.. code-block:: python

    qml.prod(qml.PauliX(0), qml.sum(qml.RX(1, 0), qml.PauliX(0)), qml.RX(1, 0))

Then the product of sums is transformed into a sum of products:

.. code-block:: python

    qml.sum(qml.prod(qml.PauliX(0), qml.RX(1, 0), qml.RX(1, 0)), qml.prod(qml.PauliX(0), qml.PauliX(0), qml.RX(1, 0)))

And finally like terms in the obtained products are grouped together, removing all identities: 

.. code-block:: python

    qml.sum(qml.prod(qml.PauliX(0), qml.RX(2, 0)), qml.RX(1, 0))

As mentioned earlier we can also simplify QNode objects to, for example, group rotation gates:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.simplify
    @qml.qnode(dev)
    def circuit(x):
        (
            qml.RX(x[0], wires=0)
            @ qml.RY(x[1], wires=1)
            @ qml.RZ(x[2], wires=2)
            @ qml.RX(-1, wires=0)
            @ qml.RY(-2, wires=1)
            @ qml.RZ(2, wires=2)
        )
        return qml.probs([0, 1, 2])

>>> x = [1, 2, 3]
>>> print(qml.draw(circuit)(x))
0: ───────────┤ ╭Probs
1: ───────────┤ ├Probs
2: ──RZ(5.00)─┤ ╰Probs

Compilation transforms for circuit optimization
-----------------------------------------------

PennyLane includes multiple transforms that take quantum functions and return new
quantum functions of optimized circuits:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.transforms.cancel_inverses
    ~pennylane.transforms.commute_controlled
    ~pennylane.transforms.merge_amplitude_embedding
    ~pennylane.transforms.cancel_inverses
    ~pennylane.transforms.merge_rotations
    ~pennylane.transforms.pattern_matching
    ~pennylane.transforms.remove_barrier
    ~pennylane.transforms.single_qubit_fusion
    ~pennylane.transforms.undo_swaps

:html:`</div>`

.. note::

    Most compilation transforms support just-in-time compilation with ``jax.jit``.

The :func:`~.pennylane.compile` transform allows you to chain together
sequences of quantum function transforms into custom circuit optimization pipelines.

For example, take the following decorated quantum function:

.. code-block:: python

    dev = qml.device('default.qubit', wires=[0, 1, 2])

    @qml.compile
    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.RZ(z, wires=2)
        qml.CNOT(wires=[2, 1])
        qml.RX(z, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RX(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RZ(-z, wires=2)
        qml.RX(y, wires=2)
        qml.PauliY(wires=2)
        qml.CZ(wires=[1, 2])
        return qml.expval(qml.PauliZ(wires=0))

The default behaviour of :func:`~.pennylane.compile` applies a sequence of three
transforms: :func:`~.pennylane.transforms.commute_controlled`, :func:`~.pennylane.transforms.cancel_inverses`,
and then :func:`~.pennylane.transforms.merge_rotations`.

>>> print(qml.draw(circuit)(0.2, 0.3, 0.4))
0: ──H──RX(0.60)─────────────────┤  <Z>
1: ──H─╭X─────────────────────╭●─┤     
2: ──H─╰●─────────RX(0.30)──Y─╰Z─┤     


The :func:`~.pennylane.compile` transform is flexible and accepts a custom pipeline
of quantum function transforms (you can even write your own!).
For example, if we wanted to only push single-qubit gates through
controlled gates and cancel adjacent inverses, we could do:

.. code-block:: python

    from pennylane.transforms import commute_controlled, cancel_inverses
    pipeline = [commute_controlled, cancel_inverses]

    @partial(qml.compile, pipeline=pipeline)
    @qml.qnode(dev)
    def qfunc(x, y, z):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.RZ(z, wires=2)
        qml.CNOT(wires=[2, 1])
        qml.RX(z, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RX(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RZ(-z, wires=2)
        qml.RX(y, wires=2)
        qml.PauliY(wires=2)
        qml.CZ(wires=[1, 2])
        return qml.expval(qml.PauliZ(wires=0))

>>> print(qml.draw(qfunc)(0.2, 0.3, 0.4))
0: ──H──RX(0.40)──RX(0.20)────────────────────────────┤  <Z>
1: ──H─╭X──────────────────────────────────────────╭●─┤     
2: ──H─╰●─────────RZ(0.40)──RZ(-0.40)──RX(0.30)──Y─╰Z─┤     

.. note::

    The :class:`~.pennylane.Barrier` operator can be used to prevent blocks of code from being merged during
    compilation.


For more details on :func:`~.pennylane.compile` and the available compilation transforms, visit
`the compilation documentation
<../code/qml_transforms.html#transforms-for-circuit-compilation>`_.

Custom decompositions
---------------------

PennyLane decomposes gates unknown to the device into other, "lower-level" gates. As a user, you may want to fine-tune this mechanism. For example, you may wish your circuit to use different fundamental gates.

For example, suppose we would like to implement the following QNode:

.. code-block:: python

    def circuit(weights):
        qml.BasicEntanglerLayers(weights, wires=[0, 1, 2])
        return qml.expval(qml.PauliZ(0))

    original_dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(circuit, original_dev)

>>> weights = np.array([[0.4, 0.5, 0.6]])
>>> print(qml.draw(original_qnode, expansion_strategy="device")(weights))
0: ──RX(0.40)─╭●────╭X─┤  <Z>
1: ──RX(0.50)─╰X─╭●─│──┤     
2: ──RX(0.60)────╰X─╰●─┤     

Now, let's swap out PennyLane's default decomposition of the ``CNOT`` gate into ``CZ``
and ``Hadamard``.
We define the custom decompositions like so, and pass them to a device:

.. code-block:: python

    def custom_cnot(wires):
        return [
            qml.Hadamard(wires=wires[1]),
            qml.CZ(wires=[wires[0], wires[1]]),
            qml.Hadamard(wires=wires[1])
        ]

    custom_decomps = {qml.CNOT: custom_cnot}

    decomp_dev = qml.device("default.qubit", wires=3, custom_decomps=custom_decomps)
    decomp_qnode = qml.QNode(circuit, decomp_dev)

Now when we draw or run a QNode on this device, the gates will be expanded
according to our specifications:

>>> print(qml.draw(decomp_qnode, expansion_strategy="device")(weights))
0: ──RX(0.40)────╭●──H───────╭Z──H─┤  <Z>
1: ──RX(0.50)──H─╰Z──H─╭●────│─────┤     
2: ──RX(0.60)──H───────╰Z──H─╰●────┤     

.. note::
    If the custom decomposition is only supposed to be used in a specific code context,
    a separate context manager :func:`~.pennylane.transforms.set_decomposition` can be used.

Circuit cutting
---------------

Circuit cutting allows you to replace a circuit with ``N`` wires by a set of circuits with less than
``N`` wires (see also `Peng et. al <https://arxiv.org/abs/1904.00102>`_). Of course this comes with a cost: The smaller circuits
require a greater number of device executions to be evaluated.

In PennyLane, circuit cutting can be
activated by positioning :class:`~.pennylane.WireCut` operators at the desired cut locations, and
by decorating the QNode with the :func:`~.pennylane.cut_circuit` transform.

The example below shows how a three-wire circuit can be run on a two-wire device:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.cut_circuit
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.RY(0.9, wires=1)
        qml.RX(0.3, wires=2)

        qml.CZ(wires=[0, 1])
        qml.RY(-0.4, wires=0)

        qml.WireCut(wires=1)

        qml.CZ(wires=[1, 2])

        return qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))

Instead of being executed directly, the circuit will be partitioned into
smaller fragments according to the :class:`~.pennylane.WireCut` locations,
and each fragment will be executed multiple times. PennyLane automatically combines the results
of the fragment executions to recover the expected output of the original uncut circuit.

>>> x = np.array(0.531, requires_grad=True)
>>> circuit(0.531)
0.47165198882111165

Circuit cutting support is also differentiable:

>>> qml.grad(circuit)(x)
-0.276982865449393

.. note::

    Simulated quantum circuits that produce samples can be cut using
    the :func:`~.pennylane.cut_circuit_mc`
    transform, which is based on the Monte Carlo method.

Groups of commuting Pauli words
-------------------------------

Mutually commuting Pauli words can be measured simultaneously on a quantum computer.
Finding groups of mutually commuting observables can therefore reduce the number of circuit executions,
and is an example of how observables can be "compiled".

PennyLane contains different functionalities for this purpose, ranging from higher-level
transforms acting on QNodes to lower-level functions acting on operators.

An example of a transform manipulating QNodes is :func:`~.pennylane.transforms.split_non_commuting`.
It turns a QNode that measures non-commuting observables into a QNode that internally
uses *multiple* circuit executions with qubit-wise commuting groups. The transform is used
by devices to make such measurements possible.

On a lower level, the :func:`~.pennylane.pauli.group_observables` function can be used to split lists of
observables and coefficients:

>>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
>>> coeffs = [1.43, 4.21, 0.97]
>>> obs_groupings, coeffs_groupings = qml.pauli.group_observables(obs, coeffs, 'anticommuting', 'lf')
>>> obs_groupings
[[PauliZ(wires=[1]), PauliX(wires=[0]) @ PauliX(wires=[1])],
 [PauliY(wires=[0])]]
>>> coeffs_groupings
[[0.97, 4.21], [1.43]]

This and more logic to manipulate Pauli observables is found in the :mod:`~.pennylane.pauli` module.

Just-in-time compilation with Catalyst
--------------------------------------

In addition to quantum circuit transformations, PennyLane also supports full
hybrid just-in-time (JIT) compilation via the :func:`~.qjit` decorator and
the `Catalyst hybrid compiler <https://github.com/pennylaneai/catalyst>`__.
Catalyst allows you to compile the entire quantum-classical workflow,
including any optimization loops, which allows for optimized performance, and
the ability to run the entire workflow on accelerator devices as
appropriate.

Currently, Catalyst must be installed separately, and only supports the JAX
interface and select devices such as ``lightning.qubit``,
``lightning.kokkos``, ``braket.local.qubit`` and ``braket.aws.qubit``. It
does **not** support ``default.qubit``.

On MacOS and Linux, Catalyst can be installed with ``pip``:

.. code-block:: console

    pip install pennylane-catalyst

Check out the Catalyst documentation for
:doc:`installation instructions <catalyst:dev/installation>`.

Using Catalyst with PennyLane is a simple as using the :func:`@qjit <.qjit>` decorator to
compile your hybrid workflows:

.. code-block:: python

    from jax import numpy as jnp

    dev = qml.device("lightning.qubit", wires=2, shots=1000)

    @qml.qjit
    @qml.qnode(dev)
    def cost(params):
        qml.Hadamard(0)
        qml.RX(jnp.sin(params[0]) ** 2, wires=1)
        qml.CRY(params[0], wires=[0, 1])
        qml.RX(jnp.sqrt(params[1]), wires=1)
        return qml.expval(qml.PauliZ(1))

The :func:`~.qjit` decorator can also be used on hybrid cost functions --
that is, cost functions that include both QNodes and classical processing. We
can even JIT compile the full optimization loop, for example when training
models:

.. code-block:: python

    import jaxopt

    @jax.jit
    def optimization():
        # initial parameter
        params = jnp.array([0.54, 0.3154])

        # define the optimizer
        opt = jaxopt.GradientDescent(cost, stepsize=0.4)
        update = lambda i, args: tuple(opt.update(*args))

        # perform optimization loop
        state = opt.init_state(params)
        (params, _) = jax.lax.fori_loop(0, 100, update, (params, state))

        return params

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

For more details on using the :func:`~.qjit` decorator and Catalyst
with PennyLane, please refer to the Catalyst
:doc:`quickstart guide <catalyst:dev/quick_start>`, as well as the :doc:`sharp
bits and debugging tips <catalyst:dev/sharp_bits>` page for an overview of
the differences between Catalyst and PennyLane, and how to best structure
your workflows to improve performance when using Catalyst.

To make your own compiler compatible with PennyLane, please see
the :mod:`~.compiler` module documentation.
