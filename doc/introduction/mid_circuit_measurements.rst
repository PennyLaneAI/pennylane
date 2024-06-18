.. role:: html(raw)
   :format: html

.. _mid_circuit_measurements:

Mid-circuit measurements
========================

.. currentmodule:: pennylane.measure

PennyLane allows using measurements in the middle of a quantum circuit.
Such measurements are called mid-circuit measurements and can be used to
shape the structure of the circuit dynamically, and to gather information
about the quantum state during the circuit execution.
The function to perform a mid-circuit measurement in PennyLane is
:func:`~.pennylane.measure`, and can be used as follows:

.. code-block:: python

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def my_qnode(x, y):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        m_0 = qml.measure(1, reset=False, postselect=None)

        qml.cond(m_0, qml.RY)(y, wires=0)
        return qml.probs(wires=[0]), qml.expval(m_0)

See the following sections for details on 
:func:`~.pennylane.measure`, :func:`~.pennylane.cond`, and statistics
of mid-circuit measurements, as well as information about simulation
strategies and how to configure them.
Additional information can be found in the documentation of the individual
methods. Also consider our
`Introduction to mid-circuit measurements
<https://pennylane.ai/qml/demos/tutorial_mcm_introduction/>`__,
`How to collect statistics of mid-circuit measurements
<https://pennylane.ai/qml/demos/tutorial_how_to_collect_mcm_stats/>`__,
and `How to create dynamci circuits with mid-circuit measurements
<https://pennylane.ai/qml/demos/tutorial_how_to_create_dynamic_mcm_circuits/>`__.

Resetting wires
***************

Wires can be reused after making mid-circuit measurements. Moreover, a measured wire can be
reset to the :math:`|0 \rangle` state by setting ``reset=True`` in :func:`~.pennylane.measure`:

.. code-block:: python3

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def func():
        qml.PauliX(1)
        m_0 = qml.measure(1, reset=True)
        qml.PauliX(1)
        return qml.probs(wires=[1])

Executing this QNode:

.. code-block:: pycon

    >>> func()
    tensor([0., 1.], requires_grad=True)

Postselecting mid-circuit measurements
**************************************

PennyLane also supports postselecting on mid-circuit measurement outcomes by specifying the
``postselect`` keyword argument of :func:`~.pennylane.measure`. By default, postselection
discards outcomes that do not match the ``postselect`` argument.
For example, specifying ``postselect=1`` is equivalent to projecting the state vector onto
the :math:`|1\rangle` state, i.e., disregarding all outcomes where :math:`|0\rangle` is measured:

.. code-block:: python3

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def func(x):
        qml.RX(x, wires=0)
        m_0 = qml.measure(0, postselect=1)
        return qml.sample(wires=0)

By postselecting on ``1``, we only consider results that measured the outcome ``1``.
Executing this QNode with 10 shots yields

.. code-block:: pycon

    >>> func(np.pi / 2, shots=10)
    array([1, 1, 1, 1, 1, 1, 1])

Note that only 7 samples are returned. This is because samples that do not meet the postselection criteria are
discarded. This behaviour can be customized, see the section
:ref:`"Configuring mid-circuit measurements" <mcm_config>`.

.. note::

    Currently, postselection is only supported on :class:`~.pennylane.devices.DefaultQubit`.

Conditional operators / Dynamic circuits
----------------------------------------

Users can create conditional operators controlled on mid-circuit measurements using
:func:`~.pennylane.cond`. The condition for a conditional operator may simply be
the measured value returned by a ``measure()`` call, or we may construct a boolean
condition based on such values and pass it to ``cond()``:

.. code-block:: python

    @qml.qnode(dev)
    @qml.defer_measurements
    def qnode_conditional_op_on_zero(x, y):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        m_0 = qml.measure(1)

        qml.cond(m_0 == 0, qml.RY)(y, wires=0)
        return qml.probs(wires=[0])

    pars = np.array([0.643, 0.246], requires_grad=True)

.. code-block:: pycon

    >>> qnode_conditional_op_on_zero(*pars)
    tensor([0.88660045, 0.11339955], requires_grad=True)

For more examples, refer to the :func:`~.cond` documentation
and the `how-to on creating dynamic circuits with mid-circuit measurements
<https://pennylane.ai/qml/demos/tutorial_how_to_create_dynamic_mcm_circuits/>`__.

.. _mid_circuit_measurements_statistics:

Mid-circuit measurement statistics
----------------------------------

Statistics of mid-circuit measurements can be collected along with terminal measurement statistics.
Currently, :func:`~.counts`, :func:`~.expval`, :func:`~.probs`, :func:`~.sample`, and :func:`~.var`
are supported, and devices that currently support collecting such
statistics are :class:`~.pennylane.devices.DefaultQubit`, :class:`~.DefaultMixed`,
and :class:`~.DefaultQubitLegacy`.

.. code-block:: python3

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m_0 = qml.measure(0)
        qml.cond(m_0, qml.RY)(y, wires=1)
        return qml.probs(wires=1), qml.probs(op=m_0)

Executing this ``QNode``:

.. code-block:: pycon

    >>> func(np.pi / 2, np.pi / 4)
    (tensor([0.9267767, 0.0732233], requires_grad=True),
     tensor([0.5, 0.5], requires_grad=True))

Users can also collect statistics on mid-circuit measurements manipulated using arithmetic/boolean operators.
This works for both unary and binary operators. To see a full list of supported operators, refer to the
:func:`~.pennylane.measure` documentation. An example for collecting such statistics is shown below:

.. code-block:: python3

    import pennylane as qml

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit(phi, theta):
        qml.RX(phi, wires=0)
        m_0 = qml.measure(wires=0)
        qml.RY(theta, wires=1)
        m_1 = qml.measure(wires=1)
        return qml.sample(~m_0 - 2 * m_1)

Executing this ``QNode``:

.. code-block:: pycon

    >>> circuit(1.23, 4.56, shots=5)
    array([-1, -2,  1, -1,  1])

Collecting statistics for mid-circuit measurements manipulated using arithmetic/boolean operators is supported
with :func:`~.counts`, :func:`~.expval`, :func:`~.sample`, and :func:`~.var`.

Moreover, statistics for multiple mid-circuit measurements can be collected by passing lists of mid-circuit
measurement values to the measurement process:

.. code-block:: python3

    import pennylane as qml

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit(phi, theta):
        qml.RX(phi, wires=0)
        m_0 = qml.measure(wires=0)
        qml.RY(theta, wires=1)
        m_1 = qml.measure(wires=1)
        return qml.sample([m_0, m_1])

Executing this ``QNode``:

>>> circuit(1.23, 4.56, shots=5)
array([[0, 1],
       [1, 1],
       [0, 1],
       [0, 0],
       [1, 1]])

Collecting statistics for sequences of mid-circuit measurements is supported with :func:`~.counts`, :func:`~.probs`, and :func:`~.sample`.

.. warning::

    When collecting statistics for a list of mid-circuit measurements, arithmetic
    expressions are not supported.


Simulation of mid-circuit measurements
--------------------------------------

PennyLane currently offers three methods to simulate mid-circuit measurements
on classical computers: the deferred measurements principle, dynamic one-shot
sampling, and a tree-traversal approach. These methods differ in their memory requirements
and computational cost, as well as their compatibility with other features such as
shots and differentiation methods.
While the requirements depend on details of the simulation, the expected 
scalings are

.. role:: gr
.. role:: or
.. role:: rd

.. raw:: html

   <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
   <script>
     $(document).ready(function() {
       $('.gr').parent().parent().addClass('gr-parent');
       $('.or').parent().parent().addClass('or-parent');
       $('.rd').parent().parent().addClass('rd-parent');
     });
   </script>
   <style>
       .gr-parent {background-color:#e1eba8}
       .or-parent {background-color:#ffe096}
       .rd-parent {background-color:#ffb3b3}
       .tb { border-collapse: collapse; }
       .tb th, .tb td { padding: 1px; border: solid 1px black; vertical-align: middle; column-width: auto; }
   </style>

.. rst-class:: tb

+--------------------------+-------------------------------------------+------------------------------------------------+-------------------------------------------+----------------------------------+
| **Simulation technique** | **Memory**                                | **Time**                                       | **Differentiation support**               | **Supports shots/analytic mode** |
+==========================+===========================================+================================================+===========================================+==================================+
| Deferred measurements    | :rd:`\ ` :math:`\mathcal{O}(2^{n_{MCM}})` | :rd:`\ ` :math:`\mathcal{O}(2^{n_{MCM}})`      | :gr:`\ ` yes                              | :gr:`\ ` yes / yes               |
+--------------------------+-------------------------------------------+------------------------------------------------+-------------------------------------------+----------------------------------+
| Dynamic one-shot         | :gr:`\ ` :math:`\mathcal{O}(1)`           | :rd:`\ ` :math:`\mathcal{O}(n_{shots})`        | :or:`\ ` finite differences\ :math:`{}^1` | :or:`\ ` yes / no                |
+--------------------------+-------------------------------------------+------------------------------------------------+-------------------------------------------+----------------------------------+
| Tree-traversal           | :or:`\ ` :math:`\mathcal{O}(n_{MCM}+1)`   | :or:`\ ` :math:`\mathcal{O}(\leq 2^{n_{MCM}})` | :or:`\ ` finite differences\ :math:`{}^1` | :or:`\ ` yes / no                |
+--------------------------+-------------------------------------------+------------------------------------------------+-------------------------------------------+----------------------------------+

:math:`{}^1` In principle, parameter-shift differentiation is supported as long as no
postselection is used. Parameters within conditionally applied operations will
fall back to finite differences, so that a proper value for ``h`` should be provided, see
:func:`~.pennylane.gradients.finite_diff`.

The strengths and weaknesses of the simulation techniques differ strongly and the best
technique will depend on details of the simulation workflow. As a rule of thumb:

- dynamic one-shot sampling excels in the many-measurements-few-shots regime,

- the tree-traversal technique can handle large-scale simulations with many shots
  and measurements, and

- deferred measurements are the generalist solution that enables mid-circuit measurement
  support under (almost) all circumstances, but at large memory cost.

By default, ``QNode``\ s use deferred measurements and dynamic one-shot sampling (if supported)
when executed without and with shots, respectively.

.. _deferred_measurements:

Deferred measurements
*********************

A quantum function with mid-circuit measurements (defined using
:func:`~.pennylane.measure`) and conditional operations (defined using
:func:`~.pennylane.cond`) can be executed by applying the `deferred measurement
principle <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`__.
Accordingly, statistics of mid-circuit measurements become conventional terminal measurement
statistics.
In PennyLane this technique is available as the transform :func:`~.pennylane.defer_measurements`,
which can be applied to :class:`~.pennylane.QNode`\ s, quantum functions and tapes
as usual. It is the default technique for ``QNode``\ s that do not execute with shots.

The deferred measurement principle provides a powerful method to simulate
mid-circuit measurements, conditional operations and measurement statistics
in a differentiable and device-independent way. It adds an auxiliary qubit 
to the circuit for each mid-circuit measurement, leading to overheads of both
memory and simulation time that scale exponentially with the number of measurements.

.. code-block:: pycon

    deferred_qnode = qml.defer_measurements(my_qnode)
    pars = np.array([0.643, 0.246])
    >>> deferred_qnode(*pars)
    (tensor([0.90165331, 0.09834669], requires_grad=True),
     tensor(0.09984972, requires_grad=True))

The effect of deferring measurements becomes clear if we draw the ``QNode``
before and after applying the transform:

.. code-block:: pycon

    >>> qml.draw(my_qnode)(*pars)
    0: ──RY(0.64)─╭●───────RY(0.25)─┤  Probs
    1: ───────────╰X──┤↗├──║────────┤
                       ╚═══╩════════╡  <MCM>
    >>> qml.draw(deferred_qnode)(*pars)
    0: ──RY(0.64)─╭●────╭RY(0.25)─┤  Probs
    1: ───────────╰X─╭●─│─────────┤
    2: ──────────────╰X─╰●────────┤  <None>

Mid-circuit measurements are deferred to the end of the circuit, and conditionally applied
operations become (quantumly) controlled operations.
``qml.defer_measurements`` can be applied as decorator equally well:

.. code-block:: python

    @qml.qnode(dev)
    @qml.defer_measurements
    def qnode(x, y):
        (...)

    @qml.defer_measurements
    @qml.qnode(dev)
    def qnode(x, y):
        (...)

.. note::

    The deferred measurements principle requires an additional wire, or qubit, for each mid-circuit
    measurement, limiting the number of measurements that can be used both on classical simulators
    and quantum hardware. 

.. _one_shot_transform:

Dynamic one-shot sampling
*************************

Devices that natively support mid-circuit measurements (defined using
:func:`~.pennylane.measure`) and conditional operations (defined using
:func:`~.pennylane.cond`) can estimate dynamic circuits by executing
them one shot at a time, sampling a dynamic circuit execution path for each
shot.
This technique is the default for a shot-based
:class:`~.pennylane.QNode` that uses a device supporting mid-circuit measurements,
as well as any :class:`~.pennylane.QNode` with the
:func:`~.pennylane.dynamic_one_shot` quantum function transform.
As the name suggests, this transform only works for a :class:`~.pennylane.QNode`
executing with finite shots and it will raise an error if the device does not support
mid-circuit measurements natively.

The :func:`~.pennylane.dynamic_one_shot` transform is usually advantageous compared
with the :func:`~.pennylane.defer_measurements` transform in the
many-mid-circuit-measurements and few-shots limit. This is because, unlike the
deferred measurement principle, the method does not need an additional wire for every
mid-circuit measurement in the circuit.
The transform can be applied to a QNode as follows:

.. code-block:: python

    @qml.dynamic_one_shot
    @qml.qnode(dev)
    def my_quantum_function(x, y):
        (...)

.. warning::

    Dynamic circuits executed with shots should be differentiated with the finite difference method.

.. _tree_traversal:

Tree-traversal algorithm
************************

Dynamic circuit execution is akin to traversing a binary tree where each mid-circuit measurement
corresponds to a node and groups of gates between them correspond to edges.
The :func:`~.pennylane.dynamic_one_shot` approach above picks a branch of the tree at random
and simulates it from beginning to end.
This is wasteful in many cases; the same branch is simulated many times
when there are more shots than branches for example, and shared information between branches
is not reused.
The tree-traversal algorithm does away with such redundancy while retaining the
exponential gains in memory of the one-shot approach compared with the deferred
measurement principle, among other advantages.

The algorithm cuts an :math:`n_{MCM}` circuit into :math:`n_{MCM}+1`
circuit segments. Each segment can be executed on either the 0- or 1-branch,
which gives rise to a binary tree with :math:`2^{n_{MCM}}` leaves. Terminal
measurements are obtained at the leaves, and propagated and combined back up at each
node up the tree. The tree is explored using a depth-first pattern. The tree-traversal
method improves on :func:`~.pennylane.dynamic_one_shot` by collecting all samples at a
node or leaf at once. Neglecting overheads, simulating all branches requires the same
amount of computations as :func:`~. pennylane.defer_measurements`, but without the
:math:`O(2^{n_{MCM}})` memory requirement. To save time, a copy of the state vector
is made at every node, or mid-circuit measurement, requiring :math:`n_{MCM}+1` state
vectors, an exponential improvement compared to :func:`~.pennylane.defer_measurements`.
Since the counts of many nodes come out to be zero for shot-based simulations,
it is often possible to ignore entire sub-trees, thereby reducing the computational
cost.

To summarize, this algorithm gives us the best of both worlds. In the limit of few
shots and/or many mid-circuit measurements, it is as fast as the naive shot-by-shot
implementation of ``dynamic_one_shot`` because few sub-trees are explored.
In the limit of many shots and/or few mid-circuit measurements, it is
equal to or faster than the deferred measurement algorithm (albeit with more
overheads in practice) because each tree edge is visited at most once, all while
exponentially reducing the memory requirements.

The tree-traversal algorithm is not a transform. Its usage is therefore specified
by passing an ``mcm_method`` option to a QNode (see section
:ref:`"Configuring mid-circuit measurements" <mcm_config>`). For example,

.. code-block:: python

    @qml.qnode(dev, mcm_method="tree-traversal")
    def my_qnode(x, y):
        (...)

.. warning::

    The tree-traversal algorithm is only implemented for the
    :class:`~.pennylane.devices.DefaultQubit` device.

.. _mcm_config:

Configuring mid-circuit measurements
************************************

As seen above, there are multiple ways in which circuits with mid-circuit measurements can be executed with
PennyLane. For ease of use, we provide the following configuration options to users when initializing a
:class:`~pennylane.QNode`:

* ``mcm_method``: To set the method used for applying mid-circuit measurements. Use ``mcm_method="deferred"``
  to apply the :ref:`deferred measurements principle <deferred_measurements>`, ``mcm_method="one-shot"`` to apply
  the :ref:`one-shot transform <one_shot_transform>` or ``mcm_method="tree-traversal"`` to execute the
  :ref:`tree-traversal algorithm <tree_traversal>`.
  When executing with finite shots, ``mcm_method="one-shot"``
  will be the default, and ``mcm_method="deferred"`` otherwise. Additionally, if using :func:`~pennylane.qjit`,
  ``mcm_method="single-branch-statistics"`` can also be used and will be the default. Using this method, a single
  branch of the execution tree will be chosen randomly.

  .. warning::

      If the ``mcm_method`` argument is provided, the :func:`~pennylane.defer_measurements` or
      :func:`~pennylane.dynamic_one_shot` transforms must not be applied manually to the :class:`~pennylane.QNode`
      as this can lead to incorrect behaviour.

* ``postselect_mode``: To configure how invalid shots are handled when postselecting mid-circuit measurements
  with finite-shot circuits. Use ``postselect_mode="hw-like"`` to discard invalid samples. In this case, the number
  of samples that are used to process results can be smaller than the total number of shots. If
  ``postselect_mode="fill-shots"`` is used, the postselected value will be sampled unconditionally, and all
  samples will be valid. This is equivalent to sampling until the number of valid samples matches the total number
  of shots. The default behaviour is ``postselect_mode="hw-like"``.

  .. code-block:: python3

      import pennylane as qml
      import numpy as np

      dev = qml.device("default.qubit", wires=3, shots=10)

      def circ(x):
          qml.RX(x, 0)
          m_0 = qml.measure(0, postselect=1)
          qml.CNOT([0, 1])
          return qml.sample(qml.PauliZ(0))

      method = "one-shot"
      fill_shots_node = qml.QNode(circ, dev, mcm_method=method, postselect_mode="fill-shots")
      hw_like_node = qml.QNode(circ, dev, mcm_method=method, postselect_mode="hw-like")

  >>> fill_shots_node(np.pi / 2)
  array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
  >>> hw_like_node(np.pi / 2)
  array([-1., -1., -1., -1., -1., -1., -1.])

  .. note::

      When using the ``jax`` interface, ``postselect_mode="hw-like"`` will have different behaviour based on the
      chosen ``mcm_method``.

      * If ``mcm_method="one-shot"``, invalid shots will not be discarded. Instead, invalid samples will be replaced
        by ``np.iinfo(np.int32).min``. These invalid samples will not be used for processing final results (like
        expectation values), but will appear in the ``QNode`` output if samples are requested directly. For example:

        .. code-block:: python3

            import jax
            import jax.numpy as jnp

            key = jax.random.PRNGKey(123)
            dev = qml.device("default.qubit", wires=3, shots=10, seed=key)

            @qml.qnode(dev, postselect_mode="hw-like", mcm_method="one-shot")
            def circuit(x):
                qml.RX(x, 0)
                qml.measure(0, postselect=1)
                return qml.sample(qml.PauliZ(0))

        >>> x = jnp.array(1.8)
        >>> f(x)
        Array([-2.1474836e+09, -1.0000000e+00, -2.1474836e+09, -2.1474836e+09,
               -1.0000000e+00, -2.1474836e+09, -1.0000000e+00, -2.1474836e+09,
               -1.0000000e+00, -1.0000000e+00], dtype=float32, weak_type=True)

      * When using ``jax.jit``, using ``mcm_method="deferred"`` is not supported with ``postselect_mode="hw-like"`` and
        an error will be raised if this configuration is requested. This is due to current limitations of the
        :func:`~pennylane.defer_measurements` transform, and this behaviour will change in the future to be more
        consistent with ``mcm_method="one-shot"``.
