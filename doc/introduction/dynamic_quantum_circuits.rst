.. role:: html(raw)
   :format: html

.. _mid_circuit_measurements:

Dynamic quantum circuits
========================

.. currentmodule:: pennylane.measure

PennyLane allows using measurements in the middle of a quantum circuit.
Such measurements are called mid-circuit measurements and can be used to
shape the structure of the circuit dynamically, and to gather information
about the quantum state during the circuit execution.

Available features
------------------

Mid-circuit measurements
************************

The function to perform a mid-circuit measurement in PennyLane is
:func:`~.pennylane.measure`, and can be used as follows:

.. code-block:: python

    dev = qp.device("default.qubit")

    @qp.qnode(dev)
    def my_qnode(x, y):
        qp.RY(x, wires=0)
        qp.CNOT(wires=[0, 1])
        m_0 = qp.measure(1, reset=False, postselect=None)

        qp.cond(m_0, qp.RY)(y, wires=0)
        return qp.probs(wires=[0]), qp.expval(m_0)

See the following sections for details on
:func:`~.pennylane.measure`, :func:`~.pennylane.cond`, and statistics
of mid-circuit measurements, as well as information about simulation
strategies and how to configure them :ref:`further below <simulation_techniques>`.
Additional information can be found in the documentation of the individual
methods. Also consider our
`Introduction to mid-circuit measurements <demos/tutorial_mcm_introduction>`_,
`how-to on collecting statistics of mid-circuit measurements <https://pennylane.ai/qml/demos/tutorial_how_to_collect_mcm_stats>`_,
and `how-to on creating dynamic circuits with mid-circuit measurements <https://pennylane.ai/qml/demos/tutorial_how_to_create_dynamic_mcm_circuits>`_.

Resetting qubits
****************

Wires can be reused after making mid-circuit measurements. Moreover, a measured wire can be
reset to the :math:`|0 \rangle` state by setting ``reset=True`` in :func:`~.pennylane.measure`:

.. code-block:: python3

    dev = qp.device("default.qubit", wires=3)

    @qp.qnode(dev)
    def func():
        qp.PauliX(1)
        m_0 = qp.measure(1, reset=True)
        qp.PauliX(1)
        return qp.probs(wires=[1])

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

    dev = qp.device("default.qubit")

    @qp.qnode(dev)
    def func(x):
        qp.RX(x, wires=0)
        m_0 = qp.measure(0, postselect=1)
        return qp.sample(wires=0)

By postselecting on ``1``, we only consider results that measured the outcome ``1``.
Executing this QNode with 10 shots yields

.. code-block:: pycon

    >>> func(np.pi / 2, shots=10)
    array([[1],
       [1],
       [1],
       [1],
       [1]])

Note that less than 10 samples are returned. This is because samples that do not meet the postselection criteria are
discarded. This behaviour can be customized, see the section
:ref:`"Configuring mid-circuit measurements" <mcm_config>`.

Conditional operators
*********************

Users can create conditional operators controlled on mid-circuit measurements using
:func:`~.pennylane.cond`. The condition for a conditional operator may simply be
the measured value returned by a :func:`~.pennylane.measure` call, or we may construct a boolean
condition based on such values and pass it to :func:`~.pennylane.cond`:

.. code-block:: python

    @qp.qnode(dev)
    def qnode_conditional_op_on_zero(x, y):
        qp.RY(x, wires=0)
        qp.CNOT(wires=[0, 1])
        m_0 = qp.measure(1)

        qp.cond(m_0 == 0, qp.RY)(y, wires=0)
        return qp.probs(wires=[0])

    pars = np.array([0.643, 0.246], requires_grad=True)

.. code-block:: pycon

    >>> qnode_conditional_op_on_zero(*pars)
    tensor([0.88660045, 0.11339955], requires_grad=True)

For more examples, refer to the :func:`~.pennylane.cond` documentation
and the `how-to on creating dynamic circuits with mid-circuit measurements
<https://pennylane.ai/qml/demos/tutorial_how_to_create_dynamic_mcm_circuits>`_.

.. _mid_circuit_measurements_statistics:

Mid-circuit measurement statistics
**********************************

Statistics of mid-circuit measurements can be collected along with terminal measurement statistics.
Currently, :func:`~.counts`, :func:`~.expval`, :func:`~.probs`, :func:`~.sample`, and :func:`~.var`
are supported.

.. code-block:: python3

    dev = qp.device("default.qubit", wires=2)

    @qp.qnode(dev)
    def func(x, y):
        qp.RX(x, wires=0)
        m_0 = qp.measure(0)
        qp.cond(m_0, qp.RY)(y, wires=1)
        return qp.probs(wires=1), qp.probs(op=m_0)

Executing this ``QNode``:

.. code-block:: pycon

    >>> func(np.pi / 2, np.pi / 4)
    (tensor([0.9267767, 0.0732233], requires_grad=True),
     tensor([0.5, 0.5], requires_grad=True))

Users can also collect statistics on mid-circuit measurements manipulated using arithmetic/boolean operators.
This works for both unary and binary operators. To see a full list of supported operators, refer to the
:func:`~.pennylane.measure` documentation. An example for collecting such statistics is shown below:

.. code-block:: python3

    import pennylane as qp

    dev = qp.device("default.qubit")

    @qp.qnode(dev)
    def circuit(phi, theta):
        qp.RX(phi, wires=0)
        m_0 = qp.measure(wires=0)
        qp.RY(theta, wires=1)
        m_1 = qp.measure(wires=1)
        return qp.sample(~m_0 - 2 * m_1)

Executing this ``QNode``:

.. code-block:: pycon

    >>> circuit(1.23, 4.56, shots=5)
    array([-1, -2,  1, -1,  1])

Collecting statistics for mid-circuit measurements manipulated using arithmetic/boolean operators is supported
with :func:`~.counts`, :func:`~.expval`, :func:`~.sample`, and :func:`~.var`.

Moreover, statistics for multiple mid-circuit measurements can be collected by passing lists of mid-circuit
measurement values to the measurement process:

.. code-block:: python3

    import pennylane as qp

    dev = qp.device("default.qubit")

    @qp.qnode(dev)
    def circuit(phi, theta):
        qp.RX(phi, wires=0)
        m_0 = qp.measure(wires=0)
        qp.RY(theta, wires=1)
        m_1 = qp.measure(wires=1)
        return qp.sample([m_0, m_1])

Executing this ``QNode``:

.. code-block:: pycon

    >>> circuit(1.23, 4.56, shots=5)
    array([[0, 1],
           [1, 1],
           [0, 1],
           [0, 0],
           [1, 1]])

Collecting statistics for sequences of mid-circuit measurements is supported with
:func:`~.counts`, :func:`~.probs`, and :func:`~.sample`.

.. warning::

    When collecting statistics for a sequence of mid-circuit measurements, the
    sequence must not contain arithmetic combinations of more than one measurement.

.. _simulation_techniques:

Simulation techniques
---------------------

PennyLane currently offers three methods to simulate mid-circuit measurements
on classical computers: the deferred measurements principle, dynamic one-shot
sampling, and a tree-traversal approach. These methods differ in their memory requirements
and computational cost, as well as their compatibility with other features such as
shots and differentiation methods.
While the requirements depend on details of the simulation, the expected
scalings with respect to the number of mid-circuit measurements (and shots) are

.. role:: gr
.. role:: or
.. role:: rd

.. raw:: html

   <script>
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.gr').forEach(function(element) {
            if (element.parentElement && element.parentElement.parentElement) {
                element.parentElement.parentElement.classList.add('gr-parent');
            }
        });

        document.querySelectorAll('.or').forEach(function(element) {
            if (element.parentElement && element.parentElement.parentElement) {
                element.parentElement.parentElement.classList.add('or-parent');
            }
        });

        document.querySelectorAll('.rd').forEach(function(element) {
            if (element.parentElement && element.parentElement.parentElement) {
                element.parentElement.parentElement.classList.add('rd-parent');
            }
        });
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

+--------------------------+-------------------------------------------+-----------------------------------------------------------+-------------------------------------------+--------------+--------------+
| **Simulation technique** | **Memory**                                | **Time**                                                  | **Differentiation**                       | **shots**    | **analytic** |
+==========================+===========================================+===========================================================+===========================================+==============+==============+
| Deferred measurements    | :rd:`\ ` :math:`\mathcal{O}(2^{n_{MCM}})` | :rd:`\ ` :math:`\mathcal{O}(2^{n_{MCM}})`                 | :gr:`\ ` yes \ :math:`{}^1`               | :gr:`\ ` yes | :gr:`\ ` yes |
+--------------------------+-------------------------------------------+-----------------------------------------------------------+-------------------------------------------+--------------+--------------+
| Dynamic one-shot         | :gr:`\ ` :math:`\mathcal{O}(1)`           | :rd:`\ ` :math:`\mathcal{O}(n_{shots})`                   | :or:`\ ` finite differences\ :math:`{}^2` | :gr:`\ ` yes | :rd:`\ ` no  |
+--------------------------+-------------------------------------------+-----------------------------------------------------------+-------------------------------------------+--------------+--------------+
| Tree-traversal           | :or:`\ ` :math:`\mathcal{O}(n_{MCM}+1)`   | :or:`\ ` :math:`\mathcal{O}(min(n_{shots}, 2^{n_{MCM}}))` | :or:`\ ` finite differences\ :math:`{}^2` | :gr:`\ ` yes | :gr:`\ ` yes |
+--------------------------+-------------------------------------------+-----------------------------------------------------------+-------------------------------------------+--------------+--------------+


:math:`{}^1` Backpropagation and finite differences are fully supported. The adjoint method
and the parameter-shift rule are supported if no postselection is used.

:math:`{}^2` In principle, parameter-shift differentiation is supported as long as no
postselection is used. Parameters within conditionally applied operations will
fall back to finite differences, so a proper value for ``h`` should be provided (see
:func:`~.pennylane.gradients.finite_diff`).

The strengths and weaknesses of the simulation techniques differ strongly and the best
technique will depend on details of the simulation workflow. As a rule of thumb:

- dynamic one-shot sampling excels in the many-measurements-few-shots regime,

- the tree-traversal technique can handle large-scale simulations with many shots
  and measurements, and

- deferred measurements are the generalist solution that enables mid-circuit measurement
  support under (almost) all circumstances, but at large memory cost. It is the only method
  supporting analytic simulations.

The method can be configured with the keyword argument ``mcm_method`` at ``QNode`` creation
(see :ref:`"Configuring mid-circuit measurements" <mcm_config>`).

.. _deferred_measurements:

Deferred measurements
*********************

A quantum function with mid-circuit measurements can be executed via the
`deferred measurement principle <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`__.
In PennyLane, this technique is available via ``mcm_method="deferred"`` or as the
transform :func:`~.pennylane.defer_measurements`.

The deferred measurement principle provides a powerful method to simulate
mid-circuit measurements, conditional operations and measurement statistics
in a differentiable and device-independent way. It adds an auxiliary qubit
to the circuit for each mid-circuit measurement, leading to overheads of both
memory and simulation time that scale exponentially with the number of measurements.

.. code-block:: pycon

    >>> deferred_qnode = qp.defer_measurements(my_qnode)
    >>> pars = np.array([0.643, 0.246])
    >>> deferred_qnode(*pars)
    (tensor([0.90165331, 0.09834669], requires_grad=True),
     tensor(0.09984972, requires_grad=True))

The effect of deferring measurements becomes clear if we draw the ``QNode``
before and after applying the transform:

.. code-block:: pycon

    >>> print(qp.draw(my_qnode)(*pars))
    0: ──RY(0.64)─╭●───────RY(0.25)─┤  Probs
    1: ───────────╰X──┤↗├──║────────┤
                       ╚═══╩════════╡  <MCM>
    >>> print(qp.draw(deferred_qnode)(*pars))
    0: ──RY(0.64)─╭●────╭RY(0.25)─┤  Probs
    1: ───────────╰X─╭●─│─────────┤
    2: ──────────────╰X─╰●────────┤  <None>

Mid-circuit measurements are deferred to the end of the circuit, and conditionally applied
operations become (quantumly) controlled operations.

.. note::

    This method requires an additional qubit for each mid-circuit measurement, which limits
    the number of measurements that can be used both on classical simulators and quantum hardware.

    Postselection with deferred measurements is only supported on
    :class:`~.pennylane.devices.DefaultQubit`.


.. _one_shot_transform:

Dynamic one-shot sampling
*************************

Devices that natively support mid-circuit measurements can evaluate dynamic circuits
by executing them one shot at a time, sampling a dynamic execution path for each shot.

In PennyLane, this technique is available via the QNode argument ``mcm_method="one-shot"``
or as the transform :func:`~.pennylane.dynamic_one_shot`.
As the name suggests, this transform only works for a :class:`~.pennylane.QNode` executing
with finite shots and it requires the device to support mid-circuit measurements natively.

The :func:`~.pennylane.dynamic_one_shot` transform is usually advantageous compared
with the :func:`~.pennylane.defer_measurements` transform in the
many-mid-circuit-measurements and few-shots limit. This is because, unlike the
deferred measurement principle, the method does not need an additional wire for every
mid-circuit measurement in the circuit.

.. warning::

    Dynamic circuits executed with shots should be differentiated with the finite difference method.

.. _tree_traversal:

Tree-traversal algorithm
************************

Dynamic circuit execution is akin to traversing a binary tree where each mid-circuit measurement
corresponds to a node and gates between them correspond to edges. The tree-traversal algorithm
explores this tree depth-first. It improves upon the dynamic one-shot approach above, which
simulates a randomly chosen branch from beginning to end for each shot, by collecting all
samples at a node or leaf at once.

In PennyLane, this technique is available via the QNode argument ``mcm_method="tree-traversal"``;
it is not a transform.

The tree-traversal algorithm combines the exponential savings of memory of the one-shot
approach with sampling efficiency of deferred measurements.
Neglecting overheads, simulating all branches requires the same
amount of computations as :func:`~.pennylane.defer_measurements`, but without the
:math:`O(2^{n_{MCM}})` memory cost. To save time, a copy of the state vector
is made at every mid-circuit measurement, requiring :math:`n_{MCM}+1` state
vectors, an exponential improvement over :func:`~.pennylane.defer_measurements`.
Since the counts of many nodes come out to be zero for shot-based simulations,
it is often possible to ignore entire sub-trees, thereby reducing the computational
cost.

.. note::

    The tree-traversal algorithm is supported by the following devices:

    * `default.qubit <https://pennylane.ai/devices/default-qubit>`_,

    * `lightning.qubit <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`_,

    * `lightning.gpu <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`_

    * `lightning.kokkos <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/device.html>`_,

    Just-in-time (JIT) compilation is not available on ``DefaultQubit`` with ``shots=None``.

.. _mcm_config:

Configuring mid-circuit measurements
************************************

As described above, there are multiple simulation techniques for circuits with
mid-circuit measurements in PennyLane. They can be configured when initializing a
:class:`~pennylane.QNode`, using the following keywords:

* ``mcm_method``: Sets the method used for applying mid-circuit measurements.
  The three techniques described above can be specified with ``"deferred"``,
  ``"one-shot"``, and ``"tree-traversal"``. When using :func:`~pennylane.qjit`,
  there is the additional option ``"single-branch-statistics"``, which
  explores a single branch of the execution tree at random. If not provided,
  the method is selected by the device. For ``default.qubit`` and ``lightning.qubit``,
  the ``"one-shot"`` method is the default for finite-shots execution, and 
  ``"deferred"`` is the default for analytic execution, i.e., when ``shots=None``.

* ``postselect_mode``: Configures how invalid shots are handled when postselecting
  mid-circuit measurements with finite-shot circuits. Use ``"hw-like"`` to discard invalid samples.
  In this case, fewer than the total number of shots may be used to process results. Use
  ``"fill-shots"`` to sample the postselected value unconditionally, creating valid samples
  only. This is equivalent to sampling until the number of valid
  samples matches the total number of shots. The default is ``"hw-like"``.

  .. code-block:: python3

      dev = qp.device("default.qubit", wires=3)

      def circ():
          qp.Hadamard(0)
          m_0 = qp.measure(0, postselect=1)
          return qp.sample(qp.PauliZ(0))

      fill_shots = qp.QNode(circ, dev, mcm_method="deferred", postselect_mode="fill-shots")
      hw_like = qp.QNode(circ, dev, mcm_method="deferred", postselect_mode="hw-like")
      fill_shots = qp.set_shots(fill_shots, shots=10)
      hw_like = qp.set_shots(hw_like, shots=10)

  .. code-block:: pycon

      >>> fill_shots()
      array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
      >>> hw_like()
      array([-1., -1., -1., -1., -1., -1., -1.])

  .. note::

      When using the ``jax`` interface, the postselection mode ``"hw-like"`` will change
      behaviour with the simulation technique.

      * For dynamic one-shot, invalid shots will not be discarded, but will be replaced
        by ``np.iinfo(np.int32).min``. They will not be used for processing final results (like
        expectation values), but they will appear in the output of ``QNode``\ s that return
        samples directly.

      * When using ``jax.jit``, the combination ``"deferred"`` and ``"hw-like"`` is not supported,
        due to limitations of the :func:`~pennylane.defer_measurements` transform. This behaviour
        will change in the future.

