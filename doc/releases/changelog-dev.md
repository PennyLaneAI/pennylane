# Release 0.45.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The ``qml.estimator.Resources`` class now has a nice string representation in Jupyter Notebooks.
  [(#8880)](https://github.com/PennyLaneAI/pennylane/pull/8880)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Removed all of the resource estimation functionality from the ``/labs/resource_estimation``
  module. Users can now directly access a more stable version of this functionality using the 
  :mod:`estimator <pennylane.estimator>` module. All experimental development of resource estimation
  will be added to ``/labs/estimator_beta``
  [(#8868)](https://github.com/PennyLaneAI/pennylane/pull/8868)

* The integration test for computing perturbation error of a compressed double-factorized (CDF)
  Hamiltonian in ``labs.trotter_error`` is upgraded to use a more realistic molecular geometry and
  a more reliable reference error.
  [(#8790)](https://github.com/PennyLaneAI/pennylane/pull/8790)

<h3>Breaking changes 💔</h3>

* The ``custom_decomps`` keyword argument to ``qml.device`` has been removed in 0.45. Instead, 
  with ``qml.decomposition.enable_graph()``, new decomposition rules can be defined as
  quantum functions with registered resources. See :mod:`pennylane.decomposition` for more details.
  [(#8928)](https://github.com/PennyLaneAI/pennylane/pull/8928)

  As an example, consider the case of running the following circuit on a device that does not natively support ``CNOT`` gates:

  .. code-block:: python3


    def circuit():
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.X(1))

  Instead of defining the ``CNOT`` decomposition as:

  .. code-block:: python3

    def custom_cnot(wires):
      return [
          qml.Hadamard(wires=wires[1]),
          qml.CZ(wires=[wires[0], wires[1]]),
          qml.Hadamard(wires=wires[1])
      ]

    dev = qml.device('default.qubit', wires=2, custom_decomps={"CNOT" : custom_cnot})
    qnode = qml.QNode(circuit, dev)
    print(qml.draw(qnode, level="device")())

  The same result would now be obtained using:

  .. code-block:: python3

    @qml.decomposition.register_resources({
        qml.H: 2,
        qml.CZ: 1
    })
    def _custom_cnot_decomposition(wires, **_):
      qml.Hadamard(wires=wires[1])
      qml.CZ(wires=[wires[0], wires[1]])
      qml.Hadamard(wires=wires[1])

    qml.decomposition.add_decomps(qml.CNOT, _custom_cnot_decomposition)

    qml.decomposition.enable_graph()

    @qml.transforms.decompose(gate_set={qml.CZ, qml.H})
    def circuit():
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.X(1))

    dev = qml.device('default.qubit', wires=2)
    qnode = qml.QNode(circuit, dev)

  >>> print(qml.draw(qnode, level="device")())
  0: ────╭●────┤
  1: ──H─╰Z──H─┤  <X>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixes an issue when binding a transform when the first positional arg
  is a `Sequence`, but not a `Sequence` of tapes.
  [(#8920)](https://github.com/PennyLaneAI/pennylane/pull/8920)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Marcus Edwards,
Omkar Sarkar,
Jay Soni
