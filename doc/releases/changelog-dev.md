:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* The PennyLane `qchem` package is now lazily imported; it will only be imported
  the first time it is accessed.
  [(#1962)](https://github.com/PennyLaneAI/pennylane/pull/1962)

* Change all instances of `"{}".format(..)` to `f"{..}"`.
  [(#1970)](https://github.com/PennyLaneAI/pennylane/pull/1970)

* Tests do not loop over automatically imported and instantiated operations any more,

* The QNode has been re-written to support batch execution across the board,
  custom gradients, better decomposition strategies, and higher-order derivatives.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)
  [(#1969)](https://github.com/PennyLaneAI/pennylane/pull/1969)

  - Internally, if multiple circuits are generated for simultaneous execution, they
    will be packaged into a single job for execution on the device. This can lead to
    significant performance improvement when executing the QNode on remote
    quantum hardware or simulator devices with parallelization capabilities.

  - Custom gradient transforms can be specified as the differentiation method:

    ```python
    @qml.gradients.gradient_transform
    def my_gradient_transform(tape):
        ...
        return tapes, processing_fn

    @qml.qnode(dev, diff_method=my_gradient_transform)
    def circuit():
    ```

  - Arbitrary :math:`n`-th order derivatives are supported on hardware using gradient transforms
    such as the parameter-shift rule. To specify that an :math:`n`-th order derivative of a QNode
    will be computed, the `max_diff` argument should be set. By default, this is set to 1
    (first-order derivatives only). Increasing this value allows for higher order derivatives to be
    extracted, at the cost of additional (classical) computational overhead during the backwards
    pass.

  - When decomposing the circuit, the default decomposition strategy `expansion_strategy="gradient"`
    will prioritize decompositions that result in the smallest number of parametrized operations
    required to satisfy the differentiation method. While this may lead to a slight increase in
    classical processing, it significantly reduces the number of circuit evaluations needed to
    compute gradients of complicated unitaries.

    To return to the old behaviour, `expansion_strategy="device"` can be specified.

  Note that the old QNode remains accessible at `@qml.qnode_old.qnode`, however this will
  be removed in the next release.

* Tests do not loop over automatically imported and instantiated operations any more,
  which was opaque and created unnecessarily many tests.
  [(#1895)](https://github.com/PennyLaneAI/pennylane/pull/1895)

* A `decompose()` method has been added to the `Operator` class such that we can
  obtain (and queue) decompositions directly from instances of operations.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

  ```pycon
  >>> op = qml.PhaseShift(0.3, wires=0)
  >>> op.decompose()
  [RZ(0.3, wires=[0])]
  ```

* ``qml.circuit_drawer.tape_mpl`` produces a matplotlib figure and axes given a tape.
  [(#1787)](https://github.com/PennyLaneAI/pennylane/pull/1787)

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

* Added a new `qml.PauliError` channel that allows the application of an arbitrary number of Pauli operators on an arbitrary number of wires.
  [(#1781)](https://github.com/PennyLaneAI/pennylane/pull/1781)

* BasicEntanglerLayers now supports `batch_params` decorator. [(#1883)](https://github.com/PennyLaneAI/pennylane/pull/1883)

* MottonenStatePreparation now supports `batch_params` decorator. [(#1893)](https://github.com/PennyLaneAI/pennylane/pull/1893)

* CircuitDrawer now supports a `max_length` argument to help prevent text overflows when printing circuits to the CLI. [#1841](https://github.com/PennyLaneAI/pennylane/pull/1841)

* `Identity` operation is now part of both the `ops.qubit` and `ops.cv` modules.
   [(#1956)](https://github.com/PennyLaneAI/pennylane/pull/1956)

* Insert transform now supports adding operation after or before certain specific gates.
  [(#1980)](https://github.com/PennyLaneAI/pennylane/pull/1980)


<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo