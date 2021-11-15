:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

* A thermal relaxation channel is added to the Noisy channels. The channel description can be
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)
  
* Added the identity observable to be an operator. Now we can explicitly call the identity 
  operation on our quantum circuits for both qubit and CV devices.
  [(#1829)](https://github.com/PennyLaneAI/pennylane/pull/1829) 

<h3>Improvements</h3>

* The QNode has been re-written to support batch execution across the board,
  custom gradients, better decomposition strategies, and higher-order derivatives.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

  - Internally, if multiple circuits are generated for execution simultaneously, they
    will be packaged into a single job for execution on the device. This can lead to
    significant performance improvement when executing the QNode on remote
    quantum hardware.

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
    compute gradients of complex unitaries.

    To return to the old behaviour, `expansion_strategy="device"` can be specified.

  Note that the old QNode remains accessible at `@qml.qnode_old.qnode`, however this will
  be removed in the next release.

* ``qml.circuit_drawer.draw_mpl`` produces a matplotlib figure and axes given a tape.
  [(#1787)](https://github.com/PennyLaneAI/pennylane/pull/1787)

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

<h3>Breaking changes</h3>

- The `mutable` keyword argument has been removed from the QNode.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

- The reversible QNode differentiation method has been removed.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

* `DiagonalOperation` is not a separate subclass any more. 
  [(#1889)](https://github.com/PennyLaneAI/pennylane/pull/1889) 

  Instead, devices can check for the diagonal 
  property using attributes:

  ``` python
  from pennylane.ops.qubit.attributes import diagonal_in_z_basis

  if op in diagonal_in_z_basis:
      # do something
  ``` 

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP` 
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order): 

Jalani Kanem, Christina Lee, Guillermo Alonso-Linaje, Alejandro Montanez, Maria Schuld, Jay Soni
