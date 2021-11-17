:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

* A thermal relaxation channel is added to the Noisy channels. The channel description can be
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

* Added the identity observable to be an operator. Now we can explicitly call the identity
  operation on our quantum circuits for both qubit and CV devices.
  [(#1829)](https://github.com/PennyLaneAI/pennylane/pull/1829)

* A function for computing generalized parameter shift rules for generators'
  whose eigenvalue frequency spectrum is known is available as `qml.gradients.get_shift_rule`.
  Given a generator's frequency spectrum of `R` unique frequencies, `qml.gradients.get_shift_rule`
  returns the parameter shift rules to compute expectation value gradients of the generator's
  variational parameter using `2R` shifted cost function evaluations. This becomes cheaper than
  the standard application of the chain rule and two-term shift rule when `R` is less than the
  number of Pauli words in the Hamiltonian generator.

  For example, a four-term shift rule is generated for the frequency spectrum `[1, 2]`, which
  corresponds to a generator eigenspectrum of e.g., `[-1, 0, 1]`:

  ```pycon
  >>> frequencies = (1,2)
  >>> grad_recipe = qml.gradients.get_shift_rule(frequencies)
  >>> grad_recipe
  ([[0.8535533905932737, 1, 0.7853981633974483], [-0.14644660940672624, 1, 2.356194490192345],
    [-0.8535533905932737, 1, -0.7853981633974483], [0.14644660940672624, 1, -2.356194490192345]],)
  ```

  As we can see, `get_shift_rule` returns a tuple containing a list of four nested lists for the
  four term parameter shift rule. Each term :math:`[c_i, a_i, s_i]` specifies a term in the
  gradient reconstructed via parameter shifts as

  .. math:: \frac{\partial}{\partial\phi_k}f = \sum_{i} c_i f(a_i \phi_k + s_i).

* A circuit template for time evolution under a commuting Hamiltonian utilizing generalized
  parameter shift rules for cost function gradients is available as `qml.CommutingEvolution`.
  [(#1788)](https://github.com/PennyLaneAI/pennylane/pull/1788)

  If the template is handed a frequency spectrum during its instantiation, then `get_shift_rule`
  is internally called to obtain the general parameter shift rules which will be used when computing
  the gradient of the cost function with respect to `CommutingEvolution`'s :math:`t` parameter.
  If a frequency spectrum is not handed to `CommutingEvolution`, then cost function gradients will
  be computed in the standard manner.

  The template can be initialized within a `qnode` as:

  ```python
  import pennylane as qml

  n_wires = 2
  dev = qml.device('default.qubit', wires=n_wires)

  coeffs = [1, -1]
  obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
  hamiltonian = qml.Hamiltonian(coeffs, obs)
  frequencies = [2,4]

  @qml.qnode(dev)
  def circuit(time):
      qml.PauliX(0)
      qml.CommutingEvolution(hamiltonian, time, frequencies)
      return qml.expval(qml.PauliZ(0))
  ```

  Note that there is no internal validation that 1) the input `qml.Hamiltonian` is fully commuting
  and 2) the eigenvalue frequency spectrum is correct, since these checks become
  prohibitively expensive for large Hamiltonians.

* Added density matrix initialization gate for mixed state simulation. [(#1686)](https://github.com/PennyLaneAI/pennylane/issues/1686)

<h3>Improvements</h3>

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

* ``qml.circuit_drawer.draw_mpl`` produces a matplotlib figure and axes given a tape.
  [(#1787)](https://github.com/PennyLaneAI/pennylane/pull/1787)

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

* The static method `decomposition()`, formerly in the `Operation` class, has
  been moved to the base `Operator` class.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

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

* `ExpvalCost` now returns corrects results shape when `optimize=True` with
  shots batch.
  [(#1897)](https://github.com/PennyLaneAI/pennylane/pull/1897)

* `qml.circuit_drawer.MPLDrawer` was slightly modified to work with
  matplotlib version 3.5.
  [(#1899)](https://github.com/PennyLaneAI/pennylane/pull/1899)

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP`
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje, Olivia Di Matteo, Jalani Kanem, Shumpei Kobayashi, Robert Lang, Christina Lee, Alejandro Montanez,
Romain Moyard, Maria Schuld, Jay Soni
