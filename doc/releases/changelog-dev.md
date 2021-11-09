:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>
* A thermal relaxation channel is added to the Noisy channels. The channel description can be
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP`
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)
  
* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

* A `lru_cache` decorated function for computing generalized parameter shift rules for generators'
  whose eigenvalue frequency spectrum is known is available as `qml.gradients.get_shift_rule`.
  Given a generator's frequency spectrum of `R` unique frequencies, `qml.gradients.get_shift_rule`
  returns the parameter shift rules to compute expectation value gradients of the generator's
  variational parameter using `2R` shifted cost function evaluations. This becomes cheaper than
  the standard application of the chain rule and two-term shift rule when `R` is less than the
  number of Pauli words in the Hamiltonian generator.
  [(#1788)](https://github.com/PennyLaneAI/pennylane/pull/1788)

  For example, a four-term shift rule is generated for the frequency spectrum `[1, 2]`, which
  corresponds to a generator eigenspectrum of e.g. `[-1, 0, 1]`:

  ```python
  import pennylane as qml

  frequencies = (1,2)
  grad_recipe = qml.gradients.get_shift_rule(frequencies)
  ```

  which gives:

  ```pycon
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
  If the template is handed a frequency spectrum during its instantiation, then `get_shift_rule`
  is internally called to obtain the general parameter shift rules which will be used when computing
  the gradient of the cost function with respect to `CommutingEvolution`'s :math:`t` parameter.
  If a frequency spectrum is not handed to `CommutingEvolution`, then cost function gradients will
  be computed in the standard manner. [(#1788)](https://github.com/PennyLaneAI/pennylane/pull/1788)

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

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order): 

Josh Izaac, Jalani Kanem, Robert Lang, Christina Lee, Guillermo Alonso-Linaje, Cedric Lin, Alejandro Montanez, David Wierichs.

