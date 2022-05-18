:orphan:

# Release 0.24.0-dev (development release)

<h3>New features since last release</h3>

* Boolean mask indexing of the parameter-shift Hessian
  [(#2538)](https://github.com/PennyLaneAI/pennylane/pull/2538)

  The `argnum` keyword argument for `param_shift_hessian` 
  is now allowed to be a twodimensional Boolean `array_like`.
  Only the indicated entries of the Hessian will then be computed.
  A particularly useful example is the computation of the diagonal
  of the Hessian:

  ```python
  dev = qml.device("default.qubit", wires=1)
  with qml.tape.QuantumTape() as tape:
      qml.RX(0.2, wires=0)
      qml.RY(-0.9, wires=0)
      qml.RX(1.1, wires=0)
      qml.expval(qml.PauliZ(0))

  argnum = qml.math.eye(3, dtype=bool)
  ```
  ```pycon
  >>> tapes, fn = qml.gradients.param_shift_hessian(tape, argnum=argnum)
  >>> fn(qml.execute(tapes, dev, None))
  array([[[-0.09928388,  0.        ,  0.        ],
        [ 0.        , -0.27633945,  0.        ],
        [ 0.        ,  0.        , -0.09928388]]])
  ```

* Speed up measuring of commuting Pauli operators
  [(#2425)](https://github.com/PennyLaneAI/pennylane/pull/2425)

  The code that checks for qubit wise commuting (QWC) got a performance boost that is noticable
  when many commuting paulis of the same type are measured.

<h3>Improvements</h3>

* The developer-facing `pow` method has been added to `Operator` with concrete implementations
  for many classes.
  [(#2225)](https://github.com/PennyLaneAI/pennylane/pull/2225)

* Test classes are created in qchem test modules to group the integrals and matrices unittests.
  [(#2545)](https://github.com/PennyLaneAI/pennylane/pull/2545)

* Introduced an `operations_only` argument to the `tape.get_parameters` method.
  [(#2543)](https://github.com/PennyLaneAI/pennylane/pull/2543)

* The `gradients` module now uses faster subroutines and uniform
  formats of gradient rules.
  [(#2452)](https://github.com/XanaduAI/pennylane/pull/2452)

* Wires can be passed as the final argument to an `Operator`, instead of requiring
  the wires to be explicitly specified with keyword `wires`. This functionality already
  existed for `Observable`'s, but now extends to all `Operator`'s.
  [(#2432)](https://github.com/PennyLaneAI/pennylane/pull/2432)

  ```pycon
  >>> qml.S(0)
  S(wires=[0])
  >>> qml.CNOT((0,1))
  CNOT(wires=[0, 1])
  ```

* Instead of checking types, objects are processed in `QuantumTape`'s based on a new `_queue_category` property.
  This is a temporary fix that will disappear in the future.
  [(#2408)](https://github.com/PennyLaneAI/pennylane/pull/2408)

* The `qml.taper` function can now be used to consistently taper any additional observables such as dipole moment,
  particle number, and spin operators using the symmetries obtained from the Hamiltonian.
  [(#2510)](https://github.com/PennyLaneAI/pennylane/pull/2510)

* The `QNode` class now contains a new method `best_method_str` that returns the best differentiation
  method for a provided device and interface, in human-readable format.
  [(#2533)](https://github.com/PennyLaneAI/pennylane/pull/2533)

<h3>Breaking changes</h3>

* The unused keyword argument `do_queue` for `Operation.adjoint` is now fully removed.
  [(#2583)](https://github.com/PennyLaneAI/pennylane/pull/2583)

* The module `qml.gradients.param_shift_hessian` has been renamed to
  `qml.gradients.parameter_shift_hessian` in order to distinguish it from the identically named
  function. Note that the `param_shift_hessian` function is unaffected by this change and can be
  invoked in the same manner as before via the `qml.gradients` module.
  [(#2528)](https://github.com/PennyLaneAI/pennylane/pull/2528)
* The properties `eigval` and `matrix` from the `Operator` class were replaced with the
  methods `eigval()` and `matrix(wire_order=None)`.
  [(#2498)](https://github.com/PennyLaneAI/pennylane/pull/2498)

* `Operator.decomposition()` is now an instance method, and no longer accepts parameters.
  [(#2498)](https://github.com/PennyLaneAI/pennylane/pull/2498)

* Adds tests, adds no-coverage directives, and removes inaccessible logic to improve code coverage.
  [(#2537)](https://github.com/PennyLaneAI/pennylane/pull/2537)

* The base classes `QubitDevice` and `DefaultQubit` now accept data-types for a statevector. This
  enables a derived class (device) in a plugin to choose correct data-types.
  [(#2448)](https://github.com/PennyLaneAI/pennylane/pull/2448)

  ```pycon
  >>> dev = qml.device("default.qubit", wires=4, r_dtype=np.float32, c_dtype=np.complex64)
  >>> dev.R_DTYPE
  <class 'numpy.float32'>
  >>> dev.C_DTYPE
  <class 'numpy.complex64'>
  ```

<h3>Bug fixes</h3>

* Fixed a bug for `diff_method="adjoint"` where incorrect gradients were
  computed for QNodes with parametrized observables (e.g., `qml.Hermitian`).
  [(#2543)](https://github.com/PennyLaneAI/pennylane/pull/2543)

* Fixed a bug where `QNGOptimizer` did not work with operators
  whose generator was a Hamiltonian.
  [(#2524)](https://github.com/PennyLaneAI/pennylane/pull/2524)

* Fixes a bug with the decomposition of `qml.CommutingEvolution`.
  [(#2542)](https://github.com/PennyLaneAI/pennylane/pull/2542)

* Fixed a bug enabling PennyLane to work with the latest version of Autoray.
  [(#2549)](https://github.com/PennyLaneAI/pennylane/pull/2549)

* Fixed a bug which caused different behaviour for `Hamiltonian @ Observable` and `Observable @ Hamiltonian`.
  [(#2570)](https://github.com/PennyLaneAI/pennylane/pull/2570)

* Fixes a bug in `DiagonalQubitUnitary._controlled` where an invalid operation was queued
  instead of the controlled version of the diagonal unitary.
  [(#2525)](https://github.com/PennyLaneAI/pennylane/pull/2525)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* The centralized [Xanadu Sphinx Theme](https://github.com/XanaduAI/xanadu-sphinx-theme)
  is now used to style the Sphinx documentation.
  [(#2450)](https://github.com/PennyLaneAI/pennylane/pull/2450)

* Added a new section in the [Gradients and Training](https://pennylane.readthedocs.io/en/stable/introduction/interfaces.html)
  page that summarizes the supported device configurations and provides justification. Also
  added [code examples](https://pennylane.readthedocs.io/en/stable/introduction/unsupported.html)
  for some selected configurations.
  [(#2540)](https://github.com/PennyLaneAI/pennylane/pull/2540)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje, Mikhail Andrenkov, Juan Miguel Arrazola, Utkarsh Azad, Christian Gogolin,
Soran Jahangiri, Edward Jiang, Christina Lee, Chae-Yeun Park, Maria Schuld, Jay Soni
