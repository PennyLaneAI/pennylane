:orphan:

# Release 0.24.0-dev (development release)

<h3>New features since last release</h3>

* Operators have new attributes `ndim_params` and `batch_size`, and `QuantumTapes` have the new
  attribute `batch_size`.
  - `Operator.ndim_params` contains the expected number of dimensions per parameter of the operator,
  - `Operator.batch_size` contains the size of an additional parameter broadcasting axis, if present,
  - `QuantumTape.batch_size` contains the `batch_size` of its operations (see below).
  [(#2575)](https://github.com/PennyLaneAI/pennylane/pull/2575)

  When providing an operator with the `ndim_params` attribute, it will
  determine whether (and with which `batch_size`) its input parameter(s)
  is/are broadcasted.
  A `QuantumTape` can then infer from its operations whether it is batched.
  For this, all `Operators` in the tape must have the same `batch_size` or `batch_size=None`.
  That is, mixing broadcasted and unbroadcasted `Operators` is allowed, but mixing broadcasted
  `Operators` with differing `batch_size` is not, similar to NumPy broadcasting.

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

* The qchem openfermion-dependent tests are localized and collected in `tests.qchem.of_tests`. The
  new module `test_structure` is created to collect the tests of the `qchem.structure` module in
  one place and remove their dependency to openfermion.
  [(#2593)](https://github.com/PennyLaneAI/pennylane/pull/2593)

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

* Using `Operation.inv()` in a queuing environment no longer updates the queue's metadata, but merely updates
  the operation in place.
  [(#2596)](https://github.com/PennyLaneAI/pennylane/pull/2596)

* Sparse Hamiltonians representation has changed from COOrdinate (COO) to Compressed Sparse Row (CSR) format. The CSR representation is more performant for arithmetic operations and matrix vector products. This change decreases the `expval()` calculation time, for `qml.SparseHamiltonian`, specially for large workflows. Also, the CRS format consumes less memory for the `qml.SparseHamiltonian` storage.
[(#2561)](https://github.com/PennyLaneAI/pennylane/pull/2561)

* A new method `safe_update_info` is added to `qml.QueuingContext`. This method is substituted
  for `qml.QueuingContext.update_info` in a variety of places.
  [(#2612)](https://github.com/PennyLaneAI/pennylane/pull/2612)

* `BasisEmbedding` can accept an int as argument instead of a list of bits (optionally). Example: `qml.BasisEmbedding(4, wires = range(4))` is now equivalent to `qml.BasisEmbedding([0,1,0,0], wires = range(4))` (because 4=0b100). 
  [(#2601)](https://github.com/PennyLaneAI/pennylane/pull/2601)

* Introduced a new `is_hermitian` property to determine if an operator can be used in a measurement process.
  [(#2629)](https://github.com/PennyLaneAI/pennylane/pull/2629)
<h3>Breaking changes</h3>

* The `qml.queuing.Queue` class is now removed.
  [(#2599)](https://github.com/PennyLaneAI/pennylane/pull/2599)

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

* Fixed a bug to make `param_shift_hessian` work with QNodes in which gates marked
  as trainable do not have any impact on the QNode output.
  [(#2584)](https://github.com/PennyLaneAI/pennylane/pull/2584)

* `QNode`'s now can interpret variations on the interface name, like `"tensorflow"` or `"jax-jit"`, when requesting backpropagation. 
  [(#2591)](https://github.com/PennyLaneAI/pennylane/pull/2591)

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

Amintor Dusko, Chae-Yeun Park, Christian Gogolin, Christina Lee, David Wierichs, Edward Jiang, Guillermo Alonso-Linaje,
Jay Soni, Juan Miguel Arrazola, Maria Schuld, Mikhail Andrenkov, Soran Jahangiri, Utkarsh Azad

