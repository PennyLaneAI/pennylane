:orphan:

# Release 0.39.0-dev (development release)

<h3>New features since last release</h3>

* Added `show_wire_labels` option to `draw` and `draw_mpl`, which hides wire labels when set to `False`.
  Defaults to `True`.
  [(#6410)](https://github.com/PennyLaneAI/pennylane/pull/6410)

* Introduced `sample_probs` function for the `qml.devices.qubit` and `qml.devices.qutrit_mixed` modules:
  - This function takes probability distributions as input and returns sampled outcomes.
  - Simplifies the sampling process by separating it from other operations in the measurement chain.
  - Improves modularity: The same code can be easily adapted for other devices (e.g., a potential `default_mixed` device).
  - Enhances maintainability by isolating the sampling logic.
  [(#6354)](https://github.com/PennyLaneAI/pennylane/pull/6354)
  
* `qml.transforms.decompose` is added for stepping through decompositions to a target gate set. 
  [(#6334)](https://github.com/PennyLaneAI/pennylane/pull/6334)

* Added `process_density_matrix` implementations to 5 `StateMeasurement` subclasses:
  `ExpVal`, `Var`, `Purity`, `MutualInformation`, and `VnEntropy`.
  This enables `process_density_matrix` to be an abstract method in `StateMeasurement`,
  facilitating future support for mixed-state devices and expanded density matrix operations. Also, there is a quick fix for the `np.sqrt` call in the `ProbabilityMP` class to be replaced by `qml.math.sqrt`.
  [(#6330)](https://github.com/PennyLaneAI/pennylane/pull/6330)

* A new class `MomentumQNGOptimizer` is added. It inherits the basic `QNGOptimizer` class and requires one additional hyperparameter (the momentum coefficient) :math:`0 \leq \rho < 1`, the default value being :math:`\rho=0.9`. For :math:`\rho=0` Momentum-QNG reduces to the basic QNG.
  [(#6240)](https://github.com/PennyLaneAI/pennylane/pull/6240)
 
* Function is added for generating the spin Hamiltonian for the
  [Kitaev](https://arxiv.org/abs/cond-mat/0506438) model on a lattice.
  [(#6174)](https://github.com/PennyLaneAI/pennylane/pull/6174)

* Function is added for generating the spin Hamiltonians for custom lattices.
  [(#6226)](https://github.com/PennyLaneAI/pennylane/pull/6226)

* Functions are added for generating spin Hamiltonians for [Emery]
  (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.58.2794) and
  [Haldane](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.61.2015) models on a lattice.
  [(#6201)](https://github.com/PennyLaneAI/pennylane/pull/6201/)

* A `has_sparse_matrix` property is added to `Operator` to indicate whether a sparse matrix is defined.
  [(#6278)](https://github.com/PennyLaneAI/pennylane/pull/6278)
  [(#6310)](https://github.com/PennyLaneAI/pennylane/pull/6310)

<h3>Improvements 🛠</h3>

* RTD support for `qml.labs` added to API.
  [(#6397)](https://github.com/PennyLaneAI/pennylane/pull/6397)

* Module-level sandboxing added to `qml.labs` via pre-commit hooks.
  [(#6369)](https://github.com/PennyLaneAI/pennylane/pull/6369)

* `qml.matrix` now works with empty objects (such as empty tapes, `QNode`s and quantum functions that do
  not call operations, single operators with empty decompositions).
  [(#6347)](https://github.com/PennyLaneAI/pennylane/pull/6347)
  
* PennyLane is now compatible with NumPy 2.0.
  [(#6061)](https://github.com/PennyLaneAI/pennylane/pull/6061)
  [(#6258)](https://github.com/PennyLaneAI/pennylane/pull/6258)
  [(#6342)](https://github.com/PennyLaneAI/pennylane/pull/6342)

* PennyLane is now compatible with Jax 0.4.28.
  [(#6255)](https://github.com/PennyLaneAI/pennylane/pull/6255)

* `qml.qchem.excitations` now optionally returns fermionic operators.
  [(#6171)](https://github.com/PennyLaneAI/pennylane/pull/6171)

* The `diagonalize_measurements` transform now uses a more efficient method of diagonalization
  when possible, based on the `pauli_rep` of the relevant observables.
  [(#6113)](https://github.com/PennyLaneAI/pennylane/pull/6113/)

* The `QuantumScript.copy` method now takes `operations`, `measurements`, `shots` and 
  `trainable_params` as keyword arguments. If any of these are passed when copying a 
  tape, the specified attributes will replace the copied attributes on the new tape.
  [(#6285)](https://github.com/PennyLaneAI/pennylane/pull/6285)
  [(#6363)](https://github.com/PennyLaneAI/pennylane/pull/6363)

* Datasets are now downloaded via Dataset API.
  [(#6126)](https://github.com/PennyLaneAI/pennylane/pull/6126)

* The `Hermitian` operator now has a `compute_sparse_matrix` implementation.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* All PL templates are now unit tested to ensure JIT compatibility.
  [(#6309)](https://github.com/PennyLaneAI/pennylane/pull/6309)

* `qml.QutritBasisStatePreparation` is now JIT compatible.
  [(#6308)](https://github.com/PennyLaneAI/pennylane/pull/6308)

* `qml.AmplitudeAmplification` is now compatible with QJIT.
  [(#6306)](https://github.com/PennyLaneAI/pennylane/pull/6306)

* The quantum arithmetic templates are now QJIT compatible.
  [(#6307)](https://github.com/PennyLaneAI/pennylane/pull/6307)
  
* The `qml.Qubitization` template is now QJIT compatible.
  [(#6305)](https://github.com/PennyLaneAI/pennylane/pull/6305)

* When an observable is repeated on a tape, `tape.diagonalizing_gates` no longer returns the 
  diagonalizing gates for each instance of the observable. Instead, the diagonalizing gates of
  each observable on the tape are included just once.
  [(#6288)](https://github.com/PennyLaneAI/pennylane/pull/6288)

* The number of diagonalizing gates returned in `qml.specs` now follows the `level` keyword argument 
  regarding whether the diagonalizing gates are modified by device, instead of always counting 
  unprocessed diagonalizing gates.
  [(#6290)](https://github.com/PennyLaneAI/pennylane/pull/6290)

* A more sensible error message is raised from a `RecursionError` encountered when accessing properties and methods of a nested `CompositeOp` or `SProd`.
  [(#6375)](https://github.com/PennyLaneAI/pennylane/pull/6375)

<h4>Capturing and representing hybrid programs</h4>

* `qml.wires.Wires` now accepts JAX arrays as input. Furthermore, a `FutureWarning` is no longer raised in `JAX 0.4.30+`
  when providing JAX tracers as input to `qml.wires.Wires`.
  [(#6312)](https://github.com/PennyLaneAI/pennylane/pull/6312)

* Differentiation of hybrid programs via `qml.grad` and `qml.jacobian` can now be captured
  into plxpr. When evaluating a captured `qml.grad` (`qml.jacobian`) instruction, it will
  dispatch to `jax.grad` (`jax.jacobian`), which differs from the Autograd implementation
  without capture. Pytree inputs and outputs are supported.
  [(#6120)](https://github.com/PennyLaneAI/pennylane/pull/6120)
  [(#6127)](https://github.com/PennyLaneAI/pennylane/pull/6127)
  [(#6134)](https://github.com/PennyLaneAI/pennylane/pull/6134)

* Improve unit testing for capturing of nested control flows.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via
  `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

* All higher order primitives now use `jax.core.Jaxpr` as metadata instead of sometimes
  using `jax.core.ClosedJaxpr` and sometimes using `jax.core.Jaxpr`.
  [(#6319)](https://github.com/PennyLaneAI/pennylane/pull/6319)

* `FermiWord` class now has a method to apply anti-commutator relations.
   [(#6196)](https://github.com/PennyLaneAI/pennylane/pull/6196)

* `FermiWord` and `FermiSentence` classes now have methods to compute adjoints.
  [(#6166)](https://github.com/PennyLaneAI/pennylane/pull/6166)

* The `SampleMP.process_samples` method is updated to support using JAX tracers
  for samples, allowing compatiblity with Catalyst workflows.
  [(#6211)](https://github.com/PennyLaneAI/pennylane/pull/6211)

* Improve `qml.Qubitization` decomposition.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The `__repr__` methods for `FermiWord` and `FermiSentence` now returns a
  unique representation of the object.
  [(#6167)](https://github.com/PennyLaneAI/pennylane/pull/6167)

* Predefined lattice shapes such as `lieb`, `cubic`, `bcc`, `fcc`, and `diamond`
  can now be generated.
  [(6237)](https://github.com/PennyLaneAI/pennylane/pull/6237)

* A `ReferenceQubit` is introduced for testing purposes and as a reference for future plugin development.
  [(#6181)](https://github.com/PennyLaneAI/pennylane/pull/6181)

* The `to_mat` methods for `FermiWord` and `FermiSentence` now optionally return
  a sparse matrix.
  [(#6173)](https://github.com/PennyLaneAI/pennylane/pull/6173)

* The `make_plxpr` function is added, to take a function and create a `Callable` that,
  when called, will return a PLxPR representation of the input function.
  [(#6326)](https://github.com/PennyLaneAI/pennylane/pull/6326)

<h3>Breaking changes 💔</h3>

* `AllWires` validation in `QNode.construct` has been removed. 
  [(#6373)](https://github.com/PennyLaneAI/pennylane/pull/6373)

* The `simplify` argument in `qml.Hamiltonian` and `qml.ops.LinearCombination` has been removed.
  Instead, `qml.simplify()` can be called on the constructed operator.
  [(#6279)](https://github.com/PennyLaneAI/pennylane/pull/6279)

* The functions `qml.qinfo.classical_fisher` and `qml.qinfo.quantum_fisher` have been removed and migrated to the `qml.gradients`
  module. Therefore, `qml.gradients.classical_fisher` and `qml.gradients.quantum_fisher` should be used instead.
  [(#5911)](https://github.com/PennyLaneAI/pennylane/pull/5911)

* Remove support for Python 3.9.
  [(#6223)](https://github.com/PennyLaneAI/pennylane/pull/6223)

* `DefaultQubitTF`, `DefaultQubitTorch`, `DefaultQubitJax`, and `DefaultQubitAutograd` are removed.
  Please use `default.qubit` for all interfaces.
  [(#6207)](https://github.com/PennyLaneAI/pennylane/pull/6207)
  [(#6208)](https://github.com/PennyLaneAI/pennylane/pull/6208)
  [(#6209)](https://github.com/PennyLaneAI/pennylane/pull/6209)
  [(#6210)](https://github.com/PennyLaneAI/pennylane/pull/6210)

* `expand_fn`, `max_expansion`, `override_shots`, and `device_batch_transform` are removed from the
  signature of `qml.execute`.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `max_expansion` and `expansion_strategy` are removed from the `QNode`.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `expansion_strategy` is removed from `qml.draw`, `qml.draw_mpl`, and `qml.specs`. `max_expansion` is removed from `qml.specs`, as it had no impact on the output.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `qml.transforms.hamiltonian_expand` and `qml.transforms.sum_expand` are removed.
  Please use `qml.transforms.split_non_commuting` instead.
  [(#6204)](https://github.com/PennyLaneAI/pennylane/pull/6204)

* The `decomp_depth` keyword argument to `qml.device` is removed.
  [(#6234)](https://github.com/PennyLaneAI/pennylane/pull/6234)

* `Operator.expand` is now removed. Use `qml.tape.QuantumScript(op.deocomposition())` instead.
  [(#6227)](https://github.com/PennyLaneAI/pennylane/pull/6227)

* Native folding method `qml.transforms.fold_global` for `qml.transforms.mitiagte_with_zne`
  transform no longer expands the circuit automatically. Instead, the user should apply `qml.transforms.decompose` to
  decompose a circuit into a target gate set before applying `fold_global` or `mitigate_with_zne`.
  [(#6382)](https://github.com/PennyLaneAI/pennylane/pull/6382)

<h3>Deprecations 👋</h3>

* The `expand_depth` and `max_expansion` arguments for `qml.transforms.compile` and
  `qml.transforms.decompositions.clifford_t_decomposition` respectively have been deprecated.
  [(#6404)](https://github.com/PennyLaneAI/pennylane/pull/6404)

* Legacy operator arithmetic has been deprecated. This includes `qml.ops.Hamiltonian`, `qml.operation.Tensor`,
  `qml.operation.enable_new_opmath`, `qml.operation.disable_new_opmath`, and `qml.operation.convert_to_legacy_H`.
  Note that when new operator arithmetic is enabled, ``qml.Hamiltonian`` will continue to dispatch to
  `qml.ops.LinearCombination`; this behaviour is not deprecated. For more information, check out the
  [updated operator troubleshooting page](https://docs.pennylane.ai/en/stable/news/new_opmath.html).
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)
  [(#6365)](https://github.com/PennyLaneAI/pennylane/pull/6365)

* `qml.pauli.PauliSentence.hamiltonian` and `qml.pauli.PauliWord.hamiltonian` are deprecated. Instead, please use
  `qml.pauli.PauliSentence.operation` and `qml.pauli.PauliWord.operation` respectively.
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)

* `qml.pauli.simplify()` is deprecated. Instead, please use `qml.simplify(op)` or `op.simplify()`.
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)

* The `qml.BasisStatePreparation` template is deprecated.
  Instead, use `qml.BasisState`.
  [(#6021)](https://github.com/PennyLaneAI/pennylane/pull/6021)

* The `'ancilla'` argument for `qml.iterative_qpe` has been deprecated. Instead, use the `'aux_wire'` argument.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* `qml.shadows.shadow_expval` has been deprecated. Instead, use the `qml.shadow_expval` measurement
  process.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* `qml.broadcast` has been deprecated. Please use `for` loops instead.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* The `qml.QubitStateVector` template is deprecated. Instead, use `qml.StatePrep`.
  [(#6172)](https://github.com/PennyLaneAI/pennylane/pull/6172)

* The `qml.qinfo` module has been deprecated. Please see the respective functions in the `qml.math` and
  `qml.measurements` modules instead.
  [(#5911)](https://github.com/PennyLaneAI/pennylane/pull/5911)

* `Device`, `QubitDevice`, and `QutritDevice` will no longer be accessible via top-level import in v0.40.
  They will still be accessible as `qml.devices.LegacyDevice`, `qml.devices.QubitDevice`, and `qml.devices.QutritDevice`
  respectively.
  [(#6238)](https://github.com/PennyLaneAI/pennylane/pull/6238/)

* `QNode.gradient_fn` is deprecated. Please use `QNode.diff_method` and `QNode.get_gradient_fn` instead.
  [(#6244)](https://github.com/PennyLaneAI/pennylane/pull/6244)

<h3>Documentation 📝</h3>

* Updated links to PennyLane.ai in the documentation to use the latest URL format, which excludes the `.html` prefix.
  [(#6412)](https://github.com/PennyLaneAI/pennylane/pull/6412)

* Update `qml.Qubitization` documentation based on new decomposition.
  [(#6276)](https://github.com/PennyLaneAI/pennylane/pull/6276)

* Fixed examples in the documentation of a few optimizers.
  [(#6303)](https://github.com/PennyLaneAI/pennylane/pull/6303)
  [(#6315)](https://github.com/PennyLaneAI/pennylane/pull/6315)

* Corrected examples in the documentation of `qml.jacobian`.
  [(#6283)](https://github.com/PennyLaneAI/pennylane/pull/6283)
  [(#6315)](https://github.com/PennyLaneAI/pennylane/pull/6315)

* Fixed spelling in a number of places across the documentation.
  [(#6280)](https://github.com/PennyLaneAI/pennylane/pull/6280)

* Add `work_wires` parameter to `qml.MultiControlledX` docstring signature.
  [(#6271)](https://github.com/PennyLaneAI/pennylane/pull/6271)

* Removed ambiguity in error raised by the `PauliRot` class.
  [(#6298)](https://github.com/PennyLaneAI/pennylane/pull/6298)

* Renamed an incorrectly named test in `test_pow_ops.py`.
  [(#6388)](https://github.com/PennyLaneAI/pennylane/pull/6388)

<h3>Bug fixes 🐛</h3>

* `default.tensor` can now handle mid circuit measurements via the deferred measurement principle.
  [(#6408)](https://github.com/PennyLaneAI/pennylane/pull/6408)

* The `validate_device_wires` transform now raises an error if abstract wires are provided.
  [(#6405)](https://github.com/PennyLaneAI/pennylane/pull/6405)

* Fixes `qml.math.expand_matrix` for qutrit and arbitrary qudit operators.
  [(#6398)](https://github.com/PennyLaneAI/pennylane/pull/6398/)

* `MeasurementValue` now raises an error when it is used as a boolean.
  [(#6386)](https://github.com/PennyLaneAI/pennylane/pull/6386)

* `default.qutrit` now returns integer samples.
  [(#6385)](https://github.com/PennyLaneAI/pennylane/pull/6385)

* `adjoint_metric_tensor` now works with circuits containing state preparation operations.
  [(#6358)](https://github.com/PennyLaneAI/pennylane/pull/6358)

* `quantum_fisher` now respects the classical Jacobian of QNodes.
  [(#6350)](https://github.com/PennyLaneAI/pennylane/pull/6350)

* `qml.map_wires` can now be applied to a batch of tapes.
  [(#6295)](https://github.com/PennyLaneAI/pennylane/pull/6295)

* Fix float-to-complex casting in various places across PennyLane.
 [(#6260)](https://github.com/PennyLaneAI/pennylane/pull/6260)
 [(#6268)](https://github.com/PennyLaneAI/pennylane/pull/6268)

* Fix a bug where zero-valued JVPs were calculated wrongly in the presence of shot vectors.
  [(#6219)](https://github.com/PennyLaneAI/pennylane/pull/6219)

* Fix `qml.PrepSelPrep` template to work with `torch`.
  [(#6191)](https://github.com/PennyLaneAI/pennylane/pull/6191)

* Now `qml.equal` compares correctly `qml.PrepSelPrep` operators.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The `qml.QSVT` template now orders the `projector` wires first and the `UA` wires second, which is the expected order of the decomposition.
  [(#6212)](https://github.com/PennyLaneAI/pennylane/pull/6212)

* The `qml.Qubitization` template now orders the `control` wires first and the `hamiltonian` wires second, which is the expected according to other templates.
  [(#6229)](https://github.com/PennyLaneAI/pennylane/pull/6229)

* The `qml.FABLE` template now returns the correct value when JIT is enabled.
  [(#6263)](https://github.com/PennyLaneAI/pennylane/pull/6263)

* Fixes a bug where a circuit using the `autograd` interface sometimes returns nested values that are not of the `autograd` interface.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* Fixes a bug where a simple circuit with no parameters or only builtin/numpy arrays as parameters returns autograd tensors.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* `qml.pauli.PauliVSpace` now uses a more stable SVD-based linear independence check to avoid running into `LinAlgError: Singular matrix`. This stabilizes the usage of `qml.lie_closure`. It also introduces normalization of the basis vector's internal representation `_M` to avoid exploding coefficients.
  [(#6232)](https://github.com/PennyLaneAI/pennylane/pull/6232)

* Fixes a bug where `csc_dot_product` is used during measurement for `Sum`/`Hamiltonian` that contains observables that does not define a sparse matrix.
  [(#6278)](https://github.com/PennyLaneAI/pennylane/pull/6278)
  [(#6310)](https://github.com/PennyLaneAI/pennylane/pull/6310)

* Fixes a bug where `None` was added to the wires in `qml.PhaseAdder`, `qml.Adder` and `qml.OutAdder`.
  [(#6360)](https://github.com/PennyLaneAI/pennylane/pull/6360)

* Fixes a test after updating to the nightly version of Catalyst.
  [(#6362)](https://github.com/PennyLaneAI/pennylane/pull/6362)

* Fixes a bug where `CommutingEvolution` with a trainable `Hamiltonian` cannot be differentiated using parameter shift.
  [(#6372)](https://github.com/PennyLaneAI/pennylane/pull/6372)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Utkarsh Azad,
Oleksandr Borysenko,
Astral Cai,
Isaac De Vlugt,
Diksha Dhawan,
Lillian M. A. Frederiksen,
Pietropaolo Frisoni,
Emiliano Godinez,
Anthony Hayes,
Austin Huang,
Korbinian Kottmann,
Christina Lee,
William Maxwell,
Erick Ochoa Lopez,
Lee J. O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Ashish Kanwar Singh,
David Wierichs,
