# Release 0.8.0-dev (development version)

<h3>New features since last release</h3>

* Added the covenience load functions ``qml.from_pyquil``, ``qml.from_quil`` and 
  ``qml.from_quil_file`` that convert pyquil objects and Quil code to PennyLane
  templates. This feature requires the latest version of the PennyLane-Forest plugin.
  [#459](https://github.com/XanaduAI/pennylane/pull/459)

* Added a `qml.inv` method that inverts templates and sequences of Operations.
  Added a `@qml.template` decorator that makes templates return the queued Operations.
  [#462](https://github.com/XanaduAI/pennylane/pull/462)

* Added a quantum chemistry package, `pennylane.qchem`, which supports
  integration with OpenFermion, Psi4, PySCF, and OpenBabel.
  [(#453)](https://github.com/XanaduAI/pennylane/pull/453)

  Features include:

  - Generate the qubit Hamiltonians directly starting with the atomic structure of the molecule.
  - Calculate the mean-field (Hartree-Fock) electronic structure of molecules.
  - Allow to define an active space based on the number of active electrons and active orbitals.
  - Perform the fermionic-to-qubit transformation of the electronic Hamiltonian by
    using different functions implemented in OpenFermion.
  - Convert OpenFermion's QubitOperator to a Pennylane `Hamiltonian` class.
  - Perform a Variational Quantum Eigensolver (VQE) computation with this Hamiltonian in PennyLane.

  Check out the [quantum chemistry quickstart](https://pennylane.readthedocs.io/en/latest/introduction/chemistry.html), as well the quantum chemistry and VQE tutorials.

* Added `QAOAEmbedding` and its parameter initialization
  as a new trainable template.
  [(#442)](https://github.com/XanaduAI/pennylane/pull/442)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/qaoa_layers.png"
  width=70%></img>

* Added the `qml.probs()` measurement function, allowing QNodes
  to differentiate variational circuit probabilities
  on simulators and hardware.
  [(#432)](https://github.com/XanaduAI/pennylane/pull/432)

  ```python
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(wires=0)
      qml.RY(x, wires=0)
      qml.RX(x, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.probs(wires=[0])
  ```
  Executing this circuit gives the marginal probability of wire 1:
  ```python
  >>> circuit(0.2)
  [0.40066533 0.59933467]
  ```
  QNodes that return probabilities fully support autodifferentiation.

* Added the `QNodeCollection` container class, that allows independent
  QNodes to be stored and evaluated simultaneously.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

* Added a high level `qml.map` function, that maps a quantum
  circuit template over a list of observables or devices, returning
  a `QNodeCollection`.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  For example:

  ```python3
  >>> def my_template(params, wires, **kwargs):
  >>>    qml.RX(params[0], wires=wires[0])
  >>>    qml.RX(params[1], wires=wires[1])
  >>>    qml.CNOT(wires=wires)

  >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(1)]
  >>> dev = qml.device("default.qubit", wires=2)
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> qnodes([0.54, 0.12])
  array([-0.06154835  0.99280864])
  ```

* Added high level `qml.sum`, `qml.dot`, `qml.apply` functions
  that act on QNode collections.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  `qml.apply` allows vectorized functions to act over the entire QNode
  collection:
  ```python
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> cost = qml.apply(np.sin, qnodes)
  >>> cost([0.54, 0.12])
  array([-0.0615095  0.83756375])
  ```

  `qml.sum` and `qml.dot` take the sum of a QNode collection, and a
  dot product of tensors/arrays/QNode collections, respectively.
  
  
* Unified the way how samples are generated on qubit based devices by refactoring the `QubitDevice`
  class and adding the `sample` and further auxiliary methods.
  [#461](https://github.com/XanaduAI/pennylane/pull/461)

<h3>Breaking changes</h3>

* Deprecated the old-style `QNode` such that only the new-style `QNode` and its syntax can be used,
  moved all related files from the `pennylane/beta` folder to `pennylane`.
  [(#440)](https://github.com/XanaduAI/pennylane/pull/440)

<h3>Improvements</h3>

* Added the ``Observable.eigvals`` method to return the eigenvalues of observables.
  [(#449)](https://github.com/XanaduAI/pennylane/pull/449)

* Added the ``Observable.diagonalizing_gates`` method to return the gates
  that diagonalize an observable in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added the ``Operator.matrix`` method to return the matrix representation
  of an operator in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added a `QubitDevice` class which implements common functionalities of plugin devices such that
  plugin devices can rely on these implementations.
  [(#452)](https://github.com/XanaduAI/pennylane/pull/452)

* Improved documentation of `AmplitudeEmbedding` and `BasisEmbedding` templates.
  [(#441)](https://github.com/XanaduAI/pennylane/pull/441)
  [(#439)](https://github.com/XanaduAI/pennylane/pull/439)

* Codeblocks in the documentation now have a 'copy' button for easily
  copying examples.
  [(#437)](https://github.com/XanaduAI/pennylane/pull/437)

<h3>Bug fixes</h3>

* Fixed a bug in `CVQNode._pd_analytic`, where non-descendant observables were not
  Heisenberg-transformed before evaluating the partial derivatives when using the
  order-2 parameter-shift method, resulting in an erroneous Jacobian for some circuits.
  [(#433)](https://github.com/XanaduAI/pennylane/pull/433)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Ville Bergholm, Alain Delgado Gran, Josh Izaac,
Soran Jahangiri, Johannes Jakob Meyer, Zeyue Niu, Maria Schuld, Antal Száva

# Release 0.7.0 (current release)

<h3>New features since last release</h3>

* Custom padding constant in `AmplitudeEmbedding` is supported (see 'Breaking changes'.)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* `StronglyEntanglingLayer` and `RandomLayer` now work with a single wire.
  [(#409)](https://github.com/XanaduAI/pennylane/pull/409)
  [(#413)](https://github.com/XanaduAI/pennylane/pull/413)

* Added support for applying the inverse of an `Operation` within a circuit.
  [(#377)](https://github.com/XanaduAI/pennylane/pull/377)

* Added an `OperationRecorder()` context manager, that allows templates
  and quantum functions to be executed while recording events. The
  recorder can be used with and without QNodes as a debugging utility.
  [(#388)](https://github.com/XanaduAI/pennylane/pull/388)

* Operations can now specify a decomposition that is used when the desired operation
  is not supported on the target device.
  [(#396)](https://github.com/XanaduAI/pennylane/pull/396)

* The ability to load circuits from external frameworks as templates
  has been added via the new `qml.load()` function. This feature
  requires plugin support --- this initial release provides support
  for Qiskit circuits and QASM files when `pennylane-qiskit` is installed,
  via the functions `qml.from_qiskit` and `qml.from_qasm`.
  [(#418)](https://github.com/XanaduAI/pennylane/pull/418)

* An experimental tensor network device has been added
  [(#416)](https://github.com/XanaduAI/pennylane/pull/416)
  [(#395)](https://github.com/XanaduAI/pennylane/pull/395)
  [(#394)](https://github.com/XanaduAI/pennylane/pull/394)
  [(#380)](https://github.com/XanaduAI/pennylane/pull/380)

* An experimental tensor network device which uses TensorFlow for
  backpropagation has been added
  [(#427)](https://github.com/XanaduAI/pennylane/pull/427)

* Custom padding constant in `AmplitudeEmbedding` is supported (see 'Breaking changes'.)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

<h3>Breaking changes</h3>

* The `pad` parameter in `AmplitudeEmbedding()`` is now either `None` (no automatic padding), or a
  number that is used as the padding constant.
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* Initialization functions now return a single array of weights per function. Utilities for multi-weight templates
  `Interferometer()` and `CVNeuralNetLayers()` are provided.
  [(#412)](https://github.com/XanaduAI/pennylane/pull/412)

* The single layer templates `RandomLayer()`, `CVNeuralNetLayer()` and `StronglyEntanglingLayer()`
  have been turned into private functions `_random_layer()`, `_cv_neural_net_layer()` and
  `_strongly_entangling_layer()`. Recommended use is now via the corresponding `Layers()` templates.
  [(#413)](https://github.com/XanaduAI/pennylane/pull/413)

<h3>Improvements</h3>

* Added extensive input checks in templates.
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* Templates integration tests are rewritten - now cover keyword/positional argument passing,
  interfaces and combinations of templates.
  [(#409)](https://github.com/XanaduAI/pennylane/pull/409)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* State vector preparation operations in the `default.qubit` plugin can now be
  applied to subsets of wires, and are restricted to being the first operation
  in a circuit.
  [(#346)](https://github.com/XanaduAI/pennylane/pull/346)

* The `QNode` class is split into a hierarchy of simpler classes.
  [(#354)](https://github.com/XanaduAI/pennylane/pull/354)
  [(#398)](https://github.com/XanaduAI/pennylane/pull/398)
  [(#415)](https://github.com/XanaduAI/pennylane/pull/415)
  [(#417)](https://github.com/XanaduAI/pennylane/pull/417)
  [(#425)](https://github.com/XanaduAI/pennylane/pull/425)

* Added the gates U1, U2 and U3 parametrizing arbitrary unitaries on 1, 2 and 3
  qubits and the Toffoli gate to the set of qubit operations.
  [(#396)](https://github.com/XanaduAI/pennylane/pull/396)

* Changes have been made to accomodate the movement of the main function
  in `pytest._internal` to `pytest._internal.main` in pip 19.3.
  [(#404)](https://github.com/XanaduAI/pennylane/pull/404)

* Added the templates `BasisStatePreparation` and `MottonenStatePreparation` that use
  gates to prepare a basis state and an arbitrary state respectively.
  [(#336)](https://github.com/XanaduAI/pennylane/pull/336)

* Added decompositions for `BasisState` and `QubitStateVector` based on state
  preparation templates.
  [(#414)](https://github.com/XanaduAI/pennylane/pull/414)

* Replaces the pseudo-inverse in the quantum natural gradient optimizer
  (which can be numerically unstable) with `np.linalg.solve`.
  [(#428)](https://github.com/XanaduAI/pennylane/pull/428)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Nathan Killoran, Angus Lowe, Johannes Jakob Meyer,
Oluwatobi Ogunbayo, Maria Schuld, Antal Száva.

# Release 0.6.1

<h3>New features since last release</h3>

* Added a `print_applied` method to QNodes, allowing the operation
  and observable queue to be printed as last constructed.
  [(#378)](https://github.com/XanaduAI/pennylane/pull/378)

<h3>Improvements</h3>

* A new `Operator` base class is introduced, which is inherited by both the
  `Observable` class and the `Operation` class.
  [(#355)](https://github.com/XanaduAI/pennylane/pull/355)

* Removed deprecated `@abstractproperty` decorators
  in `_device.py`.
  [(#374)](https://github.com/XanaduAI/pennylane/pull/374)

* The `CircuitGraph` class is updated to deal with `Operation` instances directly.
  [(#344)](https://github.com/XanaduAI/pennylane/pull/344)

* Comprehensive gradient tests have been added for the interfaces.
  [(#381)](https://github.com/XanaduAI/pennylane/pull/381)

<h3>Documentation</h3>

* The new restructured documentation has been polished and updated.
  [(#387)](https://github.com/XanaduAI/pennylane/pull/387)
  [(#375)](https://github.com/XanaduAI/pennylane/pull/375)
  [(#372)](https://github.com/XanaduAI/pennylane/pull/372)
  [(#370)](https://github.com/XanaduAI/pennylane/pull/370)
  [(#369)](https://github.com/XanaduAI/pennylane/pull/369)
  [(#367)](https://github.com/XanaduAI/pennylane/pull/367)
  [(#364)](https://github.com/XanaduAI/pennylane/pull/364)

* Updated the development guides.
  [(#382)](https://github.com/XanaduAI/pennylane/pull/382)
  [(#379)](https://github.com/XanaduAI/pennylane/pull/379)

* Added all modules, classes, and functions to the API section
  in the documentation.
  [(#373)](https://github.com/XanaduAI/pennylane/pull/373)

<h3>Bug fixes</h3>

* Replaces the existing `np.linalg.norm` normalization with hand-coded
  normalization, allowing `AmplitudeEmbedding` to be used with differentiable
  parameters. AmplitudeEmbedding tests have been added and improved.
  [(#376)](https://github.com/XanaduAI/pennylane/pull/376)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Nathan Killoran, Maria Schuld, Antal Száva

# Release 0.6.0

<h3>New features since last release</h3>

* The devices `default.qubit` and `default.gaussian` have a new initialization parameter
  `analytic` that indicates if expectation values and variances should be calculated
  analytically and not be estimated from data.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Added C-SWAP gate to the set of qubit operations
  [(#330)](https://github.com/XanaduAI/pennylane/pull/330)

* The TensorFlow interface has been renamed from `"tfe"` to `"tf"`, and
  now supports TensorFlow 2.0.
  [(#337)](https://github.com/XanaduAI/pennylane/pull/337)

* Added the S and T gates to the set of qubit operations.
  [(#343)](https://github.com/XanaduAI/pennylane/pull/343)

* Tensor observables are now supported within the `expval`,
  `var`, and `sample` functions, by using the `@` operator.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)


<h3>Breaking changes</h3>

* The argument `n` specifying the number of samples in the method `Device.sample` was removed.
  Instead, the method will always return `Device.shots` many samples.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

<h3>Improvements</h3>

* The number of shots / random samples used to estimate expectation values and variances, `Device.shots`,
  can now be changed after device creation.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Unified import shortcuts to be under qml in qnode.py
  and test_operation.py
  [(#329)](https://github.com/XanaduAI/pennylane/pull/329)

* The quantum natural gradient now uses `scipy.linalg.pinvh` which is more efficient for symmetric matrices
  than the previously used `scipy.linalg.pinv`.
  [(#331)](https://github.com/XanaduAI/pennylane/pull/331)

* The deprecated `qml.expval.Observable` syntax has been removed.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)

* Remainder of the unittest-style tests were ported to pytest.
  [(#310)](https://github.com/XanaduAI/pennylane/pull/310)

* The `do_queue` argument for operations now only takes effect
  within QNodes. Outside of QNodes, operations can now be instantiated
  without needing to specify `do_queue`.
  [(#359)](https://github.com/XanaduAI/pennylane/pull/359)

<h3>Documentation</h3>

* The docs are rewritten and restructured to contain a code introduction section as well as an API section.
  [(#314)](https://github.com/XanaduAI/pennylane/pull/275)

* Added Ising model example to the tutorials
  [(#319)](https://github.com/XanaduAI/pennylane/pull/319)

* Added tutorial for QAOA on MaxCut problem
  [(#328)](https://github.com/XanaduAI/pennylane/pull/328)

* Added QGAN flow chart figure to its tutorial
  [(#333)](https://github.com/XanaduAI/pennylane/pull/333)

* Added missing figures for gallery thumbnails of state-preparation
  and QGAN tutorials
  [(#326)](https://github.com/XanaduAI/pennylane/pull/326)

* Fixed typos in the state preparation tutorial
  [(#321)](https://github.com/XanaduAI/pennylane/pull/321)

* Fixed bug in VQE tutorial 3D plots
  [(#327)](https://github.com/XanaduAI/pennylane/pull/327)

<h3>Bug fixes</h3>

* Fixed typo in measurement type error message in qnode.py
  [(#341)](https://github.com/XanaduAI/pennylane/pull/341)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Ville Bergholm, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Angus Lowe,
Johannes Jakob Meyer, Maria Schuld, Antal Száva, Roeland Wiersema.

# Release 0.5.0

<h3>New features since last release</h3>

* Adds a new optimizer, `qml.QNGOptimizer`, which optimizes QNodes using
  quantum natural gradient descent. See https://arxiv.org/abs/1909.02108
  for more details.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#311)](https://github.com/XanaduAI/pennylane/pull/311)

* Adds a new QNode method, `QNode.metric_tensor()`,
  which returns the block-diagonal approximation to the Fubini-Study
  metric tensor evaluated on the attached device.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)

* Sampling support: QNodes can now return a specified number of samples
  from a given observable via the top-level `pennylane.sample()` function.
  To support this on plugin devices, there is a new `Device.sample` method.

  Calculating gradients of QNodes that involve sampling is not possible.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* `default.qubit` has been updated to provide support for sampling.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* Added controlled rotation gates to PennyLane operations and `default.qubit` plugin.
  [(#251)](https://github.com/XanaduAI/pennylane/pull/251)

<h3>Breaking changes</h3>

* The method `Device.supported` was removed, and replaced with the methods
  `Device.supports_observable` and `Device.supports_operation`.
  Both methods can be called with string arguments (`dev.supports_observable('PauliX')`) and
  class arguments (`dev.supports_observable(qml.PauliX)`).
  [(#276)](https://github.com/XanaduAI/pennylane/pull/276)

* The following CV observables were renamed to comply with the new Operation/Observable
  scheme: `MeanPhoton` to `NumberOperator`, `Homodyne` to `QuadOperator` and `NumberState` to `FockStateProjector`.
  [(#254)](https://github.com/XanaduAI/pennylane/pull/254)

<h3>Improvements</h3>

* The `AmplitudeEmbedding` function now provides options to normalize and
  pad features to ensure a valid state vector is prepared.
  [(#275)](https://github.com/XanaduAI/pennylane/pull/275)

* Operations can now optionally specify generators, either as existing PennyLane
  operations, or by providing a NumPy array.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#313)](https://github.com/XanaduAI/pennylane/pull/313)

* Adds a `Device.parameters` property, so that devices can view a dictionary mapping free
  parameters to operation parameters. This will allow plugin devices to take advantage
  of parametric compilation.
  [(#283)](https://github.com/XanaduAI/pennylane/pull/283)

* Introduces two enumerations: `Any` and `All`, representing any number of wires
  and all wires in the system respectively. They can be imported from
  `pennylane.operation`, and can be used when defining the `Operation.num_wires`
  class attribute of operations.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

  As part of this change:

  - `All` is equivalent to the integer 0, for backwards compatibility with the
    existing test suite

  - `Any` is equivalent to the integer -1 to allow numeric comparison
    operators to continue working

  - An additional validation is now added to the `Operation` class,
    which will alert the user that an operation with `num_wires = All`
    is being incorrectly.

* The one-qubit rotations in `pennylane.plugins.default_qubit` no longer depend on Scipy's `expm`. Instead
  they are calculated with Euler's formula.
  [(#292)](https://github.com/XanaduAI/pennylane/pull/292)

* Creates an `ObservableReturnTypes` enumeration class containing `Sample`,
  `Variance` and `Expectation`. These new values can be assigned to the `return_type`
  attribute of an `Observable`.
  [(#290)](https://github.com/XanaduAI/pennylane/pull/290)

* Changed the signature of the `RandomLayer` and `RandomLayers` templates to have a fixed seed by default.
  [(#258)](https://github.com/XanaduAI/pennylane/pull/258)

* `setup.py` has been cleaned up, removing the non-working shebang,
  and removing unused imports.
  [(#262)](https://github.com/XanaduAI/pennylane/pull/262)

<h3>Documentation</h3>

* A documentation refactor to simplify the tutorials and
  include Sphinx-Gallery.
  [(#291)](https://github.com/XanaduAI/pennylane/pull/291)

  - Examples and tutorials previously split across the `examples/`
    and `doc/tutorials/` directories, in a mixture of ReST and Jupyter notebooks,
    have been rewritten as Python scripts with ReST comments in a single location,
    the `examples/` folder.

  - Sphinx-Gallery is used to automatically build and run the tutorials.
    Rendered output is displayed in the Sphinx documentation.

  - Links are provided at the top of every tutorial page for downloading the
    tutorial as an executable python script, downloading the tutorial
    as a Jupyter notebook, or viewing the notebook on GitHub.

  - The tutorials table of contents have been moved to a single quick start page.

* Fixed a typo in `QubitStateVector`.
  [(#296)](https://github.com/XanaduAI/pennylane/pull/296)

* Fixed a typo in the `default_gaussian.gaussian_state` function.
  [(#293)](https://github.com/XanaduAI/pennylane/pull/293)

* Fixed a typo in the gradient recipe within the `RX`, `RY`, `RZ`
  operation docstrings.
  [(#248)](https://github.com/XanaduAI/pennylane/pull/248)

* Fixed a broken link in the tutorial documentation, as a
  result of the `qml.expval.Observable` deprecation.
  [(#246)](https://github.com/XanaduAI/pennylane/pull/246)

<h3>Bug fixes</h3>

* Fixed a bug where a `PolyXP` observable would fail if applied to subsets
  of wires on `default.gaussian`.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Simon Cross, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Johannes Jakob Meyer,
Rohit Midha, Nicolás Quesada, Maria Schuld, Antal Száva, Roeland Wiersema.

# Release 0.4.0

<h3>New features since last release</h3>

* `pennylane.expval()` is now a top-level *function*, and is no longer
  a package of classes. For now, the existing `pennylane.expval.Observable`
  interface continues to work, but will raise a deprecation warning.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

* Variance support: QNodes can now return the variance of observables,
  via the top-level `pennylane.var()` function. To support this on
  plugin devices, there is a new `Device.var` method.

  The following observables support analytic gradients of variances:

  - All qubit observables (requiring 3 circuit evaluations for involutory
    observables such as `Identity`, `X`, `Y`, `Z`; and 5 circuit evals for
    non-involutary observables, currently only `qml.Hermitian`)

  - First-order CV observables (requiring 5 circuit evaluations)

  Second-order CV observables support numerical variance gradients.

* `pennylane.about()` function added, providing details
  on current PennyLane version, installed plugins, Python,
  platform, and NumPy versions [(#186)](https://github.com/XanaduAI/pennylane/pull/186)

* Removed the logic that allowed `wires` to be passed as a positional
  argument in quantum operations. This allows us to raise more useful
  error messages for the user if incorrect syntax is used.
  [(#188)](https://github.com/XanaduAI/pennylane/pull/188)

* Adds support for multi-qubit expectation values of the `pennylane.Hermitian()`
  observable [(#192)](https://github.com/XanaduAI/pennylane/pull/192)

* Adds support for multi-qubit expectation values in `default.qubit`.
  [(#202)](https://github.com/XanaduAI/pennylane/pull/202)

* Organize templates into submodules [(#195)](https://github.com/XanaduAI/pennylane/pull/195).
  This included the following improvements:

  - Distinguish embedding templates from layer templates.

  - New random initialization functions supporting the templates available
    in the new submodule `pennylane.init`.

  - Added a random circuit template (`RandomLayers()`), in which rotations and 2-qubit gates are randomly
    distributed over the wires

  - Add various embedding strategies

<h3>Breaking changes</h3>

* The `Device` methods `expectations`, `pre_expval`, and `post_expval` have been
  renamed to `observables`, `pre_measure`, and `post_measure` respectively.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

<h3>Improvements</h3>

* `default.qubit` plugin now uses `np.tensordot` when applying quantum operations
  and evaluating expectations, resulting in significant speedup
  [(#239)](https://github.com/XanaduAI/pennylane/pull/239),
  [(#241)](https://github.com/XanaduAI/pennylane/pull/241)

* PennyLane now allows division of quantum operation parameters by a constant
  [(#179)](https://github.com/XanaduAI/pennylane/pull/179)

* Portions of the test suite are in the process of being ported to pytest.
  Note: this is still a work in progress.

  Ported tests include:

  - `test_ops.py`
  - `test_about.py`
  - `test_classical_gradients.py`
  - `test_observables.py`
  - `test_measure.py`
  - `test_init.py`
  - `test_templates*.py`
  - `test_ops.py`
  - `test_variable.py`
  - `test_qnode.py` (partial)

<h3>Bug fixes</h3>

* Fixed a bug in `Device.supported`, which would incorrectly
  mark an operation as supported if it shared a name with an
  observable [(#203)](https://github.com/XanaduAI/pennylane/pull/203)

* Fixed a bug in `Operation.wires`, by explicitly casting the
  type of each wire to an integer [(#206)](https://github.com/XanaduAI/pennylane/pull/206)

* Removed code in PennyLane which configured the logger,
  as this would clash with users' configurations
  [(#208)](https://github.com/XanaduAI/pennylane/pull/208)

* Fixed a bug in `default.qubit`, in which `QubitStateVector` operations
  were accidentally being cast to `np.float` instead of `np.complex`.
  [(#211)](https://github.com/XanaduAI/pennylane/pull/211)


<h3>Contributors</h3>

This release contains contributions from:

Shahnawaz Ahmed, riveSunder, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Maria Schuld.

# Release 0.3.1

<h3>Bug fixes</h3>

* Fixed a bug where the interfaces submodule was not correctly being packaged via setup.py

# Release 0.3.0

<h3>New features since last release</h3>

* PennyLane now includes a new `interfaces` submodule, which enables QNode integration with additional machine learning libraries.
* Adds support for an experimental PyTorch interface for QNodes
* Adds support for an experimental TensorFlow eager execution interface for QNodes
* Adds a PyTorch+GPU+QPU tutorial to the documentation
* Documentation now includes links and tutorials including the new [PennyLane-Forest](https://github.com/rigetti/pennylane-forest) plugin.

<h3>Improvements</h3>

* Printing a QNode object, via `print(qnode)` or in an interactive terminal, now displays more useful information regarding the QNode,
  including the device it runs on, the number of wires, it's interface, and the quantum function it uses:

  ```python
  >>> print(qnode)
  <QNode: device='default.qubit', func=circuit, wires=2, interface=PyTorch>
  ```

<h3>Contributors</h3>

This release contains contributions from:

Josh Izaac and Nathan Killoran.


# Release 0.2.0

<h3>New features since last release</h3>

* Added the `Identity` expectation value for both CV and qubit models [(#135)](https://github.com/XanaduAI/pennylane/pull/135)
* Added the `templates.py` submodule, containing some commonly used QML models to be used as ansatz in QNodes [(#133)](https://github.com/XanaduAI/pennylane/pull/133)
* Added the `qml.Interferometer` CV operation [(#152)](https://github.com/XanaduAI/pennylane/pull/152)
* Wires are now supported as free QNode parameters [(#151)](https://github.com/XanaduAI/pennylane/pull/151)
* Added ability to update stepsizes of the optimizers [(#159)](https://github.com/XanaduAI/pennylane/pull/159)

<h3>Improvements</h3>

* Removed use of hardcoded values in the optimizers, made them parameters (see [#131](https://github.com/XanaduAI/pennylane/pull/131) and [#132](https://github.com/XanaduAI/pennylane/pull/132))
* Created the new `PlaceholderExpectation`, to be used when both CV and qubit expval modules contain expectations with the same name
* Provide the plugins a way to view the operation queue _before_ applying operations. This allows for on-the-fly modifications of
  the queue, allowing hardware-based plugins to support the full range of qubit expectation values. [(#143)](https://github.com/XanaduAI/pennylane/pull/143)
* QNode return values now support _any_ form of sequence, such as lists, sets, etc. [(#144)](https://github.com/XanaduAI/pennylane/pull/144)
* CV analytic gradient calculation is now more robust, allowing for operations which may not themselves be differentiated, but have a
  well defined `_heisenberg_rep` method, and so may succeed operations that are analytically differentiable [(#152)](https://github.com/XanaduAI/pennylane/pull/152)

<h3>Bug fixes</h3>

* Fixed a bug where the variational classifier example was not batching when learning parity (see [#128](https://github.com/XanaduAI/pennylane/pull/128) and [#129](https://github.com/XanaduAI/pennylane/pull/129))
* Fixed an inconsistency where some initial state operations were documented as accepting complex parameters - all operations
  now accept real values [(#146)](https://github.com/XanaduAI/pennylane/pull/146)

<h3>Contributors</h3>

This release contains contributions from:

Christian Gogolin, Josh Izaac, Nathan Killoran, and Maria Schuld.


# Release 0.1.0

Initial public release.

<h3>Contributors</h3>
This release contains contributions from:

Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
