
# Release 0.7.0

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
  has been added via the new `qp.load()` function. This feature
  requires plugin support --- this initial release provides support
  for Qiskit circuits and QASM files when `pennylane-qiskit` is installed,
  via the functions `qp.from_qiskit` and `qp.from_qasm`.
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

* The `pad` parameter in `AmplitudeEmbedding()` is now either `None` (no automatic padding), or a
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
