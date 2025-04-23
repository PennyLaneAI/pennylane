:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

* The transform `convert_to_mbqc_gateset` is added to the `ftqc` module to convert arbitrary 
  circuits to a limited gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* The `RotXZX` operation is added to the `ftqc` module to support definition of a universal
  gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* Two new functions called `convert_to_su2` and `convert_to_su4` have been added to `qml.math`, which convert unitary matrices to SU(2) or SU(4), respectively, and optionally a global phase.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h4>Resource-efficient Decompositions 🔎</h4>

* New decomposition rules comprising rotation gates and global phases have been added to `QubitUnitary` that 
  can be accessed with the new graph-based decomposition system. The most efficient set of rotations to 
  decompose into will be chosen based on the target gate set.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

  ```python
  from functools import partial
  import numpy as np
  import pennylane as qml
  
  qml.decomposition.enable_graph()
  
  U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
  
  @partial(qml.transforms.decompose, gate_set={"RX", "RY", "GlobalPhase"})
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.QubitUnitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), wires=[0])
      return qml.expval(qml.PauliZ(0))
  ```
  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──RX(0.00)──RY(1.57)──RX(3.14)──GlobalPhase(-1.57)─┤  <Z>
  ```

* Decomposition rules can be marked as not-applicable with `qml.decomposition.DecompositionNotApplicable`, allowing for flexibility when creating conditional decomposition 
  rules based on parameters that affects the rule's resources.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

  ```python
  import pennylane as qml
  from pennylane.decomposition import DecompositionNotApplicable
  from pennylane.math.decomposition import zyz_rotation_angles
  
  def _zyz_resource(num_wires):
      if num_wires != 1:
          # This decomposition is only applicable when num_wires is 1
          raise DecompositionNotApplicable
      return {qml.RZ: 2, qml.RY: 1, qml.GlobalPhase: 1}

  def zyz_decomposition(U, wires, **__):
      phi, theta, omega, phase = zyz_rotation_angles(U, return_global_phase=True)
      qml.RZ(phi, wires=wires[0])
      qml.RY(theta, wires=wires[0])
      qml.RZ(omega, wires=wires[0])
      qml.GlobalPhase(-phase)
  ```
  
  This decomposition will be ignored for `QubitUnitary` on more than one wire.

<h3>Improvements 🛠</h3>

* Alias for Identity (`I`) is now accessible from `qml.ops`.
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)

* Two-qubit `QubitUnitary` gates no longer decompose into fundamental rotation gates; it now 
  decomposes into single-qubit `QubitUnitary` gates. This allows the decomposition system to
  further decompose single-qubit unitary gates more flexibly using different rotations.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h3>Breaking changes 💔</h3>

* Accessing terms of a tensor product (e.g., `op = X(0) @ X(1)`) via `op.obs` has been removed.
  [(#7324)](https://github.com/PennyLaneAI/pennylane/pull/7324)

* The `inner_transform` and `config` keyword arguments in `qml.execute` have been removed.
  [(#7300)](https://github.com/PennyLaneAI/pennylane/pull/7300)

* `Sum.ops`, `Sum.coeffs`, `Prod.ops` and `Prod.coeffs` have been removed.
  [(#7304)](https://github.com/PennyLaneAI/pennylane/pull/7304)

* Specifying `pipeline=None` with `qml.compile` has been removed.
  [(#7307)](https://github.com/PennyLaneAI/pennylane/pull/7307)

* `qml.tape.TapeError` has been removed.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* Introduce module dependency management using `tach`.
  [(#7185)](https://github.com/PennyLaneAI/pennylane/pull/7185)

* Add new `pennylane.exceptions` module for custom errors and warnings.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

* Clean up `__init__.py` files in `math`, `ops`, `qaoa`, `tape` and `templates` to be explicit in what they import. 
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)
  
* The `Tracker` class has been moved into the `devices` module.
  [(#7281)](https://github.com/PennyLaneAI/pennylane/pull/7281)

* Moved functions that calculate rotation angles for unitary decompositions into an internal
  module `qml.math.decomposition`
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h3>Documentation 📝</h3>

* The entry in the :doc:`/news/program_capture_sharp_bits` page for using program capture with Catalyst
  has been updated. Instead of using ``qjit(experimental_capture=True)``, Catalyst is now compatible 
  with the global toggles ``qml.capture.enable()`` and ``qml.capture.disable()`` for enabling and 
  disabling program capture.
  [(#7298)](https://github.com/PennyLaneAI/pennylane/pull/7298)

<h3>Bug fixes 🐛</h3>

* Adds an informative error if `qml.cond` is used with an abstract condition with
  jitting on `default.qubit` if capture is enabled.
  [(#7314)](https://github.com/PennyLaneAI/pennylane/pull/7314)

* Fixes a bug where using a ``StatePrep`` operation with `batch_size=1` did not work with ``default.mixed``.
  [(#7280)](https://github.com/PennyLaneAI/pennylane/pull/7280)

* Gradient transforms can now be used in conjunction with batch transforms with all interfaces.
  [(#7287)](https://github.com/PennyLaneAI/pennylane/pull/7287)

* Fixes a bug where the global phase was not being added in the ``QubitUnitary`` decomposition.  
  [(#7244)](https://github.com/PennyLaneAI/pennylane/pull/7244)
  [(#7270)](https://github.com/PennyLaneAI/pennylane/pull/7270)

* Using finite differences with program capture without x64 mode enabled now raises a warning.
  [(#7282)](https://github.com/PennyLaneAI/pennylane/pull/7282)

* When the `mcm_method` is specified to the `"device"`, the `defer_measurements` transform will 
  no longer be applied. Instead, the device will be responsible for all MCM handling.
  [(#7243)](https://github.com/PennyLaneAI/pennylane/pull/7243)

* Fixed coverage of `qml.liealg.CII` and `qml.liealg.AIII`.
  [(#7291)](https://github.com/PennyLaneAI/pennylane/pull/7291)

* Fixed a bug where the phase is used as the wire label for a `qml.GlobalPhase` when capture is enabled.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Astral Cai,
Yushao Chen,
Lillian Frederiksen,
Pietropaolo Frisoni,
Korbinian Kottmann,
Christina Lee,
Andrija Paurevic
