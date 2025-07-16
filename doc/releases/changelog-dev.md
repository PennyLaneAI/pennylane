:orphan:

# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

* Leveraging quantum just-in-time compilation to optimize parameterized hybrid workflows with the momentum
  quantum natural gradient optimizer is now possible with the new :class:`~.MomentumQNGOptimizerQJIT` optimizer.
  [(#7606)](https://github.com/PennyLaneAI/pennylane/pull/7606)

  Similar to the :class:`~.QNGOptimizerQJIT` optimizer, :class:`~.MomentumQNGOptimizerQJIT` offers a 
  `jax.jit`- and `qml.qjit`-compatible analogue to the existing :class:`~.MomentumQNGOptimizer` with an 
  Optax-like interface:

  ```python
  import pennylane as qml
  import jax.numpy as jnp

  @qml.qjit(autograph=True)
  def workflow():
      dev = qml.device("lightning.qubit", wires=2)

      @qml.qnode(dev)
      def circuit(params):
          qml.RX(params[0], wires=0)
          qml.RY(params[1], wires=1)
          return qml.expval(qml.Z(0) + qml.X(1))
      
      opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.1, momentum=0.2)
      params = jnp.array([0.1, 0.2])
      state = opt.init(params)
      
      for _ in range(100):
          params, state = opt.step(circuit, params, state)
      
      return params
  ```

  ```pycon
  >>> workflow()
  Array([ 3.14159265, -1.57079633], dtype=float64)
  ```

<h3>Improvements 🛠</h3>

<h4>OpenQASM-PennyLane interoperability</h4>

* The :func:`qml.from_qasm3` function can now convert OpenQASM 3.0 circuits that contain
  subroutines, constants, and built-in mathematical functions.
  [(#7651)](https://github.com/PennyLaneAI/pennylane/pull/7651)
  [(#7653)](https://github.com/PennyLaneAI/pennylane/pull/7653)
  [(#7676)](https://github.com/PennyLaneAI/pennylane/pull/7676)
  [(#7677)](https://github.com/PennyLaneAI/pennylane/pull/7677)

<h4>Other improvements</h4>

* The :func:`qml.workflow.set_shots` transform can now be directly applied to a QNode without the need for `functools.partial`, providing a more user-friendly syntax and negating having to import the `functools` package.
  [(#7876)](https://github.com/PennyLaneAI/pennylane/pull/7876)
  
  ```python
  @qml.set_shots(shots=1000)
  @qml.qnode(dev)
  def circuit():
      qml.H(0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> circuit()
  0.002
  ```

* Update minimum supported `pytest` version to `8.4.1`.
  [(#7853)](https://github.com/PennyLaneAI/pennylane/pull/7853)

* A compilation pass written with xDSL called `qml.compiler.python_compiler.transforms.MeasurementsFromSamplesPass`
  has been added for the experimental xDSL Python compiler integration. This pass replaces all
  terminal measurements in a program with a single :func:`pennylane.sample` measurement, and adds
  postprocessing instructions to recover the original measurement.
  [(#7620)](https://github.com/PennyLaneAI/pennylane/pull/7620)

* A combine-global-phase pass has been added to the xDSL Python compiler integration.
  Note that the current implementation can only combine all the global phase operations at
  the last global phase operation in the same region. In other words, global phase operations inside a control flow region can't be combined with those in their parent 
  region.
  [(#7675)](https://github.com/PennyLaneAI/pennylane/pull/7675)

* The `mbqc` xDSL dialect has been added to the Python compiler, which is used to represent
  measurement-based quantum-computing instructions in the xDSL framework.
  [(#7815)](https://github.com/PennyLaneAI/pennylane/pull/7815)

* The :func:`pennylane.ops.rs_decomposition` method now performs exact decomposition and returns
  complete global phase information when used for decomposing a phase gate to Clifford+T basis.
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Added state of the art resources for the `ResourceSelectPauliRot` template and the
  `ResourceQubitUnitary` templates.
  [(#7786)](https://github.com/PennyLaneAI/pennylane/pull/7786)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* Make `pennylane.io` a tertiary module.
  [(#7877)](https://github.com/PennyLaneAI/pennylane/pull/7877)

* Seeded tests for the `split_to_single_terms` transformation.
  [(#7851)](https://github.com/PennyLaneAI/pennylane/pull/7851)

* Upgrade `rc_sync.yml` to work with latest `pyproject.toml` changes.
  [(#7808)](https://github.com/PennyLaneAI/pennylane/pull/7808)
  [(#7818)](https://github.com/PennyLaneAI/pennylane/pull/7818)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Calling `QNode.update` no longer acts as if `set_shots` has been applied.
  [(#7881)](https://github.com/PennyLaneAI/pennylane/pull/7881)

* Fixes attributes and types in the quantum dialect.
  This allows for types to be inferred correctly when parsing.
  [(#7825)](https://github.com/PennyLaneAI/pennylane/pull/7825)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Joey Carter,
Simone Gasperini,
Erick Ochoa,
Andrija Paurevic,
Jay Soni,
Jake Zaia
