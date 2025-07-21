:orphan:

# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Decomposition rules that can be accessed with the new graph-based decomposition system are
  implemented. The following decompositions have been added:
    [(#7779)](https://github.com/PennyLaneAI/pennylane/pull/7779)

    * :class:`~.Adder`
    
    * :class:`~.ControlledSequence`
  
    * :class:`~.ModExp`

    * :class:`~.Multiplier`

    * :class:`~.OutAdder`

    * :class:`~.OutMultiplier`

    * :class:`~.OutPoly`

  ```python
  import pennylane as qml
  from functools import partial
  
  qml.decomposition.enable_graph()
  
  x = 3
  k = 4
  mod = 7
  
  x_wires = [0,1,2]
  work_wires = [3,4,5,6,7]
  
  dev = qml.device("default.qubit")
  
  @partial(qml.transforms.decompose)
  @partial(qml.set_shots, shots=1)
  @qml.qnode(dev)
  def circuit():
      qml.BasisEmbedding(x, wires=x_wires)
      qml.Multiplier(k, x_wires, mod, work_wires)
      return qml.sample(wires=x_wires)
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ───────────────────────────────────────────────────────────────────────────────╭●──────── ···
  1: ──X────────────────────────────────────────────────────────────────────────────│───────── ···
  2: ──X────────────────────────────────────────────────────────────────────────────│───────── ···
  3: ───────────────────────────────────────────────────────────────────────────────│───────── ···
  4: ──H─╭Rϕ(1.57)─╭Rϕ(0.79)─╭Rϕ(0.39)────────────────────────────────────────╭SWAP─╰Rϕ(12.57) ···
  5: ────╰●────────│─────────│──────────H─╭Rϕ(1.57)─╭Rϕ(0.79)─────────────────│─────╭SWAP───── ···
  6: ──────────────╰●────────│────────────╰●────────│──────────H─╭Rϕ(1.57)────│─────╰SWAP───── ···
  7: ────────────────────────╰●─────────────────────╰●───────────╰●─────────H─╰SWAP─────────── ···
  ```

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


<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Added state of the art resources for the `ResourceSelectPauliRot` template and the
  `ResourceQubitUnitary` templates.
  [(#7786)](https://github.com/PennyLaneAI/pennylane/pull/7786)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* Seeded tests for the `split_to_single_terms` transformation.
  [(#7851)](https://github.com/PennyLaneAI/pennylane/pull/7851)

* Upgrade `rc_sync.yml` to work with latest `pyproject.toml` changes.
  [(#7808)](https://github.com/PennyLaneAI/pennylane/pull/7808)
  [(#7818)](https://github.com/PennyLaneAI/pennylane/pull/7818)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixes attributes and types in the quantum dialect.
  This allows for types to be inferred correctly when parsing.
  [(#7825)](https://github.com/PennyLaneAI/pennylane/pull/7825)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Erick Ochoa,
Joey Carter,
Marcus Edwards,
Andrija Paurevic,
Jay Soni,
