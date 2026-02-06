
# Release 0.38.0

<h3>New features since last release</h3>

<h4>Registers of wires 🧸</h4>

* A new function called `qp.registers` has been added that lets you seamlessly create registers of 
  wires.
  [(#5957)](https://github.com/PennyLaneAI/pennylane/pull/5957)
  [(#6102)](https://github.com/PennyLaneAI/pennylane/pull/6102)

  Using registers, it is easier to build large algorithms and circuits by applying gates and operations 
  to predefined collections of wires. With `qp.registers`, you can create registers of wires by providing 
  a dictionary whose keys are register names and whose values are the number of wires in each register.

  ```python
  >>> wire_reg = qp.registers({"alice": 4, "bob": 3})
  >>> wire_reg
  {'alice': Wires([0, 1, 2, 3]), 'bob': Wires([4, 5, 6])}
  ```

  The resulting data structure of `qp.registers` is a dictionary with the same register names as keys, 
  but the values are `qp.wires.Wires` instances.

  Nesting registers within other registers can be done by providing a nested dictionary, where the ordering 
  of wire labels is based on the order of appearance and nestedness.

  ```python
  >>> wire_reg = qp.registers({"alice": {"alice1": 1, "alice2": 2}, "bob": {"bob1": 2, "bob2": 1}})
  >>> wire_reg
  {'alice1': Wires([0]), 'alice2': Wires([1, 2]), 'alice': Wires([0, 1, 2]), 'bob1': Wires([3, 4]), 'bob2': Wires([5]), 'bob': Wires([3, 4, 5])}
  ```

  Since the values of the dictionary are `Wires` instances, their use within quantum circuits is very 
  similar to that of a `list` of integers.

  ```python
  dev = qp.device("default.qubit")

  @qp.qnode(dev)
  def circuit():
      for w in wire_reg["alice"]:
          qp.Hadamard(w)

      for w in wire_reg["bob1"]:
          qp.RX(0.1967, wires=w)

      qp.CNOT(wires=[wire_reg["alice1"][0], wire_reg["bob2"][0]])

      return [qp.expval(qp.Y(w)) for w in wire_reg["bob1"]]

  print(qp.draw(circuit)())
  ```

  ```pycon
  0: ──H────────╭●─┤     
  1: ──H────────│──┤     
  2: ──H────────│──┤     
  3: ──RX(0.20)─│──┤  <Y>
  4: ──RX(0.20)─│──┤  <Y>
  5: ───────────╰X─┤  
  ```

  In tandem with `qp.registers`, we've also made the following improvements to `qp.wires.Wires`:

  * `Wires` instances now have a more copy-paste friendly representation when printed.
    [(#5958)](https://github.com/PennyLaneAI/pennylane/pull/5958)

    ```python
    >>> from pennylane.wires import Wires
    >>> w = Wires([1, 2, 3])
    >>> w
    Wires([1, 2, 3])
    ```

  * Python set-based combinations are now supported by `Wires`.
    [(#5983)](https://github.com/PennyLaneAI/pennylane/pull/5983)

    This new feature unlocks the ability to combine `Wires` instances in the following ways:

    * intersection with `&` or `intersection()`:

      ```python
      >>> wires1 = Wires([1, 2, 3])
      >>> wires2 = Wires([2, 3, 4])
      >>> wires1.intersection(wires2) # or wires1 & wires2
      Wires([2, 3])
      ```

    * symmetric difference with `^` or `symmetric_difference()`:

      ```python
      >>> wires1.symmetric_difference(wires2) # or wires1 ^ wires2
      Wires([1, 4])
      ```

    * union with `|` or `union()`:

      ```python
      >>> wires1.union(wires2) # or wires1 | wires2
      Wires([1, 2, 3, 4])
      ```

    * difference with `-` or `difference()`:

      ```python
      >>> wires1.difference(wires2) # or wires1 - wires2
      Wires([1])
      ```

<h4>Quantum arithmetic operations 🧮</h4>

* Several new operator templates have been added to PennyLane that let you perform quantum arithmetic 
  operations.
  [(#6109)](https://github.com/PennyLaneAI/pennylane/pull/6109)
  [(#6112)](https://github.com/PennyLaneAI/pennylane/pull/6112)
  [(#6121)](https://github.com/PennyLaneAI/pennylane/pull/6121)

  * `qp.Adder` performs in-place modular addition: 
    :math:`\text{Adder}(k, m)\vert x \rangle = \vert x + k \; \text{mod} \; m\rangle`. 

  * `qp.PhaseAdder` is similar to `qp.Adder`, but it performs in-place modular addition in the Fourier 
    basis. 

  * `qp.Multiplier` performs in-place multiplication: 
    :math:`\text{Multiplier}(k, m)\vert x \rangle = \vert x \times k \; \text{mod} \; m \rangle`.

  * `qp.OutAdder` performs out-place modular addition:
    :math:`\text{OutAdder}(m)\vert x \rangle \vert y \rangle \vert b \rangle = \vert x \rangle \vert y \rangle \vert b + x + y \; \text{mod} \; m \rangle`.

  * `qp.OutMultiplier` performs out-place modular multiplication: 
    :math:`\text{OutMultiplier}(m)\vert x \rangle \vert y \rangle \vert b \rangle = \vert x \rangle \vert y \rangle \vert b + x \times y \; \text{mod} \; m \rangle`.

  * `qp.ModExp` performs modular exponentiation: 
    :math:`\text{ModExp}(base, m) \vert x \rangle \vert k \rangle = \vert x \rangle \vert k \times base^x \; \text{mod} \; m \rangle`.

  Here is a comprehensive example that performs the following calculation: `(2 + 1) * 3 mod 7 = 2` (or 
  `010` in binary).

  ```python
  dev = qp.device("default.qubit", shots=1)

  wire_reg = qp.registers({
      "x_wires": 2, # |x>: stores the result of 2 + 1 = 3
      "y_wires": 2, # |y>: multiples x by 3
      "output_wires": 3, # stores the result of (2 + 1) * 3 m 7 = 2
      "work_wires": 2 # for qp.OutMultiplier
  })

  @qp.qnode(dev)
  def circuit():
      # In-place addition
      qp.BasisEmbedding(2, wires=wire_reg["x_wires"])
      qp.Adder(1, x_wires=wire_reg["x_wires"]) # add 1 to wires [0, 1] 

      # Out-place multiplication
      qp.BasisEmbedding(3, wires=wire_reg["y_wires"])
      qp.OutMultiplier(
          wire_reg["x_wires"], 
          wire_reg["y_wires"], 
          wire_reg["output_wires"], 
          work_wires=wire_reg["work_wires"], 
          mod=7
      ) 

      return qp.sample(wires=wire_reg["output_wires"])
  ```

  ```
  >>> circuit()
  array([0, 1, 0])
  ```

<h4>Converting noise models from Qiskit ♻️</h4>

* Convert Qiskit noise models into a PennyLane `NoiseModel` with `qp.from_qiskit_noise`.
  [(#5996)](https://github.com/PennyLaneAI/pennylane/pull/5996)

  In the last few releases, we've added substantial improvements and new features to the 
  [Pennylane-Qiskit plugin](https://docs.pennylane.ai/projects/qiskit/en/latest/installation.html).
  With this release, a new `qp.from_qiskit_noise` function allows you to convert a Qiskit noise model 
  into a PennyLane `NoiseModel`. Here is a simple example with two quantum errors that add two different 
  depolarizing errors based on the presence of different gates in the circuit:

  ```python
  import pennylane as qp
  import qiskit_aer.noise as noise

  error_1 = noise.depolarizing_error(0.001, 1) # 1-qubit noise
  error_2 = noise.depolarizing_error(0.01, 2) # 2-qubit noise

  noise_model = noise.NoiseModel()

  noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'ry'])
  noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
  ```
  
  ```pycon
  >>> qp.from_qiskit_noise(noise_model)
  NoiseModel({
    OpIn(['RZ', 'RY']): QubitChannel(num_kraus=4, num_wires=1)
    OpIn(['CNOT']): QubitChannel(num_kraus=16, num_wires=2)
  })
  ```

  Under the hood, PennyLane converts each quantum error in the Qiskit noise model into an equivalent 
  `qp.QubitChannel` operator with the same canonical 
  [Kraus representation](https://en.wikipedia.org/wiki/Quantum_operation#Kraus_operators). Currently, 
  noise models in PennyLane do not support readout errors. As such, those will be skipped during conversion
  if they are present in the Qiskit noise model.

  Make sure to `pip install pennylane-qiskit` to access this new feature!

<h4>Substantial upgrades to mid-circuit measurements using tree-traversal 🌳</h4>

* The `"tree-traversal"` algorithm for mid-circuit measurements (MCMs) on `default.qubit` has been internally redesigned for better 
  performance.
  [(#5868)](https://github.com/PennyLaneAI/pennylane/pull/5868)

  In the last release (v0.37), we introduced the tree-traversal MCM method, which was implemented in 
  a recursive way for simplicity. However, this had the unintended consequence of very deep [stack calls](https://en.wikipedia.org/wiki/Call_stack) 
  for circuits with many MCMs, resulting in [stack overflows](https://en.wikipedia.org/wiki/Stack_overflow) 
  in some cases. With this release, we've refactored the implementation of the tree-traversal method 
  into an iterative approach, which solves those inefficiencies when many MCMs are present in a circuit.
 
* The `tree-traversal` algorithm is now compatible with analytic-mode execution (`shots=None`).
  [(#5868)](https://github.com/PennyLaneAI/pennylane/pull/5868)

  ```python
  dev = qp.device("default.qubit")

  n_qubits = 5

  @qp.qnode(dev, mcm_method="tree-traversal")
  def circuit():
      for w in range(n_qubits):
          qp.Hadamard(w)
      
      for w in range(n_qubits - 1):
          qp.CNOT(wires=[w, w+1])

      for w in range(n_qubits):
          m = qp.measure(w)
          qp.cond(m == 1, qp.RX)(0.1967 * (w + 1), w)

      return [qp.expval(qp.Z(w)) for w in range(n_qubits)]
  ```

  ```pycon
  >>> circuit()
  [tensor(0.00964158, requires_grad=True),
   tensor(0.03819446, requires_grad=True),
   tensor(0.08455748, requires_grad=True),
   tensor(0.14694258, requires_grad=True),
   tensor(0.2229438, requires_grad=True)]
  ```

<h3>Improvements 🛠</h3>

<h4>Creating spin Hamiltonians</h4>

* Three new functions are now available for creating commonly-used spin Hamiltonians in PennyLane:
  [(#6106)](https://github.com/PennyLaneAI/pennylane/pull/6106)
  [(#6128)](https://github.com/PennyLaneAI/pennylane/pull/6128)

  * `qp.spin.transverse_ising` creates the [transverse-field Ising model](https://en.wikipedia.org/wiki/Transverse-field_Ising_model) Hamiltonian.
  * `qp.spin.heisenberg` creates the [Heisenberg model](https://en.wikipedia.org/wiki/Quantum_Heisenberg_model) Hamiltonian.
  * `qp.spin.fermi_hubbard` creates the [Fermi-Hubbard model](https://en.wikipedia.org/wiki/Hubbard_model) Hamiltonian.

  Each Hamiltonian can be instantiated by specifying a `lattice`, the number of [unit cells](https://en.wikipedia.org/wiki/Unit_cell), 
  `n_cells`, and the Hamiltonian parameters as keyword arguments. Here is an example with the transverse-field 
  Ising model:

  ```pycon
  >>> tfim_ham = qp.spin.transverse_ising(lattice="square", n_cells=[2, 2], coupling=0.5, h=0.2)
  >>> tfim_ham
  (
      -0.5 * (Z(0) @ Z(1))
    + -0.5 * (Z(0) @ Z(2))
    + -0.5 * (Z(1) @ Z(3))
    + -0.5 * (Z(2) @ Z(3))
    + -0.2 * X(0)
    + -0.2 * X(1)
    + -0.2 * X(2)
    + -0.2 * X(3)
  )
  ```

  The resulting object is a `qp.Hamiltonian` instance, making it easy to use in circuits like the following.

  ```python
  dev = qp.device("default.qubit", shots=1)

  @qp.qnode(dev)
  def circuit():
      return qp.expval(tfim_ham)
  ```

  ```
  >>> circuit()
  -2.0
  ```

  More features will be added to the `qp.spin` module in the coming releases, so stay tuned!

<h4>A Prep-Select-Prep template</h4>

* A new template called `qp.PrepSelPrep` has been added that implements a block-encoding of a linear
  combination of unitaries. 
  [(#5756)](https://github.com/PennyLaneAI/pennylane/pull/5756)
  [(#5987)](https://github.com/PennyLaneAI/pennylane/pull/5987)

  This operator acts as a nice wrapper for having to perform `qp.StatePrep`, `qp.Select`, and `qp.adjoint(qp.StatePrep)`
  in succession, which is quite common in many quantum algorithms (e.g., [LCU and block encoding](https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/)). Here is an example showing the equivalence
  between using `qp.PrepSelPrep` and `qp.StatePrep`, `qp.Select`, and `qp.adjoint(qp.StatePrep)`.

  ```python
  coeffs = [0.3, 0.1]
  alphas = (np.sqrt(coeffs) / np.linalg.norm(np.sqrt(coeffs)))
  unitaries = [qp.X(2), qp.Z(2)]

  lcu = qp.dot(coeffs, unitaries)
  control = [0, 1]

  def prep_sel_prep(alphas, unitaries):
      qp.StatePrep(alphas, wires=control, pad_with=0)
      qp.Select(unitaries, control=control)
      qp.adjoint(qp.StatePrep)(alphas, wires=control, pad_with=0)

  @qp.qnode(qp.device("default.qubit"))
  def circuit(lcu, control, alphas, unitaries):
      qp.PrepSelPrep(lcu, control)
      qp.adjoint(prep_sel_prep)(alphas, unitaries)
      return qp.state()
  ```

  ```pycon
  >>> import numpy as np
  >>> np.round(circuit(lcu, control, alphas, unitaries), decimals=2)
  tensor([1.+0.j -0.+0.j -0.+0.j -0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j], requires_grad=True)
  ```

<h4>QChem improvements</h4>

* Molecules and Hamiltonians can now be constructed for all the elements present in the periodic table.
  [(#5821)](https://github.com/PennyLaneAI/pennylane/pull/5821)

  This new feature is made possible by integrating with the [basis-set-exchange package](https://pypi.org/project/basis-set-exchange/).
  If loading basis sets from `basis-set-exchange` is needed for your molecule, make sure that you 
  `pip install basis-set-exchange` and set `load_data=True`.

  ```python
  symbols  = ['Ti', 'Ti']
  geometry = np.array([[0.0, 0.0, -1.1967],
                      [0.0, 0.0,  1.1967]], requires_grad=True)
  mol = qp.qchem.Molecule(symbols, geometry, load_data=True)
  ```

  ```pycon
  >>> mol.n_electrons
  44
  ```

* `qp.UCCSD` now accepts an additional optional argument, `n_repeats`, which defines the number of
  times the UCCSD template is repeated. This can improve the accuracy of the template by reducing
  the Trotter error, but would result in deeper circuits.
  [(#5801)](https://github.com/PennyLaneAI/pennylane/pull/5801)

* The `qp.qchem.qubit_observable` function has been modified to return an ascending wire order for molecular 
  Hamiltonians.
  [(#5950)](https://github.com/PennyLaneAI/pennylane/pull/5950)

* A new method called `to_mat` has been added to the `qp.FermiWord` and `qp.FermiSentence` classes, 
  which allows for computing the matrix representation of these Fermi operators.
  [(#5920)](https://github.com/PennyLaneAI/pennylane/pull/5920)

<h4>Improvements to operators</h4>

* `qp.GlobalPhase` now supports parameter broadcasting.
  [(#5923)](https://github.com/PennyLaneAI/pennylane/pull/5923)

* `qp.Hermitian` now has a `compute_decomposition` method.
  [(#6062)](https://github.com/PennyLaneAI/pennylane/pull/6062)

* The implementation of `qp.PhaseShift`, `qp.S`, and `qp.T` has been improved, resulting in faster
  circuit execution times.
  [(#5876)](https://github.com/PennyLaneAI/pennylane/pull/5876)

* The `qp.CNOT` operator no longer decomposes into itself. Instead, it raises a `qp.DecompositionUndefinedError`.
  [(#6039)](https://github.com/PennyLaneAI/pennylane/pull/6039)

<h4>Mid-circuit measurements</h4>

* The `qp.dynamic_one_shot` transform now supports circuits using the `"tensorflow"` interface.
  [(#5973)](https://github.com/PennyLaneAI/pennylane/pull/5973)

* If the conditional does not include a mid-circuit measurement, then `qp.cond`
  will automatically evaluate conditionals using standard Python control flow.
  [(#6016)](https://github.com/PennyLaneAI/pennylane/pull/6016)

  This allows `qp.cond` to be used to represent a wider range of conditionals:

  ```python
  dev = qp.device("default.qubit", wires=1)

  @qp.qnode(dev)
  def circuit(x):
      c = qp.cond(x > 2.7, qp.RX, qp.RZ)
      c(x, wires=0)
      return qp.probs(wires=0)
  ```

  ```pycon
  >>> print(qp.draw(circuit)(3.8))
  0: ──RX(3.80)─┤  Probs
  >>> print(qp.draw(circuit)(0.54))
  0: ──RZ(0.54)─┤  Probs
  ```

<h4>Transforms</h4>

* `qp.transforms.single_qubit_fusion` and `qp.transforms.merge_rotations` now respect global phases.
  [(#6031)](https://github.com/PennyLaneAI/pennylane/pull/6031)

* A new transform called `qp.transforms.diagonalize_measurements` has been added. This transform converts 
  measurements to the computational basis by applying the relevant diagonalizing gates. It can be set 
  to diagonalize only a subset of the base observables `{qp.X, qp.Y, qp.Z, qp.Hadamard}`.
  [(#5829)](https://github.com/PennyLaneAI/pennylane/pull/5829)

* A new transform called `split_to_single_terms` has been added. This transform splits expectation values 
  of sums into multiple single-term measurements on a single tape, providing better support for simulators
  that can handle non-commuting observables but don't natively support multi-term observables.
  [(#5884)](https://github.com/PennyLaneAI/pennylane/pull/5884)

* New functionality has been added to natively support exponential extrapolation when using `qp.transforms.mitigate_with_zne`. 
  This allows users to have more control over the error mitigation protocol without needing to add further 
  dependencies.
  [(#5972)](https://github.com/PennyLaneAI/pennylane/pull/5972)

<h4>Capturing and representing hybrid programs</h4>

* `qp.for_loop` now supports `range`-like syntax with default `step=1`.
  [(#6068)](https://github.com/PennyLaneAI/pennylane/pull/6068)

* Applying `adjoint` and `ctrl` to a quantum function can now be captured into plxpr. Furthermore, the 
  `qp.cond` function can be captured into plxpr.
  [(#5966)](https://github.com/PennyLaneAI/pennylane/pull/5966)
  [(#5967)](https://github.com/PennyLaneAI/pennylane/pull/5967)
  [(#5999)](https://github.com/PennyLaneAI/pennylane/pull/5999)
  [(#6058)](https://github.com/PennyLaneAI/pennylane/pull/6058)

* During experimental program capture, functions that accept and/or return `pytree` structures can now 
  be handled in the `qp.QNode` call, `qp.cond`, `qp.for_loop` and `qp.while_loop`. 
  [(#6081)](https://github.com/PennyLaneAI/pennylane/pull/6081)

* During experimental program capture, QNodes can now use closure variables.
  [(#6052)](https://github.com/PennyLaneAI/pennylane/pull/6052)

* Mid-circuit measurements can now be captured with `qp.capture` enabled.
  [(#6015)](https://github.com/PennyLaneAI/pennylane/pull/6015)

* `qp.for_loop` can now be captured into plxpr.
  [(#6041)](https://github.com/PennyLaneAI/pennylane/pull/6041)
  [(#6064)](https://github.com/PennyLaneAI/pennylane/pull/6064)

* `qp.for_loop` and `qp.while_loop` now fall back to standard Python control flow if `@qjit` is not 
  present, allowing the same code to work with and without `@qjit` without any rewrites.
  [(#6014)](https://github.com/PennyLaneAI/pennylane/pull/6014)

  ```python
  dev = qp.device("lightning.qubit", wires=3)

  @qp.qnode(dev)
  def circuit(x, n):

      @qp.for_loop(0, n, 1)
      def init_state(i):
          qp.Hadamard(wires=i)

      init_state()

      @qp.for_loop(0, n, 1)
      def apply_operations(i, x):
          qp.RX(x, wires=i)

          @qp.for_loop(i + 1, n, 1)
          def inner(j):
              qp.CRY(x**2, [i, j])

          inner()
          return jnp.sin(x)

      apply_operations(x)
      return qp.probs()
  ```

  ```pycon
  >>> print(qp.draw(circuit)(0.5, 3))
  0: ──H──RX(0.50)─╭●────────╭●──────────────────────────────────────┤  Probs
  1: ──H───────────╰RY(0.25)─│──────────RX(0.48)─╭●──────────────────┤  Probs
  2: ──H─────────────────────╰RY(0.25)───────────╰RY(0.23)──RX(0.46)─┤  Probs
  >>> circuit(0.5, 3)
  array([0.125     , 0.125     , 0.09949758, 0.15050242, 0.07594666,
       0.11917543, 0.08942104, 0.21545687])
  >>> qp.qjit(circuit)(0.5, 3)
  Array([0.125     , 0.125     , 0.09949758, 0.15050242, 0.07594666,
       0.11917543, 0.08942104, 0.21545687], dtype=float64)
  ```

<h4>Community contributions 🥳</h4>

* Fixed a bug in `qp.ThermalRelaxationError` where there was a typo from `tq` to `tg`.
  [(#5988)](https://github.com/PennyLaneAI/pennylane/issues/5988)

* Readout error has been added using parameters `readout_relaxation_probs` and `readout_misclassification_probs` 
  on the `default.qutrit.mixed` device. These parameters add a `qp.QutritAmplitudeDamping`  and a `qp.TritFlip` 
  channel, respectively, after measurement diagonalization. The amplitude damping error represents the 
  potential for relaxation to occur during longer measurements. The trit flip error represents misclassification 
  during readout.
  [(#5842)](https://github.com/PennyLaneAI/pennylane/pull/5842)

* `qp.ops.qubit.BasisStateProjector` now has a `compute_sparse_matrix` method that computes the sparse 
  CSR matrix representation of the projector onto the given basis state.
  [(#5790)](https://github.com/PennyLaneAI/pennylane/pull/5790)

<h4>Other improvements</h4>

* `qp.pauli.group_observables` now uses `rustworkx` colouring algorithms to solve the 
  [Minimum Clique Cover problem](https://en.wikipedia.org/wiki/Clique_cover), resulting in orders of
  magnitude performance improvements.
  [(#6043)](https://github.com/PennyLaneAI/pennylane/pull/6043)

  This adds two new options for the `method` argument: `dsatur` (degree of saturation) and `gis` (independent 
  set). In addition, the creation of the adjacency matrix now takes advantage of the symplectic representation 
  of the Pauli observables. 
  
  Additionally, a new function called `qp.pauli.compute_partition_indices` has been added to calculate 
  the indices from the partitioned observables more efficiently. These changes improve the wall time 
  of `qp.LinearCombination.compute_grouping` and the `grouping_type='qwc'` by orders of magnitude. 

* `qp.counts` measurements with `all_outcomes=True` can now be used with JAX jitting. Additionally, 
  measurements broadcasted across all available wires (e.g., `qp.probs()`) can now be used with JAX 
  jit and devices that allow dynamic numbers of wires (only `'default.qubit'` currently).
  [(#6108)](https://github.com/PennyLaneAI/pennylane/pull/6108/)

* `qp.ops.op_math.ctrl_decomp_zyz` can now decompose special unitaries with multiple control wires.
  [(#6042)](https://github.com/PennyLaneAI/pennylane/pull/6042)

* A new method called `process_density_matrix` has been added to the `ProbabilityMP` and `DensityMatrixMP`
  measurement processes, allowing for more efficient handling of quantum density matrices, particularly 
  with batch processing support. This method simplifies the calculation of probabilities from quantum 
  states represented as density matrices.
  [(#5830)](https://github.com/PennyLaneAI/pennylane/pull/5830)

* `SProd.terms` now flattens out the terms if the base is a multi-term observable.
  [(#5885)](https://github.com/PennyLaneAI/pennylane/pull/5885)

* `qp.QNGOptimizer` now supports cost functions with multiple arguments, updating each argument independently.
  [(#5926)](https://github.com/PennyLaneAI/pennylane/pull/5926)

* `semantic_version` has been removed from the list of required packages in PennyLane. 
  [(#5836)](https://github.com/PennyLaneAI/pennylane/pull/5836)

* `qp.devices.LegacyDeviceFacade` has been added to map the legacy devices to the new device interface, 
  making it easier for developers to develop legacy devices.
  [(#5927)](https://github.com/PennyLaneAI/pennylane/pull/5927)

* `StateMP.process_state` now defines rules in `cast_to_complex` for complex casting, avoiding a superfluous 
  statevector copy in PennyLane-Lightning simulations.
  [(#5995)](https://github.com/PennyLaneAI/pennylane/pull/5995)

* `QuantumScript.hash` is now cached, leading to performance improvements.
  [(#5919)](https://github.com/PennyLaneAI/pennylane/pull/5919)

* Observable validation for `default.qubit` is now based on execution mode (analytic vs. finite shots) 
  and measurement type (sample measurement vs. state measurement). This improves our error handling when, 
  for example, non-hermitian operators are given to `qp.expval`.
  [(#5890)](https://github.com/PennyLaneAI/pennylane/pull/5890)

* A new `is_leaf` parameter has been added to the function `flatten` in the `qp.pytrees` module. This 
  is to allow for node flattening to be stopped for any node where the `is_leaf` optional argument evaluates 
  to being `True`.
  [(#6107)](https://github.com/PennyLaneAI/pennylane/issues/6107)

* A progress bar has been added to `qp.data.load()` when downloading a dataset.
  [(#5560)](https://github.com/PennyLaneAI/pennylane/pull/5560)

* Upgraded and simplified `StatePrep` and `AmplitudeEmbedding` templates.
  [(#6034)](https://github.com/PennyLaneAI/pennylane/pull/6034)
  [(#6170)](https://github.com/PennyLaneAI/pennylane/pull/6170)

* Upgraded and simplified `BasisState` and `BasisEmbedding` templates.
  [(#6021)](https://github.com/PennyLaneAI/pennylane/pull/6021)
  
<h3>Breaking changes 💔</h3>

* `MeasurementProcess.shape(shots: Shots, device:Device)` is now
  `MeasurementProcess.shape(shots: Optional[int], num_device_wires:int = 0)`. This has been done to 
  allow for jitting when a measurement is broadcasted across all available wires, but the device does 
  not specify wires.
  [(#6108)](https://github.com/PennyLaneAI/pennylane/pull/6108/)

* If the shape of a probability measurement is affected by a `Device.cutoff` property, it will no longer 
  work with jitting.
  [(#6108)](https://github.com/PennyLaneAI/pennylane/pull/6108/)

* `qp.GlobalPhase` is considered non-differentiable with tape transforms. As a consequence, `qp.gradients.finite_diff` 
  and `qp.gradients.spsa_grad` no longer support differentiating `qp.GlobalPhase` with state-based 
  outputs.
  [(#5620)](https://github.com/PennyLaneAI/pennylane/pull/5620) 

* The `CircuitGraph.graph` `rustworkx` graph now stores indices into the circuit as the node labels,
  instead of the operator/ measurement itself. This allows the same operator to occur multiple times 
  in the circuit.
  [(#5907)](https://github.com/PennyLaneAI/pennylane/pull/5907)

* The `queue_idx` attribute has been removed from the `Operator`, `CompositeOp`, and `SymbolicOp` classes.
  [(#6005)](https://github.com/PennyLaneAI/pennylane/pull/6005)

* `qp.from_qasm` no longer removes measurements from the QASM code. Use `measurements=[]` to remove 
  measurements from the original circuit.
  [(#5982)](https://github.com/PennyLaneAI/pennylane/pull/5982)

* `qp.transforms.map_batch_transform` has been removed, since transforms can be applied directly to 
  a batch of tapes. See `qp.transform` for more information.
  [(#5981)](https://github.com/PennyLaneAI/pennylane/pull/5981)

* `QuantumScript.interface` has been removed.
  [(#5980)](https://github.com/PennyLaneAI/pennylane/pull/5980)

<h3>Deprecations 👋</h3>

* The `decomp_depth` argument in `qp.device` has been deprecated.
  [(#6026)](https://github.com/PennyLaneAI/pennylane/pull/6026)

* The `max_expansion` argument in `qp.QNode` has been deprecated.
  [(#6026)](https://github.com/PennyLaneAI/pennylane/pull/6026)

* The `expansion_strategy` attribute `qp.QNode` has been deprecated.
  [(#5989)](https://github.com/PennyLaneAI/pennylane/pull/5989)

* The `expansion_strategy` argument has been deprecated in all of `qp.draw`, `qp.draw_mpl`, and `qp.specs`.
  The `level` argument should be used instead.
  [(#5989)](https://github.com/PennyLaneAI/pennylane/pull/5989)

* `Operator.expand` has been deprecated. Users should simply use `qp.tape.QuantumScript(op.decomposition())`
  for equivalent behaviour.
  [(#5994)](https://github.com/PennyLaneAI/pennylane/pull/5994)

* `qp.transforms.sum_expand` and `qp.transforms.hamiltonian_expand` have been deprecated. Users should 
  instead use `qp.transforms.split_non_commuting` for equivalent behaviour.
  [(#6003)](https://github.com/PennyLaneAI/pennylane/pull/6003)

* The `expand_fn` argument in `qp.execute` has been deprecated. Instead, please create a `qp.transforms.core.TransformProgram` 
  with the desired preprocessing and pass it to the `transform_program` argument of `qp.execute`.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* The `max_expansion` argument in `qp.execute` has been deprecated.
  Instead, please use `qp.devices.preprocess.decompose` with the desired expansion level, add it to 
  a `qp.transforms.core.TransformProgram` and pass it to the `transform_program` argument of `qp.execute`.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* The `override_shots` argument in `qp.execute` has been deprecated.
  Instead, please add the shots to the `QuantumTape`s to be executed.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* The `device_batch_transform` argument in `qp.execute` has been deprecated.
  Instead, please create a `qp.transforms.core.TransformProgram` with the desired preprocessing and pass it to the `transform_program` argument of `qp.execute`.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* `qp.qinfo.classical_fisher` and `qp.qinfo.quantum_fisher` have been deprecated.
  Instead, use `qp.gradients.classical_fisher` and `qp.gradients.quantum_fisher`.
  [(#5985)](https://github.com/PennyLaneAI/pennylane/pull/5985)

* The legacy devices `default.qubit.{autograd,torch,tf,jax,legacy}` have been deprecated.
  Instead, use `default.qubit`, as it now supports backpropagation through the several backends.
  [(#5997)](https://github.com/PennyLaneAI/pennylane/pull/5997)

* The logic for internally switching a device for a different backpropagation
  compatible device is now deprecated, as it was in place for the deprecated
  `default.qubit.legacy`.
  [(#6032)](https://github.com/PennyLaneAI/pennylane/pull/6032)

<h3>Documentation 📝</h3>

* The docstring for `qp.qinfo.quantum_fisher`, regarding the internally used functions and potentially 
  required auxiliary wires, has been improved.
  [(#6074)](https://github.com/PennyLaneAI/pennylane/pull/6074)

* The docstring for `QuantumScript.expand` and `qp.tape.tape.expand_tape` has been improved.
  [(#5974)](https://github.com/PennyLaneAI/pennylane/pull/5974)

<h3>Bug fixes 🐛</h3>

* The sparse matrix can now be computed for a product operator when one operand is a `GlobalPhase`
  on no wires.
  [(#6197)](https://github.com/PennyLaneAI/pennylane/pull/6197)

* For `default.qubit`, JAX is now used for sampling whenever the state is a JAX array. This fixes normalization issues
  that can occur when the state uses 32-bit precision.
  [(#6190)](https://github.com/PennyLaneAI/pennylane/pull/6190)

* Fix Pytree serialization of operators with empty shot vectors
  [(#6155)](https://github.com/PennyLaneAI/pennylane/pull/6155)

* Fixes an error in the `dynamic_one_shot` transform when used with sampling a single shot.
  [(#6149)](https://github.com/PennyLaneAI/pennylane/pull/6149)

* `qp.transforms.pattern_matching_optimization` now preserves the tape measurements.
  [(#6153)](https://github.com/PennyLaneAI/pennylane/pull/6153)

* `qp.transforms.broadcast_expand` no longer squeezes out batch sizes of size 1, as a batch size of 1 is still a
  batch size.
  [(#6147)](https://github.com/PennyLaneAI/pennylane/pull/6147)

* Catalyst replaced `argnum` with `argnums` in gradient related functions, therefore we updated the Catalyst
  calls to those functions in PennyLane.
  [(#6117)](https://github.com/PennyLaneAI/pennylane/pull/6117)

* `fuse_rot_angles` now returns NaN instead of incorrect derivatives at singular points.
  [(#6031)](https://github.com/PennyLaneAI/pennylane/pull/6031)

* `qp.GlobalPhase` and `qp.Identity` can now be captured with plxpr when acting on no wires.
  [(#6060)](https://github.com/PennyLaneAI/pennylane/pull/6060)

* Fixed `jax.grad` and `jax.jit` to work for `qp.AmplitudeEmbedding`, `qp.StatePrep` and `qp.MottonenStatePreparation`.
  [(#5620)](https://github.com/PennyLaneAI/pennylane/pull/5620) 

* Fixed a bug in `qp.center` that omitted elements from the center if they were
  linear combinations of input elements.
  [(#6049)](https://github.com/PennyLaneAI/pennylane/pull/6049)

* Fix a bug where the global phase returned by `one_qubit_decomposition` gained a broadcasting dimension.
  [(#5923)](https://github.com/PennyLaneAI/pennylane/pull/5923)

* Fixed a bug in `qp.SPSAOptimizer` that ignored keyword arguments in the objective function.
  [(#6027)](https://github.com/PennyLaneAI/pennylane/pull/6027)

* Fixed `dynamic_one_shot` for use with devices using the old device API, since `override_shots` was deprecated.
  [(#6024)](https://github.com/PennyLaneAI/pennylane/pull/6024)

* `CircuitGraph` can now handle circuits with the same operation instance occuring multiple times.
  [(#5907)](https://github.com/PennyLaneAI/pennylane/pull/5907)

* `qp.QSVT` has been updated to store wire order correctly.
  [(#5959)](https://github.com/PennyLaneAI/pennylane/pull/5959)

* `qp.devices.qubit.measure_with_samples` now returns the correct result if the provided measurements
  contain a sum of operators acting on the same wire.
  [(#5978)](https://github.com/PennyLaneAI/pennylane/pull/5978)

* `qp.AmplitudeEmbedding` has better support for features using low precision integer data types.
  [(#5969)](https://github.com/PennyLaneAI/pennylane/pull/5969)

* `qp.BasisState` and `qp.BasisEmbedding` now works with jax.jit, `lightning.qubit`, and give the correct 
  decomposition.
  [(#6021)](https://github.com/PennyLaneAI/pennylane/pull/6021)

* Jacobian shape has been fixed for measurements with dimension in `qp.gradients.vjp.compute_vjp_single`.
  [(5986)](https://github.com/PennyLaneAI/pennylane/pull/5986)

* `qp.lie_closure` now works with sums of Paulis.
  [(#6023)](https://github.com/PennyLaneAI/pennylane/pull/6023)

* Workflows that parameterize the coefficients of `qp.exp` are now jit-compatible.
  [(#6082)](https://github.com/PennyLaneAI/pennylane/pull/6082)

* Fixed a bug where `CompositeOp.overlapping_ops` changes the original ordering of operators, causing 
  an incorrect matrix to be generated for `Prod` with `Sum` as operands.
  [(#6091)](https://github.com/PennyLaneAI/pennylane/pull/6091)

* `qp.qsvt` now works with "Wx" convention and any number of angles.
  [(#6105)](https://github.com/PennyLaneAI/pennylane/pull/6105)

* Basis set data from the Basis Set Exchange library can now be loaded for elements with `SPD`-type orbitals.
  [(#6159)](https://github.com/PennyLaneAI/pennylane/pull/6159)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Tarun Kumar Allamsetty,
Guillermo Alonso,
Ali Asadi,
Utkarsh Azad,
Tonmoy T. Bhattacharya,
Gabriel Bottrill,
Jack Brown,
Ahmed Darwish,
Astral Cai,
Yushao Chen,
Ahmed Darwish,
Diksha Dhawan
Maja Franz,
Lillian M. A. Frederiksen,
Pietropaolo Frisoni,
Emiliano Godinez,
Austin Huang,
Renke Huang,
Josh Izaac,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Jorge Martinez de Lejarza,
William Maxwell,
Vincent Michaud-Rioux,
Anurav Modak,
Mudit Pandey,
Andrija Paurevic,
Erik Schultheis,
nate stemen,
David Wierichs,
