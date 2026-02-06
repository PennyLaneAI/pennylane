
# Release 0.39.0

<h3>New features since last release</h3>

<h4>Creating spin Hamiltonians on lattices 💞</h4>

* Functionality for creating custom Hamiltonians on arbitrary lattices has been added.
  [(#6226)](https://github.com/PennyLaneAI/pennylane/pull/6226)
  [(#6237)](https://github.com/PennyLaneAI/pennylane/pull/6237)

  Hamiltonians beyond the available boiler-plate ones in the `qp.spin` module can be created with the 
  addition of three new functions:

  * `qp.spin.Lattice`: a new object for instantiating customized lattices via primitive translation vectors and unit cell parameters,
  * `qp.spin.generate_lattice`: a utility function for creating standard `Lattice` objects, including `'chain'`, `'square'`, `'rectangle'`, `'triangle'`, `'honeycomb'`, `'kagome'`, `'lieb'`, `'cubic'`, `'bcc'`, `'fcc'`, and `'diamond'`,
  * `qp.spin.spin_hamiltonian`: generates a spin `Hamiltonian` object given a `Lattice` object with custom edges/nodes.

  An example is shown below for a :math:`3 \times 3` triangular lattice with open boundary conditions.

  ```python
  lattice = qp.spin.Lattice(
      n_cells=[3, 3],
      vectors=[[1, 0], [np.cos(np.pi/3), np.sin(np.pi/3)]],
      positions=[[0, 0]],
      boundary_condition=False
  )
  ```

  We can validate this `lattice` against `qp.spin.generate_lattice('triangle', ...)` by checking the 
  `lattice_points` (the :math:`(x, y)` coordinates of all sites in the lattice):

  ```pycon
  >>> lp = lattice.lattice_points
  >>> triangular_lattice = qp.spin.generate_lattice('triangle', n_cells=[3, 3])
  >>> np.allclose(lp, triangular_lattice.lattice_points)
  True
  ```

  The `edges` of the `Lattice` object are nearest-neighbour by default, where we can add edges by using
  its `add_edge` method. 

  Optionally, a `Lattice` object can have interactions and fields endowed to it by specifying values 
  for its `custom_edges` and `custom_nodes` keyword arguments. The Hamiltonian can then be extracted
  with the `qp.spin.spin_hamiltonian` function. An example is shown below for the transverse-field 
  Ising model Hamiltonian on a :math:`3 \times 3` triangular lattice. Note that the `custom_edges` and 
  `custom_nodes` keyword arguments only need to be defined for one unit cell repetition.

  ```python
  edges = [
      (0, 1), (0, 3), (1, 3)
  ]

  lattice = qp.spin.Lattice(
      n_cells=[3, 3],
      vectors=[[1, 0], [np.cos(np.pi/3), np.sin(np.pi/3)]],
      positions=[[0, 0]],
      boundary_condition=False,
      custom_edges=[[edge, ("ZZ", -1.0)] for edge in edges], 
      custom_nodes=[[i, ("X", -0.5)] for i in range(3*3)],
  )
  ```

  ```pycon
  >>> tfim_ham = qp.spin.transverse_ising('triangle', [3, 3], coupling=1.0, h=0.5)
  >>> tfim_ham == qp.spin.spin_hamiltonian(lattice=lattice)
  True
  ```

* More industry-standard spin Hamiltonians have been added in the `qp.spin` module.
  [(#6174)](https://github.com/PennyLaneAI/pennylane/pull/6174)
  [(#6201)](https://github.com/PennyLaneAI/pennylane/pull/6201)

  Three new industry-standard spin Hamiltonians are now available with PennyLane v0.39:

  * `qp.spin.emery` : the [Emery model](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.58.2794)
  * `qp.spin.haldane` : the [Haldane model](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.61.2015)
  * `qp.spin.kitaev` : the [Kitaev model](https://arxiv.org/abs/cond-mat/0506438)

  These additions accompany `qp.spin.heisenberg`, `qp.spin.transverse_ising`, and `qp.spin.fermi_hubbard`, 
  which were introduced in v0.38. 

<h4>Calculating Polynomials 🔢</h4>

* Polynomial functions can now be easily encoded into quantum circuits with `qp.OutPoly`.
  [(#6320)](https://github.com/PennyLaneAI/pennylane/pull/6320)

  A new template called `qp.OutPoly` is available, which provides the ability to encode a polynomial
  function in a quantum circuit. Given a polynomial function :math:`f(x_1, x_2, \cdots, x_N)`, `qp.OutPoly` 
  requires:

  * `f` : a standard Python function that represents :math:`f(x_1, x_2, \cdots, x_N)`,
  * `input_registers` (:math:`\vert x_1 \rangle`, :math:`\vert x_2 \rangle`, ..., :math:`\vert x_N \rangle`) : a list/tuple containing ``Wires`` objects that correspond to the embedded numeric values of :math:`x_1, x_2, \cdots, x_N`,
  * `output_wires` : the `Wires` for which the numeric value of :math:`f(x_1, x_2, \cdots, x_N)` is stored.

  Here is an example of using `qp.OutPoly` to calculate :math:`f(x_1, x_2) = 3x_1^2 - x_1x_2` for :math:`f(1, 2) = 1`.

  ```python
  wires = qp.registers({"x1": 1, "x2": 2, "output": 2})

  def f(x1, x2):
      return 3 * x1 ** 2 - x1 * x2

  @qp.qnode(qp.device("default.qubit", shots = 1))
  def circuit():
      # load values of x1 and x2
      qp.BasisEmbedding(1, wires=wires["x1"])
      qp.BasisEmbedding(2, wires=wires["x2"])

      # apply the polynomial
      qp.OutPoly(
          f,
          input_registers = [wires["x1"], wires["x2"]],
          output_wires = wires["output"])

      return qp.sample(wires=wires["output"])
  ```

  ```pycon
  >>> circuit()
  array([0, 1])
  ```

  The result, `[0, 1]`, is the binary representation of :math:`1`. By default, the result is calculated
  modulo :math:`2^\text{len(output_wires)}` but can be overridden with the `mod` keyword argument.

<h4>Readout Noise 📠</h4>

* Readout errors can now be included in `qp.NoiseModel` and `qp.add_noise` with the new `qp.noise.meas_eq`
  function.
  [(#6321)](https://github.com/PennyLaneAI/pennylane/pull/6321/)

  Measurement/readout errors can be specified in a similar fashion to regular gate noise in PennyLane: 
  a newly added Boolean function called `qp.noise.meas_eq` that accepts a measurement function 
  (e.g., `qp.expval`, `qp.sample`, or any other function that can be returned from a QNode) that, 
  when present in the QNode, inserts a noisy operation via `qp.noise.partial_wires` or a custom noise 
  function. Readout noise in PennyLane also follows the insertion convention, where the specified noise 
  is inserted *before* the measurement.

  Here is an example of adding `qp.PhaseFlip` noise to any `qp.expval` measurement:

  ```python
  c0 = qp.noise.meas_eq(qp.expval)
  n0 = qp.noise.partial_wires(qp.PhaseFlip, 0.2)
  ```

  To include this in a `qp.NoiseModel`, use its `meas_map` keyword argument:

  ```python
  # gate-based noise
  c1 = qp.noise.wires_in([0, 2]) 
  n1 = qp.noise.partial_wires(qp.RY, -0.42)

  noise_model = qp.NoiseModel({c1: n1}, meas_map={c0: n0})
  ```

  ```pycon
  >>> noise_model
  NoiseModel({
    WiresIn([0, 2]): RY(phi=-0.42)
  },
  meas_map = {
      MeasEq(expval): PhaseFlip(p=0.2)
  })
  ```

  `qp.noise.meas_eq` can also be combined with other Boolean functions in `qp.noise` via bitwise operators
  for more versatility.

  To add this `noise_model` to a circuit, use the `qp.add_noise` transform as per usual. For example, 

  ```python
  @qp.qnode(qp.device("default.mixed", wires=3))
  def circuit():
      qp.RX(0.1967, wires=0)
      for i in range(3):
          qp.Hadamard(i)

      return qp.expval(qp.X(0) @ qp.X(1))
  ```

  ```pycon
  >>> noisy_circuit = qp.add_noise(circuit, noise_model)
  >>> print(qp.draw(noisy_circuit)())
  0: ──RX(0.20)──RY(-0.42)────────H──RY(-0.42)──PhaseFlip(0.20)─┤ ╭<X@X>
  1: ──H─────────PhaseFlip(0.20)────────────────────────────────┤ ╰<X@X>
  2: ──H─────────RY(-0.42)──────────────────────────────────────┤    
  >>> print(circuit(), noisy_circuit())
  0.9807168489852615 0.35305806563469433
  ```

<h4>User-friendly decompositions 📠</h4>

* A new transform called `qp.transforms.decompose` has been added to better facilitate the custom decomposition
  of operators in PennyLane circuits.
  [(#6334)](https://github.com/PennyLaneAI/pennylane/pull/6334)

  Previous to the addition of `qp.transforms.decompose`, decomposing operators in PennyLane had to 
  be done by specifying a `stopping_condition` in `qp.device.preprocess.decompose`. With `qp.transforms.decompose`, 
  the user-interface for specifying decompositions is much simpler and more versatile.
  
  Decomposing gates in a circuit can be done a few ways:

  * Specifying a `gate_set` comprising PennyLane `Operator`s to decompose into:

    ```python
    from functools import partial

    dev = qp.device('default.qubit')
    allowed_gates = {qp.Toffoli, qp.RX, qp.RZ}

    @partial(qp.transforms.decompose, gate_set=allowed_gates)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=[0])
        qp.Toffoli(wires=[0, 1, 2])
        return qp.expval(qp.Z(0))
    ```

    ```pycon
    >>> print(qp.draw(circuit)())
    0: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╭●─┤  <Z>
    1: ───────────────────────────────├●─┤
    2: ───────────────────────────────╰X─┤
    ```
  
  * Specifying a `gate_set` that is defined by a rule (Boolean function). For example, one can specify 
    an arbitrary gate set to decompose into, so long as the resulting gates only act on one or two qubits:

    ```python
    @partial(qp.transforms.decompose, gate_set = lambda op: len(op.wires) <= 2)
    @qp.qnode(dev)
    def circuit():
        qp.Toffoli(wires=[0, 1, 2])
        return qp.expval(qp.Z(0))
    ```

    ```pycon
    >>> print(qp.draw(circuit)())
    0: ───────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
    1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤     
    2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤
    ```
  
  * Specifying a value for `max_expansion`. By default, decomposition occurs recursively until the desired 
    gate set is reached, but this can be overridden to control the number of passes. 

    ```python
    phase = 1.0
    target_wires = [0]
    unitary = qp.RX(phase, wires=0).matrix()
    n_estimation_wires = 1
    estimation_wires = range(1, n_estimation_wires + 1)

    def qfunc():
        qp.QuantumPhaseEstimation(
            unitary,
            target_wires=target_wires,
            estimation_wires=estimation_wires,
        )

    decompose_once = qp.transforms.decompose(qfunc, max_expansion=1)
    decompose_twice = qp.transforms.decompose(qfunc, max_expansion=2)
    ```

    ```pycon
    >>> print(qp.draw(decompose_once)())
    0: ────╭U(M0)¹───────┤  
    1: ──H─╰●───────QFT†─┤  
    M0 = 
    [[0.87758256+0.j         0.        -0.47942554j]
    [0.        -0.47942554j 0.87758256+0.j        ]]
    >>> print(qp.draw(decompose_twice)())
    0: ──RZ(1.57)──RY(0.50)─╭X──RY(-0.50)──RZ(-6.28)─╭X──RZ(4.71)─┤  
    1: ──H──────────────────╰●───────────────────────╰●──H†───────┤ 
    ```

<h3>Improvements 🛠</h3>

<h4>qjit/jit compatibility and improvements</h4>

[Quantum just-in-time (qjit) compilation](https://pennylane.ai/qml/glossary/what-is-quantum-just-in-time-qjit-compilation/) 
is the compilation of hybrid quantum-classical programs during execution of a program, rather than before 
execution. PennyLane provides just-in-time compilation with its `@qp.qjit` decorator, powered by the 
[Catalyst](https://docs.pennylane.ai/projects/catalyst/en/stable/index.html) compiler.

* Several templates are now compatible with qjit:
  * `qp.AmplitudeAmplification`
    [(#6306)](https://github.com/PennyLaneAI/pennylane/pull/6306)
  * All quantum arithmetic templates
    [(#6307)](https://github.com/PennyLaneAI/pennylane/pull/6307)
  * `qp.Qubitization`
    [(#6305)](https://github.com/PennyLaneAI/pennylane/pull/6305)
  
* The sample-based measurements now support samples that are JAX tracers, allowing compatiblity with 
  qjit workflows.
  [(#6211)](https://github.com/PennyLaneAI/pennylane/pull/6211)

* The `qp.FABLE` template now returns the correct value when jit is enabled.
  [(#6263)](https://github.com/PennyLaneAI/pennylane/pull/6263)

* `qp.metric_tensor` is now jit compatible.
  [(#6468)](https://github.com/PennyLaneAI/pennylane/pull/6468)

* `qp.QutritBasisStatePreparation` is now jit compatible.
  [(#6308)](https://github.com/PennyLaneAI/pennylane/pull/6308)

* All PennyLane templates are now unit tested to ensure jit compatibility.
  [(#6309)](https://github.com/PennyLaneAI/pennylane/pull/6309)

<h4>Various improvements to fermionic operators</h4>

* `qp.fermi.FermiWord` and `qp.fermi.FermiSentence` are now compatible with JAX arrays.
  [(#6324)](https://github.com/PennyLaneAI/pennylane/pull/6324)

  Fermionic operators interacting, in some way, with a JAX array will no longer raise an error. For 
  example:

  ```pycon
  >>> import jax.numpy as jnp
  >>> w = qp.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
  >>> jnp.array([3]) * w
  FermiSentence({FermiWord({(0, 0): '+', (1, 1): '-'}): Array([3], dtype=int32)})
  ```

* The `qp.fermi.FermiWord` class now has a `shift_operator` method to apply anti-commutator relations.
  [(#6196)](https://github.com/PennyLaneAI/pennylane/pull/6196)

  ```pycon
  >>> w = qp.fermi.FermiWord({(0, 0): '+', (1, 1): '-'})
  >>> w
  FermiWord({(0, 0): '+', (1, 1): '-'})
  >>> w.shift_operator(0, 1)
  FermiSentence({FermiWord({(0, 1): '-', (1, 0): '+'}): -1})
  ```

* The `qp.fermi.FermiWord` and `qp.fermi.FermiSentence` classes now an `adjoint` method.
  [(#6166)](https://github.com/PennyLaneAI/pennylane/pull/6166)

  ```pycon
  >>> w = qp.fermi.FermiWord({(0, 0): '+', (1, 1): '-'})
  >>> w.adjoint()
  FermiWord({(0, 1): '+', (1, 0): '-'})
  ```

* The `to_mat` methods for `qp.fermi.FermiWord` and `qp.fermi.FermiSentence` now optionally return
  a sparse matrix with the `format` keyword argument.
  [(#6173)](https://github.com/PennyLaneAI/pennylane/pull/6173)

  ```pycon
  >>> w = qp.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'}) 
  >>> w.to_mat(format="dense")
  array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
  >>> w.to_mat(format="csr")
  <4x4 sparse matrix of type '<class 'numpy.complex128'>'
    with 1 stored elements in Compressed Sparse Row format>
  ```

* When printed, `qp.fermi.FermiWord` and `qp.fermi.FermiSentence` now return a unique representation 
  of the object.
  [(#6167)](https://github.com/PennyLaneAI/pennylane/pull/6167)

  ```pycon
  >>> w = qp.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
  >>> print(w)
  a⁺(0) a(1)
  ```

* `qp.qchem.excitations` now optionally returns fermionic operators with a new `fermionic` keyword 
  argument (defaults to `False`).
  [(#6171)](https://github.com/PennyLaneAI/pennylane/pull/6171)

  ```pycon
  >>> singles, doubles = excitations(electrons, orbitals, fermionic=True)
  >>> print(singles)
  [FermiWord({(0, 0): '+', (1, 2): '-'}), FermiWord({(0, 1): '+', (1, 3): '-'})]
  ```

<h4>A new optimizer</h4>

* A new optimizer called `qp.MomentumQNGOptimizer` has been added. It inherits from the basic `qp.QNGOptimizer` 
  optimizer and requires one additional hyperparameter: the momentum coefficient, :math:`0 \leq \rho < 1`, 
  the default value being :math:`\rho=0.9`. For :math:`\rho=0`, `qp.MomentumQNGOptimizer` reduces down
  to `qp.QNGOptimizer`.
  [(#6240)](https://github.com/PennyLaneAI/pennylane/pull/6240)

<h4>Other Improvements</h4>

* `default.tensor` can now handle mid-circuit measurements via the deferred measurement principle.
  [(#6408)](https://github.com/PennyLaneAI/pennylane/pull/6408)

* `process_density_matrix` was implemented in 5 `StateMeasurement` subclasses: `ExpVal`, `Var`, `Purity`, 
  `MutualInformation`, and `VnEntropy`. This facilitates future support for mixed-state devices and 
  expanded density matrix operations. Also, a fix was made in the `ProbabilityMP` class to use `qp.math.sqrt` 
  instead of `np.sqrt`.
  [(#6330)](https://github.com/PennyLaneAI/pennylane/pull/6330)

* The decomposition for `qp.Qubitization` has been improved to use `qp.PrepSelPrep`.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* A `ReferenceQubit` device (shortname: `"reference.qubit"`) has been introduced for testing purposes 
  and referencing for future plugin development.
  [(#6181)](https://github.com/PennyLaneAI/pennylane/pull/6181)

* `qp.transforms.mitigate_with_zne` now gives a clearer error message when being applied on circuits, not devices, 
  that include channel noise.
  [(#6346)](https://github.com/PennyLaneAI/pennylane/pull/6346)

* A new function called `get_best_diff_method` has been added to `qp.workflow`.
  [(#6399)](https://github.com/PennyLaneAI/pennylane/pull/6399)

* A new method called `construct_tape` has been added to `qp.workflow` for users to construct single 
  tapes from a `QNode`.
  [(#6419)](https://github.com/PennyLaneAI/pennylane/pull/6419)

* An infrastructural improvement was made to PennyLane datasets that allows us to upload datasets more 
  easily.
  [(#6126)](https://github.com/PennyLaneAI/pennylane/pull/6126)

* `qp.Hadamard` now has a more friendly alias: `qp.H`.
  [(#6450)](https://github.com/PennyLaneAI/pennylane/pull/6450)

* A Boolean keyword argument called `show_wire_labels` has been added to `draw` and `draw_mpl`, which 
  hides wire labels when set to `False` (the default is `True`).
  [(#6410)](https://github.com/PennyLaneAI/pennylane/pull/6410)

* A new function called `sample_probs` has been added to the `qp.devices.qubit` and `qp.devices.qutrit_mixed` 
  modules. This function takes probability distributions as input and returns sampled outcomes, simplifying 
  the sampling process by separating it from other operations in the measurement chain and improving modularity.
  [(#6354)](https://github.com/PennyLaneAI/pennylane/pull/6354)

* `qp.labs` has been added to the PennyLane documentation.
  [(#6397)](https://github.com/PennyLaneAI/pennylane/pull/6397)
  [(#6369)](https://github.com/PennyLaneAI/pennylane/pull/6369)

* A `has_sparse_matrix` property has been added to `Operator` to indicate whether a sparse matrix is 
  defined.
  [(#6278)](https://github.com/PennyLaneAI/pennylane/pull/6278)
  [(#6310)](https://github.com/PennyLaneAI/pennylane/pull/6310)

* `qp.matrix` now works with empty objects (e.g., empty tapes, QNodes and quantum functions that do
  not call operations, and single operators with empty decompositions).
  [(#6347)](https://github.com/PennyLaneAI/pennylane/pull/6347)
  
  ```python
  dev = qp.device("default.qubit", wires=1)

  @qp.qnode(dev)
  def node():
      return qp.expval(qp.Z(0))
  ```

  ```pycon
  >>> qp.matrix(node)()
  array([[1., 0.],
         [0., 1.]])
  ```

* PennyLane is now compatible with NumPy 2.0.
  [(#6061)](https://github.com/PennyLaneAI/pennylane/pull/6061)
  [(#6258)](https://github.com/PennyLaneAI/pennylane/pull/6258)
  [(#6342)](https://github.com/PennyLaneAI/pennylane/pull/6342)

* PennyLane is now compatible with Jax 0.4.28.
  [(#6255)](https://github.com/PennyLaneAI/pennylane/pull/6255)

* The `diagonalize_measurements` transform now uses a more efficient method of diagonalization when possible, 
  based on the `pauli_rep` of the relevant observables.
  [(#6113)](https://github.com/PennyLaneAI/pennylane/pull/6113/)

* The `QuantumScript.copy` method now takes `operations`, `measurements`, `shots` and `trainable_params` 
  as keyword arguments. If any of these are passed when copying a tape, the specified attributes will 
  replace the copied attributes on the new tape.
  [(#6285)](https://github.com/PennyLaneAI/pennylane/pull/6285)
  [(#6363)](https://github.com/PennyLaneAI/pennylane/pull/6363)

* The `Hermitian` operator now has a `compute_sparse_matrix` implementation.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* When an observable is repeated on a tape, `tape.diagonalizing_gates` no longer returns the
  diagonalizing gates for each instance of the observable. Instead, the diagonalizing gates of
  each observable on the tape are included just once.
  [(#6288)](https://github.com/PennyLaneAI/pennylane/pull/6288)

* The number of diagonalizing gates returned in `qp.specs` now follows the `level` keyword argument 
  regarding whether the diagonalizing gates are modified by device, instead of always counting unprocessed 
  diagonalizing gates.
  [(#6290)](https://github.com/PennyLaneAI/pennylane/pull/6290)

* A more sensible error message is raised from a `RecursionError` encountered when accessing properties 
  and methods of a nested `CompositeOp` or `SProd`.
  [(#6375)](https://github.com/PennyLaneAI/pennylane/pull/6375)

  By default, `qp.sum` and `qp.prod` set `lazy=True`, which keeps its operands nested. Given the recursive 
  nature of such structures, if there are too many levels of nesting, a `RecursionError` would occur 
  when accessing many of the properties and methods. 

* The performance of the decomposition of `qp.QFT` has been improved.
  [(#6434)](https://github.com/PennyLaneAI/pennylane/pull/6434)

<h4>Capturing and representing hybrid programs</h4>

* `qp.wires.Wires` now accepts JAX arrays as input. In many workflows with JAX, PennyLane may attempt 
  to assign a JAX array to a `Wires` object, which will cause an error since JAX arrays are not hashable. Now, 
  JAX arrays are valid `Wires` types. Furthermore, a `FutureWarning` is no longer raised in `JAX 0.4.30+` 
  when providing JAX tracers as input to `qp.wires.Wires`.
  [(#6312)](https://github.com/PennyLaneAI/pennylane/pull/6312)

* A new function called `qp.capture.make_plxpr` has been added to take a function and create a `Callable` 
  that, when called, will return a PLxPR representation of the input function.
  [(#6326)](https://github.com/PennyLaneAI/pennylane/pull/6326)`

* Differentiation of hybrid programs via `qp.grad` and `qp.jacobian` can now be captured with PLxPR. 
  When evaluating a captured `qp.grad` (`qp.jacobian`) instruction, it will dispatch to `jax.grad` 
  (`jax.jacobian`), which differs from the Autograd implementation without capture. Pytree inputs and 
  outputs are supported.
  [(#6120)](https://github.com/PennyLaneAI/pennylane/pull/6120)
  [(#6127)](https://github.com/PennyLaneAI/pennylane/pull/6127)
  [(#6134)](https://github.com/PennyLaneAI/pennylane/pull/6134)

* Unit testing for capturing nested control flow has been improved.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

* All higher order primitives now use `jax.core.Jaxpr` as metadata instead of sometimes using `jax.core.ClosedJaxpr` 
  or `jax.core.Jaxpr`.
  [(#6319)](https://github.com/PennyLaneAI/pennylane/pull/6319)

<h3>Breaking changes 💔</h3>

* Red-herring validation in `QNode.construct` has been removed, which fixed a bug with `qp.GlobalPhase`.
  [(#6373)](https://github.com/PennyLaneAI/pennylane/pull/6373)

  Removing the `AllWires` validation in `QNode.construct` was addressed as a solution to the following 
  example not being able to run:

  ```python
  @qp.qnode(qp.device('default.qubit', wires=2))
  def circuit(x):
      qp.GlobalPhase(x, wires=0)
      return qp.state()
  ```

  ```pycon
  >>> circuit(0.5)
  array([0.87758256-0.47942554j, 0.        +0.j        ,
       1.        +0.j        , 0.        +0.j        ])
  ```

* The `simplify` argument in `qp.Hamiltonian` and `qp.ops.LinearCombination` has been removed.
  Instead, `qp.simplify()` can be called on the constructed operator.
  [(#6279)](https://github.com/PennyLaneAI/pennylane/pull/6279)

* The functions `qp.qinfo.classical_fisher` and `qp.qinfo.quantum_fisher` have been removed and migrated 
  to the `qp.gradients` module. `qp.gradients.classical_fisher` and `qp.gradients.quantum_fisher` 
  should be used instead.
  [(#5911)](https://github.com/PennyLaneAI/pennylane/pull/5911)

* Python 3.9 is no longer supported. Please update to 3.10 or newer.
  [(#6223)](https://github.com/PennyLaneAI/pennylane/pull/6223)

* `default.qubit.legacy`, `default.qubit.tf`, `default.qubit.torch`, `default.qubit.jax`, and 
  `default.qubit.autograd` have been removed. Please use `default.qubit` for all interfaces.
  [(#6207)](https://github.com/PennyLaneAI/pennylane/pull/6207)
  [(#6208)](https://github.com/PennyLaneAI/pennylane/pull/6208)
  [(#6209)](https://github.com/PennyLaneAI/pennylane/pull/6209)
  [(#6210)](https://github.com/PennyLaneAI/pennylane/pull/6210)
  [(#6266)](https://github.com/PennyLaneAI/pennylane/pull/6266)

* `expand_fn`, `max_expansion`, `override_shots`, and `device_batch_transform` have been removed from 
  the signature of `qp.execute`.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `max_expansion` and `expansion_strategy` have been removed from the QNode.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `expansion_strategy` has been removed from `qp.draw`, `qp.draw_mpl`, and `qp.specs`. `max_expansion` 
  has been removed from `qp.specs`, as it had no impact on the output.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `qp.transforms.hamiltonian_expand` and `qp.transforms.sum_expand` have been removed. Please use 
  `qp.transforms.split_non_commuting` instead.
  [(#6204)](https://github.com/PennyLaneAI/pennylane/pull/6204)

* The `decomp_depth` keyword argument to `qp.device` has been removed.
  [(#6234)](https://github.com/PennyLaneAI/pennylane/pull/6234)

* `Operator.expand` has been removed. Please use `qp.tape.QuantumScript(op.decomposition())` instead.
  [(#6227)](https://github.com/PennyLaneAI/pennylane/pull/6227)

* The native folding method `qp.transforms.fold_global` for the `qp.transforms.mitigate_with_zne`
  transform no longer expands the circuit automatically. Instead, the user should apply `qp.transforms.decompose` to
  decompose a circuit into a target gate set before applying `fold_global` or `mitigate_with_zne`.
  [(#6382)](https://github.com/PennyLaneAI/pennylane/pull/6382)

* The `LightningVJPs` class has been removed, as all lightning devices now follow the new device interface.
  [(#6420)](https://github.com/PennyLaneAI/pennylane/pull/6420)

<h3>Deprecations 👋</h3>

* The `expand_depth` and `max_expansion` arguments for `qp.transforms.compile` and
  `qp.transforms.decompositions.clifford_t_decomposition` respectively have been deprecated.
  [(#6404)](https://github.com/PennyLaneAI/pennylane/pull/6404)

* Legacy operator arithmetic has been deprecated. This includes `qp.ops.Hamiltonian`, `qp.operation.Tensor`,
  `qp.operation.enable_new_opmath`, `qp.operation.disable_new_opmath`, and `qp.operation.convert_to_legacy_H`.
  Note that when new operator arithmetic is enabled, `qp.Hamiltonian` will continue to dispatch to
  `qp.ops.LinearCombination`; this behaviour is not deprecated. For more information, check out the
  [updated operator troubleshooting page](https://docs.pennylane.ai/en/stable/news/new_opmath.html).
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)
  [(#6365)](https://github.com/PennyLaneAI/pennylane/pull/6365)

* `qp.pauli.PauliSentence.hamiltonian` and `qp.pauli.PauliWord.hamiltonian` have been deprecated. 
  Instead, please use `qp.pauli.PauliSentence.operation` and `qp.pauli.PauliWord.operation`, respectively.
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)

* `qp.pauli.simplify()` has been deprecated. Instead, please use `qp.simplify(op)` or `op.simplify()`.
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)

* The `qp.BasisStatePreparation` template has been deprecated. Instead, use `qp.BasisState`.
  [(#6021)](https://github.com/PennyLaneAI/pennylane/pull/6021)

* The `'ancilla'` argument for `qp.iterative_qpe` has been deprecated. Instead, use the `'aux_wire'` 
  argument.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* `qp.shadows.shadow_expval` has been deprecated. Instead, use the `qp.shadow_expval` measurement
  process.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* `qp.broadcast` has been deprecated. Please use Python `for` loops instead.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* The `qp.QubitStateVector` template has been deprecated. Instead, use `qp.StatePrep`.
  [(#6172)](https://github.com/PennyLaneAI/pennylane/pull/6172)

* The `qp.qinfo` module has been deprecated. Please see the respective functions in the `qp.math` 
  and `qp.measurements` modules instead.
  [(#5911)](https://github.com/PennyLaneAI/pennylane/pull/5911)

* `Device`, `QubitDevice`, and `QutritDevice` will no longer be accessible via top-level import in v0.40.
  They will still be accessible as `qp.devices.LegacyDevice`, `qp.devices.QubitDevice`, and `qp.devices.QutritDevice`
  respectively.
  [(#6238)](https://github.com/PennyLaneAI/pennylane/pull/6238/)

* `QNode.gradient_fn` has been deprecated. Please use `QNode.diff_method` and `QNode.get_gradient_fn` 
  instead.
  [(#6244)](https://github.com/PennyLaneAI/pennylane/pull/6244)

<h3>Documentation 📝</h3>

* Updated `qp.spin` documentation.
  [(#6387)](https://github.com/PennyLaneAI/pennylane/pull/6387)

* Updated links to PennyLane.ai in the documentation to use the latest URL format, which excludes the `.html` prefix.
  [(#6412)](https://github.com/PennyLaneAI/pennylane/pull/6412)

* Update `qp.Qubitization` documentation based on new decomposition.
  [(#6276)](https://github.com/PennyLaneAI/pennylane/pull/6276)

* Fixed examples in the documentation of a few optimizers.
  [(#6303)](https://github.com/PennyLaneAI/pennylane/pull/6303)
  [(#6315)](https://github.com/PennyLaneAI/pennylane/pull/6315)

* Corrected examples in the documentation of `qp.jacobian`.
  [(#6283)](https://github.com/PennyLaneAI/pennylane/pull/6283)
  [(#6315)](https://github.com/PennyLaneAI/pennylane/pull/6315)

* Fixed spelling in a number of places across the documentation.
  [(#6280)](https://github.com/PennyLaneAI/pennylane/pull/6280)

* Add `work_wires` parameter to `qp.MultiControlledX` docstring signature.
  [(#6271)](https://github.com/PennyLaneAI/pennylane/pull/6271)

* Removed ambiguity in error raised by the `PauliRot` class.
  [(#6298)](https://github.com/PennyLaneAI/pennylane/pull/6298)

* Renamed an incorrectly named test in `test_pow_ops.py`.
  [(#6388)](https://github.com/PennyLaneAI/pennylane/pull/6388)

* Removed outdated Docker description from installation page.
  [(#6487)](https://github.com/PennyLaneAI/pennylane/pull/6487)

<h3>Bug fixes 🐛</h3>

* The wire order for `Snapshot`'s now matches the wire order of the device, rather than the simulation.
  [(#6461)](https://github.com/PennyLaneAI/pennylane/pull/6461)
  
* Fixed a bug where `QNSPSAOptimizer`, `QNGOptimizer` and `MomentumQNGOptimizer` calculate invalid 
  parameter updates if the metric tensor becomes singular.
  [(#6471)](https://github.com/PennyLaneAI/pennylane/pull/6471)

* The `default.qubit` device now supports parameter broadcasting with `qp.classical_shadow` and `qp.shadow_expval`.
  [(#6301)](https://github.com/PennyLaneAI/pennylane/pull/6301)

* Fixed unnecessary call of `eigvals` in `qp.ops.op_math.decompositions.two_qubit_unitary.py` that
  was causing an error in VJP. Raises warnings to users if this essentially nondifferentiable
  module is used.
  [(#6437)](https://github.com/PennyLaneAI/pennylane/pull/6437)

* Patches the `math` module to function with autoray 0.7.0.
  [(#6429)](https://github.com/PennyLaneAI/pennylane/pull/6429)

* Fixed incorrect differentiation of `PrepSelPrep` when using `diff_method="parameter-shift"`.
  [(#6423)](https://github.com/PennyLaneAI/pennylane/pull/6423)

* The `validate_device_wires` transform now raises an error if abstract wires are provided.
  [(#6405)](https://github.com/PennyLaneAI/pennylane/pull/6405)

* Fixed `qp.math.expand_matrix` for qutrit and arbitrary qudit operators.
  [(#6398)](https://github.com/PennyLaneAI/pennylane/pull/6398/)

* `MeasurementValue` now raises an error when it is used as a boolean.
  [(#6386)](https://github.com/PennyLaneAI/pennylane/pull/6386)

* `default.qutrit` now returns integer samples.
  [(#6385)](https://github.com/PennyLaneAI/pennylane/pull/6385)

* `adjoint_metric_tensor` now works with circuits containing state preparation operations.
  [(#6358)](https://github.com/PennyLaneAI/pennylane/pull/6358)

* `quantum_fisher` now respects the classical Jacobian of QNodes.
  [(#6350)](https://github.com/PennyLaneAI/pennylane/pull/6350)

* `qp.map_wires` can now be applied to a batch of tapes.
  [(#6295)](https://github.com/PennyLaneAI/pennylane/pull/6295)

* Fixed float-to-complex casting in various places across PennyLane.
  [(#6260)](https://github.com/PennyLaneAI/pennylane/pull/6260)
  [(#6268)](https://github.com/PennyLaneAI/pennylane/pull/6268)

* Fixed a bug where zero-valued JVPs were calculated wrongly in the presence of shot vectors.
  [(#6219)](https://github.com/PennyLaneAI/pennylane/pull/6219)

* Fixed `qp.PrepSelPrep` template to work with `torch`.
  [(#6191)](https://github.com/PennyLaneAI/pennylane/pull/6191)

* Fixed a bug where `qp.equal` now correctly compares `qp.PrepSelPrep` operators.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The `qp.QSVT` template now orders the `projector` wires first and the `UA` wires second, which is 
  the expected order of the decomposition.
  [(#6212)](https://github.com/PennyLaneAI/pennylane/pull/6212)

* The `qp.Qubitization` template now orders the `control` wires first and the `hamiltonian` wires second, 
  which is the expected according to other templates.
  [(#6229)](https://github.com/PennyLaneAI/pennylane/pull/6229)

* Fixed a bug where a circuit using the `autograd` interface sometimes returns nested values that are 
  not of the `autograd` interface.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* Fixed a bug where a simple circuit with no parameters or only builtin/NumPy arrays as parameters returns 
  autograd tensors.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* `qp.pauli.PauliVSpace` now uses a more stable SVD-based linear independence check to avoid running 
  into `LinAlgError: Singular matrix`. This stabilizes the usage of `qp.lie_closure`. It also introduces 
  normalization of the basis vector's internal representation `_M` to avoid exploding coefficients.
  [(#6232)](https://github.com/PennyLaneAI/pennylane/pull/6232)

* Fixed a bug where `csc_dot_product` is used during measurement for `Sum`/`Hamiltonian` that contains 
  observables that does not define a sparse matrix.
  [(#6278)](https://github.com/PennyLaneAI/pennylane/pull/6278)
  [(#6310)](https://github.com/PennyLaneAI/pennylane/pull/6310)

* Fixed a bug where `None` was added to the wires in `qp.PhaseAdder`, `qp.Adder` and `qp.OutAdder`.
  [(#6360)](https://github.com/PennyLaneAI/pennylane/pull/6360)

* Fixed a test after updating to the nightly version of Catalyst.
  [(#6362)](https://github.com/PennyLaneAI/pennylane/pull/6362)

* Fixed a bug where `CommutingEvolution` with a trainable `Hamiltonian` cannot be differentiated using 
  parameter shift.
  [(#6372)](https://github.com/PennyLaneAI/pennylane/pull/6372)

* Fixed a bug where `mitigate_with_zne` loses the `shots` information of the original tape.
  [(#6444)](https://github.com/PennyLaneAI/pennylane/pull/6444)

* Fixed a bug where `default.tensor` raises an error when applying `Identity`/`GlobalPhase` on no wires, 
  and `PauliRot`/`MultiRZ` on a single wire.
  [(#6448)](https://github.com/PennyLaneAI/pennylane/pull/6448)

* Fixes a bug where applying `qp.ctrl` and `qp.adjoint` on an operator type instead of an operator instance results in extra operators in the queue.
  [(#6284)](https://github.com/PennyLaneAI/pennylane/pull/6284)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Utkarsh Azad,
Oleksandr Borysenko,
Astral Cai,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Lillian M. A. Frederiksen,
Pietropaolo Frisoni,
Emiliano Godinez,
Anthony Hayes,
Austin Huang,
Soran Jahangiri,
Jacob Kitchen,
Korbinian Kottmann,
Christina Lee,
William Maxwell,
Erick Ochoa Lopez,
Lee J. O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Alex Preciado,
Ashish Kanwar Singh,
David Wierichs,
