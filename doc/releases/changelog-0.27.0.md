
# Release 0.27.0

<h3>New features since last release</h3>

<h4>An all-new data module 💾</h4>

* The `qp.data` module is now available, allowing users to download, load, and create quantum datasets.
  [(#3156)](https://github.com/PennyLaneAI/pennylane/pull/3156)

  Datasets are hosted on Xanadu Cloud and can be downloaded by using `qp.data.load()`:

  ```pycon
  >>> H2_datasets = qp.data.load(
  ...   data_name="qchem", molname="H2", basis="STO-3G", bondlength=1.1
  ... )
  >>> H2data = H2_datasets[0]
  >>> H2data
  <Dataset = description: qchem/H2/STO-3G/1.1, attributes: ['molecule', 'hamiltonian', ...]>
  ```

  - Datasets available to be downloaded can be listed with `qp.data.list_datasets()`.

  - To download or load only specific properties of a dataset, we can specify the desired properties in `qp.data.load` with the `attributes` keyword argument:

    ```pycon
    >>> H2_hamiltonian = qp.data.load(
    ... data_name="qchem", molname="H2", basis="STO-3G", bondlength=1.1,
    ... attributes=["molecule", "hamiltonian"]
    ... )[0]
    >>> H2_hamiltonian.hamiltonian
    <Hamiltonian: terms=15, wires=[0, 1, 2, 3]>
    ```

    The available attributes can be found using `qp.data.list_attributes()`:

  - To select data interactively, we can use `qp.data.load_interactive()`:

    ```pycon
    >>> qp.data.load_interactive()
    Please select a data name:
        1) qspin
        2) qchem
    Choice [1-2]: 1
    Please select a sysname:
        ...
    Please select a periodicity:
        ...
    Please select a lattice:
        ...
    Please select a layout:
        ...
    Please select attributes:
        ...
    Force download files? (Default is no) [y/N]: N
    Folder to download to? (Default is pwd, will download to /datasets subdirectory):

    Please confirm your choices:
    dataset: qspin/Ising/open/rectangular/4x4
    attributes: ['parameters', 'ground_states']
    force: False
    dest folder: datasets
    Would you like to continue? (Default is yes) [Y/n]:
    <Dataset = description: qspin/Ising/open/rectangular/4x4, attributes: ['parameters', 'ground_states']>
    ```

  - Once a dataset is loaded, its properties can be accessed as follows:

    ```pycon
    >>> dev = qp.device("default.qubit",wires=4)
    >>> @qp.qnode(dev)
    ... def circuit():
    ...     qp.BasisState(H2data.hf_state, wires = [0, 1, 2, 3])
    ...     for op in H2data.vqe_gates:
    ...          qp.apply(op)
    ...     return qp.expval(H2data.hamiltonian)
    >>> print(circuit())
    -1.0791430411076344
    ```

  It's also possible to create custom datasets with `qp.data.Dataset`:

  ```pycon
  >>> example_hamiltonian = qp.Hamiltonian(coeffs=[1,0.5], observables=[qp.PauliZ(wires=0),qp.PauliX(wires=1)])
  >>> example_energies, _ = np.linalg.eigh(qp.matrix(example_hamiltonian))
  >>> example_dataset = qp.data.Dataset(
  ... data_name = 'Example', hamiltonian=example_hamiltonian, energies=example_energies
  ... )
  >>> example_dataset.data_name
  'Example'
  >>> example_dataset.hamiltonian
    (0.5) [X1]
  + (1) [Z0]
  >>> example_dataset.energies
  array([-1.5, -0.5,  0.5,  1.5])
  ```

  Custom datasets can be saved and read with the `qp.data.Dataset.write()` and `qp.data.Dataset.read()` methods, respectively.

  ```pycon
  >>> example_dataset.write('./path/to/dataset.dat')
  >>> read_dataset = qp.data.Dataset()
  >>> read_dataset.read('./path/to/dataset.dat')
  >>> read_dataset.data_name
  'Example'
  >>> read_dataset.hamiltonian
    (0.5) [X1]
  + (1) [Z0]
  >>> read_dataset.energies
  array([-1.5, -0.5,  0.5,  1.5])
  ```

  We will continue to work on adding more datasets and features for `qp.data` in future releases.
 
<h4>Adaptive optimization 🏃🏋️🏊</h4>

* Optimizing quantum circuits can now be done *adaptively* with 
  `qp.AdaptiveOptimizer`.
  [(#3192)](https://github.com/PennyLaneAI/pennylane/pull/3192)

  The `qp.AdaptiveOptimizer` takes an initial circuit and a collection of operators as input and adds a selected gate to the circuit at each optimization step. The process of growing the circuit can be repeated until the circuit gradients converge to zero within a given threshold. The adaptive optimizer can be used to implement algorithms such as ADAPT-VQE as shown in the following example.

  Firstly, we define some preliminary variables needed for VQE:

  ```python
  symbols = ["H", "H", "H"]
  geometry = np.array([[0.01076341, 0.04449877, 0.0],
                      [0.98729513, 1.63059094, 0.0],
                      [1.87262415, -0.00815842, 0.0]], requires_grad=False)
  H, qubits = qp.qchem.molecular_hamiltonian(symbols, geometry, charge = 1)
  ```

  The collection of gates to grow the circuit is built to contain all single and double excitations:

  ```python
  n_electrons = 2
  singles, doubles = qp.qchem.excitations(n_electrons, qubits)
  singles_excitations = [qp.SingleExcitation(0.0, x) for x in singles]
  doubles_excitations = [qp.DoubleExcitation(0.0, x) for x in doubles]
  operator_pool = doubles_excitations + singles_excitations
  ```

  Next, an initial circuit that prepares a Hartree-Fock state and returns the expectation value of the Hamiltonian is defined:

  ```python
  hf_state = qp.qchem.hf_state(n_electrons, qubits)
  dev = qp.device("default.qubit", wires=qubits)
  @qp.qnode(dev)
  def circuit():
      qp.BasisState(hf_state, wires=range(qubits))
      return qp.expval(H)
  ```

  Finally, the optimizer is instantiated and then the circuit is created and optimized adaptively:

  ```python
  opt = qp.optimize.AdaptiveOptimizer()
  for i in range(len(operator_pool)):
      circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
      print('Energy:', energy)
      print(qp.draw(circuit)())
      print('Largest Gradient:', gradient)
      print()
      if gradient < 1e-3:
          break
  ```

  ```pycon
  Energy: -1.246549938420637
  0: ─╭BasisState(M0)─╭G²(0.20)─┤ ╭<𝓗>
  1: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
  2: ─├BasisState(M0)─│─────────┤ ├<𝓗>
  3: ─├BasisState(M0)─│─────────┤ ├<𝓗>
  4: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
  5: ─╰BasisState(M0)─╰G²(0.20)─┤ ╰<𝓗>
  Largest Gradient: 0.14399872776755085

  Energy: -1.2613740231529604
  0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)─┤ ╭<𝓗>
  1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─┤ ├<𝓗>
  2: ─├BasisState(M0)─│─────────├G²(0.19)─┤ ├<𝓗>
  3: ─├BasisState(M0)─│─────────╰G²(0.19)─┤ ├<𝓗>
  4: ─├BasisState(M0)─├G²(0.20)───────────┤ ├<𝓗>
  5: ─╰BasisState(M0)─╰G²(0.20)───────────┤ ╰<𝓗>
  Largest Gradient: 0.1349349562423238

  Energy: -1.2743971719780331
  0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)──────────┤ ╭<𝓗>
  1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─╭G(0.00)─┤ ├<𝓗>
  2: ─├BasisState(M0)─│─────────├G²(0.19)─│────────┤ ├<𝓗>
  3: ─├BasisState(M0)─│─────────╰G²(0.19)─╰G(0.00)─┤ ├<𝓗>
  4: ─├BasisState(M0)─├G²(0.20)────────────────────┤ ├<𝓗>
  5: ─╰BasisState(M0)─╰G²(0.20)────────────────────┤ ╰<𝓗>
  Largest Gradient: 0.00040841755397108586
  ```

  For a detailed breakdown of its implementation, check out the [Adaptive circuits for quantum chemistry
  demo](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits).

<h4>Automatic interface detection 🧩</h4>

* QNodes now accept an `auto` interface argument which automatically detects the machine learning library to use.
  [(#3132)](https://github.com/PennyLaneAI/pennylane/pull/3132)

  ```python
  from pennylane import numpy as np
  import torch
  import tensorflow as tf
  from jax import numpy as jnp

  dev = qp.device("default.qubit", wires=2)
  @qp.qnode(dev, interface="auto")
  def circuit(weight):
      qp.RX(weight[0], wires=0)
      qp.RY(weight[1], wires=1)
      return qp.expval(qp.PauliZ(0))

  interface_tensors = [[0, 1], np.array([0, 1]), torch.Tensor([0, 1]), tf.Variable([0, 1], dtype=float), jnp.array([0, 1])]
  for tensor in interface_tensors:
      res = circuit(weight=tensor)
      print(f"Result value: {res:.2f}; Result type: {type(res)}")
  ```

  ```pycon
  Result value: 1.00; Result type: <class 'pennylane.numpy.tensor.tensor'>
  Result value: 1.00; Result type: <class 'pennylane.numpy.tensor.tensor'>
  Result value: 1.00; Result type: <class 'torch.Tensor'>
  Result value: 1.00; Result type: <class 'tensorflow.python.framework.ops.EagerTensor'>
  Result value: 1.00; Result type: <class 'jaxlib.xla_extension.Array'>
  ```

<h4>Upgraded JAX-JIT gradient support 🏎</h4>

* JAX-JIT support for computing the gradient of QNodes that return a single vector of probabilities or multiple expectation values is now available.
  [(#3244)](https://github.com/PennyLaneAI/pennylane/pull/3244)
  [(#3261)](https://github.com/PennyLaneAI/pennylane/pull/3261)

  ```python
  import jax
  from jax import numpy as jnp
  from jax.config import config
  config.update("jax_enable_x64", True)

  dev = qp.device("lightning.qubit", wires=2)

  @jax.jit
  @qp.qnode(dev, diff_method="parameter-shift", interface="jax")
  def circuit(x, y):
      qp.RY(x, wires=0)
      qp.RY(y, wires=1)
      qp.CNOT(wires=[0, 1])
      return qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(1))

  x = jnp.array(1.0)
  y = jnp.array(2.0)
  ```

  ```pycon
  >>> jax.jacobian(circuit, argnums=[0, 1])(x, y)
  (Array([-0.84147098,  0.35017549], dtype=float64, weak_type=True),
   Array([ 4.47445479e-18, -4.91295496e-01], dtype=float64, weak_type=True))
  ```

  Note that this change depends on `jax.pure_callback`, which requires `jax>=0.3.17`.

<h4>Construct Pauli words and sentences 🔤</h4>

* We've reorganized and grouped everything in PennyLane responsible for manipulating Pauli operators into a `pauli` module. The `grouping` module has been deprecated as a result, and logic was moved from `pennylane/grouping` to `pennylane/pauli/grouping`.
  [(#3179)](https://github.com/PennyLaneAI/pennylane/pull/3179)

* `qp.pauli.PauliWord` and `qp.pauli.PauliSentence` can be used to represent tensor products and linear combinations of Pauli operators, respectively. These provide a more performant method to compute sums and products of Pauli operators.
  [(#3195)](https://github.com/PennyLaneAI/pennylane/pull/3195)

  - `qp.pauli.PauliWord` represents tensor products of Pauli operators. We can efficiently multiply and extract the matrix of these operators using this representation.

    ```pycon
    >>> pw1 = qp.pauli.PauliWord({0:"X", 1:"Z"})
    >>> pw2 = qp.pauli.PauliWord({0:"Y", 1:"Z"})
    >>> pw1, pw2
    (X(0) @ Z(1), Y(0) @ Z(1))
    >>> pw1 * pw2
    (Z(0), 1j)
    >>> pw1.to_mat(wire_order=[0,1])
    array([[ 0,  0,  1,  0],
          [ 0,  0,  0, -1],
          [ 1,  0,  0,  0],
          [ 0, -1,  0,  0]])
    ```

  - `qp.pauli.PauliSentence` represents linear combinations of Pauli words. We can efficiently add, multiply and extract the matrix of these operators in this representation.

    ```pycon
    >>> ps1 = qp.pauli.PauliSentence({pw1: 1.2, pw2: 0.5j})
    >>> ps2 = qp.pauli.PauliSentence({pw1: -1.2})
    >>> ps1
    1.2 * X(0) @ Z(1)
    + 0.5j * Y(0) @ Z(1)
    >>> ps1 + ps2
    0.0 * X(0) @ Z(1)
    + 0.5j * Y(0) @ Z(1)
    >>> ps1 * ps2
    -1.44 * I
    + (-0.6+0j) * Z(0)
    >>> (ps1 + ps2).to_mat(wire_order=[0,1])
    array([[ 0. +0.j,  0. +0.j,  0.5+0.j,  0. +0.j],
          [ 0. +0.j,  0. +0.j,  0. +0.j, -0.5+0.j],
          [-0.5+0.j,  0. +0.j,  0. +0.j,  0. +0.j],
          [ 0. +0.j,  0.5+0.j,  0. +0.j,  0. +0.j]])
    ```

<h4>(Experimental) More support for multi-measurement and gradient output types 🧪</h4>

* `qp.enable_return()` now supports QNodes returning multiple measurements, 
  including shots vectors, and gradient output types.
  [(#2886)](https://github.com/PennyLaneAI/pennylane/pull/2886)
  [(#3052)](https://github.com/PennyLaneAI/pennylane/pull/3052)
  [(#3041)](https://github.com/PennyLaneAI/pennylane/pull/3041)
  [(#3090)](https://github.com/PennyLaneAI/pennylane/pull/3090)
  [(#3069)](https://github.com/PennyLaneAI/pennylane/pull/3069)
  [(#3137)](https://github.com/PennyLaneAI/pennylane/pull/3137)
  [(#3127)](https://github.com/PennyLaneAI/pennylane/pull/3127)
  [(#3099)](https://github.com/PennyLaneAI/pennylane/pull/3099)
  [(#3098)](https://github.com/PennyLaneAI/pennylane/pull/3098)
  [(#3095)](https://github.com/PennyLaneAI/pennylane/pull/3095)
  [(#3091)](https://github.com/PennyLaneAI/pennylane/pull/3091)
  [(#3176)](https://github.com/PennyLaneAI/pennylane/pull/3176)
  [(#3170)](https://github.com/PennyLaneAI/pennylane/pull/3170)
  [(#3194)](https://github.com/PennyLaneAI/pennylane/pull/3194)
  [(#3267)](https://github.com/PennyLaneAI/pennylane/pull/3267)
  [(#3234)](https://github.com/PennyLaneAI/pennylane/pull/3234)
  [(#3232)](https://github.com/PennyLaneAI/pennylane/pull/3232)
  [(#3223)](https://github.com/PennyLaneAI/pennylane/pull/3223)
  [(#3222)](https://github.com/PennyLaneAI/pennylane/pull/3222)
  [(#3315)](https://github.com/PennyLaneAI/pennylane/pull/3315)

  In v0.25, we introduced `qp.enable_return()`, which separates measurements into their own tensors. The motivation of this change is the deprecation of ragged `ndarray` creation in NumPy.

  With this release, we're continuing to elevate this feature by adding support for:
  
  - Execution (`qp.execute`)
  - Jacobian vector product (JVP) computation
  - Gradient transforms (`qp.gradients.param_shift`, `qp.gradients.finite_diff`, `qp.gradients.hessian_transform`, `qp.gradients.param_shift_hessian`).

  - Interfaces (Autograd, TensorFlow, and JAX, although without JIT)
    
  With this added support, the JAX interface can handle multiple shots (shots vectors), measurements, and gradient output types with `qp.enable_return()`:
  
  ```python
  import jax

  qp.enable_return()
  dev = qp.device("default.qubit", wires=2, shots=(1, 10000))

  params = jax.numpy.array([0.1, 0.2])

  @qp.qnode(dev, interface="jax", diff_method="parameter-shift", max_diff=2)
  def circuit(x):
      qp.RX(x[0], wires=[0])
      qp.RY(x[1], wires=[1])
      qp.CNOT(wires=[0, 1])
      return qp.var(qp.PauliZ(0) @ qp.PauliX(1)), qp.probs(wires=[0])
  ```

  ```pycon
  >>> jax.hessian(circuit)(params)
  ((Array([[ 0.,  0.],
                [ 2., -3.]], dtype=float32),
  Array([[[-0.5,  0. ],
                [ 0. ,  0. ]],
              [[ 0.5,  0. ],
                [ 0. ,  0. ]]], dtype=float32)),
  (Array([[ 0.07677898,  0.0563341 ],
                [ 0.07238522, -1.830669  ]], dtype=float32),
  Array([[[-4.9707499e-01,  2.9999996e-04],
                [-6.2500127e-04,  1.2500001e-04]],
                [[ 4.9707499e-01, -2.9999996e-04],
                [ 6.2500127e-04, -1.2500001e-04]]], dtype=float32)))
  ```

  For more details, please [refer to the documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html?highlight=enable_return#pennylane.enable_return).

<h4>New basis rotation and tapering features in qp.qchem 🤓</h4>

* Grouped coefficients, observables, and basis rotation transformation matrices needed to construct a qubit Hamiltonian in the rotated basis of molecular orbitals are now calculable via `qp.qchem.basis_rotation()`.
  ([#3011](https://github.com/PennyLaneAI/pennylane/pull/3011))

  ```pycon
  >>> symbols  = ['H', 'H']
  >>> geometry = np.array([[0.0, 0.0, 0.0], [1.398397361, 0.0, 0.0]], requires_grad = False)
  >>> mol = qp.qchem.Molecule(symbols, geometry)
  >>> core, one, two = qp.qchem.electron_integrals(mol)()
  >>> coeffs, ops, unitaries = qp.qchem.basis_rotation(one, two, tol_factor=1.0e-5)
  >>> unitaries
  [tensor([[-1.00000000e+00, -5.46483514e-13],
         [ 5.46483514e-13, -1.00000000e+00]], requires_grad=True),
  tensor([[-1.00000000e+00,  3.17585063e-14],
          [-3.17585063e-14, -1.00000000e+00]], requires_grad=True),
  tensor([[-0.70710678, -0.70710678],
          [-0.70710678,  0.70710678]], requires_grad=True),
  tensor([[ 2.58789009e-11,  1.00000000e+00],
          [-1.00000000e+00,  2.58789009e-11]], requires_grad=True)]
  ```

* Any gate operation can now be tapered according to :math:`\mathbb{Z}_2` symmetries of the Hamiltonian via `qp.qchem.taper_operation`.
  [(#3002)](https://github.com/PennyLaneAI/pennylane/pull/3002)
  [(#3121)](https://github.com/PennyLaneAI/pennylane/pull/3121)

  ```pycon
  >>> symbols = ['He', 'H']
  >>> geometry =  np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4589]])
  >>> mol = qp.qchem.Molecule(symbols, geometry, charge=1)
  >>> H, n_qubits = qp.qchem.molecular_hamiltonian(symbols, geometry)
  >>> generators = qp.qchem.symmetry_generators(H)
  >>> paulixops = qp.qchem.paulix_ops(generators, n_qubits)
  >>> paulix_sector = qp.qchem.optimal_sector(H, generators, mol.n_electrons)
  >>> tap_op = qp.qchem.taper_operation(qp.SingleExcitation, generators, paulixops,
  ...                paulix_sector, wire_order=H.wires, op_wires=[0, 2])
  >>> tap_op(3.14159)
  [Exp(1.5707949999999993j PauliY)]
  ```

  Moreover, the obtained tapered operation can be used directly within a QNode.

  ```pycon
  >>> dev = qp.device('default.qubit', wires=[0, 1])
  >>> @qp.qnode(dev)
  ... def circuit(params):
  ...     tap_op(params[0])
  ...     return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))
  >>> drawer = qp.draw(circuit, show_all_wires=True)
  >>> print(drawer(params=[3.14159]))
  0: ──Exp(0.00+1.57j Y)─┤ ╭<Z@Z>
  1: ────────────────────┤ ╰<Z@Z>
  ```

* Functionality has been added to estimate the number of measurements required to compute an expectation value with a target error and estimate the error in computing an expectation value with a given number of measurements.
  [(#3000)](https://github.com/PennyLaneAI/pennylane/pull/3000)

<h4>New functions, operations, and observables 🤩</h4>

* Wires of operators or entire QNodes can now be mapped to other wires via `qp.map_wires()`.
  [(#3143)](https://github.com/PennyLaneAI/pennylane/pull/3143)
  [(#3145)](https://github.com/PennyLaneAI/pennylane/pull/3145)

  The `qp.map_wires()` function requires a dictionary representing a wire map. Use it with

  - arbitrary operators:

    ```pycon
    >>> op = qp.RX(0.54, wires=0) + qp.PauliX(1) + (qp.PauliZ(2) @ qp.RY(1.23, wires=3))
    >>> op
    (RX(0.54, wires=[0]) + PauliX(wires=[1])) + (PauliZ(wires=[2]) @ RY(1.23, wires=[3]))
    >>> wire_map = {0: 10, 1: 11, 2: 12, 3: 13}
    >>> qp.map_wires(op, wire_map)
    (RX(0.54, wires=[10]) + PauliX(wires=[11])) + (PauliZ(wires=[12]) @ RY(1.23, wires=[13]))
    ```

    A `map_wires` method has also been added to operators, which returns a copy
    of the operator with its wires changed according to the given wire map.

  - entire QNodes:

    ```python
    dev = qp.device("default.qubit", wires=["A", "B", "C", "D"])
    wire_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    @qp.qnode(dev)
    def circuit():
        qp.RX(0.54, wires=0)
        qp.PauliX(1)
        qp.PauliZ(2)
        qp.RY(1.23, wires=3)
        return qp.probs(wires=0)
    ```

    ```pycon
    >>> mapped_circuit = qp.map_wires(circuit, wire_map)
    >>> mapped_circuit()
    tensor([0.92885434, 0.07114566], requires_grad=True)
    >>> print(qp.draw(mapped_circuit)())
    A: ──RX(0.54)─┤  Probs
    B: ──X────────┤
    C: ──Z────────┤
    D: ──RY(1.23)─┤
    ```

* The `qp.IntegerComparator` arithmetic operation is now available.
[(#3113)](https://github.com/PennyLaneAI/pennylane/pull/3113)

  Given a basis state :math:`\vert n \rangle`, where :math:`n` is a positive integer, and a fixed positive integer :math:`L`, `qp.IntegerComparator` flips a target qubit if :math:`n \geq L`. Alternatively, the flipping condition can be :math:`n < L` as demonstrated below:

  ```python
  dev = qp.device("default.qubit", wires=2)

  @qp.qnode(dev)
  def circuit():
      qp.BasisState(np.array([0, 1]), wires=range(2))
      qp.broadcast(qp.Hadamard, wires=range(2), pattern='single')
      qp.IntegerComparator(2, geq=False, wires=[0, 1])
      return qp.state()
  ```

  ```pycon
  >>> circuit()
  [-0.5+0.j  0.5+0.j -0.5+0.j  0.5+0.j]
  ```

* The `qp.GellMann` qutrit observable, the ternary generalization of the Pauli observables, is now available.
  [(#3035)](https://github.com/PennyLaneAI/pennylane/pull/3035)

  When using `qp.GellMann`, the `index` keyword argument determines which of the 8 Gell-Mann matrices is used.

  ```python
  dev = qp.device("default.qutrit", wires=2)

  @qp.qnode(dev)
  def circuit():
      qp.TClock(wires=0)
      qp.TShift(wires=1)
      qp.TAdd(wires=[0, 1])
      return qp.expval(qp.GellMann(wires=0, index=8) + qp.GellMann(wires=1, index=3))
  ```

  ```pycon
  >>> circuit()
  -0.42264973081037416
  ```

* Controlled qutrit operations can now be performed with `qp.ControlledQutritUnitary`.
  ([#2844](https://github.com/PennyLaneAI/pennylane/pull/2844))

  The control wires and values that define the operation are defined analogously to the qubit operation.

  ```python
  dev = qp.device("default.qutrit", wires=3)

  @qp.qnode(dev)
  def circuit(U):
      qp.TShift(wires=0)
      qp.TAdd(wires=[0, 1])
      qp.ControlledQutritUnitary(U, control_wires=[0, 1], control_values='12', wires=2)
      return qp.state()
  ```

  ```pycon
  >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
  >>> circuit(U)
  tensor([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j], requires_grad=True)
  ```

<h3>Improvements</h3>

* PennyLane now supports Python 3.11!
  [(#3297)](https://github.com/PennyLaneAI/pennylane/pull/3297)

* `qp.sample` and `qp.counts` work more efficiently and track if computational basis samples are being generated when they are called without specifying an observable.
  [(#3207)](https://github.com/PennyLaneAI/pennylane/pull/3207)

* The parameters of a basis set containing a different number of Gaussian functions are now easier to differentiate.
  [(#3213)](https://github.com/PennyLaneAI/pennylane/pull/3213)

* Printing a `qp.MultiControlledX` operator now shows the `control_values` keyword argument.
  [(#3113)](https://github.com/PennyLaneAI/pennylane/pull/3113)

* `qp.simplify` and transforms like `qp.matrix`, `batch_transform`, `hamiltonian_expand`, and `split_non_commuting` now work with
  `QuantumScript` as well as `QuantumTape`.
  [(#3209)](https://github.com/PennyLaneAI/pennylane/pull/3209)

* A redundant flipping of the initial state in the UCCSD and kUpCCGSD templates has been removed.
  [(#3148)](https://github.com/PennyLaneAI/pennylane/pull/3148)

* `qp.adjoint` now supports batching if the base operation supports batching.
  [(#3168)](https://github.com/PennyLaneAI/pennylane/pull/3168)

* `qp.OrbitalRotation` is now decomposed into two `qp.SingleExcitation` operations for faster execution and more efficient parameter-shift gradient calculations on devices that natively support `qp.SingleExcitation`.
  [(#3171)](https://github.com/PennyLaneAI/pennylane/pull/3171)

* The `Exp` class decomposes into a `PauliRot` class if the coefficient is imaginary and the base operator is a Pauli Word.
  [(#3249)](https://github.com/PennyLaneAI/pennylane/pull/3249)

* Added the operator attributes `has_decomposition` and `has_adjoint` that indicate
  whether a corresponding `decomposition` or `adjoint` method is available.
  [(#2986)](https://github.com/PennyLaneAI/pennylane/pull/2986)

* Structural improvements are made to `QueuingManager`, formerly `QueuingContext`, and `AnnotatedQueue`.
  [(#2794)](https://github.com/PennyLaneAI/pennylane/pull/2794)
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)
  [(#3085)](https://github.com/PennyLaneAI/pennylane/pull/3085)

  - `QueuingContext` is renamed to `QueuingManager`.
  - `QueuingManager` should now be the global communication point for putting queuable objects into the active queue.
  - `QueuingManager` is no longer an abstract base class.
  - `AnnotatedQueue` and its children no longer inherit from `QueuingManager`.
  - `QueuingManager` is no longer a context manager.
  - Recording queues should start and stop recording via the `QueuingManager.add_active_queue` and
    `QueuingContext.remove_active_queue` class methods instead of directly manipulating the `_active_contexts` property.
  - `AnnotatedQueue` and its children no longer provide global information about actively recording queues. This information is now only available through `QueuingManager`.
  - `AnnotatedQueue` and its children no longer have the private `_append`, `_remove`, `_update_info`, `_safe_update_info`, and `_get_info` methods. The public analogues should be used instead.
  - `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info` are deprecated.  Their functionality is moved to `update_info`.

* `qp.Identity` now accepts multiple wires.
    [(#3049)](https://github.com/PennyLaneAI/pennylane/pull/3049)

    ```pycon
    >>> id_op = qp.Identity([0, 1])
    >>> id_op.matrix()
    array([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
    >>> id_op.sparse_matrix()
    <4x4 sparse matrix of type '<class 'numpy.float64'>'
        with 4 stored elements in Compressed Sparse Row format>
    >>> id_op.eigvals()
    array([1., 1., 1., 1.])
    ```

* Added `unitary_check` keyword argument to the constructor of the `QubitUnitary` class which
  indicates whether the user wants to check for unitarity of the input matrix or not. Its default
  value is `false`.
  [(#3063)](https://github.com/PennyLaneAI/pennylane/pull/3063)

* Modified the representation of `WireCut` by using `qp.draw_mpl`.
  [(#3067)](https://github.com/PennyLaneAI/pennylane/pull/3067)

* Improved the performance of `qp.math.expand_matrix` function for dense
  and sparse matrices.
  [(#3060)](https://github.com/PennyLaneAI/pennylane/pull/3060)
  [(#3064)](https://github.com/PennyLaneAI/pennylane/pull/3064)

* Added support for sums and products of operator classes with scalar tensors of any interface
  (NumPy, JAX, Tensorflow, PyTorch...).
  [(#3149)](https://github.com/PennyLaneAI/pennylane/pull/3149)

  ```pycon
  >>> s_prod = torch.tensor(4) * qp.RX(1.23, 0)
  >>> s_prod
  4*(RX(1.23, wires=[0]))
  >>> s_prod.scalar
  tensor(4)
  ```

* Added `overlapping_ops` property to the `Composite` class to improve the
  performance of the `eigvals`, `diagonalizing_gates` and `Prod.matrix` methods.
  [(#3084)](https://github.com/PennyLaneAI/pennylane/pull/3084)

* Added the `map_wires` method to the operators, which returns a copy of the operator with
  its wires changed according to the given wire map.
  [(#3143)](https://github.com/PennyLaneAI/pennylane/pull/3143)

  ```pycon
  >>> op = qp.Toffoli([0, 1, 2])
  >>> wire_map = {0: 2, 2: 0}
  >>> op.map_wires(wire_map=wire_map)
  Toffoli(wires=[2, 1, 0])
  ```

* Calling `compute_matrix` and `compute_sparse_matrix` of simple non-parametric operations is now faster and more memory-efficient with the addition of caching.
  [(#3134)](https://github.com/PennyLaneAI/pennylane/pull/3134)

* Added details to the output of `Exp.label()`.
  [(#3126)](https://github.com/PennyLaneAI/pennylane/pull/3126)

* `qp.math.unwrap` no longer creates ragged arrays. Lists remain lists.
  [(#3163)](https://github.com/PennyLaneAI/pennylane/pull/3163)

* New `null.qubit` device. The `null.qubit` performs no operations or memory allocations.
  [(#2589)](https://github.com/PennyLaneAI/pennylane/pull/2589)

* `default.qubit` favours decomposition and avoids matrix construction for `QFT` and `GroverOperator` at larger qubit numbers.
  [(#3193)](https://github.com/PennyLaneAI/pennylane/pull/3193)

* `qp.ControlledQubitUnitary` now has a `control_values` property.
  [(#3206)](https://github.com/PennyLaneAI/pennylane/pull/3206)

* Added a new `qp.tape.QuantumScript` class that contains all the non-queuing behavior of `QuantumTape`. Now, `QuantumTape` inherits from `QuantumScript` as well as `AnnotatedQueue`.
  [(#3097)](https://github.com/PennyLaneAI/pennylane/pull/3097)

* Extended the `qp.equal` function to MeasurementProcesses
  [(#3189)](https://github.com/PennyLaneAI/pennylane/pull/3189)

* `qp.drawer.draw.draw_mpl` now accepts a `style` kwarg to select a style for plotting, rather than calling
  `qp.drawer.use_style(style)` before plotting. Setting a style for `draw_mpl` does not change the global
  configuration for matplotlib plotting. If no `style` is passed, the function defaults
  to plotting with the `black_white` style.
  [(#3247)](https://github.com/PennyLaneAI/pennylane/pull/3247)

<h3>Breaking changes</h3>

* `QuantumTape._par_info` is now a list of dictionaries, instead of a dictionary whose keys are integers starting from zero.
  [(#3185)](https://github.com/PennyLaneAI/pennylane/pull/3185)

* `QueuingContext` has been renamed to `QueuingManager`.
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

* Deprecation patches for the return types enum's location and `qp.utils.expand` are removed.
  [(#3092)](https://github.com/PennyLaneAI/pennylane/pull/3092)

* `_multi_dispatch` functionality has been moved inside the `get_interface` function. This function
  can now be called with one or multiple tensors as arguments.
  [(#3136)](https://github.com/PennyLaneAI/pennylane/pull/3136)

  ```pycon
  >>> torch_scalar = torch.tensor(1)
  >>> torch_tensor = torch.Tensor([2, 3, 4])
  >>> numpy_tensor = np.array([5, 6, 7])
  >>> qp.math.get_interface(torch_scalar)
  'torch'
  >>> qp.math.get_interface(numpy_tensor)
  'numpy'
  ```

  `_multi_dispatch` previously had only one argument which contained a list of the tensors to be
  dispatched:

  ```pycon
  >>> qp.math._multi_dispatch([torch_scalar, torch_tensor, numpy_tensor])
  'torch'
  ```

  To differentiate whether the user wants to get the interface of a single tensor or multiple
  tensors, `get_interface` now accepts a different argument per tensor to be dispatched:

  ```pycon
  >>> qp.math.get_interface(*[torch_scalar, torch_tensor, numpy_tensor])
  'torch'
  >>> qp.math.get_interface(torch_scalar, torch_tensor, numpy_tensor)
  'torch'
  ```

* `Operator.compute_terms` is removed. On a specific instance of an operator, `op.terms()` can be used
  instead. There is no longer a static method for this.
  [(#3215)](https://github.com/PennyLaneAI/pennylane/pull/3215)

<h3>Deprecations</h3>

* `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info` are deprecated. Instead, `update_info` no longer raises errors
  if the object isn't in the queue.
  [(#3085)](https://github.com/PennyLaneAI/pennylane/pull/3085)

* `qp.tape.stop_recording` and `QuantumTape.stop_recording` have been moved to `qp.QueuingManager.stop_recording`. The old functions will still be available until v0.29.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

* `qp.tape.get_active_tape` has been deprecated. Use `qp.QueuingManager.active_context()` instead.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

* `Operator.compute_terms` has been removed. On a specific instance of an operator, use `op.terms()` instead. There is no longer a static method for this.
  [(#3215)](https://github.com/PennyLaneAI/pennylane/pull/3215)

* `qp.tape.QuantumTape.inv()` has been deprecated. Use `qp.tape.QuantumTape.adjoint` instead.
  [(#3237)](https://github.com/PennyLaneAI/pennylane/pull/3237)

* `qp.transforms.qcut.remap_tape_wires` has been deprecated. Use `qp.map_wires` instead.
  [(#3186)](https://github.com/PennyLaneAI/pennylane/pull/3186)

* The grouping module `qp.grouping` has been deprecated. Use `qp.pauli` or `qp.pauli.grouping` instead. The module will still be available until v0.28.
  [(#3262)](https://github.com/PennyLaneAI/pennylane/pull/3262)

<h3>Documentation</h3>

* The code block in the usage details of the UCCSD template has been updated.
  [(#3140)](https://github.com/PennyLaneAI/pennylane/pull/3140)

* Added a "Deprecations" page to the developer documentation.
  [(#3093)](https://github.com/PennyLaneAI/pennylane/pull/3093)

* The example of the `qp.FlipSign` template has been updated.
  [(#3219)](https://github.com/PennyLaneAI/pennylane/pull/3219)

<h3>Bug fixes</h3>

* `qp.SparseHamiltonian` now validates the size of the input matrix.
  [(#3278)](https://github.com/PennyLaneAI/pennylane/pull/3278)

* Users no longer see unintuitive errors when inputing sequences to `qp.Hermitian`.
  [(#3181)](https://github.com/PennyLaneAI/pennylane/pull/3181)

* The evaluation of QNodes that return either `vn_entropy` or `mutual_info` raises an
  informative error message when using devices that define a vector of shots.
  [(#3180)](https://github.com/PennyLaneAI/pennylane/pull/3180)

* Fixed a bug that made `qp.AmplitudeEmbedding` incompatible with JITting.
  [(#3166)](https://github.com/PennyLaneAI/pennylane/pull/3166)

* Fixed the `qp.transforms.transpile` transform to work correctly for all two-qubit operations.
  [(#3104)](https://github.com/PennyLaneAI/pennylane/pull/3104)

* Fixed a bug with the control values of a controlled version of a `ControlledQubitUnitary`.
  [(#3119)](https://github.com/PennyLaneAI/pennylane/pull/3119)

* Fixed a bug where `qp.math.fidelity(non_trainable_state, trainable_state)` failed unexpectedly.
  [(#3160)](https://github.com/PennyLaneAI/pennylane/pull/3160)

* Fixed a bug where `qp.QueuingManager.stop_recording` did not clean up if yielded code raises an exception.
  [(#3182)](https://github.com/PennyLaneAI/pennylane/pull/3182)

* Returning `qp.sample()` or `qp.counts()` with other measurements of non-commuting observables
  now raises a QuantumFunctionError (e.g., `return qp.expval(PauliX(wires=0)), qp.sample()`
  now raises an error).
  [(#2924)](https://github.com/PennyLaneAI/pennylane/pull/2924)

* Fixed a bug where `op.eigvals()` would return an incorrect result if the operator was a non-hermitian
  composite operator.
  [(#3204)](https://github.com/PennyLaneAI/pennylane/pull/3204)

* Fixed a bug where `qp.BasisStatePreparation` and `qp.BasisEmbedding` were not jit-compilable with JAX.
  [(#3239)](https://github.com/PennyLaneAI/pennylane/pull/3239)

* Fixed a bug where `qp.MottonenStatePreparation` was not jit-compilable with JAX.
  [(#3260)](https://github.com/PennyLaneAI/pennylane/pull/3260)

* Fixed a bug where `qp.expval(qp.Hamiltonian())` would not raise an error
  if the Hamiltonian involved some wires that are not present on the device.
  [(#3266)](https://github.com/PennyLaneAI/pennylane/pull/3266)

* Fixed a bug where `qp.tape.QuantumTape.shape()` did not account for the batch dimension of the tape
  [(#3269)](https://github.com/PennyLaneAI/pennylane/pull/3269)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Kamal Mohamed Ali,
Guillermo Alonso-Linaje,
Juan Miguel Arrazola,
Utkarsh Azad,
Thomas Bromley, 
Albert Mitjans Coma,
Isaac De Vlugt,
Olivia Di Matteo,
Amintor Dusko,
Lillian M. A. Frederiksen,
Diego Guala,
Josh Izaac,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Christina Lee,
Romain Moyard,
Lee J. O'Riordan,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
Antal Száva,
David Wierichs,
