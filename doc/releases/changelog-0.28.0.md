
# Release 0.28.0

<h3>New features since last release</h3>

<h4>Custom measurement processes 📐</h4>

* Custom measurements can now be facilitated with the addition of the `qml.measurements` module.
  [(#3286)](https://github.com/PennyLaneAI/pennylane/pull/3286)
  [(#3343)](https://github.com/PennyLaneAI/pennylane/pull/3343)
  [(#3288)](https://github.com/PennyLaneAI/pennylane/pull/3288)
  [(#3312)](https://github.com/PennyLaneAI/pennylane/pull/3312)
  [(#3287)](https://github.com/PennyLaneAI/pennylane/pull/3287)
  [(#3292)](https://github.com/PennyLaneAI/pennylane/pull/3292)
  [(#3287)](https://github.com/PennyLaneAI/pennylane/pull/3287)
  [(#3326)](https://github.com/PennyLaneAI/pennylane/pull/3326)
  [(#3327)](https://github.com/PennyLaneAI/pennylane/pull/3327)
  [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)
  [(#3439)](https://github.com/PennyLaneAI/pennylane/pull/3439)
  [(#3466)](https://github.com/PennyLaneAI/pennylane/pull/3466)

  Within `qml.measurements` are new subclasses that allow for the possibility to create *custom* measurements:

  * `SampleMeasurement`: represents a sample-based measurement
  * `StateMeasurement`: represents a state-based measurement
  * `MeasurementTransform`: represents a measurement process that requires the application of a batch transform

  Creating a custom measurement involves making a class that inherits from one of the classes above. An example is given below. Here, the measurement computes the number of samples obtained of a given state:
  
  ```python
  from pennylane.measurements import SampleMeasurement

  class CountState(SampleMeasurement):
      def __init__(self, state: str):
          self.state = state  # string identifying the state, e.g. "0101"
          wires = list(range(len(state)))
          super().__init__(wires=wires)

      def process_samples(self, samples, wire_order, shot_range, bin_size):
          counts_mp = qml.counts(wires=self._wires)
          counts = counts_mp.process_samples(samples, wire_order, shot_range, bin_size)
          return counts.get(self.state, 0)

      def __copy__(self):
          return CountState(state=self.state)
  ```

  We can now execute the new measurement in a QNode as follows.

  ```python
  dev = qml.device("default.qubit", wires=1, shots=10000)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return CountState(state="1")
  ```

  ```pycon
  >>> circuit(1.23)
  tensor(3303., requires_grad=True)
  ```

  Differentiability is also supported for this new measurement process:

  ```pycon
  >>> x = qml.numpy.array(1.23, requires_grad=True)
  >>> qml.grad(circuit)(x)
  4715.000000000001
  ```

  For more information about these new features, see the documentation for [`qml.measurements`](https://docs.pennylane.ai/en/stable/code/qml_measurements.html).

<h4>ZX Calculus 🧮</h4>

* QNodes can now be converted into ZX diagrams via the PyZX framework.
  [(#3446)](https://github.com/PennyLaneAI/pennylane/pull/3446)

  ZX diagrams are the medium for which we can envision a quantum circuit as a graph in the ZX-calculus language, showing properties of quantum protocols in a visually compact and logically complete fashion.

  QNodes decorated with `@qml.transforms.to_zx` will return a PyZX graph that represents the computation in the ZX-calculus language.

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.transforms.to_zx
  @qml.qnode(device=dev)
  def circuit(p):
      qml.RZ(p[0], wires=1),
      qml.RZ(p[1], wires=1),
      qml.RX(p[2], wires=0),
      qml.PauliZ(wires=0),
      qml.RZ(p[3], wires=1),
      qml.PauliX(wires=1),
      qml.CNOT(wires=[0, 1]),
      qml.CNOT(wires=[1, 0]),
      qml.SWAP(wires=[0, 1]),
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  ```pycon
  >>> params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
  >>> circuit(params)
  Graph(20 vertices, 23 edges)
  ```

  Information about PyZX graphs can be found in the [PyZX Graphs API](https://pyzx.readthedocs.io/en/stable/api.html).

<h4>QChem databases and basis sets ⚛️</h4>

* The symbols and geometry of a compound from the PubChem database can now be accessed via `qchem.mol_data()`.
  [(#3289)](https://github.com/PennyLaneAI/pennylane/pull/3289)
  [(#3378)](https://github.com/PennyLaneAI/pennylane/pull/3378)

  ```pycon
  >>> import pennylane as qp
  >>> from pennylane.qchem import mol_data
  >>> mol_data("BeH2")
  (['Be', 'H', 'H'],
   tensor([[ 4.79404621,  0.29290755,  0.        ],
                [ 3.77945225, -0.29290755,  0.        ],
                [ 5.80882913, -0.29290755,  0.        ]], requires_grad=True))
  >>> mol_data(223, "CID")
  (['N', 'H', 'H', 'H', 'H'],
   tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.82264085,  0.52836742,  0.40402345],
                [ 0.01417295, -1.67429735, -0.98038991],
                [-0.98927163, -0.22714508,  1.65369933],
                [-0.84773114,  1.373075  , -1.07733286]], requires_grad=True))
  ```

* Perform quantum chemistry calculations with two new basis sets: `6-311g` and `CC-PVDZ`.
  [(#3279)](https://github.com/PennyLaneAI/pennylane/pull/3279)

  ```pycon
  >>> symbols = ["H", "He"] 
  >>> geometry = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=False)
  >>> charge = 1
  >>> basis_names = ["6-311G", "CC-PVDZ"] 
  >>> for basis_name in basis_names:
  ...     mol = qml.qchem.Molecule(symbols, geometry, charge=charge, basis_name=basis_name)
  ...     print(qml.qchem.hf_energy(mol)())
  [-2.84429531] 
  [-2.84061284]
  ```

<h4>A bunch of new operators 👀</h4>

* The controlled CZ gate and controlled Hadamard gate are now available via `qml.CCZ` and `qml.CH`, respectively.
  [(#3408)](https://github.com/PennyLaneAI/pennylane/pull/3408)

  ```pycon
  >>> ccz = qml.CCZ(wires=[0, 1, 2])
  >>> qml.matrix(ccz)
  [[ 1  0  0  0  0  0  0  0]
   [ 0  1  0  0  0  0  0  0]
   [ 0  0  1  0  0  0  0  0]
   [ 0  0  0  1  0  0  0  0]
   [ 0  0  0  0  1  0  0  0]
   [ 0  0  0  0  0  1  0  0]
   [ 0  0  0  0  0  0  1  0]
   [ 0  0  0  0  0  0  0 -1]]
  >>> ch = qml.CH(wires=[0, 1])
  >>> qml.matrix(ch)
  [[ 1.          0.          0.          0.        ]
   [ 0.          1.          0.          0.        ]
   [ 0.          0.          0.70710678  0.70710678]
   [ 0.          0.          0.70710678 -0.70710678]]
  ```

* Three new parametric operators, `qml.CPhaseShift00`, `qml.CPhaseShift01`, and `qml.CPhaseShift10`, are now available. Each of these operators performs a phase shift akin to `qml.ControlledPhaseShift` but on different positions of the state vector.
  [(#2715)](https://github.com/PennyLaneAI/pennylane/pull/2715)

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...     qml.PauliX(wires=1)
  ...     qml.CPhaseShift01(phi=1.23, wires=[0,1])
  ...     return qml.state()
  ...
  >>> circuit()
  tensor([0.        +0.j       , 0.33423773+0.9424888j, 
          1.        +0.j       , 0.        +0.j       ], requires_grad=True)
  ```

* A new gate operation called `qml.FermionicSWAP` has been added. This implements the exchange of spin orbitals
  representing fermionic-modes while maintaining proper anti-symmetrization.
  [(#3380)](https://github.com/PennyLaneAI/pennylane/pull/3380)

  ```python
  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  def circuit(phi):
      qml.BasisState(np.array([0, 1]), wires=[0, 1])
      qml.FermionicSWAP(phi, wires=[0, 1])
      return qml.state()
  ```

  ```pycon
  >>> circuit(0.1)
  tensor([0.        +0.j        , 0.99750208+0.04991671j,
        0.00249792-0.04991671j, 0.        +0.j        ], requires_grad=True)
  ```

* Create operators defined from a generator via `qml.ops.op_math.Evolution`.
  [(#3375)](https://github.com/PennyLaneAI/pennylane/pull/3375)

  `qml.ops.op_math.Evolution` defines the exponential of an operator $\hat{O}$ of the form $e^{ix\hat{O}}$, with a single trainable parameter, $x$. Limiting to a single trainable parameter allows the use of `qml.gradients.param_shift` to find the gradient with respect to the parameter $x$.

  ```python
  dev = qml.device('default.qubit', wires=2)
  
  @qml.qnode(dev, diff_method=qml.gradients.param_shift)
  def circuit(phi):
      qml.ops.op_math.Evolution(qml.PauliX(0), -.5 * phi)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> phi = np.array(1.2)
  >>> circuit(phi)
  tensor(0.36235775, requires_grad=True)
  >>> qml.grad(circuit)(phi)
  -0.9320390495504149
  ```

* The qutrit Hadamard gate, `qml.THadamard`, is now available.
  [(#3340)](https://github.com/PennyLaneAI/pennylane/pull/3340)

  The operation accepts a `subspace` keyword argument which determines which variant of the qutrit Hadamard to use.

  ```pycon
  >>> th = qml.THadamard(wires=0, subspace=[0, 1])
  >>> qml.matrix(th)
  array([[ 0.70710678+0.j,  0.70710678+0.j,  0.        +0.j],
        [ 0.70710678+0.j, -0.70710678+0.j,  0.        +0.j],
        [ 0.        +0.j,  0.        +0.j,  1.        +0.j]])
  ```

<h4>New transforms, functions, and more 😯</h4>

* Calculating the purity of arbitrary quantum states is now supported.
  [(#3290)](https://github.com/PennyLaneAI/pennylane/pull/3290)

  The purity can be calculated in an analogous fashion to, say, the Von Neumann entropy:

  * `qml.math.purity` can be used as an in-line function:

    ```pycon
    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> qml.math.purity(x, [0, 1])
    1.0
    >>> qml.math.purity(x, [0])
    0.5

    >>> x = [[1 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 / 2]]
    >>> qml.math.purity(x, [0, 1])
    0.5
    ```

  * `qml.qinfo.transforms.purity` can transform a QNode returning a state to a
    function that returns the purity:

    ```python3
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.IsingXX(x, wires=[0, 1])
        return qml.state()
    ```

    ```pycon
    >>> qml.qinfo.transforms.purity(circuit, wires=[0])(np.pi / 2)
    0.5
    >>> qml.qinfo.transforms.purity(circuit, wires=[0, 1])(np.pi / 2)
    1.0
    ```

  As with the other methods in `qml.qinfo`, the purity is fully differentiable:

  ```pycon
  >>> param = np.array(np.pi / 4, requires_grad=True)
  >>> qml.grad(qml.qinfo.transforms.purity(circuit, wires=[0]))(param)
  -0.5
  ```

* A new gradient transform, `qml.gradients.spsa_grad`, that is based on the idea of SPSA is now available.
  [(#3366)](https://github.com/PennyLaneAI/pennylane/pull/3366)

  This new transform allows users to compute a single estimate of a quantum gradient using simultaneous perturbation of parameters and a stochastic approximation. A QNode that takes, say, an argument `x`, the approximate gradient can be computed as follows.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> x = np.array(0.4, requires_grad=True)
  >>> @qml.qnode(dev)
  ... def circuit(x):
  ...     qml.RX(x, 0)
  ...     qml.RX(x, 1)
  ...     return qml.expval(qml.PauliZ(0))
  >>> grad_fn = qml.gradients.spsa_grad(circuit, h=0.1, num_directions=1)
  >>> grad_fn(x)
  array(-0.38876964)
  ```

  The argument `num_directions` determines how many directions of simultaneous perturbation are used, which is proportional to the number of circuit evaluations. See the [SPSA gradient transform documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.spsa_grad.html) for details. Note that the full SPSA optimizer is already available as `qml.SPSAOptimizer`.
  
* Multiple mid-circuit measurements can now be combined arithmetically to create new conditionals.
  [(#3159)](https://github.com/PennyLaneAI/pennylane/pull/3159)

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(wires=0)
      qml.Hadamard(wires=1)
      m0 = qml.measure(wires=0)
      m1 = qml.measure(wires=1)
      combined = 2 * m1 + m0
      qml.cond(combined == 2, qml.RX)(1.3, wires=2)
      return qml.probs(wires=2)
  ```

  ```pycon
  >>> circuit()
  [0.90843735 0.09156265]  
  ```

* A new method called `pauli_decompose()` has been added to the `qml.pauli` module, which takes a hermitian matrix, decomposes it in the Pauli basis, and returns it either as a `qml.Hamiltonian` or `qml.PauliSentence` instance.
  [(#3384)](https://github.com/PennyLaneAI/pennylane/pull/3384)

* `Operation` or `Hamiltonian` instances can now be generated from a `qml.PauliSentence` or `qml.PauliWord` via the new `operation()` and `hamiltonian()` methods.
  [(#3391)](https://github.com/PennyLaneAI/pennylane/pull/3391)

* A `sum_expand` function has been added for tapes, which splits a tape measuring a `Sum` expectation into mutliple tapes of summand expectations, and provides a function to recombine the results.
  [(#3230)](https://github.com/PennyLaneAI/pennylane/pull/3230)
  
<h4>(Experimental) More interface support for multi-measurement and gradient output types 🧪</h4>

* The autograd and Tensorflow interfaces now support devices with shot vectors when `qml.enable_return()` has been called.
  [(#3374)](https://github.com/PennyLaneAI/pennylane/pull/3374)
  [(#3400)](https://github.com/PennyLaneAI/pennylane/pull/3400)

  Here is an example using Tensorflow:

  ```python
  import tensorflow as tf
  qml.enable_return()

  dev = qml.device("default.qubit", wires=2, shots=[1000, 2000, 3000])

  @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
  def circuit(a):
      qml.RY(a, wires=0)
      qml.RX(0.2, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])
  ```

  ```pycon
  >>> a = tf.Variable(0.4)
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(a)
  ...     res = tf.stack([tf.experimental.numpy.hstack(r) for r in res])
  ...
  >>> res
  <tf.Tensor: shape=(3, 5), dtype=float64, numpy=
  array([[0.902, 0.951, 0.   , 0.   , 0.049],
         [0.898, 0.949, 0.   , 0.   , 0.051],
         [0.892, 0.946, 0.   , 0.   , 0.054]])>
  >>> tape.jacobian(res, a)
  <tf.Tensor: shape=(3, 5), dtype=float64, numpy=
  array([[-0.345     , -0.1725    ,  0.        ,  0.        ,  0.1725    ],
         [-0.383     , -0.1915    ,  0.        ,  0.        ,  0.1915    ],
         [-0.38466667, -0.19233333,  0.        ,  0.        ,  0.19233333]])>
  ```

* The PyTorch interface is now fully supported when `qml.enable_return()` has been called, allowing the calculation of the Jacobian and the Hessian using custom differentiation methods (e.g., parameter-shift, finite difference, or adjoint).
  [(#3416)](https://github.com/PennyLaneAI/pennylane/pull/3414)
  
  ```python
  import torch

  qml.enable_return()

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, diff_method="parameter-shift", interface="torch")
  def circuit(a, b):
      qml.RY(a, wires=0)
      qml.RX(b, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])
  ```

  ```pycon
  >>> a = torch.tensor(0.1, requires_grad=True)
  >>> b = torch.tensor(0.2, requires_grad=True)
  >>> torch.autograd.functional.jacobian(circuit, (a, b))
  ((tensor(-0.0998), tensor(0.)), (tensor([-0.0494, -0.0005,  0.0005,  0.0494]), tensor([-0.0991,  0.0991,  0.0002, -0.0002])))
  ```

* The JAX-JIT interface now supports first-order gradient computation when `qml.enable_return()` has been called.
  [(#3235)](https://github.com/PennyLaneAI/pennylane/pull/3235)
  [(#3445)](https://github.com/PennyLaneAI/pennylane/pull/3445)

  ```python
  import jax
  from jax import numpy as jnp

  jax.config.update("jax_enable_x64", True)

  qml.enable_return()

  dev = qml.device("lightning.qubit", wires=2)

  @jax.jit
  @qml.qnode(dev, interface="jax-jit", diff_method="parameter-shift")
  def circuit(a, b):
      qml.RY(a, wires=0)
      qml.RX(b, wires=0)
      return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

  a, b = jnp.array(1.0), jnp.array(2.0)
  ```

  ```pycon
  >>> jax.jacobian(circuit, argnums=[0, 1])(a, b)
  ((Array(0.35017549, dtype=float64, weak_type=True),
  Array(-0.4912955, dtype=float64, weak_type=True)),
  (Array(5.55111512e-17, dtype=float64, weak_type=True),
  Array(0., dtype=float64, weak_type=True)))
  ```

<h3>Improvements 🛠</h3>

* `qml.pauli.is_pauli_word` now supports instances of `qml.Hamiltonian`.
  [(#3389)](https://github.com/PennyLaneAI/pennylane/pull/3389)

* When `qml.probs`, `qml.counts`, and `qml.sample` are called with no arguments, they measure all
  wires. Calling any of the aforementioned measurements with an empty wire list (e.g., `qml.sample(wires=[])`) will raise an error.
  [(#3299)](https://github.com/PennyLaneAI/pennylane/pull/3299)

* Made `qml.gradients.finite_diff` more convenient to use with custom data type observables/devices by reducing the
  number of magic methods that need to be defined in the custom data type to support `finite_diff`.
  [(#3426)](https://github.com/PennyLaneAI/pennylane/pull/3426)

* The `qml.ISWAP` gate is now natively supported on `default.mixed`, improving on its efficiency.
  [(#3284)](https://github.com/PennyLaneAI/pennylane/pull/3284)

* Added more input validation to `qml.transforms.hamiltonian_expand` such that Hamiltonian objects with no terms raise an error.
  [(#3339)](https://github.com/PennyLaneAI/pennylane/pull/3339)

* Continuous integration checks are now performed for Python 3.11 and Torch v1.13. Python 3.7 is dropped.
  [(#3276)](https://github.com/PennyLaneAI/pennylane/pull/3276)

* `qml.Tracker` now also logs results in `tracker.history` when tracking the execution of a circuit.
  [(#3306)](https://github.com/PennyLaneAI/pennylane/pull/3306)

* The execution time of `Wires.all_wires` has been improved by avoiding data type changes and making use of `itertools.chain`.
  [(#3302)](https://github.com/PennyLaneAI/pennylane/pull/3302)

* Printing an instance of `qml.qchem.Molecule` is now more concise and informational.
  [(#3364)](https://github.com/PennyLaneAI/pennylane/pull/3364)

* The error message for `qml.transforms.insert` when it fails to diagonalize non-qubit-wise-commuting observables is now more detailed.
  [(#3381)](https://github.com/PennyLaneAI/pennylane/pull/3381)

* Extended the `qml.equal` function to `qml.Hamiltonian` and `Tensor` objects.
  [(#3390)](https://github.com/PennyLaneAI/pennylane/pull/3390)

* `QuantumTape._process_queue` has been moved to `qml.queuing.process_queue` to disentangle its functionality from the `QuantumTape` class.
  [(#3401)](https://github.com/PennyLaneAI/pennylane/pull/3401)

* QPE can now accept a target operator instead of a matrix and target wires pair.
  [(#3373)](https://github.com/PennyLaneAI/pennylane/pull/3373)

* The `qml.ops.op_math.Controlled.map_wires` method now uses `base.map_wires` internally instead of the private `_wires` property setter.
  [(#3405)](https://github.com/PennyLaneAI/pennylane/pull/3405)

* A new function called `qml.tape.make_qscript` has been created for converting a quantum function into a quantum script. This replaces `qml.transforms.make_tape`.
  [(#3429)](https://github.com/PennyLaneAI/pennylane/pull/3429)

* Add a `_pauli_rep` attribute to operators to integrate the new Pauli arithmetic classes with native PennyLane objects.
  [(#3443)](https://github.com/PennyLaneAI/pennylane/pull/3443)

* Extended the functionality of `qml.matrix` to qutrits.
  [(#3508)](https://github.com/PennyLaneAI/pennylane/pull/3508)

* The `qcut.py` file in `pennylane/transforms/` has been reorganized into multiple files that are now in `pennylane/transforms/qcut/`.
  [(#3413)](https://github.com/PennyLaneAI/pennylane/pull/3413)

* A warning now appears when creating a `Tensor` object with overlapping wires, informing that this can lead to undefined behaviour.
  [(#3459)](https://github.com/PennyLaneAI/pennylane/pull/3459)

* Extended the `qml.equal` function to `qml.ops.op_math.Controlled` and `qml.ops.op_math.ControlledOp` objects.
  [(#3463)](https://github.com/PennyLaneAI/pennylane/pull/3463)

* Nearly every instance of `with QuantumTape()` has been replaced with `QuantumScript` construction.
  [(#3454)](https://github.com/PennyLaneAI/pennylane/pull/3454)
  
* Added `validate_subspace` static method to `qml.Operator` to check the validity of the subspace of certain
  qutrit operations.
  [(#3340)](https://github.com/PennyLaneAI/pennylane/pull/3340)

* `qml.equal` now supports operators created via `qml.s_prod`, `qml.pow`, `qml.exp`, and `qml.adjoint`.
  [(#3471)](https://github.com/PennyLaneAI/pennylane/pull/3471)

* Devices can now disregard observable grouping indices in Hamiltonians through the optional `use_grouping` attribute.
  [(#3456)](https://github.com/PennyLaneAI/pennylane/pull/3456)

* Add the optional argument `lazy=True` to functions `qml.s_prod`, `qml.prod` and `qml.op_sum` to allow simplification.
  [(#3483)](https://github.com/PennyLaneAI/pennylane/pull/3483)

* Updated the `qml.transforms.zyz_decomposition` function such that it now supports broadcast operators. This means that single-qubit `qml.QubitUnitary` operators, instantiated from a batch of unitaries, can now be decomposed.
  [(#3477)](https://github.com/PennyLaneAI/pennylane/pull/3477)

* The performance of executing circuits under the `jax.vmap` transformation has been improved by being able to leverage the batch-execution capabilities of some devices.
  [(#3452)](https://github.com/PennyLaneAI/pennylane/pull/3452)

* The tolerance for converting openfermion Hamiltonian complex coefficients to real ones has been modified to prevent conversion errors.
  [(#3367)](https://github.com/PennyLaneAI/pennylane/pull/3367)

* `OperationRecorder` now inherits from `AnnotatedQueue` and `QuantumScript` instead of `QuantumTape`.
  [(#3496)](https://github.com/PennyLaneAI/pennylane/pull/3496)

* Updated `qml.transforms.split_non_commuting` to support the new return types.
  [(#3414)](https://github.com/PennyLaneAI/pennylane/pull/3414)

* Updated `qml.transforms.mitigate_with_zne` to support the new return types.
  [(#3415)](https://github.com/PennyLaneAI/pennylane/pull/3415)

* Updated `qml.transforms.metric_tensor`, `qml.transforms.adjoint_metric_tensor`, `qml.qinfo.classical_fisher`, and `qml.qinfo.quantum_fisher` to support the new return types.
  [(#3449)](https://github.com/PennyLaneAI/pennylane/pull/3449)

* Updated `qml.transforms.batch_params` and `qml.transforms.batch_input` to support the new return types.
  [(#3431)](https://github.com/PennyLaneAI/pennylane/pull/3431)

* Updated `qml.transforms.cut_circuit` and `qml.transforms.cut_circuit_mc` to support the new return types.
  [(#3346)](https://github.com/PennyLaneAI/pennylane/pull/3346)

* Limit NumPy version to `<1.24`.
  [(#3346)](https://github.com/PennyLaneAI/pennylane/pull/3346)

<h3>Breaking changes 💔</h3>

* Python 3.7 support is no longer maintained. PennyLane will be maintained for versions 3.8 and up.
  [(#3276)](https://github.com/PennyLaneAI/pennylane/pull/3276)

* The `log_base` attribute has been moved from `MeasurementProcess` to the new `VnEntropyMP` and `MutualInfoMP` classes, which inherit from `MeasurementProcess`.
  [(#3326)](https://github.com/PennyLaneAI/pennylane/pull/3326)

* `qml.utils.decompose_hamiltonian()` has been removed. Please use `qml.pauli.pauli_decompose()` instead.
  [(#3384)](https://github.com/PennyLaneAI/pennylane/pull/3384)

* The `return_type` attribute of `MeasurementProcess` has been removed where possible. Use `isinstance` checks instead.
  [(#3399)](https://github.com/PennyLaneAI/pennylane/pull/3399)

* Instead of having an `OrderedDict` attribute called `_queue`, `AnnotatedQueue` now inherits from `OrderedDict` and encapsulates the queue. Consequentially, this also applies to the `QuantumTape` class which inherits from `AnnotatedQueue`.
  [(#3401)](https://github.com/PennyLaneAI/pennylane/pull/3401)

* The `ShadowMeasurementProcess` class has been renamed to `ClassicalShadowMP`.
  [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)

* The `qml.Operation.get_parameter_shift` method has been removed. The `gradients` module should be used for general parameter-shift rules instead.
  [(#3419)](https://github.com/PennyLaneAI/pennylane/pull/3419)

* The signature of the `QubitDevice.statistics` method has been changed from

  ```python
  def statistics(self, observables, shot_range=None, bin_size=None, circuit=None):
  ```

  to

  ```python
  def statistics(self, circuit: QuantumTape, shot_range=None, bin_size=None):
  ```
  [(#3421)](https://github.com/PennyLaneAI/pennylane/pull/3421)

* The `MeasurementProcess` class is now an abstract class and `return_type` is now a property of the class.
  [(#3434)](https://github.com/PennyLaneAI/pennylane/pull/3434)

<h3>Deprecations 👋</h3>

Deprecations cycles are tracked at [doc/developement/deprecations.rst](https://docs.pennylane.ai/en/stable/development/deprecations.html).

* The following methods are deprecated:
  [(#3281)](https://github.com/PennyLaneAI/pennylane/pull/3281/)

  * `qml.tape.get_active_tape`: Use `qml.QueuingManager.active_context()` instead
  * `qml.transforms.qcut.remap_tape_wires`: Use `qml.map_wires` instead
  * `qml.tape.QuantumTape.inv()`: Use `qml.tape.QuantumTape.adjoint()` instead
  * `qml.tape.stop_recording()`: Use `qml.QueuingManager.stop_recording()` instead
  * `qml.tape.QuantumTape.stop_recording()`: Use `qml.QueuingManager.stop_recording()` instead
  * `qml.QueuingContext` is now `qml.QueuingManager`
  * `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info`: Use `update_info` instead.

* `qml.transforms.measurement_grouping` has been deprecated. Use `qml.transforms.hamiltonian_expand` instead.
  [(#3417)](https://github.com/PennyLaneAI/pennylane/pull/3417)

* The `observables` argument in `QubitDevice.statistics` is deprecated. Please use `circuit` instead.
  [(#3433)](https://github.com/PennyLaneAI/pennylane/pull/3433)

* The `seed_recipes` argument in `qml.classical_shadow` and `qml.shadow_expval` is deprecated. A new argument `seed` has been added, which defaults to None and can contain an integer with the wanted seed.
  [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)

* `qml.transforms.make_tape` has been deprecated. Please use `qml.tape.make_qscript` instead.
  [(#3478)](https://github.com/PennyLaneAI/pennylane/pull/3478)

<h3>Documentation 📝</h3>

* Added documentation on parameter broadcasting regarding both its usage and technical aspects.
  [(#3356)](https://github.com/PennyLaneAI/pennylane/pull/3356)

  The [quickstart guide on circuits](https://docs.pennylane.ai/en/stable/introduction/circuits.html#parameter-broadcasting-in-qnodes) as well as the the documentation of [QNodes](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) and [Operators](https://docs.pennylane.ai/en/stable/code/api/pennylane.operation.Operator.html) now contain introductions and details on parameter broadcasting. The QNode documentation mostly contains usage details, the Operator documentation is concerned with implementation details and a guide to support broadcasting in custom operators.

* The return type statements of gradient and Hessian transforms and a series of other functions that are a `batch_transform` have been corrected.
  [(#3476)](https://github.com/PennyLaneAI/pennylane/pull/3476)

* Developer documentation for the queuing module has been added.
  [(#3268)](https://github.com/PennyLaneAI/pennylane/pull/3268)

* More mentions of diagonalizing gates for all relevant operations have been corrected.
  [(#3409)](https://github.com/PennyLaneAI/pennylane/pull/3409)

  The docstrings for `compute_eigvals` used to say that the diagonalizing gates implemented $U$, the unitary such that $O = U \Sigma U^{\dagger}$, where $O$ is the original observable and $\Sigma$ a diagonal matrix. However, the diagonalizing gates actually implement $U^{\dagger}$, since $\langle \psi | O | \psi \rangle = \langle \psi | U \Sigma U^{\dagger} | \psi \rangle$, making $U^{\dagger} | \psi \rangle$ the actual state being measured in the $Z$-basis.

* A warning about using ``dill`` to pickle and unpickle datasets has been added.
  [(#3505)](https://github.com/PennyLaneAI/pennylane/pull/3505)

<h3>Bug fixes 🐛</h3>

* Fixed a bug that prevented `qml.gradients.param_shift` from being used for broadcasted tapes.
  [(#3528)](https://github.com/PennyLaneAI/pennylane/pull/3528)

* Fixed a bug where `qml.transforms.hamiltonian_expand` didn't preserve the type of the input results in its output.
  [(#3339)](https://github.com/PennyLaneAI/pennylane/pull/3339)

* Fixed a bug that made `qml.gradients.param_shift` raise an error when used with unshifted terms only in a custom recipe, and when using any unshifted terms at all under the new return type system.
  [(#3177)](https://github.com/PennyLaneAI/pennylane/pull/3177)

* The original tape `_obs_sharing_wires` attribute is updated during its expansion.
  [(#3293)](https://github.com/PennyLaneAI/pennylane/pull/3293)

* An issue with `drain=False` in the adaptive optimizer has been fixed. Before the fix, the operator pool needed to be reconstructed inside the optimization pool when `drain=False`. With this fix, this reconstruction is no longer needed.
  [(#3361)](https://github.com/PennyLaneAI/pennylane/pull/3361)

* If the device originally has no shots but finite shots are dynamically specified, Hamiltonian expansion now occurs.
  [(#3369)](https://github.com/PennyLaneAI/pennylane/pull/3369)

* `qml.matrix(op)` now fails if the operator truly has no matrix (e.g., `qml.Barrier`) to match `op.matrix()`.
  [(#3386)](https://github.com/PennyLaneAI/pennylane/pull/3386)

* The `pad_with` argument in the `qml.AmplitudeEmbedding` template is now compatible with all interfaces.
  [(#3392)](https://github.com/PennyLaneAI/pennylane/pull/3392)

* `Operator.pow` now queues its constituents by default.
  [(#3373)](https://github.com/PennyLaneAI/pennylane/pull/3373)

* Fixed a bug where a QNode returning `qml.sample` would produce incorrect results when run on a device defined with a shot vector.
  [(#3422)](https://github.com/PennyLaneAI/pennylane/pull/3422)

* The `qml.data` module now works as expected on Windows.
  [(#3504)](https://github.com/PennyLaneAI/pennylane/pull/3504)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Juan Miguel Arrazola,
Utkarsh Azad,
Samuel Banning,
Thomas Bromley,
Astral Cai,
Albert Mitjans Coma,
Ahmed Darwish,
Isaac De Vlugt,
Olivia Di Matteo,
Amintor Dusko,
Pieter Eendebak,
Lillian M. A. Frederiksen,
Diego Guala,
Katharine Hyatt,
Josh Izaac,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Christina Lee,
Romain Moyard,
Lee James O'Riordan,
Mudit Pandey,
Kevin Shen,
Matthew Silverman,
Jay Soni,
Antal Száva,
David Wierichs,
Moritz Willmann, and
Filippo Vicentini.
