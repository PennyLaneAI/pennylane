
# Release 0.24.0

<h3>New features since last release</h3>

<h4>All new quantum information quantities 📏</h4>

* Functionality for computing quantum information quantities for QNodes has been added.
  [(#2554)](https://github.com/PennyLaneAI/pennylane/pull/2554)
  [(#2569)](https://github.com/PennyLaneAI/pennylane/pull/2569)
  [(#2598)](https://github.com/PennyLaneAI/pennylane/pull/2598)
  [(#2617)](https://github.com/PennyLaneAI/pennylane/pull/2617)
  [(#2631)](https://github.com/PennyLaneAI/pennylane/pull/2631)
  [(#2640)](https://github.com/PennyLaneAI/pennylane/pull/2640)
  [(#2663)](https://github.com/PennyLaneAI/pennylane/pull/2663)
  [(#2684)](https://github.com/PennyLaneAI/pennylane/pull/2684)
  [(#2688)](https://github.com/PennyLaneAI/pennylane/pull/2688)
  [(#2695)](https://github.com/PennyLaneAI/pennylane/pull/2695)
  [(#2710)](https://github.com/PennyLaneAI/pennylane/pull/2710)
  [(#2712)](https://github.com/PennyLaneAI/pennylane/pull/2712)

  This includes two new QNode measurements:

  - The [Von Neumann entropy](https://en.wikipedia.org/wiki/Von_Neumann_entropy) via `qp.vn_entropy`:
  
    ```pycon
    >>> dev = qp.device("default.qubit", wires=2)
    >>> @qp.qnode(dev)
    ... def circuit_entropy(x):
    ...     qp.IsingXX(x, wires=[0,1])
    ...     return qp.vn_entropy(wires=[0], log_base=2)
    >>> circuit_entropy(np.pi/2)
    1.0
    ```

  - The [mutual information](https://en.wikipedia.org/wiki/Quantum_mutual_information) via `qp.mutual_info`:
  
    ```pycon
    >>> dev = qp.device("default.qubit", wires=2)
    >>> @qp.qnode(dev)
    ... def circuit(x):
    ...     qp.IsingXX(x, wires=[0,1])
    ...     return qp.mutual_info(wires0=[0], wires1=[1], log_base=2)
    >>> circuit(np.pi/2)
    2.0
    ```

  New differentiable transforms are also available in the `qp.qinfo` module:

  - The classical and quantum [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) via `qp.qinfo.classical_fisher`, `qp.qinfo.quantum_fisher`, respectively:
  
    ```python3
    dev = qp.device("default.qubit", wires=3)

    @qp.qnode(dev)
    def circ(params):
        qp.RY(params[0], wires=1)
        qp.CNOT(wires=(1,0))
        qp.RY(params[1], wires=1)
        qp.RZ(params[2], wires=1)
        return qp.expval(qp.PauliX(0) @ qp.PauliX(1) - 0.5 * qp.PauliZ(1))

    params = np.array([0.5, 1., 0.2], requires_grad=True)
    cfim = qp.qinfo.classical_fisher(circ)(params)
    qfim = qp.qinfo.quantum_fisher(circ)(params)
    ```

    These quantities are typically employed in variational optimization schemes to tilt the gradient in a more favourable direction  --- producing what is known as the [natural gradient](https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient). For example:

    ```pycon
    >>> grad = qp.grad(circ)(params)
    >>> cfim @ grad  # natural gradient
    [ 5.94225615e-01 -2.61509542e-02 -1.18674655e-18]
    >>> qfim @ grad  # quantum natural gradient
    [ 0.59422561 -0.02615095 -0.03989212]
    ```

  - The fidelity between two arbitrary states via `qp.qinfo.fidelity`:

    ```python
    dev = qp.device('default.qubit', wires=1)
  
    @qp.qnode(dev)
    def circuit_rx(x):
        qp.RX(x[0], wires=0)
        qp.RZ(x[1], wires=0)
        return qp.state()
  
    @qp.qnode(dev)
    def circuit_ry(y):
        qp.RY(y, wires=0)
        return qp.state()
    ```

    ```pycon
    >>> x = np.array([0.1, 0.3], requires_grad=True)
    >>> y = np.array(0.2, requires_grad=True) 
    >>> fid_func = qp.qinfo.fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])
    >>> fid_func(x, y)
    0.9905158135644924
    >>> df = qp.grad(fid_func)
    >>> df(x, y)
    (array([-0.04768725, -0.29183666]), array(-0.09489803))
    ```

  - [Reduced density matrices](https://en.wikipedia.org/wiki/Quantum_entanglement#Reduced_density_matrices) of arbitrary states via `qp.qinfo.reduced_dm`:

    ```python
    dev = qp.device("default.qubit", wires=2)
    @qp.qnode(dev)
    def circuit(x):
        qp.IsingXX(x, wires=[0,1])
        return qp.state()
    ```

    ```pycon
    >>> qp.qinfo.reduced_dm(circuit, wires=[0])(np.pi/2)
    [[0.5+0.j 0.+0.j]
      [0.+0.j 0.5+0.j]]
    ```

  - Similar transforms, `qp.qinfo.vn_entropy` and `qp.qinfo.mutual_info` exist
    for transforming QNodes.

  Currently, all quantum information measurements and transforms are differentiable, but only
  support statevector devices, with hardware support to come in a future release (with the
  exception of `qp.qinfo.classical_fisher` and `qp.qinfo.quantum_fisher`, which are both hardware
  compatible).

  For more information, check out the new [qinfo module](https://pennylane.readthedocs.io/en/stable/code/qml_qinfo.html) and
  [measurements page](https://pennylane.readthedocs.io/en/stable/introduction/measurements.html).

* In addition to the QNode transforms and measurements above, functions for computing and
  differentiating quantum information metrics with numerical statevectors and density matrices have
  been added to the `qp.math` module. This enables flexible custom post-processing.

  Added functions include:

  - `qp.math.reduced_dm`
  - `qp.math.vn_entropy`
  - `qp.math.mutual_info`
  - `qp.math.fidelity`
  
  For example:

  ```pycon
  >>> x = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)
  >>> en = qp.math.vn_entropy(x / np.sqrt(2.), indices=[0])
  >>> en
  tensor(0.6931, dtype=torch.float64, grad_fn=<DivBackward0>)
  >>> en.backward()
  >>> x.grad
  tensor([-0.3069,  0.0000,  0.0000, -0.3069])
  ```

<h4>Faster mixed-state training with backpropagation 📉</h4>

* The `default.mixed` device now supports differentiation via backpropagation with the Autograd,
  TensorFlow, and PyTorch (CPU) interfaces, leading to significantly more performant optimization
  and training.
  [(#2615)](https://github.com/PennyLaneAI/pennylane/pull/2615)
  [(#2670)](https://github.com/PennyLaneAI/pennylane/pull/2670)
  [(#2680)](https://github.com/PennyLaneAI/pennylane/pull/2680)

  As a result, the default differentiation method for the device is now `"backprop"`. To continue
  using the old default `"parameter-shift"`, explicitly specify this differentiation method in the
  QNode:

  ```python
  dev = qp.device("default.mixed", wires=2)

  @qp.qnode(dev, interface="autograd", diff_method="backprop")
  def circuit(x):
      qp.RY(x, wires=0)
      qp.CNOT(wires=[0, 1])
      return qp.expval(qp.PauliZ(wires=1))
  ```
  ```pycon
  >>> x = np.array(0.5, requires_grad=True)
  >>> circuit(x)
  array(0.87758256)
  >>> qp.grad(circuit)(x)
  -0.479425538604203
  ```

<h4>Support for quantum parameter broadcasting 📡</h4>

* Quantum operators, functions, and tapes now support broadcasting across
  parameter dimensions, making it more convenient for developers to execute their PennyLane
  programs with multiple sets of parameters.
  [(#2575)](https://github.com/PennyLaneAI/pennylane/pull/2575)
  [(#2609)](https://github.com/PennyLaneAI/pennylane/pull/2609)

  Parameter broadcasting refers to passing tensor parameters with additional leading dimensions
  to quantum operators; additional dimensions will flow through the computation,
  and produce additional dimensions at the output.

  For example, instantiating a rotation gate with a one-dimensional array leads to a broadcasted
  `Operation`:

  ```pycon
  >>> x = np.array([0.1, 0.2, 0.3], requires_grad=True)
  >>> op = qp.RX(x, 0)
  >>> op.batch_size
  3
  ```

  Its matrix correspondingly is augmented by a leading dimension of size `batch_size`:

  ```pycon
  >>> np.round(qp.matrix(op), 4)
  tensor([[[0.9988+0.j    , 0.    -0.05j  ],
         [0.    -0.05j  , 0.9988+0.j    ]],
        [[0.995 +0.j    , 0.    -0.0998j],
         [0.    -0.0998j, 0.995 +0.j    ]],
        [[0.9888+0.j    , 0.    -0.1494j],
         [0.    -0.1494j, 0.9888+0.j    ]]], requires_grad=True)
  >>> qp.matrix(op).shape
  (3, 2, 2)
  ```
  
  This can be extended to quantum functions, where we may mix-and-match operations
  with batched parameters and those without. However, the `batch_size` of each batched
  `Operator` within the quantum function must be the same:

  ```pycon
  >>> dev = qp.device('default.qubit', wires=1)
  >>> @qp.qnode(dev)
  ... def circuit_rx(x, z):
  ...     qp.RX(x, wires=0)
  ...     qp.RZ(z, wires=0)
  ...     qp.RY(0.3, wires=0)
  ...     return qp.probs(wires=0)
  >>> circuit_rx([0.1, 0.2], [0.3, 0.4])
  tensor([[0.97092256, 0.02907744],
          [0.95671515, 0.04328485]], requires_grad=True)
  ```

  Parameter broadcasting is supported on all devices, hardware and simulator. Note that
  if not natively supported by the underlying device, parameter broadcasting may result
  in additional quantum device evaluations.

* A new transform, `qp.transforms.broadcast_expand`, has been added,
  which automates the process of transforming quantum functions (and tapes) to
  multiple quantum evaluations with no parameter broadcasting.
  [(#2590)](https://github.com/PennyLaneAI/pennylane/pull/2590)

  ```pycon
  >>> dev = qp.device('default.qubit', wires=1)
  >>> @qp.transforms.broadcast_expand()
  >>> @qp.qnode(dev)
  ... def circuit_rx(x, z):
  ...     qp.RX(x, wires=0)
  ...     qp.RZ(z, wires=0)
  ...     qp.RY(0.3, wires=0)
  ...     return qp.probs(wires=0)
  >>> print(qp.draw(circuit_rx)([0.1, 0.2], [0.3, 0.4]))
  0: ──RX(0.10)──RZ(0.30)──RY(0.30)─┤  Probs
  \
  0: ──RX(0.20)──RZ(0.40)──RY(0.30)─┤  Probs
  ```

  Under-the-hood, this transform is used for devices that don't natively support
  parameter broadcasting.

* To specify that a device natively supports broadcasted tapes,
  the new flag `Device.capabilities()["supports_broadcasting"]` should be set to `True`.

* To support parameter broadcasting for new or custom operations, the following
  new `Operator` class attributes must be specified:

  - `Operator.ndim_params` specifies expected number of dimensions for each parameter

  Once set, `Operator.batch_size` and `QuantumTape.batch_size` will dynamically compute the
  parameter broadcasting axis dimension, if present.

<h4>Improved JAX JIT support 🏎</h4>

* JAX just-in-time (JIT) compilation now supports vector-valued QNodes,
  enabling new types of workflows and significant performance boosts.
  [(#2034)](https://github.com/PennyLaneAI/pennylane/pull/2034)
  
  Vector-valued QNodes include those with:
  * `qp.probs`;
  * `qp.state`;
  * `qp.sample` or
  * multiple `qp.expval` / `qp.var` measurements.

  Consider a QNode that returns basis-state probabilities:

  ```python
  dev = qp.device('default.qubit', wires=2)
  x = jnp.array(0.543)
  y = jnp.array(-0.654)

  @jax.jit
  @qp.qnode(dev, diff_method="parameter-shift", interface="jax")
  def circuit(x, y):
      qp.RX(x, wires=[0])
      qp.RY(y, wires=[1])
      qp.CNOT(wires=[0, 1])
      return qp.probs(wires=[1])
  ```
  ```pycon
  >>> circuit(x, y)
  Array([0.8397495 , 0.16025047], dtype=float32)
  ```

  Note that computing the jacobian of vector-valued QNode is not supported with JAX
  JIT. The output of vector-valued QNodes can, however, be used in
  the definition of scalar-valued cost functions whose gradients can be
  computed.

  For example, one can define a cost function that outputs the first element of
  the probability vector:

  ```python
  def cost(x, y):
      return circuit(x, y)[0]
  ```
  
  ```pycon
  >>> jax.grad(cost, argnums=[0])(x, y)
  (Array(-0.2050439, dtype=float32),)
  ```

<h4>More drawing styles 🎨</h4>

* New `solarized_light` and `solarized_dark` styles are available for drawing circuit diagram graphics. 
  [(#2662)](https://github.com/PennyLaneAI/pennylane/pull/2662)

<h4>New operations & transforms 🤖</h4>  
  
* The `qp.IsingXY` gate is now available (see [1912.04424](https://arxiv.org/abs/1912.04424)). 
  [(#2649)](https://github.com/PennyLaneAI/pennylane/pull/2649)

* The `qp.ECR` (echoed cross-resonance) operation is now available (see
  [2105.01063](https://arxiv.org/pdf/2105.01063.pdf)). This gate is a maximally-entangling gate and
  is equivalent to a CNOT gate up to single-qubit pre-rotations.
  [(#2613)](https://github.com/PennyLaneAI/pennylane/pull/2613)

* The adjoint transform `adjoint` can now accept either a single instantiated operator or a quantum
  function. It returns an entity of the same type / call signature as what it was given:
  [(#2222)](https://github.com/PennyLaneAI/pennylane/pull/2222)
  [(#2672)](https://github.com/PennyLaneAI/pennylane/pull/2672)

  ```pycon
  >>> qp.adjoint(qp.PauliX(0))
  Adjoint(PauliX)(wires=[0])
  >>> qp.adjoint(qp.RX)(1.23, wires=0)
  Adjoint(RX)(1.23, wires=[0])
  ```

  Now, `adjoint` wraps operators in a symbolic operator class `qp.ops.op_math.Adjoint`. This class
  should not be constructed directly; the `adjoint` constructor should always be used instead. The
  class behaves just like any other `Operator`:

  ```pycon
  >>> op = qp.adjoint(qp.S(0))
  >>> qp.matrix(op)
  array([[1.-0.j, 0.-0.j],
        [0.-0.j, 0.-1.j]])
  >>> qp.eigvals(op)
  array([1.-0.j, 0.-1.j])
  ```

* A new symbolic operator class `qp.ops.op_math.Pow` represents an operator raised to a power. 
When `decomposition()` is called, a list of new operators equal to this one raised to the given power is given:
  [(#2621)](https://github.com/PennyLaneAI/pennylane/pull/2621)

  ```pycon
  >>> op = qp.ops.op_math.Pow(qp.PauliX(0), 0.5)
  >>> op.decomposition()
  [SX(wires=[0])]
  >>> qp.matrix(op)
  array([[0.5+0.5j, 0.5-0.5j],
       [0.5-0.5j, 0.5+0.5j]])
  ```

* A new transform `qp.batch_partial` is available which behaves similarly to `functools.partial`,
  but supports batching in the unevaluated parameters.
  [(#2585)](https://github.com/PennyLaneAI/pennylane/pull/2585)

  This is useful for executing a circuit with a batch dimension in some of its parameters:

  ```python
  dev = qp.device("default.qubit", wires=1)

  @qp.qnode(dev)
  def circuit(x, y):
     qp.RX(x, wires=0)
     qp.RY(y, wires=0)
     return qp.expval(qp.PauliZ(wires=0))
  ```

  ```pycon
  >>> batched_partial_circuit = qp.batch_partial(circuit, x=np.array(np.pi / 4))
  >>> y = np.array([0.2, 0.3, 0.4])
  >>> batched_partial_circuit(y=y)
  tensor([0.69301172, 0.67552491, 0.65128847], requires_grad=True)
  ```

* A new transform `qp.split_non_commuting` is available, which splits a quantum
  function or tape into multiple functions/tapes determined by groups of commuting
  observables:
  [(#2587)](https://github.com/PennyLaneAI/pennylane/pull/2587)

  ```python
  dev = qp.device("default.qubit", wires=1)

  @qp.transforms.split_non_commuting
  @qp.qnode(dev)
  def circuit(x):
      qp.RX(x,wires=0)
      return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(0))]
  ```

  ```pycon
  >>> print(qp.draw(circuit)(0.5))
  0: ──RX(0.50)─┤  <X>
  \
  0: ──RX(0.50)─┤  <Z>
  ```

<h3>Improvements</h3>

* Expectation values of multiple non-commuting observables from within a single QNode are now
  supported:
  [(#2587)](https://github.com/PennyLaneAI/pennylane/pull/2587)

  ```
  >>> dev = qp.device('default.qubit', wires=1)
  >>> @qp.qnode(dev)
  ... def circuit_rx(x, z):
  ...     qp.RX(x, wires=0)
  ...     qp.RZ(z, wires=0)
  ...     return qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0))
  >>> circuit_rx(0.1, 0.3)
  tensor([ 0.02950279, -0.09537451], requires_grad=True)
  ```

* Selecting which parts of parameter-shift Hessians are computed is now possible.
  [(#2538)](https://github.com/PennyLaneAI/pennylane/pull/2538)

  The `argnum` keyword argument for `qp.gradients.param_shift_hessian`
  is now allowed to be a two-dimensional Boolean `array_like`.
  Only the indicated entries of the Hessian will then be computed.

  A particularly useful example is the computation of the diagonal
  of the Hessian:

  ```python
  dev = qp.device("default.qubit", wires=1)

  @qp.qnode(dev)
  def circuit(x):
      qp.RX(x[0], wires=0)
      qp.RY(x[1], wires=0)
      qp.RX(x[2], wires=0)
      return qp.expval(qp.PauliZ(0))

  argnum = qp.math.eye(3, dtype=bool)
  x = np.array([0.2, -0.9, 1.1], requires_grad=True)
  ```

  ```pycon
  >>> qp.gradients.param_shift_hessian(circuit, argnum=argnum)(x)
  tensor([[-0.09928388,  0.        ,  0.        ],
          [ 0.        , -0.27633945,  0.        ],
          [ 0.        ,  0.        , -0.09928388]], requires_grad=True)
  ```

* Commuting Pauli operators are now measured faster.
  [(#2425)](https://github.com/PennyLaneAI/pennylane/pull/2425)

  The logic that checks for qubit-wise commuting (QWC) observables has been improved, resulting in
  a performance boost that is noticable when many commuting Pauli operators of the same type are
  measured.

* It is now possible to add `Observable` objects to the integer `0`, for example
  `qp.PauliX(wires=[0]) + 0`.
  [(#2603)](https://github.com/PennyLaneAI/pennylane/pull/2603)

* Wires can now be passed as the final argument to an `Operator`, instead of requiring the wires to
  be explicitly specified with keyword `wires`. This functionality already existed for
  `Observable`s, but now extends to all `Operator`s:
  [(#2432)](https://github.com/PennyLaneAI/pennylane/pull/2432)

  ```pycon
  >>> qp.S(0)
  S(wires=[0])
  >>> qp.CNOT((0,1))
  CNOT(wires=[0, 1])
  ```

* The `qp.taper` function can now be used to consistently taper any additional observables such as
  dipole moment, particle number, and spin operators using the symmetries obtained from the
  Hamiltonian.
  [(#2510)](https://github.com/PennyLaneAI/pennylane/pull/2510)

* Sparse Hamiltonians' representation has changed from Coordinate (COO) to Compressed Sparse Row
  (CSR) format.
  [(#2561)](https://github.com/PennyLaneAI/pennylane/pull/2561)

  The CSR representation is more performant for arithmetic operations and matrix-vector products.
  This change decreases the `expval()` calculation time for `qp.SparseHamiltonian`, specially for
  large workflows. In addition, the CSR format consumes less memory for `qp.SparseHamiltonian`
  storage.

* IPython now displays the `str` representation of a `Hamiltonian`, rather than the `repr`. This
  displays more information about the object.
  [(#2648)](https://github.com/PennyLaneAI/pennylane/pull/2648)

* The `qp.qchem` tests have been restructured.
  [(#2593)](https://github.com/PennyLaneAI/pennylane/pull/2593)
  [(#2545)](https://github.com/PennyLaneAI/pennylane/pull/2545)

  - OpenFermion-dependent tests are now localized and collected in
    `tests.qchem.of_tests`. The new module `test_structure` is created to collect the tests of the
    `qchem.structure` module in one place and remove their dependency to OpenFermion.

  - Test classes have been created to group the integrals and matrices
    unit tests.

* An `operations_only` argument is introduced to the `tape.get_parameters` method.
  [(#2543)](https://github.com/PennyLaneAI/pennylane/pull/2543)

* The `gradients` module now uses faster subroutines and uniform formats of gradient rules.
  [(#2452)](https://github.com/XanaduAI/pennylane/pull/2452)

* Instead of checking types, objects are now processed in the `QuantumTape` based on a new
  `_queue_category` property. This is a temporary fix that will disappear in the future.
  [(#2408)](https://github.com/PennyLaneAI/pennylane/pull/2408)

* The `QNode` class now contains a new method `best_method_str` that returns the best
  differentiation method for a provided device and interface, in human-readable format.
  [(#2533)](https://github.com/PennyLaneAI/pennylane/pull/2533)
  
* Using `Operation.inv()` in a queuing environment no longer updates the queue's metadata, but
  merely updates the operation in place.
  [(#2596)](https://github.com/PennyLaneAI/pennylane/pull/2596)

* A new method `safe_update_info` is added to `qp.QueuingContext`. This method is substituted
  for `qp.QueuingContext.update_info` in a variety of places.
  [(#2612)](https://github.com/PennyLaneAI/pennylane/pull/2612)
  [(#2675)](https://github.com/PennyLaneAI/pennylane/pull/2675)

* `BasisEmbedding` can accept an int as argument instead of a list of bits.
  [(#2601)](https://github.com/PennyLaneAI/pennylane/pull/2601)

  For example, `qp.BasisEmbedding(4, wires = range(4))` is now equivalent to
  `qp.BasisEmbedding([0,1,0,0], wires = range(4))` (as `4==0b100`).

* Introduced a new `is_hermitian` property to `Operator`s to determine if an operator can be used in a measurement
  process.
  [(#2629)](https://github.com/PennyLaneAI/pennylane/pull/2629)

* Added separate `requirements_dev.txt` for separation of concerns for code development and just
  using PennyLane.
  [(#2635)](https://github.com/PennyLaneAI/pennylane/pull/2635)

* The performance of building sparse Hamiltonians has been improved by accumulating the sparse
  representation of coefficient-operator pairs in a temporary storage and by eliminating
  unnecessary `kron` operations on identity matrices.
  [(#2630)](https://github.com/PennyLaneAI/pennylane/pull/2630)

* Control values are now displayed distinctly in text and matplotlib drawings of circuits.
  [(#2668)](https://github.com/PennyLaneAI/pennylane/pull/2668)

* The `TorchLayer` `init_method` argument now accepts either a `torch.nn.init` function or a
  dictionary which should specify a `torch.nn.init`/`torch.Tensor` for each different weight.
  [(#2678)](https://github.com/PennyLaneAI/pennylane/pull/2678)

* The unused keyword argument `do_queue` for `Operation.adjoint` is now fully removed.
  [(#2583)](https://github.com/PennyLaneAI/pennylane/pull/2583)

* Several non-decomposable `Adjoint` operators are added to the device test suite.
  [(#2658)](https://github.com/PennyLaneAI/pennylane/pull/2658)

* The developer-facing `pow` method has been added to `Operator` with concrete implementations for
  many classes.
  [(#2225)](https://github.com/PennyLaneAI/pennylane/pull/2225)

* The `ctrl` transform and `ControlledOperation` have been moved to the new `qp.ops.op_math`
  submodule. The developer-facing `ControlledOperation` class is no longer imported top-level.
  [(#2656)](https://github.com/PennyLaneAI/pennylane/pull/2656)

<h3>Deprecations</h3>

* `qp.ExpvalCost` has been deprecated, and usage will now raise a warning.
  [(#2571)](https://github.com/PennyLaneAI/pennylane/pull/2571)

  Instead, it is recommended to simply
  pass Hamiltonians to the `qp.expval` function inside QNodes:

  ```python
  @qp.qnode(dev)
  def ansatz(params):
      some_qfunc(params)
      return qp.expval(Hamiltonian)
  ```

<h3>Breaking changes</h3>

* When using `qp.TorchLayer`, weights with negative shapes will now raise an error, while weights
  with `size = 0` will result in creating empty Tensor objects.
  [(#2678)](https://github.com/PennyLaneAI/pennylane/pull/2678)
  
* PennyLane no longer supports TensorFlow `<=2.3`.
  [(#2683)](https://github.com/PennyLaneAI/pennylane/pull/2683)

* The `qp.queuing.Queue` class has been removed. 
  [(#2599)](https://github.com/PennyLaneAI/pennylane/pull/2599)

* The `qp.utils.expand` function is now removed; `qp.operation.expand_matrix` should be used
  instead.
  [(#2654)](https://github.com/PennyLaneAI/pennylane/pull/2654)

* The module `qp.gradients.param_shift_hessian` has been renamed to
  `qp.gradients.parameter_shift_hessian` in order to distinguish it from the identically named
  function. Note that the `param_shift_hessian` function is unaffected by this change and can be
  invoked in the same manner as before via the `qp.gradients` module.
  [(#2528)](https://github.com/PennyLaneAI/pennylane/pull/2528)
  
* The properties `eigval` and `matrix` from the `Operator` class were replaced with the methods
  `eigval()` and `matrix(wire_order=None)`.
  [(#2498)](https://github.com/PennyLaneAI/pennylane/pull/2498)

* `Operator.decomposition()` is now an instance method, and no longer accepts parameters.
  [(#2498)](https://github.com/PennyLaneAI/pennylane/pull/2498)

* Adds tests, adds no-coverage directives, and removes inaccessible logic to improve code coverage.
  [(#2537)](https://github.com/PennyLaneAI/pennylane/pull/2537)

* The base classes `QubitDevice` and `DefaultQubit` now accept data-types for a statevector. This
  enables a derived class (device) in a plugin to choose correct data-types:
  [(#2448)](https://github.com/PennyLaneAI/pennylane/pull/2448)

  ```pycon
  >>> dev = qp.device("default.qubit", wires=4, r_dtype=np.float32, c_dtype=np.complex64)
  >>> dev.R_DTYPE
  <class 'numpy.float32'>
  >>> dev.C_DTYPE
  <class 'numpy.complex64'>
  ```

<h3>Bug fixes</h3>

* Fixed a bug where returning `qp.density_matrix` using the PyTorch interface would return a
  density matrix with wrong shape.
  [(#2643)](https://github.com/PennyLaneAI/pennylane/pull/2643)

* Fixed a bug to make `param_shift_hessian` work with QNodes in which gates marked
  as trainable do not have any impact on the QNode output.
  [(#2584)](https://github.com/PennyLaneAI/pennylane/pull/2584)

* QNodes can now interpret variations on the interface name, like `"tensorflow"`
  or `"jax-jit"`, when requesting backpropagation.
  [(#2591)](https://github.com/PennyLaneAI/pennylane/pull/2591)

* Fixed a bug for `diff_method="adjoint"` where incorrect gradients were computed for QNodes with
  parametrized observables (e.g., `qp.Hermitian`).
  [(#2543)](https://github.com/PennyLaneAI/pennylane/pull/2543)

* Fixed a bug where `QNGOptimizer` did not work with operators whose generator was a Hamiltonian.
  [(#2524)](https://github.com/PennyLaneAI/pennylane/pull/2524)

* Fixed a bug with the decomposition of `qp.CommutingEvolution`.
  [(#2542)](https://github.com/PennyLaneAI/pennylane/pull/2542)

* Fixed a bug enabling PennyLane to work with the latest version of Autoray.
  [(#2549)](https://github.com/PennyLaneAI/pennylane/pull/2549)

* Fixed a bug which caused different behaviour for `Hamiltonian @ Observable` and `Observable @
  Hamiltonian`.
  [(#2570)](https://github.com/PennyLaneAI/pennylane/pull/2570)

* Fixed a bug in `DiagonalQubitUnitary._controlled` where an invalid operation was queued instead
  of the controlled version of the diagonal unitary.
  [(#2525)](https://github.com/PennyLaneAI/pennylane/pull/2525)

* Updated the gradients fix to only apply to the `strawberryfields.gbs` device, since the original
  logic was breaking some devices.
  [(#2485)](https://github.com/PennyLaneAI/pennylane/pull/2485)
  [(#2595)](https://github.com/PennyLaneAI/pennylane/pull/2595)

* Fixed a bug in `qp.transforms.insert` where operations were not inserted after gates within a
  template.
  [(#2704)](https://github.com/PennyLaneAI/pennylane/pull/2704)
  
* `Hamiltonian.wires` is now properly updated after in place operations.
  [(#2738)](https://github.com/PennyLaneAI/pennylane/pull/2738)

<h3>Documentation</h3>

* The centralized [Xanadu Sphinx Theme](https://github.com/XanaduAI/xanadu-sphinx-theme)
  is now used to style the Sphinx documentation.
  [(#2450)](https://github.com/PennyLaneAI/pennylane/pull/2450)

* Added a reference to `qp.utils.sparse_hamiltonian` in `qp.SparseHamiltonian` to clarify
  how to construct sparse Hamiltonians in PennyLane.
  [(2572)](https://github.com/PennyLaneAI/pennylane/pull/2572)

* Added a new section in the [Gradients and Training](https://pennylane.readthedocs.io/en/stable/introduction/interfaces.html)
  page that summarizes the supported device configurations and provides justification.
  In addition, [code examples](https://pennylane.readthedocs.io/en/stable/introduction/unsupported.html) were added for some selected configurations.
  [(#2540)](https://github.com/PennyLaneAI/pennylane/pull/2540)

* Added a note for the [Depolarization Channel](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.DepolarizingChannel.html)
  that specifies how the channel behaves for the different values of depolarization probability `p`.
  [(#2669)](https://github.com/PennyLaneAI/pennylane/pull/2669)

* The quickstart documentation has been improved.
  [(#2530)](https://github.com/PennyLaneAI/pennylane/pull/2530)
  [(#2534)](https://github.com/PennyLaneAI/pennylane/pull/2534)
  [(#2564](https://github.com/PennyLaneAI/pennylane/pull/2564)
  [(#2565](https://github.com/PennyLaneAI/pennylane/pull/2565)
  [(#2566)](https://github.com/PennyLaneAI/pennylane/pull/2566)
  [(#2607)](https://github.com/PennyLaneAI/pennylane/pull/2607)
  [(#2608)](https://github.com/PennyLaneAI/pennylane/pull/2608)
  
* The quantum chemistry quickstart documentation has been improved.
  [(#2500)](https://github.com/PennyLaneAI/pennylane/pull/2500)
  
* Testing documentation has been improved.
  [(#2536)](https://github.com/PennyLaneAI/pennylane/pull/2536)
  
* Documentation for the `pre-commit` package has been added.
  [(#2567)](https://github.com/PennyLaneAI/pennylane/pull/2567)
  
* Documentation for draw control wires change has been updated.
  [(#2682)](https://github.com/PennyLaneAI/pennylane/pull/2682)
  
<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje, Mikhail Andrenkov, Juan Miguel Arrazola, Ali Asadi, Utkarsh Azad, Samuel Banning, 
Avani Bhardwaj, Thomas Bromley, Albert Mitjans Coma, Isaac De Vlugt, Amintor Dusko, Trent Fridey, Christian Gogolin,
Qi Hu, Katharine Hyatt, David Ittah, Josh Izaac, Soran Jahangiri, Edward Jiang, Nathan Killoran, Korbinian Kottmann, 
Ankit Khandelwal, Christina Lee, Chae-Yeun Park, Mason Moreland, Romain Moyard, Maria Schuld, Jay Soni, Antal Száva, 
tal66, David Wierichs, Roeland Wiersema, WingCode.