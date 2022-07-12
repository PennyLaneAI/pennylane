:orphan:

# Release 0.25.0-dev (development release)

<h3>New features since last release</h3>

* Added new feature `qml.ops.op_math.Product.py` that was created using Sum.py as a template. This new 
  feature allows for the creation of a product operation between operators. The product of rotation operatiors is 
  commutative so the order of the products does not matter. 
  **Example**
    The product of qml.PauliX and qml.PauliZ must be equal to -i*qml.PauliY. Which means that the matrix must be equal to 
    -i*array([[0,-i],[i,0]])
    >>> product_op = op_product(qml.PauliX(0), qml.PauliZ(0))
    >>> product_op
    PauliX(wires=[0]) * PauliZ(wires=[0])
    >>> prouct_op.matrix()
    array([[ 0,  -1],
           [ 1, 0]])
    >>> summed_op.terms()
    ([1.0, 1.0], (PauliX(wires=[0]), PauliZ(wires=[0])))

    .. details::
        :title: Usage Details

        We can have products of operators in the same Hilbert Space. 
        For example, multiplying operators whose matrices have the same dimensions. 
        Support for operators with different dimensions is not available yet.
    """
* Changed the definition of qml.BasicEntanglerLayers to take in multiple single qubit rotations as parameters instead of just 1 
  If the parameter rotation is set as a single qubit rotation, the class still works as usual, if it passes as parameter a 
  list of single wubit rotations, the class wil automatically check for the correct format for the weights parameter that represent the rotation angle of each operator.
   !!! Update for more than one single-qubit rotation gates !!! 
         - The basic entangler can now take more than one gate as parameter for repetition. The code works as follows:
         For a given list of operations, rotations = [RX,RY,RZ], the model takes in parameters in shape (num_layers,num_wires*num_rotations)
         where num_rotations = len(rotations). Meaning that each rotation takes a parameter per layer per wire. 
        - The function compute decomposition checks the form of rotations, if it is a single roation element the code works as it is used to.
        For a list, it checks the length of the list, if the length is equal to 1, the code gets the element and works as usual. If the length of 
        rotations list is more than 1, the code implements the operation product between the single-qubit operations. 

        For more information refer to pennylane.ops.op_math.product.py

        Example: 
        For a two layer, 3 rotations and 4 wires system, the params have the form 
        
        tensor([[0.98490185, 0.48615071, 0.65416114, 0.76073784, 0.4379965 ,
         0.91467668, 0.37770095, 0.91138513, 0.14018763, 0.48878116,
         0.94855556, 0.67714962],
        [0.54151177, 0.05728717, 0.94766153, 0.43230254, 0.49035082,
         0.50956715, 0.56727017, 0.57852111, 0.86937769, 0.03215202,
         0.78536781, 0.81338788]], requires_grad=True)

         Where the first index of the tensor indicates the layer, the second one indicates the position of the parameter. For a layer,
         the first 4 parameters correspond to the parameters for the first rotation on each wire, the next 4 parameters correspond to the 
         parameters for the second rotation on each wire, and so on. 

        ** Example Usage **
            >>> import pennylane as qml
            >>> from pennylane import numpy as np

            >>> n_wires = 4
            ... dev = qml.device('default.qubit', wires=n_wires)
            ... rotations = [qml.RX, qml.RZ,qml.RY]
            ... n_layers = 2
            ... @qml.qnode(dev)
            ... def circuit(weights):
            ...     qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires), rotation = rotations)
            ...     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]
            ... params = np.random.random(size=(n_layers, n_wires*len(rotations)))
            >>> circuit(params)
            tensor([1., 1., 1., 1.], requires_grad=True)


* Added the new optimizer, `qml.SPSAOptimizer` that implements the simultaneous
  perturbation stochastic approximation method based on
  [An Overview of the Simultaneous Perturbation Method for Efficient Optimization](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF).
  [(#2661)](https://github.com/PennyLaneAI/pennylane/pull/2661)

  It is a suitable optimizer for cost functions whose evaluation may involve
  noise, as optimization with SPSA may significantly decrease the number of
  quantum executions for the entire optimization.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=1)
  >>> def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  >>> coeffs = [1, 1]
  >>> obs = [qml.PauliX(0), qml.PauliZ(0)]
  >>> H = qml.Hamiltonian(coeffs, obs)
  >>> @qml.qnode(dev)
  ... def cost(params):
  ...     circuit(params)
  ...     return qml.expval(H)
  >>> params = np.random.normal(0, np.pi, (2), requires_grad=True)
  >>> print(params)
  [-5.92774911 -4.26420843]
  >>> print(cost(params))
  0.43866366253270167
  >>> max_iterations = 50
  >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
  >>> for _ in range(max_iterations):
  ...     params, energy = opt.step_and_cost(cost, params)
  >>> print(params)
  [-6.21193761 -2.99360548]
  >>> print(energy)
  -1.1258709813834058
  ```

* The quantum information module now supports computation of relative entropy.
  [(#2772)](https://github.com/PennyLaneAI/pennylane/pull/2772)

  It includes a function in `qml.math`:

  ```pycon
  >>> rho = np.array([[0.3, 0], [0, 0.7]])
  >>> sigma = np.array([[0.5, 0], [0, 0.5]])
  >>> qml.math.relative_entropy(rho, sigma)
  tensor(0.08228288, requires_grad=True)
  ```

  as well as a QNode transform:

  ```python
  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  def circuit(param):
      qml.RY(param, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.state()
  ```
  ```pycon
  >>> relative_entropy_circuit = qml.qinfo.relative_entropy(circuit, circuit, wires0=[0], wires1=[0])
  >>> x, y = np.array(0.4), np.array(0.6)
  >>> relative_entropy_circuit((x,), (y,))
  0.017750012490703237
  ```


* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for drawing circuit diagram graphics.
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)


**Operator Arithmetic:**

* Adds a base class `qml.ops.op_math.SymbolicOp` for single-operator symbolic
  operators such as `Adjoint` and `Pow`.
  [(#2721)](https://github.com/PennyLaneAI/pennylane/pull/2721)

* Added operation `qml.QutritUnitary` for applying user-specified unitary operations on qutrit devices.
  [(#2699)](https://github.com/PennyLaneAI/pennylane/pull/2699)

* A `Sum` symbolic class is added that allows users to represent the sum of operators.
  [(#2475)](https://github.com/PennyLaneAI/pennylane/pull/2475)

  The `Sum` class provides functionality like any other PennyLane operator. We can
  get the matrix, eigenvalues, terms, diagonalizing gates and more.

  ```pycon
  >>> summed_op = qml.op_sum(qml.PauliX(0), qml.PauliZ(0))
  >>> summed_op
  PauliX(wires=[0]) + PauliZ(wires=[0])
  >>> qml.matrix(summed_op)
  array([[ 1,  1],
         [ 1, -1]])
  >>> summed_op.terms()
  ([1.0, 1.0], (PauliX(wires=[0]), PauliZ(wires=[0])))
  ```

  The `summed_op` can also be used inside a `qnode` as an observable.
  If the circuit is parameterized, then we can also differentiate through the
  sum observable.

  ```python
  sum_op = Sum(qml.PauliX(0), qml.PauliZ(1))
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, grad_method="best")
  def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(sum_op)
  ```

  ```
  >>> weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
  >>> qml.grad(circuit)(weights)
  tensor([-0.09347337, -0.18884787, -0.28818254], requires_grad=True)
  ```
* New FlipSign operator that flips the sign for a given basic state. [(#2780)](https://github.com/PennyLaneAI/pennylane/pull/2780)


<h3>Improvements</h3>

* Samples can be grouped into counts by passing the `counts=True` flag to `qml.sample`.
  [(#2686)](https://github.com/PennyLaneAI/pennylane/pull/2686)

  Note that the change included creating a new `Counts` measurement type in `measurements.py`.

  `counts=True` can be set when obtaining raw samples in the computational basis:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...     qml.Hadamard(wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     # passing the counts flag
  ...     return qml.sample(counts=True)   
  >>> result = circuit()
  >>> print(result)
  {'00': 495, '11': 505}
  ```

  Counts can also be obtained when sampling the eigenstates of an observable:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...   qml.Hadamard(wires=0)
  ...   qml.CNOT(wires=[0, 1])
  ...   return qml.sample(qml.PauliZ(0), counts=True), qml.sample(qml.PauliZ(1), counts=True)
  >>> result = circuit()
  >>> print(result)
  [tensor({-1: 526, 1: 474}, dtype=object, requires_grad=True)
   tensor({-1: 526, 1: 474}, dtype=object, requires_grad=True)]
  ```

* The `qml.state` and `qml.density_matrix` measurements now support custom wire
  labels.
  [(#2779)](https://github.com/PennyLaneAI/pennylane/pull/2779)

* Adds a new function to compare operators. `qml.equal` can be used to compare equality of parametric operators taking 
  into account their interfaces and trainability.
  [(#2651)](https://github.com/PennyLaneAI/pennylane/pull/2651)

* The `default.mixed` device now supports backpropagation with the `"jax"` interface.
  [(#2754)](https://github.com/PennyLaneAI/pennylane/pull/2754)

* Quantum channels such as `qml.BitFlip` now support abstract tensors. This allows
  their usage inside QNodes decorated by `tf.function`, `jax.jit`, or `jax.vmap`:

  ```python
  dev = qml.device("default.mixed", wires=1)

  @qml.qnode(dev, diff_method="backprop", interface="jax")
  def circuit(t):
      qml.PauliX(wires=0)
      qml.ThermalRelaxationError(0.1, t, 1.4, 0.1, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```
  ```pycon
  >>> x = jnp.array([0.8, 1.0, 1.2])
  >>> jax.vmap(circuit)(x)
  DeviceArray([-0.78849435, -0.8287073 , -0.85608006], dtype=float32)
  ```

* Added an `are_pauli_words_qwc` function which checks if certain 
  Pauli words are pairwise qubit-wise commuting. This new function improves performance when measuring hamiltonians 
  with many commuting terms. 
  [(#2789)](https://github.com/PennyLaneAI/pennylane/pull/2798)

<h3>Breaking changes</h3>

* PennyLane now depends on newer versions (>=2.7) of the `semantic_version` package,
  which provides an updated API that is incompatible which versions of the package prior to 2.7.
  If you run into issues relating to this package, please reinstall PennyLane.
  [(#2744)](https://github.com/PennyLaneAI/pennylane/pull/2744)
  [(#2767)](https://github.com/PennyLaneAI/pennylane/pull/2767)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* Optimization examples of using JAXopt and Optax with the JAX interface have
  been added.
  [(#2769)](https://github.com/PennyLaneAI/pennylane/pull/2769)

<h3>Bug fixes</h3>

* `qml.grouping.group_observables` now works when individual wire
  labels are iterable.
  [(#2752)](https://github.com/PennyLaneAI/pennylane/pull/2752)

* The adjoint of an adjoint has a correct `expand` result.
  [(#2766)](https://github.com/PennyLaneAI/pennylane/pull/2766)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):


David Ittah, Edward Jiang, Ankit Khandelwal, Christina Lee, Sergio Martínez-Losa, Ixchel Meza Chavez, 
Mudit Pandey, Bogdan Reznychenko, Jay Soni, Antal Száva, Moritz Willmann
