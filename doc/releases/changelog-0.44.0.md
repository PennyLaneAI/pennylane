# Release 0.44.0 (current release)

<h3>New features since last release</h3>

<h4>Quantum Random Access Memory (QRAM) 💾</h4>

* Three implementations of QRAM are now available in PennyLane, including Bucket Brigade QRAM 
  (:class:`~.BBQRAM`), a Select-Only QRAM (:class:`~.SelectOnlyQRAM`), and a Hybrid QRAM 
  (:class:`~.HybridQRAM`) that combines behaviour from both :class:`~.BBQRAM` and 
  :class:`~.SelectOnlyQRAM`. The choice of QRAM implementation depends on the application, ranging
  from width versus depth tradeoffs to noise resilience.
  [(#8670)](https://github.com/PennyLaneAI/pennylane/pull/8670)
  [(#8679)](https://github.com/PennyLaneAI/pennylane/pull/8679)
  [(#8680)](https://github.com/PennyLaneAI/pennylane/pull/8680)
  [(#8801)](https://github.com/PennyLaneAI/pennylane/pull/8801)

  Irrespective of the specific implementation, QRAM encodes bitstrings, :math:`b_i`, corresponding to a 
  given entry, :math:`i`, of a data set of length :math:`N`, and can do so in superposition: 
  :math:`\text{QRAM} \sum_i c_i \vert i \rangle \vert 0 \rangle = \sum_i c_i \vert i \rangle \vert b_i \rangle`.
  Here, the first register representing :math:`\vert i \rangle` is called the ``control_wires`` register 
  (often referred to as the "address" register in literature), and the second register containing 
  :math:`\vert b_i \rangle` is called the ``target_wires`` register (where the 
  :math:`i^{\text{th}}` entry of the data set is loaded).

  Each QRAM implementation available in this release can be briefly described as follows:

  * :class:`~.BBQRAM` : a bucket-brigade style QRAM implementation that is also resilient to noise.
  * :class:`~.SelectOnlyQRAM` : a QRAM implementation that comprises a series of :class:`~.MultiControlledX` gates.
  * :class:`~.HybridQRAM` : a QRAM implementation that combines :class:`~.BBQRAM` and :class:`~.SelectOnlyQRAM` in a manner that allows for tradeoffs between depth and width.

  An example of using :class:`~.BBQRAM` to read data into a target register is given below, where 
  the data set in question is given by a list of ``bitstrings`` and we wish to read its second entry
  (``"110"``):

  ```python
  import pennylane as qml

  bitstrings = ["010", "111", "110", "000"]
  bitstring_size = 3

  num_control_wires = 2 # len(bistrings) = 4 = 2**2
  num_work_wires = 1 + 3 * ((1 << num_control_wires) - 1) # 10

  reg = qml.registers(
      {
          "control": num_control_wires,
          "target": bitstring_size,
          "work_wires": num_work_wires
      }
  )

  dev = qml.device("default.qubit")
  @qml.qnode(dev)
  def bb_quantum():
      # prepare an address, e.g., |10> (index 2)
      qml.BasisEmbedding(2, wires=reg["control"])

      qml.BBQRAM(
          bitstrings,
          control_wires=reg["control"],
          target_wires=reg["target"],
          work_wires=reg["work_wires"],
      )
      return qml.probs(wires=reg["target"])
  ```

  ```pycon
  >>> import numpy as np
  >>> print(np.round(bb_quantum()))  
  [0. 0. 0. 0. 0. 0. 1. 0.]
  ```

  Note that ``"110"`` in binary is equal to 6 in decimal, which is the position of the only 
  non-zero entry in the ``target_wires`` register.

  For more information on each implementation of QRAM in this release, check out their respective
  documentation pages: :class:`~.BBQRAM`, :class:`~.SelectOnlyQRAM`, and:class:`~.HybridQRAM`.

* A lightweight representation of the :class:`~.BBQRAM` template called ``qml.estimator.BBQRAM`` has 
  been added for fast and efficient resource estimation.
  [(#8825)](https://github.com/PennyLaneAI/pennylane/pull/8825)

  Like with other existing lightweight representations of PennyLane operations, leveraging 
  ``qml.estimator.BBQRAM`` for fast resource estimation can be done in two ways: 

  * Using ``qml.estimator.BBQRAM`` directly inside of a function and then calling 
    :func:`estimate <pennylane.estimator.estimate.estimate>`:

    ```python
    import pennylane.estimator as qre

    def circuit():
        qre.CNOT()
        qre.QFT(num_wires=4)
        qre.BBQRAM(num_bitstrings=30, size_bitstring=8, num_wires=100)
        qre.Hadamard()
    ```

    ```
    >>> print(qre.estimate(circuit)())
    --- Resources: ---
    Total wires: 100
      algorithmic wires: 100
      allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 4.504E+3
      'Toffoli': 1.096E+3,
      'T': 792,
      'CNOT': 2.475E+3,
      'Z': 120,
      'Hadamard': 21
    ```

  * On a simulatable circuit with detailed information:

    ```python
    bitstrings = ["010", "111", "110", "000"]
    bitstring_size = 3

    num_control_wires = 2 # len(bistrings) = 4 = 2**2
    num_work_wires = 1 + 3 * ((1 << num_control_wires) - 1) # 10

    reg = qml.registers(
        {
            "control": num_control_wires,
            "target": bitstring_size,
            "work_wires": num_work_wires
        }
    )

    dev = qml.device("default.qubit")
    @qml.qnode(dev)
    def bb_quantum():
        # prepare an address, e.g., |10> (index 2)
        qml.BasisEmbedding(2, wires=reg["control"])

        qml.BBQRAM(
            bitstrings,
            control_wires=reg["control"],
            target_wires=reg["target"],
            work_wires=reg["work_wires"],
        )
        return qml.probs(wires=reg["target"])
    ```

    ```pycon
    >>> print(qre.estimate(bb_quantum)())
    --- Resources: ---
    Total wires: 15
      algorithmic wires: 15
      allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 181
      'Toffoli': 40,
      'CNOT': 128,
      'X': 1,
      'Z': 6,
      'Hadamard': 6
    ```

<h4>Quantum Automatic Differentiation 🤖</h4>

* The Hadamard test gradient method (``diff_method="hadamard"``) in PennyLane now has an ``"auto"`` 
  mode, which automatically chooses the most efficient mode of differentiation.
  [(#8640)](https://github.com/PennyLaneAI/pennylane/pull/8640)
  [(#8875)](https://github.com/PennyLaneAI/pennylane/pull/8875)

  The Hadamard test gradient method is a hardware-compatible differentation method that can 
  differentiate a broad range of parameterized gates. Using the ``"auto"`` mode with 
  ``diff_method="hadamard"`` will result in an automatic selection of the method (either 
  ``"standard"``, ``"reversed"``, ``"direct"``, or ``"reversed-direct"``) which results in the 
  fewest total executions. This takes into account the number of observables, the number of 
  generators, the number of measurements, and the presence of available auxiliary wires. For more 
  details on how ``"auto"`` works, consult the section titled 
  "Variants of the standard hadamard gradient" in the documentation for the Hadamard test gradient 
  (:func:`qml.gradients.hadamard_grad <pennylane.gradients.hadamard_grad>`).

  The ``"auto"`` method can be accessed by specifying it in ``gradient_kwargs`` in the QNode when 
  using ``diff_method="hadamard"``:

  ```python
  dev = qml.device('default.qubit')
  @qml.qnode(dev, diff_method="hadamard", gradient_kwargs={"mode": "auto"})
  def circuit(x):
      qml.evolve(qml.X(0) @ qml.X(1) + qml.Z(0) @ qml.Z(1) + qml.H(0), x)
      return qml.expval(qml.Z(0) @ qml.Z(1) + qml.Y(0))
  ```

  ```pycon
  >>> print(qml.grad(circuit)(qml.numpy.array(0.5)))
  0.7342549405478683
  ```

  Theoretical information on how each mode works can be found in 
  [arXiv:2408.05406](https://arxiv.org/pdf/2408.05406). 

<h4>Instantaneous Quantum Polynomial Circuits 💨</h4>

* A new template for defining an Instantaneous Quantum Polynomial (:class:`~.IQP`) circuit has been 
  added, as well as an associated :class:`~.estimator.resource_operator.ResourceOperator` for 
  resource estimation in the :mod:`~.estimator` module. These new features facilitate the simulation 
  and resource estimation of large-scale generative quantum machine learning tasks.
  [(#8748)](https://github.com/PennyLaneAI/pennylane/pull/8748)
  [(#8807)](https://github.com/PennyLaneAI/pennylane/pull/8807)
  [(#8749)](https://github.com/PennyLaneAI/pennylane/pull/8749)
  
  While :class:`~.IQP` circuits belong to a class of circuits that are believed to be hard to sample 
  from using classical algorithms, Recio-Armengol et al. showed in a recent paper titled 
  [Train on classical, deploy on quantum](https://arxiv.org/abs/2503.02934) that such circuits can 
  still be optimized efficiently.

  Here is a simple example showing how to define an :class:`~.IQP` circuit and how to estimate the 
  required quantum resources using the :func:`~.estimator.estimate.estimate` function:

  ```python
  import pennylane as qml
  import pennylane.estimator as qre

  pattern = [[[0]],[[1]],[[0,1]]]

  @qml.qnode(qml.device('lightning.qubit', wires=2))
  def circuit():
    qml.IQP(
        weights=[1., 2., 3.],
        num_wires=2,
        pattern=pattern,
        spin_sym=False,
    )
    return qml.state()
  ```

  ```pycon
  >>> res = qre.estimate(circuit)()
  >>> print(res)
  --- Resources: ---
    Total wires: 2
      algorithmic wires: 2
      allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 138
      'T': 132,
      'CNOT': 2,
      'Hadamard': 4
  ```
  
  The expectation values of Pauli-Z type observables for parameterized :class:`~.IQP` circuits can 
  be efficeintly evaluated with the :func:`pennylane.qnn.iqp_expval` function. This estimator 
  function is based on a randomized method allowing for the efficient optimization of circuits with 
  thousands of qubits and millions of gates.

  ```python
    from pennylane.qnn import iqp_expval
    import jax

    num_wires = 2
    ops = np.array([[0, 1], [1, 0], [1, 1]]) # binary array representing ops Z1, Z0, Z0Z1
    n_samples = 1000
    key = jax.random.PRNGKey(42)

    weights = np.ones(len(pattern))
    pattern = [[[0]], [[1]], [[0, 1]]]

    expvals, stds = iqp_expval(ops, weights, pattern, num_wires, n_samples, key)
  ```

  ```pycon
  >>> print(expvals, stds)
  [0.14506625 0.17813912 0.18971463] [0.02614436 0.02615901 0.02615425]
  ```

  For more theoretical details, check out our 
  [Fast optimization of instantaneous quantum polynomial circuits](https://pennylane.ai/qml/demos/tutorial_iqp_circuit_optimization_jax) 
  demo.

<h4>Arbitrary State Preparation 😎</h4>

* A new template :class:`~.MultiplexerStatePreparation` is now available, allowing for the 
  preparation of arbitrary states using :class:`~.SelectPauliRot` operations.
  [(#8581)](https://github.com/PennyLaneAI/pennylane/pull/8581)

  Using :class:`~.MultiplexerStatePreparation` is analogous to using other state preparation 
  techniques in PennyLane.

  ```python
  probs_vector = np.array([0.5, 0., 0.25, 0.25])

  dev = qml.device("default.qubit", wires = 2)
  wires = [0, 1]

  @qml.qnode(dev)
  def circuit():
    qml.MultiplexerStatePreparation(np.sqrt(probs_vector), wires)
    return qml.probs(wires)
  ```
  
  ```pycon
  >>> np.round(circuit(), 2)
  array([0.5 , 0.  , 0.25, 0.25])
  ```

For theoretical details, see [arXiv:0208112](https://arxiv.org/abs/quant-ph/0208112).

<h4>Pauli-based computation 💻 </h4>
  
New tools dedicated to fault-tolerant quantum computing (FTQC) research based on the Pauli-based
computation (PBC) framework are now available! With this release, you can express, compile, and 
inspect workflows written in terms of Pauli product rotations (PPRs) and Pauli product measurements 
(PPMs), which are the building blocks for the PBC framework.

* Writing circuits in terms of 
  `Pauli product measurements <https://pennylane.ai/compilation/pauli-product-measurement>`_ (PPMs) 
  in PennyLane is now possible with the new :func:`~.pauli_measure` function. Using this function in 
  tandem with :class:`~.PauliRot` to represent PPRs unlocks surface-code FTQC research spurred from 
  `A Game of Surface Codes <http://arxiv.org/abs/1808.02892>`_.
  [(#8461)](https://github.com/PennyLaneAI/pennylane/pull/8461)
  [(#8631)](https://github.com/PennyLaneAI/pennylane/pull/8631)
  [(#8623)](https://github.com/PennyLaneAI/pennylane/pull/8623)
  [(#8663)](https://github.com/PennyLaneAI/pennylane/pull/8663)
  [(#8692)](https://github.com/PennyLaneAI/pennylane/pull/8692)

  The new :func:`~.pauli_measure` function is currently only for analysis on the ``null.qubit`` device,
  which allows for circuit inspection with :func:`~.specs` and :func:`~.drawer.draw`.

  Using :func:`~.pauli_measure` in a circuit is similar to :func:`~.measure` (a mid-circuit measurement),
  but requires that a ``pauli_word`` be specified for the measurement basis:

  ```python
  import pennylane as qml

  dev = qml.device("null.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(0)
      qml.Hadamard(2)
      qml.PauliRot(np.pi / 4, pauli_word="XYZ", wires=[0, 1, 2])
      ppm = qml.pauli_measure(pauli_word="XY", wires=[0, 2])
      qml.cond(ppm, qml.X)(wires=1)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──H─╭RXYZ(0.79)─╭┤↗X├────┤  <Z>
  1: ────├RXYZ(0.79)─│──────X─┤
  2: ──H─╰RXYZ(0.79)─╰┤↗Y├──║─┤
                       ╚════╝
  ```

  You can use the :func:`~.specs` function to easily determine the circuit's resources. In this 
  case, in addition to other gates, we can see that the circuit includes one PPR and one PPM 
  operation (represented by the :class:`~.PauliRot` and ``PauliMeasure`` gate types, respectively):

  ```pycon
  >>> print(qml.specs(circuit)()['resources'])
  Total wire allocations: 3
  Total gates: 5
  Circuit depth: 4

  Gate types:
    Hadamard: 2
    PauliRot: 1
    PauliMeasure: 1
    Conditional(PauliX): 1

  Measurements:
    expval(PauliZ): 1
  ```

* Several ``qjit``-compatible compilation passes designed for Pauli-based computation are now available
  with this release, and are designed to work directly with :func:`~.pauli_measure` and :class:`~.PauliRot` operations.
  [(#8609)](https://github.com/PennyLaneAI/pennylane/pull/8609)
  [(#8764)](https://github.com/PennyLaneAI/pennylane/pull/8764)
  [(#8762)](https://github.com/PennyLaneAI/pennylane/pull/8762)
  
  The compilation passes included in this release are:

  * :func:`~.gridsynth`: This pass decomposes :math:`Z`-basis rotations and :class:`~.PhaseShift`
  gates to either the Clifford+T basis or to other PPRs.

    ```python
    @qml.qjit
    @qml.transforms.gridsynth
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x):
        qml.Hadamard(0)
        qml.RZ(x, 0)
        qml.PhaseShift(x * 0.2, 0)
        return qml.state()

    ```
    ```pycon
    >>> circuit(1.1)
    [0.60284353-0.36960984j 0.5076425 +0.4922066j ]
    ```

  * Seven transforms for compiling Clifford+T gates, PPRs, and/or PPMs, including :func:`~.transforms.to_ppr`,
  :func:`~.transforms.commute_ppr`, :func:`~.transforms.merge_ppr_ppm`,
  :func:`~.transforms.ppr_to_ppm`, :func:`~.transforms.ppm_compilation`,
  :func:`~.transforms.reduce_t_depth`, and :func:`~.transforms.decompose_arbitrary_ppr`.
  
    ```python
    @qml.qjit(target="mlir")
    @qml.transforms.to_ppr
    @qml.qnode(qml.device("null.qubit", wires=2))
    def circuit():
        qml.H(0)
        qml.CNOT([0, 1])
        qml.T(0)
        return qml.expval(qml.Z(0))
    ```

    ```pycon
    >>> print(qml.specs(circuit, level=2)())
    ...
    Resource specifications:
      Total wire allocations: 2
      Total gates: 7
      Circuit depth: Not computed

      Gate types:
        PPR-pi/4: 6
        PPR-pi/8: 1
    ...
    ```

* Directly decomposing Clifford+T gates and other small gates into PPRs is possible using the 
  :func:`~.transforms.decompose` transform with graph-based decompositions enabled 
  (:func:`~.decomposition.enable_graph`). This allows direct decomposition of certain operators 
  without the need to use approximate methods such as those found in the 
  :func:`~.clifford_t_decomposition` transform, which can sometimes be less efficient.
  [(#8700)](https://github.com/PennyLaneAI/pennylane/pull/8700)
  [(#8704)](https://github.com/PennyLaneAI/pennylane/pull/8704)
  [(#8857)](https://github.com/PennyLaneAI/pennylane/pull/8857)

  The following operations have newly added decomposition rules in terms of PPRs 
  (:class:`~.PauliRot`):
  - :class:`~.CRX`, :class:`~.CRY`, :class:`~.CRZ`
  - :class:`~.ControlledPhaseShift`
  - :class:`~.IsingXX`, :class:`~.IsingYY`, :class:`~.IsingZZ`
  - :class:`~.PSWAP`
  - :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`
  - :class:`~.SingleExcitation`, :class:`~.DoubleExcitation`
  - :class:`~.SWAP`, :class:`~.ISWAP`, :class:`~.SISWAP`
  - :class:`~.CY`, :class:`~.CZ`, :class:`~.CSWAP`, :class:`~.CNOT`, :class:`~.Toffoli`

  To access these decompositions, simply specify a target gate set including :class:`~.PauliRot` and 
  :class:`~.GlobalPhase`. The following example illustrates how the :class:`~.CNOT` gate can be 
  represented in terms of three :math:`\tfrac{\pi}{2}` PPRs (``IX``, ``ZI`` and ``ZX``) acting on 
  two wires:

  ```python
  from functools import partial
  
  qml.decomposition.enable_graph()

  @partial(qml.transforms.decompose, gate_set={qml.PauliRot, qml.GlobalPhase})
  @qml.qnode(qml.device("null.qubit", wires=2))
  def circuit():
      qml.CNOT([0, 1])
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──RZ(-1.57)─╭RZX(1.57)─╭GlobalPhase(0.79)───────────┤  <Z>
  1: ──RX(-1.57)─╰RZX(1.57)─╰GlobalPhase(0.79)──RZ(3.14)─┤    
  ```

<h4>Flexible and modular compilation pipelines 🦋</h4>

* Defining large and complex compilation pipelines in intuitive, modular, and flexible ways is now 
  possible with the new :class:`~.CompilePipeline` class.
  [(#8735)](https://github.com/PennyLaneAI/pennylane/pull/8735)
  [(#8750)](https://github.com/PennyLaneAI/pennylane/pull/8750)
  [(#8731)](https://github.com/PennyLaneAI/pennylane/pull/8731)
  [(#8817)](https://github.com/PennyLaneAI/pennylane/pull/8817)
  [(#8703)](https://github.com/PennyLaneAI/pennylane/pull/8703)
  [(#8730)](https://github.com/PennyLaneAI/pennylane/pull/8730)
  [(#8751)](https://github.com/PennyLaneAI/pennylane/pull/8751)
  [(#8774)](https://github.com/PennyLaneAI/pennylane/pull/8774)
  [(#8781)](https://github.com/PennyLaneAI/pennylane/pull/8781)
  [(#8834)](https://github.com/PennyLaneAI/pennylane/pull/8834)

  The :class:`~.CompilePipeline` class allows you to chain together multiple transforms
  to create custom circuit optimization pipelines with ease. For example, :class:`~.CompilePipeline` objects can compound:

  ```pycon
  >>> pipeline = qml.CompilePipeline(qml.transforms.commute_controlled, qml.transforms.cancel_inverses)
  >>> qml.CompilePipeline(pipeline, qml.transforms.merge_rotations)
  CompilePipeline(commute_controlled, cancel_inverses, merge_rotations)
  ```

  They can be added together with ``+``:

  ```pycon
  >>> pipeline += qml.transforms.merge_rotations
  >>> pipeline
  CompilePipeline(commute_controlled, cancel_inverses, merge_rotations)
  ```

  They can be multiplied by scalars via ``*`` to repeat compilation passes a predetermined number of times:
  
  ```pycon
  >>> pipeline += 2 * qml.transforms.cancel_inverses(recursive=True)
  >>> pipeline
  CompilePipeline(commute_controlled, cancel_inverses, merge_rotations, cancel_inverses, cancel_inverses)
  ```

  Finally, they can be modified via ``list`` operations like ``append``, ``extend``, and ``insert``:

  ```pycon
  >>> pipeline.insert(0, qml.transforms.remove_barrier)
  >>> pipeline
  CompilePipeline(remove_barrier, commute_controlled, cancel_inverses, merge_rotations, cancel_inverses, cancel_inverses)
  ```

  By applying a created ``pipeline`` directly on a quantum function as a decorator, each compilation pass
  therein will be applied to the circuit:

  ```python
  import pennylane as qml

  pipeline = qml.transforms.merge_rotations + qml.transforms.cancel_inverses(recursive=True)

  @pipeline
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
    qml.H(0)
    qml.H(0)
    qml.RX(0.5, 1)
    qml.RX(0.2, 1)
    return qml.expval(qml.Z(0) @ qml.Z(1))
  ```
  ```pycon
  >>> print(qml.draw(circuit)())
  0: ───────────┤ ╭<Z@Z>
  1: ──RX(0.70)─┤ ╰<Z@Z>
  ```

<h4>Analyzing algorithms quickly and easily with resource estimation 📖</h4>

* A new :func:`~pennylane.resource.algo_error` function has been added to compute algorithm-specific 
  errors from quantum circuits. This provides a dedicated entry point for retrieving error information 
  that was previously accessible through :func:`~pennylane.specs`.
  [(#8787)](https://github.com/PennyLaneAI/pennylane/pull/8787)
  
  The function works with QNodes and 
  returns a dictionary of error types and their computed values:

  ```python
  import pennylane as qml
  
  Hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])
  
  dev = qml.device("default.qubit")
  
  @qml.qnode(dev)
  def circuit():
      qml.TrotterProduct(Hamiltonian, time=1.0, n=4, order=2)
      return qml.state()
  ```

  ```pycon
  >>> qml.resource.algo_error(circuit)()
  {'SpectralNormError': SpectralNormError(0.25)}
  ```

* Fast resource estimation is now available for many algorithms, including:
   
  * The **Generalized Quantum Signal Processing** (GQSP) algorithm and its time evolution via the 
    ``qml.estimator.GQSP`` and ``qml.estimator.GQSPTimeEvolution`` resource operations.
    [(#8675)](https://github.com/PennyLaneAI/pennylane/pull/8675)

  * The **Qubitization** algorithm via two new resource operators: ``qml.estimator.Reflection`` and
    ``qml.estimator.Qubitization``.
    [(#8675)](https://github.com/PennyLaneAI/pennylane/pull/8675)

  * The **Quantum Signal Processing** (QSP) and **Quantum Singular Value Transformation** (QSVT) 
    algorithms via two new resource operators: ``qml.estimator.QSP``.
    [(#8733)](https://github.com/PennyLaneAI/pennylane/pull/8733)

  * The **unary iteration implementation of QPE** via the new ``qml.estimator.UnaryIterationQPE``
    subroutine, which makes it possible to reduce ``T`` and ``Toffoli`` gate counts in exchange for 
    using additional qubits.
    [(#8708)](https://github.com/PennyLaneAI/pennylane/pull/8708)

  * **Trotterization** for Pauli Hamiltonians, using the new ``qml.estimator.PauliHamiltonian``
    resource Hamiltonian class and the new ``qml.estimator.TrotterPauli`` resource operator. 
    [(#8546)](https://github.com/PennyLaneAI/pennylane/pull/8546)
    [(#8761)](https://github.com/PennyLaneAI/pennylane/pull/8761)
    
    ```pycon
    >>> import pennylane.estimator as qre
    >>> pauli_terms = {"X": 10, "XX": 5, "XXXX": 3, "YY": 5, "ZZ":5, "Z": 2}
    >>> pauli_ham = qre.PauliHamiltonian(num_qubits=10, pauli_terms=pauli_terms)
    >>> res = qre.estimate(qre.TrotterPauli(pauli_ham, num_steps=1, order=2))
    >>> res.total_gates
    2844
    ```
    The ``PauliHamiltonian`` object also makes it easy to access the total number of terms (Pauli 
    words) in the Hamiltonians with the ``PauliHamiltonian.num_terms`` property:

    ```pycon
    >>> pauli_ham.num_terms
    30
    ```
  
  * **Linear combination of unitaries** (LCU) representations of ``qml.estimator.PauliHamiltonian`` 
    Hamiltonians via the new ``qml.estimator.SelectPauli`` operator.
    [(#8675)](https://github.com/PennyLaneAI/pennylane/pull/8675)


* The new ``resource_key`` keyword argument of the
  :meth:`ResourceConfig.set_precision <pennylane.estimator.resource_config.ResourceConfig.set_precision>`
  method makes it possible to set precisions for a larger variety of `ResourceOperator`s in the 
  :mod:`estimator <pennylane.estimator>` module, including ``phase_grad_precision`` and 
  ``coeff_precision`` for ``TrotterVibronic`` and ``TrotterVibrational``, ``rotation_precision`` for 
  ``GQSP`` and ``QSP`` and ``poly_approx_precision`` for ``GQSPTimeEvolution``. 
  [(#8561)](https://github.com/PennyLaneAI/pennylane/pull/8561)
  
  ```pycon
  >>> vibration_ham = qre.VibrationalHamiltonian(num_modes=2, grid_size=4, taylor_degree=2)
  >>> trotter = qre.TrotterVibrational(vibration_ham, num_steps=10, order=2)
  >>> config = qre.ResourceConfig()
  >>> qre.estimate(trotter, config = config).total_gates
  123867.0
  >>> config.set_precision(qre.TrotterVibrational, precision=1e-10, resource_key='phase_grad_precision')
  >>> qre.estimate(trotter, config = config).total_gates
  124497.0
  ```

<h4>Seamless inspection for compiled programs 👓</h4>

* Analyzing resources throughout each step of a compilation pipeline can now be done on ``qjit``'d 
  workflows with :func:`~.specs`, providing a pass-by-pass overview of quantum circuit resources.
  [(#8606)](https://github.com/PennyLaneAI/pennylane/pull/8606)
  [(#8860)](https://github.com/PennyLaneAI/pennylane/pull/8860)
  
  Consider the following :func:`qjit <pennylane.qjit>`'d circuit with two compilation passes 
  applied:
  
  ```python
  @qml.qjit
  @qml.transforms.merge_rotations
  @qml.transforms.cancel_inverses
  @qml.qnode(qml.device('lightning.qubit', wires=2))
  def circuit():
      qml.RX(1.23, wires=0)
      qml.RX(1.23, wires=0)
      qml.X(0)
      qml.X(0)
      qml.CNOT([0, 1])
      return qml.probs()
  ```

  The supplied ``level`` to :func:`~.specs` can be an individual ``int`` value or an iterable of 
  multiple levels. Additionally, the strings ``"all"`` and ``"all-mlir"`` are allowed, returning 
  circuit resources for all user-applied transforms and MLIR passes, or all user-applied MLIR passes 
  only, respectively.

  ```pycon
  >>> print(qml.specs(circuit, level=[2, 3])())
  Device: lightning.qubit
  Device wires: 2
  Shots: Shots(total=None)
  Level: ['cancel-inverses (MLIR-1)', 'merge-rotations (MLIR-2)']

  Resource specifications:
  Level = cancel-inverses (MLIR-1):
    Total wire allocations: 2
    Total gates: 3
    Circuit depth: Not computed

    Gate types:
      RX: 2
      CNOT: 1

    Measurements:
      probs(all wires): 1

  ------------------------------------------------------------

  Level = merge-rotations (MLIR-2):
    Total wire allocations: 2
    Total gates: 2
    Circuit depth: Not computed

    Gate types:
      RX: 1
      CNOT: 1

    Measurements:
      probs(all wires): 1
  ```

* A new :func:`~.marker` function allows for easy inspection at particular points in a set of 
  applied compilation passes with :func:`~.specs` and :func:`~.drawer.draw` instead of having to 
  increment ``level`` by integer amounts.
  [(#8684)](https://github.com/PennyLaneAI/pennylane/pull/8684)

  The :func:`~.marker` function works like a transform in PennyLane, and can be deployed as
  a decorator on top of QNodes:

  ```python
  @qml.marker(level="rotations-merged")
  @qml.transforms.merge_rotations
  @qml.marker(level="my-level")
  @qml.transforms.cancel_inverses
  @qml.transforms.decompose(gate_set={qml.RX})
  @qml.qnode(qml.device('lightning.qubit'))
  def circuit():
      qml.RX(0.2,0)
      qml.X(0)
      qml.X(0)
      qml.RX(0.2, 0)
      return qml.state()
  ```

  The string supplied to :func:`~.marker` can then be used as an argument to ``level`` in ``draw``
  and ``specs``, showing the cumulative result of applying transforms up to the marker:

  ```pycon
  >>> print(qml.draw(circuit, level="my-level")())
  0: ──RX(0.20)──RX(3.14)──RX(3.14)──RX(0.20)─┤  State
  >>> print(qml.draw(circuit, level="rotations-merged")())
  0: ──RX(6.68)─┤  State
  ```

  Note that :func:`~.marker` is currently not compatible with programs compiled with 
  :func:`~pennylane.qjit`.

<h3>Improvements 🛠</h3>

<h4>Resource estimation</h4>

* It is now easier to access the total gates and wires in resource estimates with the 
  ``total_wires`` and ``total_gates`` properties in the ``qml.estimator.Resources`` class.
  [(#8761)](https://github.com/PennyLaneAI/pennylane/pull/8761)

  ```python
  import pennylane.estimator as qre

  def circuit():
      qml.X(0)
      qml.Z(0)
      qml.Y(1)
  ```

  ```pycon
  >>> resources = qre.estimate(circuit)()
  >>> resources.total_gates
  3
  >>> resources.total_wires
  2
  ```

* The ``QROM`` template now uses fewer resources when argument values are ``restored=True`` and 
  ``sel_swap_depth=1``.
  [(#8761)](https://github.com/PennyLaneAI/pennylane/pull/8761)

* The resource decomposition of ``PauliRot`` now matches the optimal resources when the 
  ``pauli_string`` argument is ``XX`` or ``YY``.
  [(#8562)](https://github.com/PennyLaneAI/pennylane/pull/8562)

* It is now possible to estimate the resources for quantum circuits that contain or decompose into
  any of the following symbolic operators: :class:`~.ChangeOpBasis`, :class:`~.Prod`,
  :class:`~.Controlled`, :class:`~.ControlledOp`, :class:`~.Pow`, and/or :class:`~.Adjoint`.
  [(#8464)](https://github.com/PennyLaneAI/pennylane/pull/8464)

* Qualtran call graphs built via :func:`qml.to_bloq <pennylane.to_bloq>` now provide faster resource
  counting by using PennyLane's resource estimation module. 
  To use the previous behaviour based on PennyLane decompositions, set 
  ``call_graph='decomposition'``.
  [(#8390)](https://github.com/PennyLaneAI/pennylane/pull/8390)

  The old behaviour was the following:

  ```pycon
  >>> qml.to_bloq(qml.QFT(wires=range(5)), map_ops=False, call_graph='decomposition').call_graph()[1]
  {Hadamard(): 5,
   ZPowGate(exponent=-0.15915494309189535, eps=1e-11): 10,
   ZPowGate(exponent=-0.15915494309189535, eps=5e-12): 10,
   ZPowGate(exponent=0.15915494309189535, eps=5e-12): 10,
   CNOT(): 20,
   TwoBitSwap(): 2
  }
  ```

  The new behaviour is now this:

  ```pycon
  >>> qml.to_bloq(qml.QFT(wires=range(5)), map_ops=False).call_graph()[1]
  {Hadamard(): 5, CNOT(): 26, TGate(is_adjoint=False): 1320}
  ```

* The :class:`~pennylane.estimator.compact_hamiltonian.CDFHamiltonian`, 
  :class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`,  
  :class:`~pennylane.estimator.compact_hamiltonian.VibrationalHamiltonian`, and 
  :class:`~pennylane.estimator.compact_hamiltonian.VibronicHamiltonian` classes have been modified 
  to take the 1-norm of the Hamiltonian as an optional argument.
  [(#8697)](https://github.com/PennyLaneAI/pennylane/pull/8697)

* Input validation has been added to various operators and functions in the 
  :mod:`estimator <pennylane.estimator>` module to raise more informative errors.
  [(#8835)](https://github.com/PennyLaneAI/pennylane/pull/8835)

* The ``ResourcesUndefinedError`` has been removed from the ``adjoint``, ``ctrl``, and ``pow`` 
  resource decomposition methods of ``ResourceOperator`` to avoid using errors as control flow.
  [(#8598)](https://github.com/PennyLaneAI/pennylane/pull/8598)
  [(#8811)](https://github.com/PennyLaneAI/pennylane/pull/8811)

<h4>Decompositions</h4>

* The graph-based decomposition system now supports basis-changing Clifford gates and decomposing
  ``RX``, ``RY`` and ``RZ`` rotations into each other. 
  [(#8569)](https://github.com/PennyLaneAI/pennylane/pull/8569)

* A new decomposition has been added for the Controlled :class:`~.SemiAdder`, which reduces the 
  number of gates in its decomposition by controlling fewer gates.
  [(#8423)](https://github.com/PennyLaneAI/pennylane/pull/8423)

* A new ``gate_set`` method has been added to ``DeviceCapabilities`` that makes it easy to produce a 
  set of gate names that are directly compatible with the given device.
  [(#8522)](https://github.com/PennyLaneAI/pennylane/pull/8522)

  ```pycon
  >>> dev = qml.device('lightning.qubit')
  >>> dev.capabilities.gate_set()
  {'Adjoint(CNOT)',
  'Adjoint(CRX)',
  'Adjoint(CRY)',
  'Adjoint(CRZ)',
  'Adjoint(CRot)',
  ...
  ```

* It is now possible to minimize the number of work wires in decompositions by activating the new 
  graph-based decomposition system (:func:`~pennylane.decomposition.enable_graph`) and setting 
  ``minimize_work_wires=True`` in the :func:`~pennylane.transforms.decompose` transform. The 
  decomposition system will select decomposition rules that minimize the maximum number of 
  simultaneously allocated work wires.
  [(#8729)](https://github.com/PennyLaneAI/pennylane/pull/8729)
  [(#8734)](https://github.com/PennyLaneAI/pennylane/pull/8734)

* A new decomposition rule has been added to :class:`~pennylane.QubitUnitary` which reduces
  the number of CNOTs used to decompose certain two-qubit :class:`~pennylane.QubitUnitary` 
  operations.
  [(#8717)](https://github.com/PennyLaneAI/pennylane/pull/8717)

* Operator decompositions now only need to be defined in the graph decomposition system, as
  ``Operator.decomposition`` will fallback to the first entry in ``qml.list_decomps`` if the 
  ``Operator.compute_decomposition`` method is not overridden.
  [(#8686)](https://github.com/PennyLaneAI/pennylane/pull/8686)

* The `~.BasisRotation` graph decomposition can now scale to larger workflows with ``qjit`` as it 
  has been re-written in a ``qjit`` friendly way using PennyLane control flow.
  [(#8560)](https://github.com/PennyLaneAI/pennylane/pull/8560)
  [(#8608)](https://github.com/PennyLaneAI/pennylane/pull/8608)
  [(#8620)](https://github.com/PennyLaneAI/pennylane/pull/8620)

* The graph-based decompositions system enabled via :func:`~.decomposition.enable_graph` now 
  additionally supports many existing templates.
  [(#8520)](https://github.com/PennyLaneAI/pennylane/pull/8520)
  [(#8515)](https://github.com/PennyLaneAI/pennylane/pull/8515)
  [(#8516)](https://github.com/PennyLaneAI/pennylane/pull/8516)
  [(#8555)](https://github.com/PennyLaneAI/pennylane/pull/8555)
  [(#8558)](https://github.com/PennyLaneAI/pennylane/pull/8558)
  [(#8538)](https://github.com/PennyLaneAI/pennylane/pull/8538)
  [(#8534)](https://github.com/PennyLaneAI/pennylane/pull/8534)
  [(#8582)](https://github.com/PennyLaneAI/pennylane/pull/8582)
  [(#8543)](https://github.com/PennyLaneAI/pennylane/pull/8543)
  [(#8554)](https://github.com/PennyLaneAI/pennylane/pull/8554)
  [(#8616)](https://github.com/PennyLaneAI/pennylane/pull/8616)
  [(#8602)](https://github.com/PennyLaneAI/pennylane/pull/8602)
  [(#8600)](https://github.com/PennyLaneAI/pennylane/pull/8600)
  [(#8601)](https://github.com/PennyLaneAI/pennylane/pull/8601)
  [(#8595)](https://github.com/PennyLaneAI/pennylane/pull/8595)
  [(#8586)](https://github.com/PennyLaneAI/pennylane/pull/8586)
  [(#8614)](https://github.com/PennyLaneAI/pennylane/pull/8614)

  The supported templates with this release include: 

  - :class:`~.QSVT`
  - :class:`~.AmplitudeEmbedding`
  - :class:`~.AllSinglesDoubles`
  - :class:`~.SimplifiedTwoDesign`
  - :class:`~.GateFabric`
  - :class:`~.AngleEmbedding`
  - :class:`~.IQPEmbedding`
  - :class:`~.kUpCCGSD`
  - :class:`~.QAOAEmbedding`
  - :class:`~.BasicEntanglerLayers`
  - :class:`~.HilbertSchmidt`
  - :class:`~.LocalHilbertSchmidt`
  - :class:`~.QuantumMonteCarlo`
  - :class:`~.ArbitraryUnitary`
  - :class:`~.ApproxTimeEvolution`
  - :class:`~.ParticleConservingU2`
  - :class:`~.ParticleConservingU1`
  - :class:`~.CommutingEvolution`

* A new decomposition has been added to :class:`~.Toffoli`. This decomposition uses one work wire 
  and :class:`~.TemporaryAND` operators to reduce the resources needed.
  [(#8549)](https://github.com/PennyLaneAI/pennylane/pull/8549)

* The :func:`~pennylane.pauli_decompose` now supports decomposing scipy's sparse matrices,
  allowing for efficient decomposition of large matrices that cannot fit in memory when written as
  dense arrays.
  [(#8612)](https://github.com/PennyLaneAI/pennylane/pull/8612)

  ```python
  import scipy
  import numpy as np

  arr = np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0]])
  sparse = scipy.sparse.csr_array(arr)
  ```

  ```pycon
  >>> qml.pauli_decompose(sparse)
  1.0 * (I(0) @ X(1) @ X(2))
  ```

* The graph-based decomposition system now supports decomposition rules that contain mid-circuit 
  measurements.
  [(#8079)](https://github.com/PennyLaneAI/pennylane/pull/8079)
  
* A new decomposition has been added to the adjoint of :class:`~.TemporaryAND` that relies on 
  mid-circuit measurments and does not require any ``T`` gates.
  [(#8633)](https://github.com/PennyLaneAI/pennylane/pull/8633)

* The decompositions for several templates have been updated to use
  :class:`~.ops.op_math.ChangeOpBasis`, which makes their decompositions more resource-efficient by 
  eliminating unnecessary controlled operations. The templates include :class:`~.PhaseAdder`,
  :class:`~.TemporaryAND`, :class:`~.QSVT`, and :class:`~.SelectPauliRot`.
  [(#8490)](https://github.com/PennyLaneAI/pennylane/pull/8490)
  [(#8577)](https://github.com/PennyLaneAI/pennylane/pull/8577)
  [(#8721)](https://github.com/PennyLaneAI/pennylane/issues/8721)

* The :func:`~pennylane.clifford_t_decomposition` transform now uses the [Ross-Selinger](https://arxiv.org/abs/1403.2975)
  algorithm (``method="gridsynth"``) as the default method for decomposing single-qubit Pauli 
  rotation gates in the Clifford+T basis. The 
  [Solovay-Kitaev](https://arxiv.org/abs/quant-ph/0505030v2) algorithm (``method="sk"``) was used as 
  default in previous releases.
  [(#8862)](https://github.com/PennyLaneAI/pennylane/pull/8862)

<h4>Other improvements</h4>

* Quantum compilation passes in MLIR and xDSL can now be applied using the core PennyLane transform
  infrastructure, instead of using Catalyst-specific tools. This is made possible by a new argument in
  :func:`~pennylane.transform` and :class:`~.transforms.core.Transform` called ``pass_name``, which 
  accepts a string corresponding to the name of the compilation pass. The ``pass_name`` argument 
  ensures that the given compilation pass will be used when `qjit` is applied to a workflow, where the 
  pass is performed in MLIR or xDSL.
  [(#8539)](https://github.com/PennyLaneAI/pennylane/pull/8539)
  [(#8810)](https://github.com/PennyLaneAI/pennylane/pull/8810)

  ```python
  my_transform = qml.transform(pass_name="cancel-inverses")

  @qml.qjit
  @my_transform
  @qml.qnode(qml.device('lightning.qubit', wires=4))
  def circuit():
      qml.X(0)
      qml.X(0)
      return qml.expval(qml.Z(0))
  ```

  For additional details see the "Transforms with Catalyst" section in :func:`~pennylane.transform`.

* When program capture is enabled, ``qml.adjoint`` and ``qml.ctrl`` can now be called on operators 
  that were constructed ahead of time and used as closure variables.
  [(#8816)](https://github.com/PennyLaneAI/pennylane/pull/8816)

* The constant to convert the length unit Bohr to Angstrom in ``qml.qchem`` has been updated to use 
  scipy constants, leading to more consistent and standardized conversion.
  [(#8537)](https://github.com/PennyLaneAI/pennylane/pull/8537)

* Transform decorator arguments can now be defined without ``@partial``, leading to a simpler 
  interface. 
  [(#8730)](https://github.com/PennyLaneAI/pennylane/pull/8730)
  [(#8754)](https://github.com/PennyLaneAI/pennylane/pull/8754)

  For example, the following two usages are equivalent:

  ```python
  @partial(qml.transforms.decompose, gate_set={qml.RX, qml.CNOT})
  @qml.qnode(qml.device('default.qubit', wires=2))
  def circuit():
      qml.Hadamard(wires=0)
      qml.CZ(wires=[0,1])
      return qml.expval(qml.Z(0))
  ```

  ```python
  @qml.transforms.decompose(gate_set={qml.RX, qml.CNOT})
  @qml.qnode(qml.device('default.qubit', wires=2))
  def circuit():
      qml.Hadamard(wires=0)
      qml.CZ(wires=[0,1])
      return qml.expval(qml.Z(0))
  ```

* :class:`~.transforms.core.TransformContainer` has been renamed to 
  :class:`~.transforms.core.BoundTransform`. The old name is still available in the same location.
  [(#8753)](https://github.com/PennyLaneAI/pennylane/pull/8753)

* More programs can be captured because ``qml.for_loop`` now falls back to a standard Python ``for`` 
  loop if capturing a condensed, structured loop fails with program capture enabled.
  [(#8615)](https://github.com/PennyLaneAI/pennylane/pull/8615)

* ``qml.cond`` will now use standard Python logic if all predicates have concrete values, leading to 
  shorter, more efficient jaxpr programs. Nested control flow primitives will no longer be captured 
  as they are not needed.
  [(#8634)](https://github.com/PennyLaneAI/pennylane/pull/8634)

* Added a keyword argument ``recursive`` to ``qml.transforms.cancel_inverses`` that enables
  recursive cancellation of nested pairs of mutually inverse gates. This allows the transform to 
  cancel larger blocks of inverse gates without having to scan the circuit from scratch. By default, 
  the recursive cancellation is enabled (``recursive=True``). To obtain the previous behaviour, 
  disable it by setting ``recursive=False``.
  [(#8483)](https://github.com/PennyLaneAI/pennylane/pull/8483)

* ``qml.while_loop`` and ``qml.for_loop`` can now lazily dispatch to Catalyst when called, instead 
  of dispatching upon creation.
  [(#8786)](https://github.com/PennyLaneAI/pennylane/pull/8786)

* ``qml.grad`` and ``qml.jacobian`` now lazily dispatch to Catalyst and program capture, allowing 
  for ``qml.qjit(qml.grad(c))`` and ``qml.qjit(qml.jacobian(c))`` to work.
  [(#8382)](https://github.com/PennyLaneAI/pennylane/pull/8382)

* Both the generic and transform-specific application behavior of a 
  ``qml.transforms.core.TransformDispatcher`` can now be overwritten with 
  ``TransformDispatcher.generic_register`` and ``my_transform.register``, leading to easier 
  customization of transforms.
  [(#7797)](https://github.com/PennyLaneAI/pennylane/pull/7797)

* With capture enabled, measurements can now be performed on ``Operator`` instances passed as 
  closure variables from outside the workflow scope. This makes it possible to define observables 
  outside of a QNode and still measure them inside the QNode. 
  [(#8504)](https://github.com/PennyLaneAI/pennylane/pull/8504)

* Wires can now be specified via the ``range`` function with program capture enabled and Autograph 
  activated via ``@qml.qjit(autograph=True)``.
  [(#8500)](https://github.com/PennyLaneAI/pennylane/pull/8500)

* The :func:`~pennylane.transforms.decompose` transform no longer raises an error if both 
  ``gate_set`` and ``stopping_condition`` are provided, or if ``gate_set`` is a dictionary, when the 
  new graph-based decomposition system is disabled.
  [(#8532)](https://github.com/PennyLaneAI/pennylane/pull/8532)

* The :class:`~.pennylane.estimator.templates.SelectTHC` resource operation is upgraded to allow for 
  a trade-off between the number of qubits and T-gates. This provides more flexibility in optimizing 
  algorithms.
  [(#8682)](https://github.com/PennyLaneAI/pennylane/pull/8682)

* Added a custom solver to ``qml.transforms.intermediate_reps.rowcol`` for linear systems over 
  :math:`\mathbb{Z}_2` based on Gauss-Jordan elimination. This removes the need to install the 
  ``galois`` package for this single function and provides a performance improvement.
  [(#8771)](https://github.com/PennyLaneAI/pennylane/pull/8771)

* ``qml.measure`` can now be used as a frontend for ``catalyst.measure``.
  [(#8782)](https://github.com/PennyLaneAI/pennylane/pull/8782)

* ``qml.cond`` will also accept a partial of an operator type as the true function without a false 
  function when capture is enabled.
  [(#8776)](https://github.com/PennyLaneAI/pennylane/pull/8776)

* Solovay-Kitaev decomposition using the :func:`~.clifford_t_decomposition` transform
  with ``method="sk"`` or directly via :func:`~.ops.sk_decomposition` now raises a more
  informative ``RuntimeError`` when used with JAX-JIT or :func:`~.qjit`.
  [(#8489)](https://github.com/PennyLaneAI/pennylane/pull/8489)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* A new transform ``qml.labs.transforms.select_pauli_rot_phase_gradient`` has been added. This 
  transform may reduce the number of ``T`` gates in circuits with :class:`~.SelectPauliRot` 
  rotations by implementing them with a phase gradient resource state and semi-in-place addition 
  (:class:`~.SemiAdder`).
  [(#8738)](https://github.com/PennyLaneAI/pennylane/pull/8738)

  ```python
  import pennylane as qml
  from pennylane.labs.transforms import select_pauli_rot_phase_gradient
  import numpy as np

  @qml.qnode(qml.device("default.qubit"))
  def select_pauli_rot_circ(phis):
      # prepare phase gradient state
      for i, w in enumerate([6,7,8,9]):
          qml.H(w)
          qml.PhaseShift(-np.pi / 2**i, w)

      for wire in [0,1]:
          qml.Hadamard(wire)

      qml.SelectPauliRot(phis, [0,1], 13, rot_axis="X")

      return qml.probs(13)

  phase_grad = select_pauli_rot_phase_gradient(select_pauli_rot_circ,
      angle_wires=[2,3,4,5],
      phase_grad_wires=[6,7,8,9],
      work_wires=[10,11,12],
  )

  phis = [
      (1 / 2 + 1 / 4 + 1 / 8) * 2 * np.pi,
      (1 / 2 + 1 / 4 + 0 / 8) * 2 * np.pi,
      (1 / 2 + 0 / 4 + 1 / 8) * 2 * np.pi,
      (0 / 2 + 1 / 4 + 1 / 8) * 2 * np.pi,
  ]

  clifford_T = qml.clifford_t_decomposition(select_pauli_rot_circ)
  clifford_T_phase_gradient = qml.clifford_t_decomposition(phase_grad)
  ```
  ```pycon
  >>> qml.specs(clifford_T)(phis).resources.gate_types['T']
  16630
  >>> qml.specs(clifford_T_phase_gradient)(phis).resources.gate_types['T']
  3462
  ```

<h3>Breaking changes 💔</h3>

* The ``TransformProgram`` class has been renamed to :class:`~.CompilePipeline`. For
  backward compatibility, the ``TransformProgram`` class can still be accessed from `pennylane.transforms.core`.
  For naming consistency, uses of the term "transform program" have been updated to "compile pipeline"
  across the codebase. Correspondingly, the module `pennylane.transforms.core.transform_program` has
  been renamed to `pennylane.transforms.core.compile_pipeline`, and the old name is no longer available.
  [(#8735)](https://github.com/PennyLaneAI/pennylane/pull/8735)

* The class to dispatch transforms, the ``TransformDispatcher`` class, has been renamed to :class:`~.transforms.core.Transform`
  and is now available as `qml.transform`. For backward compatibility, the ``TransformDispatcher`` class
  can still be accessed from `pennylane.transforms.core`.
  [(#8756)](https://github.com/PennyLaneAI/pennylane/pull/8756)

* The `final_transform` property of the :class:`~.transforms.core.BoundTransform` has been renamed 
  to `is_final_transform` to better follow the naming convention for boolean properties. The `transform` 
  property of the :class:`~.transforms.core.Transform` and :class:`~.transforms.core.BoundTransform` 
  has been renamed to `tape_transform` to avoid ambiguity.
  [(#8756)](https://github.com/PennyLaneAI/pennylane/pull/8756)

* The output format of :func:`~.specs` has been restructured into a dataclass to streamline the outputs.
  Some legacy information has been removed from the output, such as gradient and interface information.
  [(#8713)](https://github.com/PennyLaneAI/pennylane/pull/8713)
  
  Consider the following circuit:

  ```python
  dev = qml.device("default.qubit")
  @qml.qnode(dev)
  def circuit():
    qml.X(0)
    qml.Y(1)
    qml.Z(2)
    return qml.state()
  ```

  The new :func:`~.specs` output format is:

  ```pycon
  >>> qml.specs(circuit)()
  Device: default.qubit
  Device wires: None
  Shots: Shots(total=None)
  Level: gradient

  Resource specifications:
    Total wire allocations: 3
    Total gates: 3
    Circuit depth: 1

    Gate types:
      PauliX: 1
      PauliY: 1
      PauliZ: 1

    Measurements:
      state(all wires): 1
  ```

  Whereas previously, :func:`~.specs` provided:

  ```pycon
  >>> qml.specs(circuit)()
  {'resources': Resources(num_wires=3, num_gates=3, gate_types=defaultdict(<class 'int'>, {'PauliX': 1, 'PauliY': 1, 'PauliZ': 1}), gate_sizes=defaultdict(<class 'int'>, {1: 3}), depth=1, shots=Shots(total_shots=None, shot_vector=())),
  'errors': {},
  'num_observables': 1,
  'num_trainable_params': 0,
  'num_device_wires': 3,
  'num_tape_wires': 3,
  'device_name': 'default.qubit',
  'level': 'gradient',
  'gradient_options': {},
  'interface': 'auto',
  'diff_method': 'best',
  'gradient_fn': 'backprop'}
  ```

* The value ``level=None`` is no longer a valid argument in the following:
  :func:`~.workflow.get_transform_program`, :func:`~.workflow.construct_batch`,
  :func:`~.drawer.draw`, :func:`~.drawer.draw_mpl`, and :func:`~.specs`.
  Please use ``level='device'`` instead to apply all transforms.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* The ``max_work_wires`` argument of the :func:`~pennylane.transforms.decompose` transform has been renamed to ``num_work_wires``. This change is only relevant with graph-based decompositions (enabled via :func:`~.decomposition.enable_graph`).
  [(#8769)](https://github.com/PennyLaneAI/pennylane/pull/8769)

* The ``final_transform`` property of the :class:`~.transforms.core.BoundTransform` has been renamed 
  to ``is_final_transform`` to better follow the naming convention for boolean properties. The ``transform`` 
  property of the :class:`~.transforms.core.Transform` and :class:`~.transforms.core.BoundTransform` 
  has been renamed to ``tape_transform`` to avoid ambiguity.
  [(#8756)](https://github.com/PennyLaneAI/pennylane/pull/8756)

* ``QuantumScript.to_openqasm`` has been removed. Please use :func:`~.to_openqasm` instead. This removes duplicated
  functionality for converting a circuit to OpenQASM code.
  [(#8499)](https://github.com/PennyLaneAI/pennylane/pull/8499)

* Providing ``num_steps`` to :func:`~.evolve`, :func:`~.exp`, :class:`~.ops.op_math.Evolution`,
  and :class:`~.ops.op_math.Exp` has been disallowed. Instead, use :class:`~.TrotterProduct` for approximate
  methods, providing the ``n`` parameter to perform the Suzuki-Trotter product approximation of a Hamiltonian
  with the specified number of Trotter steps.
  [(#8474)](https://github.com/PennyLaneAI/pennylane/pull/8474)

  As a concrete example, consider the following case:

  .. code-block:: python

    coeffs = [0.5, -0.6]
    ops = [qml.X(0), qml.X(0) @ qml.Y(1)]
    H_flat = qml.dot(coeffs, ops)

  Instead of computing the Suzuki-Trotter product approximation as:

  ```pycon
  >>> qml.evolve(H_flat, num_steps=2).decomposition()
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```

  The same result would now be obtained using :class:`~.TrotterProduct` as follows:

  ```pycon
  >>> decomp_ops = qml.adjoint(qml.TrotterProduct(H_flat, time=1.0, n=2)).decomposition()
  >>> [simp_op for op in decomp_ops for simp_op in map(qml.simplify, op.decomposition())]
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```

* Access to ``add_noise``, ``insert`` and noise mitigation transforms from the :mod:`pennylane.transforms` module has been removed.
  Instead, these functions should be imported from the :mod:`~.noise` module.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* ``qml.qnn.cost.SquaredErrorLoss`` has been removed. Instead, this hybrid workflow can be accomplished
  with a function such as ``loss = lambda *args: (circuit(*args) - target)**2``.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* Some unnecessary methods of the :class:`~.CircuitGraph` class have been removed:
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

  - ``print_contents`` was removed in favor of ``print(obj)``
  - ``observables_in_order`` was removed in favor of ``observables``
  - ``operations_in_order`` was removed in favor of ``operations``
  - ``ancestors_in_order(obj)`` was removed in favor of ``ancestors(obj, sort=True)``
  - ``descendants_in_order(obj)`` was removed in favor of ``descendants(obj, sort=True)``

* ``pennylane.devices.DefaultExecutionConfig`` has been removed. Instead, use
  :class:`~.devices.ExecutionConfig()` to create a default execution configuration.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* Specifying the ``work_wire_type`` argument in :func:`~.ctrl` and other controlled operators as ``"clean"`` or
  ``"dirty"`` is disallowed. Use ``"zeroed"`` to indicate that the work wires are initially in the :math:`|0\rangle`
  state, and ``"borrowed"`` to indicate that the work wires can be in any arbitrary state. In both cases, the
  work wires are assumed to be restored to their original state upon completing the decomposition.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* `QuantumScript.shape` and `QuantumScript.numeric_type` are removed.
  The corresponding :class:`~.measurements.MeasurementProcess`
  attributes and methods should be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* ``MeasurementProcess.expand`` has been removed.
``qml.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])``
should be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* The `qml.QNode.add_transform` method is removed.
  Instead, please use `QNode.transform_program.push_back(transform_container=transform_container)`.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* The :func:`~.dynamic_one_shot` transform can no longer be applied directly on a QNode. Instead, specify the mid-circuit measurement method in QNode: ``@qml.qnode(..., mcm_method="one-shot")``.
  [(8781)](https://github.com/PennyLaneAI/pennylane/pull/8781)

* The `qml.compiler.python_compiler` submodule has been removed from PennyLane.
  It has been migrated to Catalyst, available as `catalyst.python_interface`.
  [(#8662)](https://github.com/PennyLaneAI/pennylane/pull/8662)

* `qml.transforms.map_wires` no longer supports transforming jaxpr directly.
  [(#8683)](https://github.com/PennyLaneAI/pennylane/pull/8683)

* `qml.cond`, the `QNode`, transforms, `qml.grad`, and `qml.jacobian` no longer treat all keyword arguments as static
  arguments. They are instead treated as dynamic, numerical inputs, matching the behaviour of JAX and Catalyst.
  [(#8290)](https://github.com/PennyLaneAI/pennylane/pull/8290)

<h3>Deprecations 👋</h3>

* Maintenance support of NumPy<2.0 is deprecated as of v0.44 and will be completely dropped in v0.45.
  Future versions of PennyLane will only work with NumPy>=2.0.
  We recommend upgrading your version of NumPy to benefit from enhanced support and features.
  [(#8578)](https://github.com/PennyLaneAI/pennylane/pull/8578)
  [(#8497)](https://github.com/PennyLaneAI/pennylane/pull/8497)

* Passing a function to the ``gate_set`` argument in the :func:`~pennylane.transforms.decompose` transform
  is deprecated. The ``gate_set`` argument expects a static iterable of operator type and/or operator names,
  and the function should be passed to the ``stopping_condition`` argument instead.
  [(#8533)](https://github.com/PennyLaneAI/pennylane/pull/8533)

  The example below illustrates how you can provide a function as the ``stopping_condition`` in addition to providing a
  ``gate_set``. The decomposition of each operator will halt upon reaching the gates in the ``gate_set`` or when the
  ``stopping_condition`` is satisfied.

  ```python
  import pennylane as qml

  @qml.transforms.decompose(gate_set={"H", "T", "CNOT"}, stopping_condition=lambda op: len(op.wires) <= 2)
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.Hadamard(wires=[0])
      qml.Toffoli(wires=[0,1,2])
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──H────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
  1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
  2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤
  ```

* Access to the follow functions and classes from the :mod:`~.resources` module are deprecated.
  Instead, these functions must be imported from the :mod:`~.estimator` module.
  [(#8484)](https://github.com/PennyLaneAI/pennylane/pull/8484)

  - ``qml.estimator.estimate_shots`` rather than ``qml.resources.estimate_shots``
  - ``qml.estimator.estimate_error`` rather than ``qml.resources.estimate_error``
  - ``qml.estimator.FirstQuantization`` rather than ``qml.resources.FirstQuantization``
  - ``qml.estimator.DoubleFactorization`` rather than ``qml.resources.DoubleFactorization``

* The ``argnum`` parameter has been renamed to ``argnums`` for
  :class:`~.grad`, :class:`~.jacobian`, :func:`~.gradients.jvp` and :func:`~.gradients.vjp` to better adhere to conventions in JAX and Catalyst.
  [(#8496)](https://github.com/PennyLaneAI/pennylane/pull/8496)
  [(#8481)](https://github.com/PennyLaneAI/pennylane/pull/8481)

* The ``custom_decomps`` keyword argument to ``qml.device`` has been deprecated and will be removed
  in 0.45. Instead, with :func:`~.decomposition.enable_graph`, new decomposition rules can be defined as
  quantum functions with registered resources. See :mod:`pennylane.decomposition` for more details.

* `qml.measure`, `qml.measurements.MidMeasureMP`, `qml.measurements.MeasurementValue`,
  and `qml.measurements.get_mcm_predicates` are now located in `qml.ops.mid_measure`.
  `MidMeasureMP` has been renamed to `MidMeasure`.
  `qml.measurements.find_post_processed_mcms` is now `qml.devices.qubit.simulate._find_post_processed_mcms`,
  and is being made private, as it is an utility for tree-traversal mid-circuit measurements.
  [(#8466)](https://github.com/PennyLaneAI/pennylane/pull/8466)

* The ``pennylane.operation.Operator.is_hermitian`` property has been deprecated and renamed
  to ``pennylane.operation.Operator.is_verified_hermitian`` as it better reflects the functionality of this property.
  Access through ``pennylane.operation.Operator.is_hermitian`` is deprecated and will be removed in v0.45.
  Alternatively, consider using the :func:`~.is_hermitian` function instead for a thorough verification of hermiticity,
  at a higher computational cost.
  [(#8494)](https://github.com/PennyLaneAI/pennylane/pull/8494)

* The :func:`pennylane.devices.preprocess.mid_circuit_measurements` transform is deprecated. Instead,
  the device should determine which MCM method to use,
  and explicitly include relevant preprocessing transforms if necessary.
  [(#8467)](https://github.com/PennyLaneAI/pennylane/pull/8467)

<h3>Internal changes ⚙️</h3>

* The `_grad.py` file has been split into multiple files within
  a folder for improved source code organization.
  [(#8800)](https://github.com/PennyLaneAI/pennylane/pull/8800)

* The `pyproject.toml` has been updated with project dependencies to replace the requirements files. Workflows have also been updated to use installations from `pyproject.toml`.
  [(8702)](https://github.com/PennyLaneAI/pennylane/pull/8702)

* Some error handling has been updated in tests, to adjust to Python 3.14; ``get_type_str`` added a special branch to handle ``Union``.
  The import of ``networkx`` is softened to not occur on import of PennyLane to work around a bug in Python 3.14.1.
  [(#8568)](https://github.com/PennyLaneAI/pennylane/pull/8568)
  [(#8737)](https://github.com/PennyLaneAI/pennylane/pull/8737)

* The ``jax`` version has been updated to ``0.7.1`` for the ``capture`` module.
  [(#8715)](https://github.com/PennyLaneAI/pennylane/pull/8715)
  [(#8701)](https://github.com/PennyLaneAI/pennylane/pull/8701)

* Improved error handling when using PennyLane's experimental program capture functionality with an incompatible JAX version.
  [(#8723)](https://github.com/PennyLaneAI/pennylane/pull/8723)

* The ``autoray`` package version has been updated to ``0.8.2``.
  [(#8674)](https://github.com/PennyLaneAI/pennylane/pull/8674)

* Updated the schedule of nightly TestPyPI uploads to occur at the end of all weekdays rather than the beginning of all weekdays.
  [(#8672)](https://github.com/PennyLaneAI/pennylane/pull/8672)

* A github workflow was added to bump Catalyst and Lightning versions in the release candidate (RC) branch, create a new release tag and draft release,
  tag the RC branch, and create a PR to merge the RC branch into master.
  [(#8352)](https://github.com/PennyLaneAI/pennylane/pull/8352)

* Added ``MCM_METHOD`` and ``POSTSELECT_MODE`` ``StrEnum`` objects to improve validation and handling of ``MCMConfig`` creation.
  [(#8596)](https://github.com/PennyLaneAI/pennylane/pull/8596)

* In program capture, transforms now have a single transform primitive with a ``transform`` param that stores
  the ``Transform``. Before, each transform had its own primitive stored on the
  ``Transform._primitive`` private property.
  [(#8576)](https://github.com/PennyLaneAI/pennylane/pull/8576)
  [(#8639)](https://github.com/PennyLaneAI/pennylane/pull/8639)

* Updated documentation check workflow to run on pull requests on `v[0-9]+\.[0-9]+\.[0-9]+-docs` branches.
  [(#8590)](https://github.com/PennyLaneAI/pennylane/pull/8590)

* When program capture is enabled, there is no longer caching of the jaxpr on the QNode.
  [(#8629)](https://github.com/PennyLaneAI/pennylane/pull/8629)

* The `grad` and `jacobian` primitives now store the function under `fn`. There is also now a single `jacobian_p`
  primitive for use in program capture.
  [(#8357)](https://github.com/PennyLaneAI/pennylane/pull/8357)

* The versions for ``pylint``, ``isort`` and ``black`` in ``format.yml`` have been updated.
  [(#8506)](https://github.com/PennyLaneAI/pennylane/pull/8506)

* Reclassified ``registers`` as a tertiary module for use with ``tach``.
  [(#8513)](https://github.com/PennyLaneAI/pennylane/pull/8513)

* The :class:`~pennylane.devices.LegacyDeviceFacade` was refactored to implement ``setup_execution_config`` and ``preprocess_transforms``
  separately as opposed to implementing a single ``preprocess`` method. Additionally, the ``mid_circuit_measurements`` transform has been removed
  from the preprocess transform program. Instead, the best mcm method is chosen in ``setup_execution_config``. By default, the ``_capabilities``
  dictionary is queried for the ``"supports_mid_measure"`` property. If the underlying device defines a TOML file, the ``supported_mcm_methods``
  field in the TOML file is used as the source of truth.
  [(#8469)](https://github.com/PennyLaneAI/pennylane/pull/8469)
  [(#8486)](https://github.com/PennyLaneAI/pennylane/pull/8486)
  [(#8495)](https://github.com/PennyLaneAI/pennylane/pull/8495)

* The various private functions of the :class:`~pennylane.estimator.FirstQuantization` class have
  been modified to avoid using `numpy.matrix` as this function is deprecated.
  [(#8523)](https://github.com/PennyLaneAI/pennylane/pull/8523)

* The `ftqc` module now includes dummy transforms for several Catalyst/MLIR passes (`to-ppr`, `commute-ppr`, `merge-ppr-ppm`,
  `decompose-clifford-ppr`, `decompose-non-clifford-ppr`, `ppr-to-ppm`, `ppr-to-mbqc` and `reduce-t-depth`), to allow them to
  be captured as primitives in PLxPR and mapped to the MLIR passes in Catalyst. This enables using the passes with the unified
  compiler and program capture.
  [(#8519)](https://github.com/PennyLaneAI/pennylane/pull/8519)
  [(#8544)](https://github.com/PennyLaneAI/pennylane/pull/8544)

* Added a ``skip_decomp_matrix_check`` argument to :func:`pennylane.ops.functions.assert_valid` that
  allows the test to skip the matrix check part of testing a decomposition rule but still verify
  that the resource function is correct.
  [(#8687)](https://github.com/PennyLaneAI/pennylane/pull/8687)

* Simplified the decomposition pipeline for the ``estimator`` module. ``estimator.estimate()``
  was updated to call the base class's `symbolic_resource_decomp` method directly.
  [(#8641)](https://github.com/PennyLaneAI/pennylane/pull/8641)
  
* Disabled autograph for the ``PauliRot`` decomposition rule, as it should not be used. 
  [(#8765)](https://github.com/PennyLaneAI/pennylane/pull/8765)

<h3>Documentation 📝</h3>

* The code example in the documentation for ``qml.decomposition.register_resources`` has been
  updated to adhere to renamed keyword arguments and default behaviour of ``num_work_wires``.
  [(#8550)](https://github.com/PennyLaneAI/pennylane/pull/8550)

* A note clarifying that the factors of a ``~.ChangeOpBasis`` are iterated in reverse order has been
  added to the documentation of ``~.ChangeOpBasis``.
  [(#8757)](https://github.com/PennyLaneAI/pennylane/pull/8757)

* The documentation of ``qml.transforms.rz_phase_gradient`` has been updated with respect to the
  sign convention of phase gradient states, how it prepares the phase gradient state in the code
  example, and the verification of the code example result.
  [(#8536)](https://github.com/PennyLaneAI/pennylane/pull/8536)

* The docstring for ``qml.device`` has been updated to include a section on custom decompositions,
  and a warning about the removal of the ``custom_decomps`` kwarg in v0.45. Additionally, the
  :doc:`Building a plugin <../development/plugins>` page now includes instructions on using
  the :func:`~pennylane.devices.preprocess.decompose` transform for device-level decompositions.
  The documentation for :doc:`Compiling circuits <../introduction/compiling_circuits>` has also been
  updated with a warning message about ``custom_decomps`` future removal.
  [(#8492)](https://github.com/PennyLaneAI/pennylane/pull/8492)
  [(#8564)](https://github.com/PennyLaneAI/pennylane/pull/8564)


* The documentation for :class:`~.GeneralizedAmplitudeDamping` has been updated to match
  the standard convention in literature for the definition of the Kraus matrices.
  [(#8707)](https://github.com/PennyLaneAI/pennylane/pull/8707)

* Improved documentation in the :mod:`pennylane.transforms` module and added documentation testing.
  [(#8557)](https://github.com/PennyLaneAI/pennylane/pull/8557)

* Updated various docstring examples in the :mod:`~.fourier` module
  to be compatible with the new documentation testing approach.
  [(#8635)](https://github.com/PennyLaneAI/pennylane/pull/8635)
  
* The :mod:`estimator <pennylane.estimator>` module documentation has been revised for clarity.
  [(#8827)](https://github.com/PennyLaneAI/pennylane/pull/8827)
  [(#8829)](https://github.com/PennyLaneAI/pennylane/pull/8829)
  [(#8830)](https://github.com/PennyLaneAI/pennylane/pull/8830)
  [(#8832)](https://github.com/PennyLaneAI/pennylane/pull/8832)

<h3>Bug fixes 🐛</h3>

* Fixed the difference between the output dimensions of the dynamic one-shot and single-branch-statistics mid-circuit
  measurement methods.
  [(#8856)](https://github.com/PennyLaneAI/pennylane/pull/8856)

* Fixed a bug in ``qml.estimator.QubitizeTHC`` where specified arguments for Prepare and Select resource
  operators were being ignored in favor of default ones.
  [(#8858)] (https://github.com/PennyLaneAI/pennylane/pull/8858)

* Fixed a bug in ``torch.vmap`` that produced an error when it was used with native parameter broadcasting and ``qml.RZ``
  [(#8760)](https://github.com/PennyLaneAI/pennylane/pull/8760)

* Fixed various incorrect decomposition rules.
  [(#8812)](https://github.com/PennyLaneAI/pennylane/pull/8812)

* Fixed a bug where ``_double_factorization_compressed`` of ``pennylane/qchem/factorization.py`` used to use ``X``
  for ``Z`` parameter initialization.
  [(#8689)](https://github.com/PennyLaneAI/pennylane/pull/8689)

* Fixed some numerical stability issues of ``_apply_uniform_rotation_dagger``
  by using a fixed floating-point number tolerance from ``np.finfo``.
  [(#8780)](https://github.com/PennyLaneAI/pennylane/pull/8780)

* Fixed handling of floating-point errors in the norm of the state when applying
  mid-circuit measurements.
  [(#8741)](https://github.com/PennyLaneAI/pennylane/pull/8741)

* Updated ``interface-unit-tests.yml`` to use the input parameter ``pytest_additional_args`` when running pytest.
  [(#8705)](https://github.com/PennyLaneAI/pennylane/pull/8705)

* Fixed a bug in ``resolve_work_wire_type`` which incorrectly returned a value of ``zeroed`` if ``both work_wires``
  and ``base_work_wires`` were empty, causing an incorrect work wire type.
  [(#8718)](https://github.com/PennyLaneAI/pennylane/pull/8718)

* Fixed the warnings-as-errors CI action which was failing
  due to an incompatibility between ``pytest-xdist`` and ``pytest-benchmark``.
  Disabling the benchmark package allows the tests to be collected and executed.
  [(#8699)](https://github.com/PennyLaneAI/pennylane/pull/8699)

* Added a missing `expand_transform` to `param_shift_hessian` to pre-decompose
  operations until they are supported.
  [(#8698)](https://github.com/PennyLaneAI/pennylane/pull/8698)

* Fixed a bug in the ``default.mixed`` device where certain diagonal operations were incorrectly
  reshaped during application when using parameter broadcasting.
  [(#8593)](https://github.com/PennyLaneAI/pennylane/pull/8593)

* If :class:`.allocation.Allocate` or :class:`.allocation.Deallocate` instructions are encountered with graph-based decompositions enabled,
  they are now ignored instead of raising a warning.
  [(#8553)](https://github.com/PennyLaneAI/pennylane/pull/8553)

* Fixed a bug in ``clifford_t_decomposition`` with ``method="gridsynth"`` and ``qjit``,
  where using a cached decomposition with the same parameter caused an error.
  [(#8535)](https://github.com/PennyLaneAI/pennylane/pull/8535)

* Fixed a bug in :class:`~.SemiAdder` where the results were incorrect when more ``work_wires`` than required were passed.
 [(#8423)](https://github.com/PennyLaneAI/pennylane/pull/8423)

* Fixed a bug where the deferred-measurement method was used silently even if ``mcm_method="one-shot"`` was explicitly requested,
  when a device that extends the ``LegacyDevice`` does not declare support for mid-circuit measurements.
  [(#8486)](https://github.com/PennyLaneAI/pennylane/pull/8486)

* Fixed a bug where a ``KeyError`` was raised when querying the decomposition rule for an operator
  in the gate set from a :class:`~pennylane.decomposition.DecompGraphSolution`.
  [(#8526)](https://github.com/PennyLaneAI/pennylane/pull/8526)

* Fixed a bug where mid-circuit measurements were generating incomplete QASM.
  [(#8556)](https://github.com/PennyLaneAI/pennylane/pull/8556)

* Fixed a bug where :func:`~.specs` incorrectly computed the circuit depth in the presence of classically controlled operators.
  [(#8668)](https://github.com/PennyLaneAI/pennylane/pull/8668)

* Fixed a bug where an error was raised when trying to decompose a nested composite operator with capture and the new graph system enabled.
  [(#8695)](https://github.com/PennyLaneAI/pennylane/pull/8695)

* Fixed a bug where the :func:`~.change_op_basis` function could not be captured when the ``uncompute_op`` argument is left out.
  [(#8695)](https://github.com/PennyLaneAI/pennylane/pull/8695)

* Fixed a bug in the :func:`~.ops.rs_decomposition` function where valid solution candidates were being rejected.
  [(#8625)](https://github.com/PennyLaneAI/pennylane/pull/8625)

* Fixed a bug where decomposition rules were sometimes incorrectly disregarded by the ``DecompositionGraph`` when a higher level
  decomposition rule uses dynamically allocated work wires via ``qml.allocate``.
  [(#8725)](https://github.com/PennyLaneAI/pennylane/pull/8725)

* Fixed a bug where :class:`~.ops.op_math.ChangeOpBasis` was not correctly reconstructed using ``qml.pytrees.unflatten(*qml.pytrees.flatten(op))``.
  [(#8721)](https://github.com/PennyLaneAI/pennylane/issues/8721)

* Fixed a bug where ``qml.estimator.SelectTHC``, ``qml.estimator.QubitizeTHC``, and ``qml.estimator.PrepTHC`` were not accounting for auxiliary
  wires correctly.
  [(#8719)](https://github.com/PennyLaneAI/pennylane/pull/8719)

* Fixed a bug where the associated ``expand_transform`` does not stay with the original :class:`~.transforms.core.Transform` in a :class:`~.CompilePipeline`
  during manipulations of the :class:`~.CompilePipeline`.
  [(#8774)](https://github.com/PennyLaneAI/pennylane/pull/8774)

* Fixed a bug where an error was raised when ``to_openqasm`` is used with ``qml.decomposition.enable_graph()``.
  [(#8809)](https://github.com/PennyLaneAI/pennylane/pull/8809)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Runor Agbaire,
Guillermo Alonso,
Utkarsh Azad,
Joseph Bowles,
Astral Cai,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Marcus Edwards,
Lillian Frederiksen,
Diego Guala,
Sengthai Heng,
Austin Huang,
Soran Jahangiri,
Jeffrey Kam,
Jacob Kitchen,
Christina Lee,
Joseph Lee,
Anton Naim Ibrahim,
Lee J. O'Riordan,
Mudit Pandey,
Gabriela Sanchez Diaz,
Shuli Shu,
Jay Soni,
Nate Stemen,
Theodoros Trochatos,
David Wierichs,
Shifan Xu,
Hongsheng Zheng,
Zinan Zhou.
