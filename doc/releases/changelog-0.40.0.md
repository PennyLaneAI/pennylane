
# Release 0.40.0

<h3>New features since last release</h3>

<h4>Efficient state preparation methods 🦾</h4>

* State preparation tailored for matrix product states (MPS) is now supported with 
  :class:`qml.MPSPrep <pennylane.MPSPrep>` on the `lightning.tensor` device.
  [(#6431)](https://github.com/PennyLaneAI/pennylane/pull/6431)

  Given a list of :math:`n` tensors that represents an MPS, :math:`[A^{(0)}, ..., A^{(n-1)}]`, 
  :class:`qml.MPSPrep <pennylane.MPSPrep>` lets you directly inject the MPS into a QNode as the initial 
  state of the circuit without any need for pre-processing. The first and last tensors in the list 
  must be rank-2, while all intermediate tensors should be rank-3.

  ```python
  import pennylane as qp
  import numpy as np

  mps = [
      np.array([[0.0, 0.107], [0.994, 0.0]]),
      np.array(
          [
              [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
              [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
          ]
      ),
      np.array(
          [
              [[-1.0, 0.0], [0.0, 0.0]],
              [[0.0, 0.0], [0.0, 1.0]],
              [[0.0, -1.0], [0.0, 0.0]],
              [[0.0, 0.0], [1.0, 0.0]],
          ]
      ),
      np.array([[-1.0, -0.0], [-0.0, -1.0]]),
  ]

  dev = qml.device("lightning.tensor", wires = [0, 1, 2, 3])
  @qml.qnode(dev)
  def circuit():
      qml.MPSPrep(mps, wires = [0,1,2])
      return qml.state()
  ```

  ```pycon
  >>> print(circuit())
  [ 0.    +0.j  0.    +0.j  0.    +0.j -0.1066+0.j  0.    +0.j  0.    +0.j
    0.    +0.j  0.    +0.j  0.    +0.j  0.    +0.j  0.    +0.j  0.    +0.j
    0.9943+0.j  0.    +0.j  0.    +0.j  0.    +0.j]
  ```

  At this time, :class:`qml.MPSPrep <pennylane.MPSPrep>` is only supported on the `lightning.tensor` device.

* Custom-made state preparation for linear combinations of quantum states is now available with 
  :class:`qml.Superposition <pennylane.Superposition>`.
  [(#6670)](https://github.com/PennyLaneAI/pennylane/pull/6670)

  Given a list of :math:`m` coefficients :math:`c_i` and basic states :math:`|b_i\rangle`, 
  :class:`qml.Superposition <pennylane.Superposition>` prepares 
  :math:`|\phi\rangle = \sum_i^m c_i |b_i\rangle`. Here is a simple example showing how to use 
  :class:`qml.Superposition <pennylane.Superposition>` to prepare 
  :math:`\tfrac{1}{\sqrt{2}} |00\rangle + \tfrac{1}{\sqrt{2}} |10\rangle`.

  ```python
  coeffs = np.array([0.70710678, 0.70710678])
  basis =  np.array([[0, 0], [1, 0]])

  @qml.qnode(qml.device('default.qubit'))
  def circuit():
      qml.Superposition(coeffs, basis, wires=[0, 1], work_wire=[2])
      return qml.state()
  ```

  ```
  >>> circuit()
  Array([0.7071068 +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
         0.70710677+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j],      dtype=complex64)
  ```

  Note that specification of one `work_wire` is required.

<h4>Enhanced QSVT functionality 🤩</h4>

* New functionality to calculate and convert phase angles for QSP and QSVT has been added
  with :func:`qml.poly_to_angles <pennylane.poly_to_angles>` and 
  :func:`qml.transform_angles <pennylane.transform_angles>`.
  [(#6483)](https://github.com/PennyLaneAI/pennylane/pull/6483)

  The :func:`qml.poly_to_angles <pennylane.poly_to_angles>` function calculates phase angles 
  directly given polynomial coefficients and the routine in which the angles will be used (`"QSVT"`, 
  `"QSP"`, or `"GQSP"`):
  ```pycon
  >>> poly = [0, 1.0, 0, -1/2, 0, 1/3]
  >>> qsvt_angles = qml.poly_to_angles(poly, "QSVT")
  >>> print(qsvt_angles)
  [-5.49778714  1.57079633  1.57079633  0.5833829   1.61095884  0.74753829]
  ```

  The :func:`qml.transform_angles <pennylane.transform_angles>` function can be used to convert 
  angles from one routine to another:
  ```pycon
  >>> qsp_angles = np.array([0.2, 0.3, 0.5])
  >>> qsvt_angles = qml.transform_angles(qsp_angles, "QSP", "QSVT")
  >>> print(qsvt_angles)
  [-6.86858347  1.87079633 -0.28539816]
  ```

* The :func:`qml.qsvt <pennylane.qsvt>` function has been improved to be more user-friendly,
with enhanced capabilities.
  [(#6520)](https://github.com/PennyLaneAI/pennylane/pull/6520)
  [(#6693)](https://github.com/PennyLaneAI/pennylane/pull/6693)

  Block encoding and phase angle computation are now handled automatically,
  given a matrix to encode, polynomial coefficients, and a block encoding method
  (`"prepselprep"`, `"qubitization"`, `"embedding"`, or `"fable"`, all implemented with their 
  corresponding operators in PennyLane).
  ```python
  # P(x) = -x + 0.5 x^3 + 0.5 x^5
  poly = np.array([0, -1, 0, 0.5, 0, 0.5])
  hamiltonian = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1) @ qml.Z(2)])
  
  dev = qml.device("default.qubit")
  @qml.qnode(dev)
  def circuit():
      qml.qsvt(hamiltonian, poly, encoding_wires=[0], block_encoding="prepselprep")
      return qml.state()
  
  matrix = qml.matrix(circuit, wire_order=[0, 1, 2])()
  ```
  ```pycon
  >>> print(matrix[:4, :4].real)
  [[-0.1625  0.     -0.3793  0.    ]
   [ 0.     -0.1625  0.      0.3793]
   [-0.3793  0.      0.1625  0.    ]
   [ 0.      0.3793  0.      0.1625]]
  ```
  The old functionality can still be accessed with :func:`qml.qsvt_legacy <pennylane.qsvt_legacy>`.

* A new :class:`qml.GQSP <pennylane.GQSP>` template has been added to perform Generalized Quantum Signal Processing (GQSP).
  [(#6565)](https://github.com/PennyLaneAI/pennylane/pull/6565)
  Similar to QSVT, GQSP is an algorithm that polynomially transforms an input unitary operator,
  but with fewer restrictions on the chosen polynomial.

  You can also use :func:`qml.poly_to_angles <pennylane.poly_to_angles>` to obtain angles for GQSP!

  ```python
  # P(x) = 0.1 + 0.2j x + 0.3 x^2
  poly = [0.1, 0.2j, 0.3]
  angles = qml.poly_to_angles(poly, "GQSP")
  
  @qml.prod # transforms the qfunc into an Operator
  def unitary(wires):
      qml.RX(0.3, wires)
  
  dev = qml.device("default.qubit")
  @qml.qnode(dev)
  def circuit(angles):
      qml.GQSP(unitary(wires = 1), angles, control = 0)
      return qml.state()
  
  matrix = qml.matrix(circuit, wire_order=[0, 1])(angles)
  ```
  ```pycon
  >>> print(np.round(matrix,3)[:2, :2])
  [[0.387+0.198j 0.03 -0.089j]
  [0.03 -0.089j 0.387+0.198j]]
  ```

<h4>Generalized Trotter products 🐖</h4>

* Trotter products that work on exponentiated operators directly instead of full system hamiltonians 
  can now be encoded into circuits with the addition of 
  :class:`qml.TrotterizedQfunc <pennylane.TrotterizedQfunc>` and :func:`qml.trotterize <pennylane.trotterize>`. This
  allows for custom specification of the first-order expansion of the Suzuki-Trotter product formula
  and extrapolating it to the :math:`n^{\text{th}}` order.
  [(#6627)](https://github.com/PennyLaneAI/pennylane/pull/6627)

  If the first-order of the Suzuki-Trotter product formula for a given problem is known, 
  :class:`qml.TrotterizedQfunc <pennylane.TrotterizedQfunc>` and :func:`qml.trotterize <pennylane.trotterize>`
  let you implement the :math:`n^{\text{th}}`-order product formula while only specifying the 
  first-order term as a quantum function.

  ```python
  def my_custom_first_order_expansion(time, theta, phi, wires, flip):
    qml.RX(time * theta, wires[0])
    qml.RY(time * phi, wires[1])
    if flip:
        qml.CNOT(wires=wires[:2])
  ```

  :func:`qml.trotterize <pennylane.trotterize>` requires the quantum function representing the first-order
  product formula, the number of Trotter steps, and the desired order. It returns a function with 
  the same call signature as the first-order product formula quantum function:

  ```python
  @qml.qnode(qml.device("default.qubit"))
  def my_circuit(time, theta, phi, num_trotter_steps):
      qml.trotterize(
          first_order_expansion,
          n=num_trotter_steps,
          order=2,
      )(time, theta, phi, wires=['a', 'b'], flip=True)
      return qml.state()
  ```
  
  Alternatively, :class:`qml.TrotterizedQfunc <pennylane.TrotterizedQfunc>` can be used as follows:

  ```python
  @qml.qnode(qml.device("default.qubit"))
  def my_circuit(time, theta, phi, num_trotter_steps):
      qml.TrotterizedQfunc(
          time,
          theta,
          phi,
          qfunc=my_custom_first_order_expansion,
          n=num_trotter_steps,
          order=2,
          wires=['a', 'b'],
          flip=True,
      )
      return qml.state()
  ```

  ```pycon
  >>> time = 0.1
  >>> theta, phi = (0.12, -3.45)
  >>> print(qml.draw(my_circuit, level="device")(time, theta, phi, num_trotter_steps=1))
  a: ──RX(0.01)──╭●─╭●──RX(0.01)──┤  State
  b: ──RY(-0.17)─╰X─╰X──RY(-0.17)─┤  State
  ```

  Both methods produce the same results, but offer different interfaces based on the application or overall
  preference.

<h4>Bosonic operators 🎈</h4>

A new module, :mod:`qml.bose <pennylane.bose>`, has been added to PennyLane that includes support 
for constructing and manipulating Bosonic operators and converting between Bosonic operators and 
qubit operators.

* Bosonic operators analogous to `qml.FermiWord` and `qml.FermiSentence` are now available with 
  :class:`qml.BoseWord <pennylane.BoseWord>` and :class:`qml.BoseSentence <pennylane.BoseSentence>`.
  [(#6518)](https://github.com/PennyLaneAI/pennylane/pull/6518)

  :class:`qml.BoseWord <pennylane.BoseWord>` and :class:`qml.BoseSentence <pennylane.BoseSentence>` 
  operate similarly to their fermionic counterparts. To create a Bose word, a dictionary 
  is required as input, where the keys are tuples of boson indices and values are `'+/-'` (denoting 
  the bosonic creation/annihilation operators). For example, the :math:`b^{\dagger}_0 b_1` can be 
  constructed as follows.

  ```pycon
  >>> w = qml.BoseWord({(0, 0) : '+', (1, 1) : '-'})
  >>> print(w)
  b⁺(0) b(1)
  ```

  Multiple Bose words can then be combined to form a Bose sentence:

  ```pycon
  >>> w1 = qml.BoseWord({(0, 0) : '+', (1, 1) : '-'})
  >>> w2 = qml.BoseWord({(0, 1) : '+', (1, 2) : '-'})
  >>> s = qml.BoseSentence({w1 : 1.2, w2: 3.1})
  >>> print(s)
  1.2 * b⁺(0) b(1)
  + 3.1 * b⁺(1) b(2)
  ```

* Functionality for converting bosonic operators to qubit operators is available with 
  :func:`qml.unary_mapping <pennylane.unary_mapping>`, :func:`qml.binary_mapping <pennylane.binary_mapping>`, 
  and :func:`qml.christiansen_mapping <pennylane.christiansen_mapping>`.
  [(#6623)](https://github.com/PennyLaneAI/pennylane/pull/6623)
  [(#6576)](https://github.com/PennyLaneAI/pennylane/pull/6576)
  [(#6564)](https://github.com/PennyLaneAI/pennylane/pull/6564)

  All three mappings follow the same syntax, where a :class:`qml.BoseWord <pennylane.BoseWord>` or 
  :class:`qml.BoseSentence <pennylane.BoseSentence>` is required as input.

  ```python
  >>> w = qml.BoseWord({(0, 0): "+"})
  >>> qml.binary_mapping(w, n_states=4)
  0.6830127018922193 * X(0)
  + -0.1830127018922193 * X(0) @ Z(1)
  + -0.6830127018922193j * Y(0)
  + 0.1830127018922193j * Y(0) @ Z(1)
  + 0.3535533905932738 * X(0) @ X(1)
  + -0.3535533905932738j * X(0) @ Y(1)
  + 0.3535533905932738j * Y(0) @ X(1)
  + (0.3535533905932738+0j) * Y(0) @ Y(1)
  ```

  Additional fine-tuning is available within each function, such as the maximum number of allowed
  bosonic states and a tolerance for discarding imaginary parts of the coefficients.

<h4>Construct vibrational Hamiltonians 🔨</h4>

* Several new features are available in the :mod:`qml.qchem <pennylane.qchem>` module to help with 
  the construction of vibrational Hamiltonians. This includes:

  * The :class:`~.qchem.VibrationalPES` class to store potential energy surface information. 
    [(#6652)](https://github.com/PennyLaneAI/pennylane/pull/6652)

    ```python
    pes_onemode = np.array([[0.309, 0.115, 0.038, 0.008, 0.000, 0.006, 0.020, 0.041, 0.070]])
    pes_twomode = np.zeros((1, 1, 9, 9))
    dipole_onemode = np.zeros((1, 9, 3))
    gauss_weights = np.array([3.96e-05, 4.94e-03, 8.85e-02,
                                    4.33e-01, 7.20e-01, 4.33e-01,
                                    8.85e-02, 4.94e-03, 3.96e-05])
    grid = np.array([-3.19, -2.27, -1.47, -0.72,  0.0,  0.72,  1.47,  2.27,  3.19])
    pes_object = qml.qchem.VibrationalPES(
            freqs=np.array([0.025]),
            grid=grid,
            uloc=np.array([[1.0]]),
            gauss_weights=gauss_weights,
            pes_data=[pes_onemode, pes_twomode],
            dipole_data=[dipole_onemode],
            localized=False,
            dipole_level=1,
        )
    ```

  * The :func:`~.qchem.taylor_hamiltonian` function to build a Taylor Hamiltonian from a
    :class:`~.qchem.VibrationalPES` object.
    [(#6523)](https://github.com/PennyLaneAI/pennylane/pull/6523)

    ```pycon
    >>> qml.qchem.taylor_hamiltonian(pes_object, 4, 2)
    (
        0.016867926879358452 * I(0)
      + -0.007078617919572303 * Z(0)
      + 0.0008679410939323631 * X(0)
    )
    ```

  * The :func:`~.qchem.taylor_bosonic` function to build a Taylor Hamiltonian in terms of Bosonic
    operators.
    [(#6523)](https://github.com/PennyLaneAI/pennylane/pull/6523)

    ```pycon
    >>> coeffs_arr = qml.qchem.taylor_coeffs(pes_object)
    >>> bose_op = qml.qchem.taylor_bosonic(coeffs_arr, pes_object.freqs, is_local=pes_object.localized, uloc=pes_object.uloc)
    >>> type(bose_op)
    pennylane.bose.bosonic.BoseSentence
    ```

  Additional functionality is also available to optimize molecular geometries and convert between
  representations:

  * Convert Christiansen Hamiltonian integrals in the harmonic oscillator basis to integrals in the 
    vibrational self-consistent field (VSCF) basis with the :func:`~.qchem.vscf_integrals` function. 
    [(#6688)](https://github.com/PennyLaneAI/pennylane/pull/6688)

    ```pycon
    >>> h1 = np.array([[[0.00968289, 0.00233724, 0.0007408,  0.00199125],
                        [0.00233724, 0.02958449, 0.00675431, 0.0021936],
                        [0.0007408,  0.00675431, 0.0506012,  0.01280986],
                        [0.00199125, 0.0021936,  0.01280986, 0.07282307]]])
    >>> qml.qchem.vscf_integrals(h_integrals=[h1], modals=[4,4,4])
    (
      [array([[[ 9.36124041e-03,  3.63798208e-19, -3.42019607e-19,
        -3.83743044e-19],
        [ 9.59982270e-19,  2.77803512e-02,  5.18290259e-18,
        -4.82000376e-18],
        [-2.73826508e-19,  4.88583546e-18,  4.63297357e-02,
        -2.87022759e-18],
        [-1.94549340e-19, -5.48544743e-18, -1.41379640e-18,
          7.92203227e-02]]])], None
    )
    ```

  * Find the lowest energy configuration of molecules with :func:`~.qchem.optimize_geometry`.
    [(#6453)](https://github.com/PennyLaneAI/pennylane/pull/6453)
    [(#6666)](https://github.com/PennyLaneAI/pennylane/pull/6666)

    ```pycon
    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0,  1.0]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> eq_geom = qml.qchem.optimize_geometry(mol)
    >>> eq_geom
    array([[ 0.        ,  0.        , -0.40277116],
           [ 0.        ,  0.        ,  1.40277116]])
    ```

  * Separate normal mode frequencies and localize them with :func:`~.qchem.localize_normal_modes`.
    [(#6453)](https://github.com/PennyLaneAI/pennylane/pull/6453)

    ```pycon
    >>> freqs = np.array([1326.66001461, 2297.26736859, 2299.65032901])
    >>> vectors = np.array([[[ 5.71518696e-18, -4.55642350e-01,  5.20920552e-01],
                            [ 1.13167924e-17,  4.55642350e-01,  5.20920552e-01],
                            [-1.23163569e-17,  5.09494945e-12, -3.27565762e-02]],
                            [[-4.53008817e-17,  4.90364125e-01,  4.90363894e-01],
                            [-1.98591028e-16,  4.90361513e-01, -4.90361744e-01],
                            [-2.78235498e-18, -3.08350419e-02, -6.75886679e-08]],
                            [[ 5.75393451e-17,  5.37047963e-01,  4.41957355e-01],
                            [ 6.53049347e-17, -5.37050348e-01,  4.41959740e-01],
                            [-5.49709883e-17,  7.49851221e-08, -2.77912798e-02]]])
    >>> freqs_loc, vecs_loc, uloc = qml.qchem.localize_normal_modes(freqs, vectors)
    >>> freqs_loc
    array([1332.62008773, 2296.73455892, 2296.7346082 ])
    ```

<h3>Labs: a place for unified and rapid prototyping of research software 🧑‍🔬</h3>

The new :mod:`qml.labs <pennylane.labs>` module will house experimental research software 🔬. 
Features here may be useful for state-of-the-art research, beta testing, or getting a sneak peek 
into *potential* new features before they are added to PennyLane.

.. warning:: 
    This module is **experimental**! This means that features may not integrate well with other 
    PennyLane staples like differentiability, JAX, or JIT compatibility. There may also be 
    unexpected sharp bits 🔪 and errors ❌. Breaking changes and removals will happen without 
    warning.

Please use these features carefully and let us know your thoughts. Your feedback will inform
how these features become a part of mainline PennyLane.

<h4>Resource estimation</h4>

* Resource estimation functionality in Labs is focused on being light-weight and flexible. 
  The Labs :mod:`qml.labs.resource_estimation <pennylane.labs.resource_estimation>` module involves 
  modifications to core PennyLane that reduce the memory requirements and computational time of 
  resource estimation. These include new or modified base classes and one new function:
  * :class:`~.labs.resource_estimation.Resources` - This class is simplified in `labs`, removing the arguments: `gate_sizes`, `depth`,
    and `shots`. 
    [(#6428)](https://github.com/PennyLaneAI/pennylane/pull/6428)

  * :class:`~.labs.resource_estimation.ResourceOperator` - Replaces 
    :class:`~.resource.ResourceOperation`, expanded to include decompositions. 
    [(#6428)](https://github.com/PennyLaneAI/pennylane/pull/6428)

  * :class:`~.labs.resource_estimation.CompressedResourceOp` - A new class with the minimum 
    information to estimate resources: the operator type and the parameters needed to decompose it. 
    [(#6428)](https://github.com/PennyLaneAI/pennylane/pull/6428)

  * :class:`~.labs.resource_estimation.ResourceOperator` - versions of many existing PennyLane 
    operations, like Pauli operators, :class:`~.labs.resource_estimation.ResourceHadamard`, and 
    :class:`~.labs.resource_estimation.ResourceCNOT`. 
    [(#6447)](https://github.com/PennyLaneAI/pennylane/pull/6447)
    [(#6579)](https://github.com/PennyLaneAI/pennylane/pull/6579)
    [(#6538)](https://github.com/PennyLaneAI/pennylane/pull/6538)
    [(#6592)](https://github.com/PennyLaneAI/pennylane/pull/6592)

  * :func:`~.labs.resource_estimation.get_resources()` - The new entry point to efficiently obtain 
    the resources of quantum circuits.
    [(#6500)](https://github.com/PennyLaneAI/pennylane/pull/6500)

  Using new resource versions of existing operations and 
  :func:`~.labs.resource_estimation.get_resources`, we can estimate resources quickly:

  ```python
  import pennylane.labs.resource_estimation as re
  
  def my_circuit():
      for w in range(2):
          re.ResourceHadamard(w)
      re.ResourceCNOT([0, 1])
      re.ResourceRX(1.23, 0)
      re.ResourceRY(-4.56, 1)
      re.ResourceQFT(wires=[0, 1, 2])
      return qml.expval(re.ResourceHadamard(2))
  ```
  ```pycon
  >>> res = re.get_resources(my_circuit)()
  >>> print(res)
  wires: 3
  gates: 202
  gate_types:
  {'Hadamard': 5, 'CNOT': 10, 'T': 187}
  ```

  We can also set custom gate sets for decompositions:

  ````pycon
  >>> gate_set={"Hadamard","CNOT","RZ", "RX", "RY", "SWAP"}
  >>> res = re.get_resources(my_circuit, gate_set=gate_set)()
  >>> print(res)
  wires: 3
  gates: 24
  gate_types:
  {'Hadamard': 5, 'CNOT': 7, 'RX': 1, 'RY': 1, 'SWAP': 1, 'RZ': 9}
  ````

  Alternatively, it is possible to manually substitute associated resources:

  ```pycon
  >>> new_resources = re.substitute(res, "SWAP", re.Resources(2, 3, {"CNOT":3}))
  >>> print(new_resources)
  {'Hadamard': 5, 'CNOT': 10, 'RX': 1, 'RY': 1, 'RZ': 9}
  ```

<h4>Experimental functionality for handling dynamical Lie algebras (DLAs)</h4>

* Use the :mod:`qml.labs.dla <pennylane.labs.dla>` module to perform the
  [KAK decomposition](https://pennylane.ai/qml/demos/tutorial_kak_decomposition):
  * :func:`~.labs.dla.cartan_decomp`: obtain a **Cartan decomposition** of an input **Lie algebra** 
    via an **involution**.
    [(#6392)](https://github.com/PennyLaneAI/pennylane/pull/6392)

  * We provide a variety of **involutions** like :func:`~.labs.dla.concurrence_involution`, 
    :func:`~.labs.dla.even_odd_involution` and canonical Cartan involutions.
    [(#6392)](https://github.com/PennyLaneAI/pennylane/pull/6392)
    [(#6396)](https://github.com/PennyLaneAI/pennylane/pull/6396)

  * :func:`~.labs.dla.cartan_subalgebra`: compute a horizontal **Cartan subalgebra**.
    [(#6403)](https://github.com/PennyLaneAI/pennylane/pull/6403)

  * :func:`~.labs.dla.variational_kak_adj` : compute a 
    [variational KAK decomposition](https://pennylane.ai/qml/demos/tutorial_fixed_depth_hamiltonian_simulation_via_cartan_decomposition) 
    of a Hermitian operator using a **Cartan decomposition** and the adjoint representation of a 
    horizontal **Cartan subalgebra**.
    [(#6446)](https://github.com/PennyLaneAI/pennylane/pull/6446)

  To use this functionality we start with a set of Hermitian operators.

  ```pycon
  >>> n = 3
  >>> gens = [qml.X(i) @ qml.X(i + 1) for i in range(n - 1)]
  >>> gens += [qml.Z(i) for i in range(n)]
  >>> H = qml.sum(*gens)
  ```

  We then generate its Lie algebra by computing the Lie closure.

  ```pycon
  >>> g = qml.lie_closure(gens)
  >>> g = [op.pauli_rep for op in g]
  >>> print(g)
  [1 * X(0) @ X(1), 1 * X(1) @ X(2), 1.0 * Z(0), ...]
  ```

  We then choose an involution (e.g. :func:`~.labs.dla.concurrence_involution`) that defines a 
  Cartan decomposition `g = k + m`. `k` is the vertical subalgebra, and `m` its horizontal 
  complement (not a subalgebra).

  ```pycon
  >>> from pennylane.labs.dla import concurrence_involution, cartan_decomp
  >>> involution = concurrence_involution
  >>> k, m = cartan_decomp(g, involution=involution)
  ```

  The next step is just re-ordering the basis elements in `g` and computing its 
  `structure_constants`.

  ```pycon
  >>> g = k + m
  >>> adj = qml.structure_constants(g)
  ```

  We can then compute a (horizontal) Cartan subalgebra `a`, that is, a maximal Abelian subalgebra of 
  `m`.

  ```pycon
  >>> from pennylane.labs.dla import cartan_subalgebra
  >>> g, k, mtilde, a, adj = cartan_subalgebra(g, k, m, adj)
  ```

  Having determined both subalgebras `k` and `a`, we can compute the KAK decomposition variationally 
  like in [2104.00728](https://arxiv.org/abs/2104.00728), see our 
  [demo on KAK decomposition in practice](https://pennylane.ai/qml/demos/tutorial_fixed_depth_hamiltonian_simulation_via_cartan_decomposition).

  ```pycon
  >>> from pennylane.labs.dla import variational_kak_adj
  >>> dims = (len(k), len(mtilde), len(a))
  >>> adjvec_a, theta_opt = variational_kak_adj(H, g, dims, adj, opt_kwargs={"n_epochs": 3000})
  ```

* We also provide some additional features that are useful for handling dynamical Lie algebras.
  * :func:`~.labs.dla.recursive_cartan_decomp`: perform consecutive recursive Cartan decompositions.
    [(#6396)](https://github.com/PennyLaneAI/pennylane/pull/6396)

  * :func:`~.labs.dla.lie_closure_dense`: extension of `qml.lie_closure` using dense matrices.
    [(#6371)](https://github.com/PennyLaneAI/pennylane/pull/6371)
    [(#6695)](https://github.com/PennyLaneAI/pennylane/pull/6695)

  * :func:`~.labs.dla.structure_constants_dense`: extension of `qml.structure_constants` using dense 
    matrices.
    [(#6396)](https://github.com/PennyLaneAI/pennylane/pull/6396) [(#6376)](https://github.com/PennyLaneAI/pennylane/pull/6376)


<h4>Vibrational Hamiltonians</h4>

* New functionality in labs helps with the construction of vibrational Hamiltonians.
  * Generate potential energy surfaces (PES) with `qml.labs.vibrational.vibrational_pes`.
    [(#6616)](https://github.com/PennyLaneAI/pennylane/pull/6616)
    [(#6676)](https://github.com/PennyLaneAI/pennylane/pull/6676)
  
    ```pycon
    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = vibrational_pes(mol)
    ```
  * Use the `qml.labs.vibrational.christiansen_hamiltonian` function and potential energy surfaces 
    to generate Hamiltonians in the Ch  ristiansen form.
    [(#6560)](https://github.com/PennyLaneAI/pennylane/pull/6560)

<h3>Improvements 🛠</h3>

<h4>QChem improvements</h4>

* The `qml.qchem.factorize` function now supports new methods for double factorization:
  Cholesky decomposition (`cholesky=True`) and compressed double factorization (`compressed=True`).
  [(#6573)](https://github.com/PennyLaneAI/pennylane/pull/6573)
  [(#6611)](https://github.com/PennyLaneAI/pennylane/pull/6611)

* A new function for performing the [block-invariant symmetry shift](https://arxiv.org/pdf/2304.13772) 
  on electronic integrals has been added with `qml.qchem.symmetry_shift`.
  [(#6574)](https://github.com/PennyLaneAI/pennylane/pull/6574)

* The differentiable Hartree-Fock workflow is now compatible with JAX.
  [(#6096)](https://github.com/PennyLaneAI/pennylane/pull/6096)
  [(#6707)](https://github.com/PennyLaneAI/pennylane/pull/6707)

<h4>Transform for combining GlobalPhase instances</h4>

* A new transform called `qml.transforms.combine_global_phases` has been added. It combines all 
  `qml.GlobalPhase` gates in a circuit into a single one applied at the end. This can be useful for 
  circuits that include a lot of `qml.GlobalPhase` gates that are introduced directly during 
  circuit creation, decompositions that include `qml.GlobalPhase` gates, etc.
  [(#6686)](https://github.com/PennyLaneAI/pennylane/pull/6686)

<h4>Better drawing functionality</h4>

* `qml.draw_mpl` now has a `wire_options` keyword argument, which allows for global- and per-wire 
  customization with options like `color`, `linestyle`, and `linewidth`.
  [(#6486)](https://github.com/PennyLaneAI/pennylane/pull/6486)

  Here is an example that would make all wires cyan and bold except for wires 2 and 6, which are 
  dashed and a different colour.

  ```python
  @qml.qnode(qml.device("default.qubit"))
  def circuit(x):
      for w in range(5):
          qml.Hadamard(w) 
      return qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

  wire_options = {"color": "cyan", 
                  "linewidth": 5, 
                  2: {"linestyle": "--", "color": "red"}, 
                  6: {"linestyle": "--", "color": "orange"}
              }
  print(qml.draw_mpl(circuit, wire_options=wire_options)(0.52))
  ```

<h4>New device capabilities 💾</h4>

* Two new methods, `setup_execution_config` and `preprocess_transforms`, have been added to the 
  `Device` class. Device developers are encouraged to override these two methods separately instead of the
  `preprocess` method. For now, to avoid ambiguity, a device is allowed to override either these
  two methods or `preprocess`, but not both. In the long term, we will slowly phase out the use of
  `preprocess` in favour of these two methods for better separation of concerns.
  [(#6617)](https://github.com/PennyLaneAI/pennylane/pull/6617)

* Developers of plugin devices now have the option of providing a TOML-formatted configuration file
  to declare the capabilities of the device. See
  [Device Capabilities](https://docs.pennylane.ai/en/latest/development/plugins.html#device-capabilities) 
  for details.

* An internal module called `qml.devices.capabilities` has been added that defines a new 
  `DeviceCapabilites` data class, as well as functions that load and parse the TOML-formatted 
  configuration files.
  [(#6407)](https://github.com/PennyLaneAI/pennylane/pull/6407)

  ```pycon
  >>> from pennylane.devices.capabilities import DeviceCapabilities
  >>> capabilities = DeviceCapabilities.from_toml_file("my_device.toml")
  >>> isinstance(capabilities, DeviceCapabilities)
  True
  ```

* Devices that extend `qml.devices.Device` now have an optional class attribute called 
  `capabilities`, which is an instance of the `DeviceCapabilities` data class constructed from the 
  configuration file if it exists. Otherwise, it is set to `None`.
  [(#6433)](https://github.com/PennyLaneAI/pennylane/pull/6433)

  ```python
  from pennylane.devices import Device

  class MyDevice(Device):
      config_filepath = "path/to/config.toml"
      ...
  ```

  ```pycon
  >>> isinstance(MyDevice.capabilities, DeviceCapabilities)
  True
  ```

* Default implementations of `Device.setup_execution_config` and `Device.preprocess_transforms`
  have been added to the device API for devices that provide a TOML configuration file and, thus, 
  have a `capabilities` property.
  [(#6632)](https://github.com/PennyLaneAI/pennylane/pull/6632)
  [(#6653)](https://github.com/PennyLaneAI/pennylane/pull/6653)

<h4>Capturing and representing hybrid programs</h4>

* Support has been added for `if`/`else` statements and `for` and `while` loops in circuits executed 
  with `qml.capture.enabled`, via Autograph. Autograph conversion is now used by default in 
  `make_plxpr`, but can be skipped with `autograph=False`.
  [(#6406)](https://github.com/PennyLaneAI/pennylane/pull/6406)
  [(#6413)](https://github.com/PennyLaneAI/pennylane/pull/6413)
  [(#6426)](https://github.com/PennyLaneAI/pennylane/pull/6426)
  [(#6645)](https://github.com/PennyLaneAI/pennylane/pull/6645)
  [(#6685)](https://github.com/PennyLaneAI/pennylane/pull/6685)

* `qml.transform` now accepts a `plxpr_transform` argument. This argument must be a function that 
  can transform plxpr. Note that executing a transformed function will currently raise a 
  `NotImplementedError`. To see more details, check out the 
  :func:`documentation of qml.transform <pennylane.transform>`.
  [(#6633)](https://github.com/PennyLaneAI/pennylane/pull/6633)
  [(#6722)](https://github.com/PennyLaneAI/pennylane/pull/6722)

* Users can now apply transforms with program capture enabled. Transformed functions cannot be
  executed by default. To apply the transforms (and to be able to execute the function), it must be 
  decorated with the new `qml.capture.expand_plxpr_transforms` function, which accepts a callable as 
  input and returns a new function for which all present transforms have been applied.
  [(#6722)](https://github.com/PennyLaneAI/pennylane/pull/6722)

  ```python
  from functools import partial

  qml.capture.enable()
  wire_map = {0: 3, 1: 6, 2: 9}

  @partial(qml.map_wires, wire_map=wire_map)
  def circuit(x, y):
      qml.RX(x, 0)
      qml.CNOT([0, 1])
      qml.CRY(y, [1, 2])
      return qml.expval(qml.Z(2))
  ```
  ```pycon
  >>> qml.capture.make_plxpr(circuit)(1.2, 3.4)
  { lambda ; a:f32[] b:f32[]. let
      c:AbstractMeasurement(n_wires=None) = _map_wires_transform_transform[
      args_slice=slice(0, 2, None)
      consts_slice=slice(2, 2, None)
      inner_jaxpr={ lambda ; d:f32[] e:f32[]. let
          _:AbstractOperator() = RX[n_wires=1] d 0
          _:AbstractOperator() = CNOT[n_wires=2] 0 1
          _:AbstractOperator() = CRY[n_wires=2] e 1 2
          f:AbstractOperator() = PauliZ[n_wires=1] 2
          g:AbstractMeasurement(n_wires=None) = expval_obs f
        in (g,) }
      targs_slice=slice(2, None, None)
      tkwargs={'wire_map': {0: 3, 1: 6, 2: 9}, 'queue': False}
      ] a b
    in (c,) }
  >>> transformed_circuit = qml.capture.expand_plxpr_transforms(circuit)
  >>> jax.make_jaxpr(transformed_circuit)(1.2, 3.4)
  { lambda ; a:f32[] b:f32[]. let
      _:AbstractOperator() = RX[n_wires=1] a 3
      _:AbstractOperator() = CNOT[n_wires=2] 3 6
      _:AbstractOperator() = CRY[n_wires=2] b 6 9
      c:AbstractOperator() = PauliZ[n_wires=1] 9
      d:AbstractMeasurement(n_wires=None) = expval_obs c
    in (d,) }
  ```

* The `qml.iterative_qpe` function can now be compactly captured into plxpr.
  [(#6680)](https://github.com/PennyLaneAI/pennylane/pull/6680)

* Three new plxpr interpreters have been added that allow for functions and plxpr to be natively 
  transformed with the same API as the corresponding existing transforms in PennyLane when program
  capture is enabled:

  * `qml.capture.transforms.CancelInterpreter`:this class cancels operators appearing consecutively 
    that are adjoints of each other following the same API as `qml.transforms.cancel_inverses`.
    [(#6692)](https://github.com/PennyLaneAI/pennylane/pull/6692)

  * `qml.capture.transforms.DecomposeInterpreter`: this class decomposes pennylane operators 
    following the same API as `qml.transforms.decompose`.
    [(#6691)](https://github.com/PennyLaneAI/pennylane/pull/6691)

  * `qml.capture.transforms.MapWiresInterpreter`: this class maps wires to new values following the 
    same API as `qml.map_wires`.
    [(#6697)](https://github.com/PennyLaneAI/pennylane/pull/6697)

* A `qml.tape.plxpr_to_tape` function is now available that converts plxpr to a tape.
  [(#6343)](https://github.com/PennyLaneAI/pennylane/pull/6343)

* Execution with capture enabled now follows a new execution pipeline and natively passes the
  captured plxpr to the device. Since it no longer falls back to the old pipeline, execution only 
  works with a reduced feature set.
  [(#6655)](https://github.com/PennyLaneAI/pennylane/pull/6655)
  [(#6596)](https://github.com/PennyLaneAI/pennylane/pull/6596)

* PennyLane transforms can now be captured as primitives with experimental program capture enabled.
  [(#6633)](https://github.com/PennyLaneAI/pennylane/pull/6633)

* `jax.vmap` can be captured with `qml.capture.make_plxpr` and is compatible with quantum circuits.
  [(#6349)](https://github.com/PennyLaneAI/pennylane/pull/6349)
  [(#6422)](https://github.com/PennyLaneAI/pennylane/pull/6422)
  [(#6668)](https://github.com/PennyLaneAI/pennylane/pull/6668)

* A `qml.capture.PlxprInterpreter` base class has been added for easy transformation and execution 
  of plxpr.
  [(#6141)](https://github.com/PennyLaneAI/pennylane/pull/6141)

* A `DefaultQubitInterpreter` class has been added to provide plxpr execution using python based 
  tools, and the `DefaultQubit.eval_jaxpr` method has been implemented.
  [(#6594)](https://github.com/PennyLaneAI/pennylane/pull/6594)
  [(#6328)](https://github.com/PennyLaneAI/pennylane/pull/6328)

* An optional method, `eval_jaxpr`, has been added to the device API for native execution of plxpr 
  programs.
  [(#6580)](https://github.com/PennyLaneAI/pennylane/pull/6580)

* `qml.capture.qnode_call` has been made private and moved to the `workflow` module.
  [(#6620)](https://github.com/PennyLaneAI/pennylane/pull/6620/)

<h4>Other Improvements</h4>

* `qml.math.grad` and `qml.math.jacobian` have been added to differentiate a function with inputs of 
  any interface in a JAX-like manner.
  [(#6741)](https://github.com/PennyLaneAI/pennylane/pull/6741)

* `qml.GroverOperator` now has a `work_wires` property.
  [(#6738)](https://github.com/PennyLaneAI/pennylane/pull/6738)

* The `Wires` object's usage across Pennylane source code has been tidied up for internal 
  consistency.
  [(#6689)](https://github.com/PennyLaneAI/pennylane/pull/6689)

* `qml.equal` now supports `qml.PauliWord` and `qml.PauliSentence` instances.
  [(#6703)](https://github.com/PennyLaneAI/pennylane/pull/6703)

* Redundant commutator computations from `qml.lie_closure` have been removed.
  [(#6724)](https://github.com/PennyLaneAI/pennylane/pull/6724)

* A comprehensive error is now raised when using `qml.fourier.qnode_spectrum` with standard Numpy
  arguments and `interface="auto"`.
  [(#6622)](https://github.com/PennyLaneAI/pennylane/pull/6622)

* Pauli string representations for the gates `{X, Y, Z, S, T, SX, SWAP, ISWAP, ECR, SISWAP}` have 
  been added, and a shape error in the matrix conversion of `qml.PauliSentence`s with `list` or 
  `array` inputs has been fixed.
  [(#6562)](https://github.com/PennyLaneAI/pennylane/pull/6562)
  [(#6587)](https://github.com/PennyLaneAI/pennylane/pull/6587)
  
* `qml.QNode` and `qml.execute` now forbid certain keyword arguments from being passed positionally.
  [(#6610)](https://github.com/PennyLaneAI/pennylane/pull/6610)

* The string representations for the `qml.S`, `qml.T`, and `qml.SX` have been shortened.
  [(#6542)](https://github.com/PennyLaneAI/pennylane/pull/6542)

* Internal class functions and dunder methods have been added to allow for multiplying Resources 
  objects in series and in parallel.
  [(#6567)](https://github.com/PennyLaneAI/pennylane/pull/6567)

* The `diagonalize_measurements` transform no longer raises an error for unknown observables. Instead,
  they are left un-diagonalized, with the expectation that observable validation will catch any un-diagonalized
  observables that are also unsupported by the device.
  [(#6653)](https://github.com/PennyLaneAI/pennylane/pull/6653)

* A `qml.wires.Wires` object can now be converted to a JAX array, if all wire labels are supported as 
  JAX array elements.
  [(#6699)](https://github.com/PennyLaneAI/pennylane/pull/6699)

* PennyLane is compatible with `quimb 1.10.0`.
  [(#6630)](https://github.com/PennyLaneAI/pennylane/pull/6630)
  [(#6736)](https://github.com/PennyLaneAI/pennylane/pull/6736)

* A developer focused `run` function has been added to the `qml.workflow` module for a cleaner and 
  standardized approach to executing tapes on an ML interface.
  [(#6657)](https://github.com/PennyLaneAI/pennylane/pull/6657)

* Internal changes have been made to standardize execution interfaces, which resolves ambiguities in 
  how the `interface` value is handled during execution.
  [(#6643)](https://github.com/PennyLaneAI/pennylane/pull/6643)

* All interface handling logic has been moved to `interface_utils.py` in the `qml.math` module.
  [(#6649)](https://github.com/PennyLaneAI/pennylane/pull/6649)

* `qml.execute` can now be used with `diff_method="best"`.
  Classical cotransform information is now handled lazily by the workflow. Gradient method
  validation and program setup are now handled inside of `qml.execute`, instead of in `QNode`.
  [(#6716)](https://github.com/PennyLaneAI/pennylane/pull/6716)

* PyTree support for measurements in a circuit has been added.
  [(#6378)](https://github.com/PennyLaneAI/pennylane/pull/6378)

  ```python
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.Hadamard(0)
      qml.CNOT([0,1])
      return {"Probabilities": qml.probs(), "State": qml.state()}
  ```
  ```pycon
  >>> circuit()
  {'Probabilities': array([0.5, 0. , 0. , 0.5]), 'State': array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])}
  ```

* The `_cache_transform` transform has been moved to its own file located in 
  `pennylane/workflow/_cache_transform.py`.
  [(#6624)](https://github.com/PennyLaneAI/pennylane/pull/6624)

* The `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)
  [(#6779)](https://github.com/PennyLaneAI/pennylane/pull/6779)

* The Jaxpr primitives for `for_loop`, `while_loop` and `cond` now store slices instead of
  numbers of arguments. This helps with keeping track of what order the arguments come in.
  [(#6521)](https://github.com/PennyLaneAI/pennylane/pull/6521)

* The `ExecutionConfig.gradient_method` function has been expanded to store `TransformDispatcher` 
  type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

* The string representation of `Resources` instances has been improved to match the attribute names.
  [(#6581)](https://github.com/PennyLaneAI/pennylane/pull/6581)

* The documentation for the `dynamic_one_shot` transform has been improved, and a warning is raised 
  when a user-applied `dynamic_one_shot` transform is ignored in favour of the existing transform in 
  a device's preprocessing transform program.
  [(#6701)](https://github.com/PennyLaneAI/pennylane/pull/6701)

* A `qml.devices.qubit_mixed` module has been added for mixed-state qubit device support. This 
  module introduces an `apply_operation` helper function that features:
  * Two density matrix contraction methods using `einsum` and `tensordot`
  * Optimized handling of special cases including: Diagonal operators, Identity operators, CX 
    (controlled-X), Multi-controlled X gates, Grover operators
  [(#6379)](https://github.com/PennyLaneAI/pennylane/pull/6379)

* A function called `create_initial_state` has been added to allow for initializing a circuit with a 
  density matrix using `qml.StatePrep` or `qml.QubitDensityMatrix`.
  [(#6503)](https://github.com/PennyLaneAI/pennylane/pull/6503)

* Several additions have been made to eventually migrate the `"default.mixed"` device to the new 
  device API:
  * A `preprocess` method has been added to the `QubitMixed` device class to preprocess the quantum 
    circuit before execution.
    [(#6601)](https://github.com/PennyLaneAI/pennylane/pull/6601)
  * A new class called `DefaultMixedNewAPI` has been added to the `qml.devices.qubit_mixed` module, 
    which will replace the legacy `DefaultMixed`. 
    [(#6607)](https://github.com/PennyLaneAI/pennylane/pull/6607)
  * A new submodule called `devices.qubit_mixed.measure` has been added, featuring a `measure` 
    function for measuring qubits in mixed-state devices.
    [(#6637)](https://github.com/PennyLaneAI/pennylane/pull/6637)
  * A new submodule called `devices.qubit_mixed.simulate` has been added, featuring a `simulate` 
    function for simulating mixed states in analytic mode.
    [(#6618)](https://github.com/PennyLaneAI/pennylane/pull/6618)
  * A new submodule called `devices.qubit_mixed.sampling` has been added, featuring functions 
    `sample_state`, `measure_with_samples` and `sample_probs` for sampling qubits in mixed-state 
    devices.
    [(#6639)](https://github.com/PennyLaneAI/pennylane/pull/6639)
  * The finite-shot branch of `devices.qubit_mixed.simulate` has been added, which allows for 
    accepting stochastic arguments such as `shots`, `rng` and `prng_key`.
    [(#6665)](https://github.com/PennyLaneAI/pennylane/pull/6665)
  * Support for `qml.Snapshot` has been added.
    [(#6659)](https://github.com/PennyLaneAI/pennylane/pull/6659)

* Reporting of test warnings as failures has been added.
  [(#6217)](https://github.com/PennyLaneAI/pennylane/pull/6217)

* A warning message in the Gradients and training documentation has been added that pertains to 
  `ComplexWarning`s.
  [(#6543)](https://github.com/PennyLaneAI/pennylane/pull/6543)

* A new figure was added to the landing page of the PennyLane website.
  [(#6696)](https://github.com/PennyLaneAI/pennylane/pull/6696)

<h3>Breaking changes 💔</h3>

* The default graph coloring method of `qml.dot`, `qml.sum`, and `qml.pauli.optimize_measurements` 
  for grouping observables was changed from `"rlf"` to `"lf"`. Internally, 
  `qml.pauli.group_observables` has been replaced with `qml.pauli.compute_partition_indices` in 
  several places to improve efficiency.
  [(#6706)](https://github.com/PennyLaneAI/pennylane/pull/6706)

* `qml.fourier.qnode_spectrum` no longer automatically converts pure Numpy parameters to the
  Autograd framework. As the function uses automatic differentiation for validation, parameters
  from such a framework have to be used.
  [(#6622)](https://github.com/PennyLaneAI/pennylane/pull/6622)

* `qml.math.jax_argnums_to_tape_trainable` has been moved and made private to avoid an unnecessary 
  QNode dependency in the `qml.math` module.
  [(#6609)](https://github.com/PennyLaneAI/pennylane/pull/6609)

* Gradient transforms are now applied after the user's transform program. This ensures user 
  transforms work as expected on initial structures (e.g., embeddings or entangling layers), 
  guarantees that gradient transforms only process compatible operations, aligns transform order 
  with user expectations, and avoids confusion.
  [(#6590)](https://github.com/PennyLaneAI/pennylane/pull/6590)

* Legacy operator arithmetic has been removed. This includes `qml.ops.Hamiltonian`, 
  `qml.operation.Tensor`, `qml.operation.enable_new_opmath`, `qml.operation.disable_new_opmath`, and 
  `qml.operation.convert_to_legacy_H`. Note that `qml.Hamiltonian` will continue to dispatch to 
  `qml.ops.LinearCombination`. For more information, check out the 
  [updated operator troubleshooting page](https://docs.pennylane.ai/en/stable/news/new_opmath.html).
  [(#6548)](https://github.com/PennyLaneAI/pennylane/pull/6548)
  [(#6602)](https://github.com/PennyLaneAI/pennylane/pull/6602)
  [(#6589)](https://github.com/PennyLaneAI/pennylane/pull/6589)

* The developer-facing `qml.utils` module has been removed. 
  [(#6588)](https://github.com/PennyLaneAI/pennylane/pull/6588):

  Specifically, the following 4 sets of functions have been either moved or removed:
    * `qml.utils._flatten`, `qml.utils.unflatten` has been moved and renamed to `qml.optimize.qng._flatten_np` and `qml.optimize.qng._unflatten_np` respectively.
    * `qml.utils._inv_dict` and `qml._get_default_args` have been removed.
    * `qml.utils.pauli_eigs` has been moved to `qml.pauli.utils`.
    * `qml.utils.expand_vector` has been moved to `qml.math.expand_vector`.

* The `qml.qinfo` module has been removed. Please use the corresponding functions in the `qml.math` 
  and `qml.measurements` modules instead.
  [(#6584)](https://github.com/PennyLaneAI/pennylane/pull/6584)

* Top level access to `Device`, `QubitDevice`, and `QutritDevice` have been removed. Instead, they
  are available as `qml.devices.LegacyDevice`, `qml.devices.QubitDevice`, and 
  `qml.devices.QutritDevice`, respectively.
  [(#6537)](https://github.com/PennyLaneAI/pennylane/pull/6537)

* The `'ancilla'` argument for `qml.iterative_qpe` has been removed. Instead, use the `'aux_wire'` 
  argument.
  [(#6532)](https://github.com/PennyLaneAI/pennylane/pull/6532)

* The `qml.BasisStatePreparation` template has been removed. Instead, use `qml.BasisState`.
  [(#6528)](https://github.com/PennyLaneAI/pennylane/pull/6528)

* The `qml.workflow.set_shots` helper function has been removed. We no longer interact with the 
  legacy device interface in our code. Instead, shots should be specified on the tape, and the 
  device should use these shots.
  [(#6534)](https://github.com/PennyLaneAI/pennylane/pull/6534)

* `QNode.gradient_fn` has been removed. Please use `QNode.diff_method` instead. 
  `QNode.get_gradient_fn` can also be used to process the differentiation method.
  [(#6535)](https://github.com/PennyLaneAI/pennylane/pull/6535)

* The `qml.QubitStateVector` template has been removed. Instead, use `qml.StatePrep`.
  [(#6525)](https://github.com/PennyLaneAI/pennylane/pull/6525)

* `qml.broadcast` has been removed. Users should use `for` loops instead.
  [(#6527)](https://github.com/PennyLaneAI/pennylane/pull/6527)

* The `max_expansion` argument for `qml.transforms.clifford_t_decomposition` has been removed.
  [(#6571)](https://github.com/PennyLaneAI/pennylane/pull/6571)

* The `expand_depth` argument for `qml.compile` has been removed.
  [(#6531)](https://github.com/PennyLaneAI/pennylane/pull/6531)

* The `qml.shadows.shadow_expval` transform has been removed. Instead, please use the
  `qml.shadow_expval` measurement process.
  [(#6530)](https://github.com/PennyLaneAI/pennylane/pull/6530)
  [(#6561)](https://github.com/PennyLaneAI/pennylane/pull/6561)

* The developer-facing ``qml.drawer.MPLDrawer`` argument `n_wires` has been replaced with `wire_map`,
  which contains more complete information about wire labels and order. This allows the new functionality 
  to specify `wire_options` for specific wires when using string wire labels or non-sequential wire ordering.
  [(#6805)](https://github.com/PennyLaneAI/pennylane/pull/6805)

<h3>Deprecations 👋</h3>

* The `tape` and `qtape` properties of `QNode` have been deprecated. Instead, use the 
  `qml.workflow.construct_tape` function.
  [(#6583)](https://github.com/PennyLaneAI/pennylane/pull/6583)
  [(#6650)](https://github.com/PennyLaneAI/pennylane/pull/6650)

* The `max_expansion` argument in `qml.devices.preprocess.decompose` is deprecated and will be 
  removed in v0.41.
  [(#6400)](https://github.com/PennyLaneAI/pennylane/pull/6400)

* The `decomp_depth` argument in `qml.transforms.set_decomposition` is deprecated and will be 
  removed in v0.41.
  [(#6400)](https://github.com/PennyLaneAI/pennylane/pull/6400)

* The `output_dim` property of `qml.tape.QuantumScript` has been deprecated. Instead, use method 
  `shape` of `QuantumScript` or `MeasurementProcess` to get the same information.
  [(#6577)](https://github.com/PennyLaneAI/pennylane/pull/6577)

* The `QNode.get_best_method` and `QNode.best_method_str` methods have been deprecated. Instead, use 
  the `qml.workflow.get_best_diff_method` function.
  [(#6418)](https://github.com/PennyLaneAI/pennylane/pull/6418)

* The `qml.execute` `gradient_fn` keyword argument has been renamed to `diff_method` to better 
  align with the termionology used by the QNode. `gradient_fn` will be removed in v0.41.
  [(#6549)](https://github.com/PennyLaneAI/pennylane/pull/6549)

* The old `qml.qsvt` functionality is moved to `qml.qsvt_legacy`
  and is now deprecated. It will be removed in v0.41.
  [(#6520)](https://github.com/PennyLaneAI/pennylane/pull/6520)
  

<h3>Documentation 📝</h3>

* The docstrings for `qml.qchem.Molecule` and `qml.qchem.molecular_hamiltonian` have been updated to 
  include a note that says that they are not compatible with `qjit` or `jit`.  
  [(#6702)](https://github.com/PennyLaneAI/pennylane/pull/6702)

* The documentation of `TrotterProduct` has been updated to include the impact of the operands in 
  the Hamiltonian on the structure of the created circuit.
  [(#6629)](https://github.com/PennyLaneAI/pennylane/pull/6629)

* The documentation of `QSVT` has been updated to include examples for different block encodings.
  [(#6673)](https://github.com/PennyLaneAI/pennylane/pull/6673)

* The link to `qml.ops.one_qubit_transform` was fixed in the `QubitUnitary` docstring.
  [(#6745)](https://github.com/PennyLaneAI/pennylane/pull/6745)

<h3>Bug fixes 🐛</h3>

* Validation has been added to ensure that the device vjp is only used when the device actually 
  supports it.
  [(#6755)](https://github.com/PennyLaneAI/pennylane/pull/6755/)

* `qml.counts` now returns all outcomes when the `all_outcomes` argument is `True` and mid-circuit 
  measurements are present.
  [(#6732)](https://github.com/PennyLaneAI/pennylane/pull/6732)

* `qml.ControlledQubitUnitary` now has consistent behaviour with program capture enabled. 
  [(#6719)](https://github.com/PennyLaneAI/pennylane/pull/6719)

* The `Wires` object now throws a `TypeError` if `wires=None`. 
  [(#6713)](https://github.com/PennyLaneAI/pennylane/pull/6713)
  [(#6720)](https://github.com/PennyLaneAI/pennylane/pull/6720)

* The `qml.Hermitian` class no longer checks that the provided matrix is hermitian. The reason for
  this removal is to allow for faster execution and avoid incompatibilities with `jax.jit`.
  [(#6642)](https://github.com/PennyLaneAI/pennylane/pull/6642)

* Subclasses of `qml.ops.Controlled` no longer bind the primitives of their base operators when 
  program capture is enabled.
  [(#6672)](https://github.com/PennyLaneAI/pennylane/pull/6672)

* The `qml.HilbertSchmidt` and `qml.LocalHilbertSchmidt` templates now apply the complex conjugate
  of the unitaries instead of the adjoint, providing the correct result.
  [(#6604)](https://github.com/PennyLaneAI/pennylane/pull/6604)

* QNode return behaviour is now consistent for lists and tuples.
  [(#6568)](https://github.com/PennyLaneAI/pennylane/pull/6568)

* QNodes now accept arguments with types defined in libraries that are not necessarily in the list 
  of supported interfaces, such as the `Graph` class defined in `networkx`.
  [(#6600)](https://github.com/PennyLaneAI/pennylane/pull/6600)

* `qml.math.get_deep_interface` now works properly for Autograd arrays.
  [(#6557)](https://github.com/PennyLaneAI/pennylane/pull/6557)

* Printing instances of `qml.Identity` now returns the correct wires list.
  [(#6506)](https://github.com/PennyLaneAI/pennylane/pull/6506)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Shiwen An,
Utkarsh Azad,
Astral Cai,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Lasse Dierich,
Lillian Frederiksen,
Pietropaolo Frisoni,
Simone Gasperini,
Diego Guala, 
Austin Huang,
Korbinian Kottmann,
Christina Lee,
Alan Martin,
William Maxwell,
Anton Naim Ibrahim,
Andrija Paurevic,
Justin Pickering,
Jay Soni,
David Wierichs.