
# Release 0.42.0

<h3>New features since last release</h3>

<h4>State-of-the-art templates and decompositions 🐝</h4>

* A new decomposition using [unary iteration](https://arxiv.org/pdf/1805.03662) has been added to :class:`~.Select`.
  This state-of-the-art decomposition reduces the :class:`~.T`-count significantly, and uses :math:`c-1` auxiliary wires,
  where :math:`c` is the number of control wires of the `Select` operator.
  [(#7623)](https://github.com/PennyLaneAI/pennylane/pull/7623)
  [(#7744)](https://github.com/PennyLaneAI/pennylane/pull/7744)
  [(#7842)](https://github.com/PennyLaneAI/pennylane/pull/7842)

  Unary iteration leverages auxiliary wires to store intermediate values for reuse among the 
  different multi-controlled operators, avoiding unnecessary recomputation and leading to more 
  efficient decompositions to elementary gates. This decomposition uses a template called
  :class:`~.TemporaryAND`, which was also added in this release (see the next changelog entry).
  
  This decomposition rule for :class:`~.Select` is available when the graph-based
  decomposition system is enabled via :func:`~.decomposition.enable_graph`:

  ```python
  import pennylane as qp
  from functools import partial

  qml.decomposition.enable_graph()
  ```

  To demonstrate the resource-efficiency of this new decomposition, let's use 
  :func:`~.transforms.decompose` to decompose an instance of :class:`~.Select` using the new unary iterator decomposition rule, and further decompose these gates into the Clifford+T gate set
  using :func:`~.clifford_t_decomposition` so that we can count the number of `T` gates required:
  
  ```python
  reg = qml.registers({"targ": 2, "control": 2, "work": 1})
  targ, control, work = (reg[k] for k in reg.keys())

  dev = qml.device('default.qubit')
  ops = [qml.X(targ[0]), qml.X(targ[1]), qml.Y(targ[0]), qml.SWAP(targ)]

  @qml.clifford_t_decomposition
  @partial(qml.transforms.decompose, gate_set={
          qml.X, qml.CNOT, qml.TemporaryAND, "Adjoint(TemporaryAND)", "CY", "CSWAP"
      }
  )
  @qml.qnode(dev)
  def circuit():
      qml.Select(ops, control=control, work_wires=work)
      return qml.state()
  ```

  ```pycon
  >>> unary_specs = qml.specs(circuit)()
  >>> print(unary_specs['resources'].gate_types["T"])
  16
  >>> print(unary_specs['resources'].gate_types["Adjoint(T)"])
  13
  ```

  Go check out the *Unary iterator decomposition* section in the :class:`~.Select` documentation for 
  more information!

* A new template called :class:`~.TemporaryAND` has been added. :class:`~.TemporaryAND` enables more 
  efficient circuit decompositions, such as the newest decomposition of the :class:`~.Select` template.
  [(#7472)](https://github.com/PennyLaneAI/pennylane/pull/7472)

  The :class:`~.TemporaryAND` operation is a three-qubit gate equivalent to a logical ``AND`` operation 
  (or a reversible :class:`~.Toffoli`): it assumes that the target qubit is initialized in the 
  ``|0〉`` state, while ``Adjoint(TemporaryAND)`` assumes the target qubit will be output into the 
  ``|0〉`` state. For more details, see Fig. 4 in [arXiv:1805.03662](https://arxiv.org/abs/1805.03662).

  ```python
  from functools import partial

  dev = qml.device("default.qubit")

  @partial(qml.set_shots, shots=1)
  @qml.qnode(dev)
  def circuit():
      # |0000⟩
      qml.X(0) # |1000⟩
      qml.X(1) # |1100⟩
      # The target wire is in state |0>, so we can apply TemporaryAND
      qml.TemporaryAND([0, 1, 2]) # |1110⟩
      qml.CNOT([2, 3]) # |1111⟩
      # The target wire will be in state |0> after adjoint(TemporaryAND) gate is applied
      # so we can apply adjoint(TemporaryAND)
      qml.adjoint(qml.TemporaryAND([0, 1, 2])) # |1101⟩
      return qml.sample(wires=[0, 1, 2, 3])
  ```
  
  ```pycon
  >>> print(circuit())
  [1 1 0 1]
  ```

* A new template called :class:`~.SemiAdder` has been added, which provides state-of-the-art 
  resource-efficiency (fewer :class:`~.T` gates) when performing addition on a quantum computer.
  [(#7494)](https://github.com/PennyLaneAI/pennylane/pull/7494)

  Based on [arXiv:1709.06648](https://arxiv.org/abs/1709.06648), :class:`~.SemiAdder` performs the plain 
  addition of two integers in the computational basis. Here is an example of performing `3 + 4 = 7`
  with 5 additional work wires:

  ```python
  from functools import partial

  x = 3
  y = 4

  wires = qml.registers({"x": 3, "y": 6, "work": 5})

  dev = qml.device("default.qubit")

  @partial(qml.set_shots, shots=1)
  @qml.qnode(dev)
  def circuit():
      qml.BasisEmbedding(x, wires=wires["x"])
      qml.BasisEmbedding(y, wires=wires["y"])
      qml.SemiAdder(wires["x"], wires["y"], wires["work"])
      return qml.sample(wires=wires["y"])
  ```
  
  ```pycon
  >>> print(circuit()) 
  [0 0 0 1 1 1]
  ```

  The result `[0 0 0 1 1 1]` is the binary representation of `7`.

* A new template called :class:`~.SelectPauliRot` is available, which applies a sequence of 
  uniformly controlled rotations on a target qubit. This operator appears frequently in unitary 
  decompositions and block-encoding techniques. 
  [(#7206)](https://github.com/PennyLaneAI/pennylane/pull/7206)
  [(#7617)](https://github.com/PennyLaneAI/pennylane/pull/7617)

  As input, :class:`~.SelectPauliRot` requires the `angles` of rotation to be applied to the target 
  qubit for each control register configuration, as well as the `control_wires`, the `target_wire`,
  and the axis of rotation (`rot_axis`) for which each rotation is performed (the default is the `"Z"` 
  axis).

  ```python
  import numpy as np
  angles = np.array([1.0, 2.0, 3.0, 4.0])

  wires = qml.registers({"control": 2, "target": 1})
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.SelectPauliRot(
        angles,
        control_wires=wires["control"],
        target_wire=wires["target"],
        rot_axis="Z"
      )
      return qml.state()
  ```
  
  ```pycon
  >>> print(qml.draw(circuit, level="device")())
  0: ─────────────────────────╭●───────────────╭●─┤ ╭State
  1: ───────────╭●────────────│──╭●────────────│──┤ ├State
  2: ──RZ(2.50)─╰X──RZ(-0.50)─╰X─╰X──RZ(-1.00)─╰X─┤ ╰State
  ```

* The decompositions of :class:`~.SingleExcitation`, :class:`~.SingleExcitationMinus` and 
  :class:`~.SingleExcitationPlus` have been made more efficient by reducing the number of rotations 
  gates and ``CNOT``, ``CZ``, and ``CY`` gates (where applicable). This leads to lower circuit depth 
  when decomposing these gates.
  [(#7771)](https://github.com/PennyLaneAI/pennylane/pull/7771)

<h4>QSVT & QSP angle solver for large polynomials 🕸️</h4>

Effortlessly perform QSVT and QSP with polynomials of large degrees, using our new iterative angle 
solver.

* A new iterative angle solver for QSVT and QSP is available in the 
  :func:`poly_to_angles <pennylane.poly_to_angles>` function, designed for angle computation for 
  polynomials with degrees larger than 1000.
  [(6694)](https://github.com/PennyLaneAI/pennylane/pull/6694)

  Simply set `angle_solver="iterative"` in the :func:`poly_to_angles  <pennylane.poly_to_angles>` 
  function to use it.

  ```python
  import numpy as np

  # P(x) = x - 0.5 x^3 + 0.25 x^5
  poly = np.array([0, 1.0, 0, -1/2, 0, 1/4])

  qsvt_angles = qml.poly_to_angles(poly, "QSVT", angle_solver="iterative")
  ```

  ```pycon
  >>> print(qsvt_angles)
  [-4.72195208  1.59759022  1.12953398  1.12953403  1.59759046 -0.00956271]
  ```

  This functionality can also be accessed directly from :func:`qml.qsvt <pennylane.qsvt>` with the 
  same keyword argument:

  ```python
  # P(x) = -x + 0.5 x^3 + 0.5 x^5
  poly = np.array([0, -1, 0, 0.5, 0, 0.5])

  hamiltonian = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1) @ qml.Z(2)])

  dev = qml.device("default.qubit")
  @qml.qnode(dev)
  def circuit():
      qml.qsvt(
          hamiltonian, poly, encoding_wires=[0], block_encoding="prepselprep", angle_solver="iterative"
      )
      return qml.state()

  matrix = qml.matrix(circuit, wire_order=[0, 1, 2])()
  ```

  ```pycon
  >>> print(matrix[:4, :4].real)
  [[-0.16253996  0.         -0.37925991  0.        ]
   [ 0.         -0.16253996  0.          0.37925991]
   [-0.37925991  0.          0.16253996  0.        ]
   [ 0.          0.37925991  0.          0.16253996]]
  ```

<h4>Qualtran integration 🔗</h4>

* It's now possible to convert PennyLane circuits and operators to 
  [Qualtran](https://qualtran.readthedocs.io/en/latest/) circuits and Bloqs with the new 
  :func:`qml.to_bloq <pennylane.to_bloq>` function. This function translates PennyLane circuits (qfuncs or 
  QNodes) and operations into equivalent
  [Qualtran bloqs](https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library), 
  enabling a new way to estimate 
  the resource requirements of PennyLane quantum circuits via Qualtran's abstractions and tools. 
  [(#7197)](https://github.com/PennyLaneAI/pennylane/pull/7197)
  [(#7604)](https://github.com/PennyLaneAI/pennylane/pull/7604)
  [(#7536)](https://github.com/PennyLaneAI/pennylane/pull/7536)
  [(#7814)](https://github.com/PennyLaneAI/pennylane/pull/7814)

  :func:`qml.to_bloq <pennylane.to_bloq>` can be used in the following ways:

  * Wrap PennyLane circuits and operations to give them Qualtran features, like obtaining
    [bloq_counts](https://qualtran.readthedocs.io/en/latest/reference/qualtran/Bloq.html#:~:text=bloq_counts) 
    and drawing a 
    [call_graph](https://qualtran.readthedocs.io/en/latest/drawing/drawing_call_graph.html), but 
    preserve PennyLane's definition of the circuit/operator. This is done by setting `map_ops` to `False`, 
    which instead wraps operations as a :class:`~.ToBloq`:
  
    ```pycon
    >>> def circuit():
    ...     qml.X(0)
    ...     qml.Y(1)
    ...     qml.Z(2)
    ...
    >>> cbloq = qml.to_bloq(circuit, map_ops=False)
    >>> type(cbloq)
    pennylane.io.qualtran_io.ToBloq
    >>> cbloq.bloq_counts()
    {XGate(): 1, ZGate(): 1, YGate(): 1}
    ```

  * Use smart default mapping of PennyLane circuits and operations to Qualtran Bloqs by setting 
    `map_ops=True` (the default value):

    ```pycon
    >>> PL_op = qml.X(0)
    >>> qualtran_op = qml.to_bloq(PL_op)
    >>> type(qualtran_op)
    qualtran.bloqs.basic_gates.x_basis.XGate
    ```

  * Use custom user-defined mapping of PennyLane circuits and operations to Qualtran Bloqs by 
    setting a `custom_mapping` dictionary:

    ```python
    from qualtran.bloqs.basic_gates import XGate

    def circuit():
        qml.QubitUnitary([[0, 1],[1, 0]], wires=0)
        qml.QubitUnitary([[0, 1],[1, 0]], wires=0)
        qml.QubitUnitary([[0, 1],[1, 0]], wires=0)
    ```

    ```pycon
    >>> PL_op = qml.QubitUnitary([[0, 1],[1, 0]], wires=0)
    >>> qualtran_op = XGate()
    >>> custom_map = {PL_op: qualtran_op}
    >>> bloq = qml.to_bloq(circuit, custom_mapping=custom_map)
    >>> bloq.bloq_counts()
    {XGate(): 3}
    ```

<h4>Resource-efficient Clifford-T decompositions 🍃</h4>

* The [Ross-Selinger algorithm](https://arxiv.org/abs/1403.2975),
  also known as Gridsynth, can now be accessed in :func:`~.clifford_t_decomposition` by setting
  `method="gridsynth"`. This is a newer Clifford-T decomposition method that can produce orders of 
  magnitude fewer gates than using `method="sk"` (Solovay-Kitaev algorithm). 
  [(#7588)](https://github.com/PennyLaneAI/pennylane/pull/7588)
  [(#7641)](https://github.com/PennyLaneAI/pennylane/pull/7641)
  [(#7611)](https://github.com/PennyLaneAI/pennylane/pull/7611)
  [(#7711)](https://github.com/PennyLaneAI/pennylane/pull/7711)
  [(#7770)](https://github.com/PennyLaneAI/pennylane/pull/7770)
  [(#7791)](https://github.com/PennyLaneAI/pennylane/pull/7791)

  In the following example, decomposing with `method="gridsynth"` instead of `method="sk"` gives a
  significant reduction in overall gate counts, specifically the :class:`~.T` count:

  ```python
  @qml.qnode(qml.device("lightning.qubit", wires=2))
  def circuit():

      qml.RX(0.12, 0)
      qml.CNOT([0, 1])
      qml.RY(0.34, 0)

      return qml.expval(qml.Z(0))
  ```

  We can inspect the gate counts resulting from both decomposition methods with :func:`~.specs`:

  ```pycon
  >>> gridsynth_circuit = qml.clifford_t_decomposition(circuit, method="gridsynth")
  >>> sk_circuit = qml.clifford_t_decomposition(circuit, method="sk")
  >>> gridsynth_specs = qml.specs(gridsynth_circuit)()["resources"]
  >>> sk_specs = qml.specs(sk_circuit)()["resources"]
  >>> print(gridsynth_specs.num_gates, sk_specs.num_gates)
  239 47942
  >>> print(gridsynth_specs.gate_types['T'], sk_specs.gate_types['T'])
  90 8044
  ```

<h4>OpenQASM 🤝 PennyLane</h4>

PennyLane now offers improved support for [OpenQASM 2.0 & 3.0](https://openqasm.com/).

* Use the new :func:`qml.from_qasm3 <pennylane.from_qasm3>` function to convert your OpenQASM 3.0 
  circuits into quantum functions which can then be loaded into QNodes and executed.
  [(#7495)](https://github.com/PennyLaneAI/pennylane/pull/7495)
  [(#7486)](https://github.com/PennyLaneAI/pennylane/pull/7486)
  [(#7488)](https://github.com/PennyLaneAI/pennylane/pull/7488)
  [(#7593)](https://github.com/PennyLaneAI/pennylane/pull/7593)
  [(#7498)](https://github.com/PennyLaneAI/pennylane/pull/7498)
  [(#7469)](https://github.com/PennyLaneAI/pennylane/pull/7469)
  [(#7543)](https://github.com/PennyLaneAI/pennylane/pull/7543)
  [(#7783)](https://github.com/PennyLaneAI/pennylane/pull/7783)
  [(#7789)](https://github.com/PennyLaneAI/pennylane/pull/7789)
  [(#7802)](https://github.com/PennyLaneAI/pennylane/pull/7802)

  ```python
  import pennylane as qp

  dev = qml.device("default.qubit", wires=[0, 1, 2])
  
  @qml.qnode(dev)
  def my_circuit():
      qml.from_qasm3(
          """
          qubit q0; 
          qubit q1;
          qubit q2;
  
          float theta = 0.2;
          int power = 2;
  
          ry(theta / 2) q0; 
          rx(theta) q1; 
          pow(power) @ x q0;
  
          def random(qubit q) -> bit {
            bit b = "0";
            h q;
            measure q -> b;
            return b;
          }
  
          bit m = random(q2);
  
          if (m) {
            int i = 0;
            while (i < 5) {
              i = i + 1;
              rz(i) q1;
              break;
            }
          }
          """,
          {'q0': 0, 'q1': 1, 'q2': 2},
      )()
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(my_circuit)())
  0: ──RY(0.10)──X²────────────┤  <Z>
  1: ──RX(0.20)───────RZ(1.00)─┤     
  2: ──H─────────┤↗├──║────────┤     
                  ╚═══╝      
  ```
  
  Some gates and operations in OpenQASM 3.0 programs are not currently supported. For more details, please consult the documentation
  for :func:`qml.from_qasm3 <pennylane.from_qasm3>` and ensure that you have installed `openqasm3` and `'openqasm3[parser]'`
  in your environment by following the [OpenQASM 3.0 installation instructions](https://pypi.org/project/openqasm3/).

* The new :func:`qml.to_openqasm <pennylane.to_openqasm>` function enables conversion of PennyLane 
  circuits to OpenQASM 2.0 programs.
  [(#7393)](https://github.com/PennyLaneAI/pennylane/pull/7393)

  Consider this simple circuit in PennyLane:

  ```python
  from functools import partial

  dev = qml.device("default.qubit", wires=2)

  @partial(qml.set_shots, shots=100)
  @qml.qnode(dev)
  def circuit(theta, phi):
      qml.RX(theta, wires=0)
      qml.CNOT(wires=[0,1])
      qml.RZ(phi, wires=1)
      return qml.sample()
  ```

  This can be easily converted to OpenQASM 2.0 with :func:`qml.to_openqasm <pennylane.to_openqasm>`:
  
  ```pycon
  >>> openqasm_circ = qml.to_openqasm(circuit)(1.2, 0.9)
  >>> print(openqasm_circ)
  OPENQASM 2.0;
  include "qelib1.inc";
  qreg q[2];
  creg c[2];
  rx(1.2) q[0];
  cx q[0],q[1];
  rz(0.9) q[1];
  measure q[0] -> c[0];
  measure q[1] -> c[1];
  ```

<h3>Improvements 🛠</h3>

<h4>A quantum optimizer that works with QJIT</h4>

* Leveraging quantum just-in-time compilation to optimize parameterized hybrid workflows with the quantum 
  natural gradient optimizer is now possible with the new :class:`~.QNGOptimizerQJIT` optimizer. 
  [(#7452)](https://github.com/PennyLaneAI/pennylane/pull/7452)
  
  The :class:`~.QNGOptimizerQJIT` optimizer offers a `jax.jit`- and :func:`qml.qjit <pennylane.qjit>`-compatible analogue to the existing 
  :class:`~.QNGOptimizer` with an Optax-like interface:

  ```python
  import jax.numpy as jnp

  @qml.qjit(autograph=True)
  def workflow():
      dev = qml.device("lightning.qubit", wires=2)
  
      @qml.qnode(dev)
      def circuit(params):
          qml.RX(params[0], wires=0)
          qml.RY(params[1], wires=1)
          return qml.expval(qml.Z(0) + qml.X(1))
  
      opt = qml.QNGOptimizerQJIT(stepsize=0.2)
  
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

<h4>Resource-efficient decompositions 🔎</h4>

* With graph-based decomposition enabled via :func:`~.decomposition.enable_graph`, the 
  :func:`~.transforms.decompose` transform now supports weighting gates in the target `gate_set`, 
  allowing for preferential treatment of certain gates in a target `gate_set` over others.
  [(#7389)](https://github.com/PennyLaneAI/pennylane/pull/7389)

  Gates specified in `gate_set` can be given a numerical weight associated with their effective cost 
  to have in a circuit:
  
  * Gate weights that are greater than 1 indicate a *greater cost* (less preferred).
  * Gate weights that are less than 1 indicate a *lower cost* (more preferred).

  Consider the following toy example, where `CZ` gates are highly preferred to decompose into, but
  `H` and `CRZ` gates are quite costly.

  ```python
  from functools import partial

  qml.decomposition.enable_graph()
  
  @partial(
      qml.transforms.decompose, gate_set={
          qml.Toffoli: 1.23, qml.RX: 4.56, qml.CZ: 0.01, qml.H: 420, qml.CRZ: 100
      }
  )
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.CRX(0.1, wires=[0, 1])
      qml.Toffoli(wires=[0, 1, 2])
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())

  0: ───────────╭●────────────╭●─╭●─┤  <Z>
  1: ──RX(0.05)─╰Z──RX(-0.05)─╰Z─├●─┤     
  2: ────────────────────────────╰X─┤     
  ```

  By reducing the `H` and `CRZ` weights, the circuit decomposition changes:

  ```python
  qml.decomposition.enable_graph()

  @partial(
      qml.transforms.decompose, gate_set={
          qml.Toffoli: 1.23, qml.RX: 4.56, qml.CZ: 0.01, qml.H: 0.1, qml.CRZ: 0.1
      }
  )
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.CRX(0.1, wires=[0, 1])
      qml.Toffoli(wires=[0, 1, 2])
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())

  0: ────╭●───────────╭●─┤  <Z>
  1: ──H─╰RZ(0.10)──H─├●─┤     
  2: ─────────────────╰X─┤  
  ```

* Decomposition rules that can be accessed with the new graph-based decomposition system have been
  implemented for the following operators:

  * :class:`~.QubitUnitary`
    [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

  * :class:`~.ControlledQubitUnitary`
    [(#7371)](https://github.com/PennyLaneAI/pennylane/pull/7371)

  * :class:`~.DiagonalQubitUnitary`
    [(#7625)](https://github.com/PennyLaneAI/pennylane/pull/7625)

  * :class:`~.MultiControlledX`
    [(#7405)](https://github.com/PennyLaneAI/pennylane/pull/7405)

  * :class:`~.ops.Exp` 
    [(#7489)](https://github.com/PennyLaneAI/pennylane/pull/7489)
    Specifically, the following decompositions have been added:
    
    - Suzuki-Trotter decomposition when the `num_steps` keyword argument is specified.
    - Decomposition to a :class:`~.PauliRot` when the base is a single-term Pauli word.

  * :class:`~.PCPhase`
    [(#7591)](https://github.com/PennyLaneAI/pennylane/pull/7591)

  * :class:`~.QuantumPhaseEstimation`
    [(#7637)](https://github.com/PennyLaneAI/pennylane/pull/7637)

  * :class:`~.BasisRotation`
    [(#7074)](https://github.com/PennyLaneAI/pennylane/pull/7074)

  * :class:`~.PhaseAdder`
    [(#7070)](https://github.com/PennyLaneAI/pennylane/pull/7070)

  * :class:`~.IntegerComparator`
    [(#7636)](https://github.com/PennyLaneAI/pennylane/pull/7636)

* With graph-based decomposition enabled with :func:`~.decomposition.enable_graph`, a new 
  decomposition rule that uses a single work wire for decomposing multi-controlled operators can be
  accessed.
  [(#7383)](https://github.com/PennyLaneAI/pennylane/pull/7383)

* With graph-based decomposition enabled with :func:`~.decomposition.enable_graph`, a new decorator 
  called :func:`~.decomposition.register_condition` can be used to bind a condition to a 
  decomposition rule denoting when it is applicable.
  [(#7439)](https://github.com/PennyLaneAI/pennylane/pull/7439)

  The condition should be a function that takes the resource parameters of an operator as arguments 
  and returns `True` or `False` based on whether these parameters satisfy the condition for when 
  this rule can be applied.

  Here is an example of adding a decomposition rule to :class:`~.QubitUnitary`, where the condition
  for which this decomposition rule applies is when the number of wires :class:`~.QubitUnitary` acts
  on is exactly one:

  ```python
  from pennylane.math.decomposition import zyz_rotation_angles
  
  qml.decomposition.enable_graph()

  # The parameters must be consistent with ``qml.QubitUnitary.resource_keys``
  def _zyz_condition(num_wires):
    return num_wires == 1

  @qml.register_condition(_zyz_condition)
  @qml.register_resources({qml.RZ: 2, qml.RY: 1, qml.GlobalPhase: 1})
  def zyz_decomposition(U, wires, **__):
      # Assumes that U is a 2x2 unitary matrix
      phi, theta, omega, phase = zyz_rotation_angles(U, return_global_phase=True)
      qml.RZ(phi, wires=wires[0])
      qml.RY(theta, wires=wires[0])
      qml.RZ(omega, wires=wires[0])
      qml.GlobalPhase(-phase)
  
  # This decomposition will be ignored for `QubitUnitary` on more than one wire.
  qml.add_decomps(qml.QubitUnitary, zyz_decomposition)
  ```

* Symbolic operator types (e.g., `Adjoint`, `Controlled`, and `Pow`) can now be specified as strings
  in various parts of the new graph-based decomposition system:

  * The `gate_set` argument of the :func:`~.transforms.decompose` transform now supports adding 
    symbolic operators in the target gate set.
    [(#7331)](https://github.com/PennyLaneAI/pennylane/pull/7331)

    ```python
    from functools import partial

    qml.decomposition.enable_graph()
    
    @partial(qml.transforms.decompose, gate_set={"T", "Adjoint(T)", "H", "CNOT"})
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.Toffoli(wires=[0, 1, 2])
    ```
    ```pycon
    >>> print(qml.draw(circuit)())
    0: ───────────╭●───────────╭●────╭●──T──╭●─┤
    1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
    2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤
    ```

  * Symbolic operator types can now be given as strings to the `op_type` argument of 
    :func:`~.decomposition.add_decomps`, or as keys of the dictionaries passed to the `alt_decomps` 
    and `fixed_decomps` arguments of the :func:`~.transforms.decompose` transform, allowing custom 
    decomposition rules to be defined and registered for symbolic operators.
    [(#7347)](https://github.com/PennyLaneAI/pennylane/pull/7347)
    [(#7352)](https://github.com/PennyLaneAI/pennylane/pull/7352)
    [(#7362)](https://github.com/PennyLaneAI/pennylane/pull/7362)
    [(#7499)](https://github.com/PennyLaneAI/pennylane/pull/7499)

    ```python
    @qml.register_resources({qml.RY: 1})
    def my_adjoint_ry(phi, wires, **_):
        qml.RY(-phi, wires=wires)

    @qml.register_resources({qml.RX: 1})
    def my_adjoint_rx(phi, wires, **__):
        qml.RX(-phi, wires)

    # Registers a decomposition rule for the adjoint of RY globally
    qml.add_decomps("Adjoint(RY)", my_adjoint_ry)

    @partial(
        qml.transforms.decompose,
        gate_set={"RX", "RY", "CNOT"},
        fixed_decomps={"Adjoint(RX)": my_adjoint_rx}
    )
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.adjoint(qml.RX(0.5, wires=[0]))
        qml.CNOT(wires=[0, 1])
        qml.adjoint(qml.RY(0.5, wires=[1]))
        return qml.expval(qml.Z(0))
    ```
    ```pycon
    >>> print(qml.draw(circuit)())
    0: ──RX(-0.50)─╭●────────────┤  <Z>
    1: ────────────╰X──RY(-0.50)─┤
    ```

* A `work_wire_type` argument has been added to :func:`~pennylane.ctrl` and 
  :class:`~pennylane.ControlledQubitUnitary` for more fine-grained control over the type of work 
  wire used in their decompositions.
  [(#7612)](https://github.com/PennyLaneAI/pennylane/pull/7612)

* The :func:`~.transforms.decompose` transform now accepts a `stopping_condition` argument with 
  graph-based decomposition enabled, which must be a function that returns `True` if an operator 
  does not need to be decomposed (it meets the requirements as described in `stopping_condition`).
  See the documentation for more details.
  [(#7531)](https://github.com/PennyLaneAI/pennylane/pull/7531)

* Two-qubit `QubitUnitary` gates no longer decompose into fundamental rotation gates; it now 
  decomposes into single-qubit `QubitUnitary` gates. This allows the graph-based decomposition 
  system to further decompose single-qubit unitary gates more flexibly using different rotations.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* A new decomposition for two-qubit unitaries has been implemented in :func:`~.ops.two_qubit_decomposition`.
  It ensures the correctness of the decomposition in some edge cases but uses 3 CNOT gates even if 2 CNOTs
  would suffice theoretically.
  [(#7474)](https://github.com/PennyLaneAI/pennylane/pull/7474)

* The `gate_set` argument of :func:`~.transforms.decompose` now accepts `"X"`, `"Y"`, `"Z"`, `"H"`, 
  `"I"` as aliases for `"PauliX"`, `"PauliY"`, `"PauliZ"`, `"Hadamard"`, and `"Identity"`. These 
  aliases are also recognized as part of symbolic operators. For example, `"Adjoint(H)"` is now 
  accepted as an alias for `"Adjoint(Hadamard)"`.
  [(#7331)](https://github.com/PennyLaneAI/pennylane/pull/7331)

<h4>Setting shots 🔁</h4>

* A new QNode transform called :func:`~.transforms.set_shots` has been added to set or update the 
  number of shots to be performed, overriding shots specified in the device.
  [(#7337)](https://github.com/PennyLaneAI/pennylane/pull/7337)
  [(#7358)](https://github.com/PennyLaneAI/pennylane/pull/7358)
  [(#7415)](https://github.com/PennyLaneAI/pennylane/pull/7415)
  [(#7500)](https://github.com/PennyLaneAI/pennylane/pull/7500)
  [(#7627)](https://github.com/PennyLaneAI/pennylane/pull/7627)

  The :func:`~.workflow.set_shots` transform can be used as a decorator:

  ```python
  @partial(qml.set_shots, shots=2)
  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit():
      qml.RX(1.23, wires=0)
      return qml.sample(qml.Z(0))
  ```

  ```pycon
  >>> circuit()
  array([1., -1.])
  ```
  
  Or, it can be used in-line to update a circuit's `shots`:

  ```pycon
  >>> new_circ = qml.set_shots(circuit, shots=(4, 10)) # shot vector
  >>> new_circ()
  (array([-1.,  1., -1.,  1.]), array([ 1.,  1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1.]))
  ```

<h4>QChem</h4>

* The `qchem` module has been upgraded with new functions to construct a vibrational Hamiltonian in 
  the Christiansen representation. 
  [(#7491)](https://github.com/PennyLaneAI/pennylane/pull/7491)
  [(#7596)](https://github.com/PennyLaneAI/pennylane/pull/7596)
  [(#7785)](https://github.com/PennyLaneAI/pennylane/pull/7785)

  The new functions :func:`christiansen_hamiltonian` and :func:`qml.qchem.christiansen_bosonic` can
  be used to create the qubit and bosonic form of the Christiansen Hamiltonian, respectively. These
  functions need input parameters that can be easily obtained by using the
  :func:`christiansen_integrals` and :func:`vibrational_pes` functions. Similarly, a Christiansen
  dipole operator can be created by using the :func:`christiansen_dipole` and
  :func:`christiansen_integrals_dipole` functions.

  ```python
  import numpy as np

  symbols  = ['H', 'F']
  geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
  mol = qml.qchem.Molecule(symbols, geometry)
  pes = qml.qchem.vibrational_pes(mol, optimize=False)
  ham = qml.qchem.vibrational.christiansen_hamiltonian(pes, n_states = 4)
  ```

  ```pycon
  >>> ham
  (
      0.08527499987546708 * I(0)
    + -0.0051774006335491545 * Z(0)
    + 0.0009697024705108074 * (X(0) @ X(1))
    + 0.0009697024705108074 * (Y(0) @ Y(1))
    + 0.0002321787923591865 * (X(0) @ X(2))
    + 0.0002321787923591865 * (Y(0) @ Y(2))
    + 0.0008190498635406456 * (X(0) @ X(3))
    + 0.0008190498635406456 * (Y(0) @ Y(3))
    + -0.015699890427524253 * Z(1)
    + 0.002790002362847834 * (X(1) @ X(2))
    + 0.002790002362847834 * (Y(1) @ Y(2))
    + 0.000687929225764568 * (X(1) @ X(3))
    + 0.000687929225764568 * (Y(1) @ Y(3))
    + -0.026572392417060237 * Z(2)
    + 0.005239546276220405 * (X(2) @ X(3))
    + 0.005239546276220405 * (Y(2) @ Y(3))
    + -0.037825316397333435 * Z(3)
  )
  ```  

<h4>Experimental FTQC module</h4>

* Commutation rules for a Clifford gate set (`qml.H`, `qml.S`, `qml.CNOT`) have been added to the 
  `ftqc.pauli_tracker` module, accessible via the `commute_clifford_op` function.
  [(#7444)](https://github.com/PennyLaneAI/pennylane/pull/7444)

* Offline byproduct correction support has been added to the `ftqc` module.
  [(#7447)](https://github.com/PennyLaneAI/pennylane/pull/7447)

* The `ftqc` module `measure_arbitrary_basis`, `measure_x` and `measure_y` functions can now be 
  captured when program capture is enabled.
  [(#7219)](https://github.com/PennyLaneAI/pennylane/pull/7219)
  [(#7368)](https://github.com/PennyLaneAI/pennylane/pull/7368)

* Functions called `pauli_to_xz`, `xz_to_pauli` and `pauli_prod` that are related to `xz`-encoding 
  have been added to the `ftqc` module.
  [(#7433)](https://github.com/PennyLaneAI/pennylane/pull/7433)

* A new transform called `convert_to_mbqc_formalism` has been added to the `ftqc` module to convert 
  a circuit already expressed in a limited, compatible gate set into the MBQC formalism. Circuits 
  can be converted to the relevant gate set with the `convert_to_mbqc_gateset` transform.
  [(#7355)](https://github.com/PennyLaneAI/pennylane/pull/7355)
  [(#7586)](https://github.com/PennyLaneAI/pennylane/pull/7586)

* The `RotXZX` operation has been added to the `ftqc` module to support the definition of a 
  universal gate set that can be translated to the MBQC formalism.
  [(#7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

<h4>Other improvements</h4>

* `qml.evolve` now errors out if the first argument is not a valid type.
  [(#7768)](https://github.com/PennyLaneAI/pennylane/pull/7768)

* `qml.PauliError` now accepts Pauli strings that include the identity operator.
  [(#7760)](https://github.com/PennyLaneAI/pennylane/pull/7760)

* Caching with finite shots now always warns about the lack of expected noise.
  [(#7644)](https://github.com/PennyLaneAI/pennylane/pull/7644)

* `cache` now defaults to `"auto"` with `qml.execute`, matching the behavior of `QNode` and 
  increasing the performance of using `qml.execute` for standard executions.
  [(#7644)](https://github.com/PennyLaneAI/pennylane/pull/7644)

* `qml.grad` and `qml.jacobian` can now handle inputs with dynamic shapes being captured into plxpr.
  [(#7544)](https://github.com/PennyLaneAI/pennylane/pull/7544/)

* The drawing of `GlobalPhase`, `ctrl(GlobalPhase)`, `Identity` and `ctrl(Identity)` operations has 
  been improved. The labels are grouped together as in other multi-qubit operations, and the drawing 
  no longer depends on the wires of `GlobalPhase` or `Identity`. Control nodes of controlled global 
  phases and identities no longer receive the operator label, which is in line with other controlled 
  operations.
  [(#7457)](https://github.com/PennyLaneAI/pennylane/pull/7457)

* The decomposition of :class:`~.PCPhase` is now significantly more efficient for more than 2 qubits.
  [(#7166)](https://github.com/PennyLaneAI/pennylane/pull/7166)

* The decomposition of :class:`~.IntegerComparator` is now significantly more efficient.
  [(#7636)](https://github.com/PennyLaneAI/pennylane/pull/7636)

* :class:`~.QubitUnitary` now supports a decomposition that is compatible with an arbitrary number 
  of qubits. This represents a fundamental improvement over the previous implementation, which was
  limited to two-qubit systems.
  [(#7277)](https://github.com/PennyLaneAI/pennylane/pull/7277)

* Setting up the configuration of a workflow, including the determination of the best diff method, 
  is now done *after* user transforms have been applied. This allows transforms to update the shots 
  and change measurement processes with fewer issues.
  [(#7358)](https://github.com/PennyLaneAI/pennylane/pull/7358)
  [(#7461)](https://github.com/PennyLaneAI/pennylane/pull/7461)

* The decomposition of `DiagonalQubitUnitary` has been updated from a recursive decomposition into a 
  smaller `DiagonalQubitUnitary` and a `SelectPauliRot` operation. This is a known decomposition 
  from [Theorem 7 in Shende et al.](https://arxiv.org/abs/quant-ph/0406176) that contains fewer 
  gates.
  [(#7370)](https://github.com/PennyLaneAI/pennylane/pull/7370)
 
* An experimental integration for a Python compiler using [xDSL](https://xdsl.dev/index) has been introduced.
  This is similar to 
  [Catalyst's MLIR dialects](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html#mlir-dialects-in-catalyst), 
  but it is coded in Python instead of C++. Compiler passes written using xDSL can be registered as 
  compatible passes via the `@compiler_transform` decorator.
  [(#7509)](https://github.com/PennyLaneAI/pennylane/pull/7509)
  [(#7357)](https://github.com/PennyLaneAI/pennylane/pull/7357)
  [(#7367)](https://github.com/PennyLaneAI/pennylane/pull/7367)
  [(#7462)](https://github.com/PennyLaneAI/pennylane/pull/7462)
  [(#7470)](https://github.com/PennyLaneAI/pennylane/pull/7470)
  [(#7510)](https://github.com/PennyLaneAI/pennylane/pull/7510)
  [(#7590)](https://github.com/PennyLaneAI/pennylane/pull/7590)
  [(#7706)](https://github.com/PennyLaneAI/pennylane/pull/7706)

* An xDSL pass called `qml.compiler.python_compiler.transforms.MergeRotationsPass` has been added 
  for applying `merge_rotations` to an xDSL module for the experimental xDSL Python compiler 
  integration.
  [(#7364)](https://github.com/PennyLaneAI/pennylane/pull/7364)
  [(#7595)](https://github.com/PennyLaneAI/pennylane/pull/7595)
  [(#7664)](https://github.com/PennyLaneAI/pennylane/pull/7664)

* An xDSL pass called `qml.compiler.python_compiler.transforms.IterativeCancelInversesPass` has been 
  added for applying `cancel_inverses` iteratively to an xDSL module for the experimental xDSL 
  Python compiler integration. This pass is optimized to cancel self-inverse operations iteratively.
  [(#7363)](https://github.com/PennyLaneAI/pennylane/pull/7363)
  [(#7595)](https://github.com/PennyLaneAI/pennylane/pull/7595)

* PennyLane now supports `jax == 0.6.0` and `0.5.3`.
  [(#6919)](https://github.com/PennyLaneAI/pennylane/pull/6919)
  [(#7299)](https://github.com/PennyLaneAI/pennylane/pull/7299)

* The alias for `Identity` (`I`) is now accessible from `qml.ops`.
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)
  
* A new `allocation` module containing `allocate` and `deallocate` instructions has been added for 
  requesting dynamic wires. This is currently experimental and not integrated into any execution 
  pipelines.
  [(#7704)](https://github.com/PennyLaneAI/pennylane/pull/7704)
  [(#7710)](https://github.com/PennyLaneAI/pennylane/pull/7710)

* Computing the angles for uniformly controlled rotations, used in 
  :class:`~.MottonenStatePreparation` and :class:`~.SelectPauliRot` now takes much less 
  computational effort and memory.
  [(#7377)](https://github.com/PennyLaneAI/pennylane/pull/7377)

* Classical shadows with mixed quantum states are now computed with a dedicated method that uses an
  iterative algorithm similar to the handling of shadows with state vectors. This makes shadows with 
  density matrices much more performant.
  [(#7458)](https://github.com/PennyLaneAI/pennylane/pull/7458)

* Two new functions called :func:`~.math.convert_to_su2` and :func:`~.math.convert_to_su4` have been 
  added to `qml.math`, which convert unitary matrices to SU(2) or SU(4), respectively, and 
  optionally a global phase.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* `Operator.num_wires` now defaults to `None` to indicate that the operator can be on any number of 
  wires.
  [(#7312)](https://github.com/PennyLaneAI/pennylane/pull/7312)

* Shots can now be overridden for specific `qml.Snapshot` instances via a `shots` keyword argument.
  [(#7326)](https://github.com/PennyLaneAI/pennylane/pull/7326)

  ```python
  from functools import partial

  dev = qml.device("default.qubit", wires=2)

  @partial(qml.set_shots, shots=10)
  @qml.qnode(dev)
  def circuit():
      qml.Snapshot("sample", measurement=qml.sample(qml.X(0)), shots=5)
      return qml.sample(qml.X(0))
  ```

  ```pycon
  >>> qml.snapshots(circuit)()
  {'sample': array([-1., -1., -1., -1., -1.]),
   'execution_results': array([ 1., -1., -1., -1., -1.,  1., -1., -1.,  1., -1.])}
  ```

* PennyLane no longer validates that an operation has at least one wire, as having this check 
  reduced performance by requiring the abstract interface to maintain a list of special 
  implementations.
  [(#7327)](https://github.com/PennyLaneAI/pennylane/pull/7327)

* Two new device-developer transforms have been added to `devices.preprocess`: 
  :func:`~.devices.preprocess.measurements_from_counts` and 
  :func:`~.devices.preprocess.measurements_from_samples`.
  These transforms modify the tape to instead contain a `counts` or `sample` measurement process, 
  deriving the original measurements from the raw counts/samples in post-processing. This allows 
  expanded measurement support for devices that only support counts/samples at execution, like real 
  hardware devices.
  [(#7317)](https://github.com/PennyLaneAI/pennylane/pull/7317)

* The Sphinx version was updated to 8.1. 
  [(7212)](https://github.com/PennyLaneAI/pennylane/pull/7212)

* The `setup.py` package build and install has been migrated to `pyproject.toml`.
  [(#7375)](https://github.com/PennyLaneAI/pennylane/pull/7375)

* GitHub actions and workflows (`rtd.yml`, `readthedocs.yml`, and `docs.yml`) have been updated to 
  use `ubuntu-24.04` runners.
  [(#7396)](https://github.com/PennyLaneAI/pennylane/pull/7396)

* The requirements and pyproject files have been updated to include other packages.  
  [(#7417)](https://github.com/PennyLaneAI/pennylane/pull/7417)

* Documentation checks have been updated to remove duplicate docstring references. 
  [(#7453)](https://github.com/PennyLaneAI/pennylane/pull/7453)

* The performance of `qml.clifford_t_decomposition` has been improved by introducing caching support 
  and changing the default basis set of `qml.ops.sk_decomposition` to `(H, S, T)`, resulting in 
  shorter decomposition sequences.
  [(#7454)](https://github.com/PennyLaneAI/pennylane/pull/7454)

* The decomposition of `qml.BasisState` with capture and the graph-based decomposition systems 
  enabled is more efficient.
  [(#7722)](https://github.com/PennyLaneAI/pennylane/pull/7722)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* The imports of dependencies introduced by ``labs`` functionalities have been modified such that
  these dependencies only have to be installed for the functions that use them, not to use
  ``labs`` functionalities in general. This decouples the various submodules, and even functions
  within the same submodule, from each other.
  [(#7650)](https://github.com/PennyLaneAI/pennylane/pull/7650)

* A new module called :mod:`qml.labs.intermediate_reps <pennylane.labs.intermediate_reps>` has been 
  added to provide functionality to compute intermediate representations for particular circuits.
  The :func:`parity_matrix <pennylane.labs.intermediate_reps.parity_matrix>` function computes
  the parity matrix intermediate representation for `CNOT` circuits, and the
  :func:`phase_polynomial <pennylane.labs.intermediate_reps.phase_polynomial>` function computes
  the phase polynomial intermediate representation for `{CNOT, RZ}` circuits. These efficient 
  intermediate representations are important for CNOT routing algorithms and other quantum 
  compilation routines.
  [(#7229)](https://github.com/PennyLaneAI/pennylane/pull/7229)
  [(#7333)](https://github.com/PennyLaneAI/pennylane/pull/7333)
  [(#7629)](https://github.com/PennyLaneAI/pennylane/pull/7629)
  
* The `pennylane.labs.vibrational` module has been upgraded to use features from the `concurrency` 
  module to perform multiprocess and multithreaded execution. 
  [(#7401)](https://github.com/PennyLaneAI/pennylane/pull/7401)

* A `rowcol` function is now available in `qml.labs.intermediate_reps`. Given the parity matrix of a 
  `CNOT` circuit and a qubit connectivity graph, it synthesizes a possible implementation of the 
  parity matrix that respects the connectivity.
  [(#7394)](https://github.com/PennyLaneAI/pennylane/pull/7394)

* `qml.labs.QubitManager`, `qml.labs.AllocWires`, and `qml.labs.FreeWires` classes have been added 
  to track and manage auxiliary qubits.
  [(#7404)](https://github.com/PennyLaneAI/pennylane/pull/7404)

* A new function called `qml.labs.map_to_resource_op` has been added to map PennyLane Operations to 
  their resource equivalents.
  [(#7434)](https://github.com/PennyLaneAI/pennylane/pull/7434)

* A new class called `qml.labs.Resources` has been added to store and track the quantum resources 
  from a circuit.
  [(#7406)](https://github.com/PennyLaneAI/pennylane/pull/7406)
  
* A new class called `qml.labs.CompressedResourceOp` class has been added to store information about 
  the operator type and parameters for the purposes of resource estimation.
  [(#7408)](https://github.com/PennyLaneAI/pennylane/pull/7408)

* A base class called `qml.labs.ResourceOperator` has been added which will be used to implement all 
  quantum operators for resource estimation.
  [(#7399)](https://github.com/PennyLaneAI/pennylane/pull/7399)
  [(#7526)](https://github.com/PennyLaneAI/pennylane/pull/7526)
  [(#7540)](https://github.com/PennyLaneAI/pennylane/pull/7540)
  [(#7541)](https://github.com/PennyLaneAI/pennylane/pull/7541)
  [(#7584)](https://github.com/PennyLaneAI/pennylane/pull/7584)
  [(#7549)](https://github.com/PennyLaneAI/pennylane/pull/7549)

* A new function called `qml.labs.estimate_resources` has been added which will be used to perform 
  resource estimation on circuits, `qml.labs.ResourceOperator`, and `qml.labs.Resources` objects.
  [(#7407)](https://github.com/PennyLaneAI/pennylane/pull/7407)

* A new class called `qml.labs.resource_estimation.CompactHamiltonian` has been added to unblock the 
  need to pass a full Hamiltonian for the purposes of resource estimation. In addition, similar 
  templates called `qml.labs.resource_estimation.ResourceTrotterCDF` and 
  `qml.labs.resource_estimation.ResourceTrotterTHC`
  have been added, which will be used to perform resource estimation for trotterization of CDF and 
  THC Hamiltonians, respectively.
  [(#7705)](https://github.com/PennyLaneAI/pennylane/pull/7705)

* A new template called `qml.labs.ResourceQubitize` has been added which can be used to perform 
  resource estimation for qubitization of the THC Hamiltonian.
  [(#7730)](https://github.com/PennyLaneAI/pennylane/pull/7730)

* Two new templates called `qml.labs.resource_estimation.ResourceTrotterVibrational` and 
  `qml.labs.resource_estimation.ResourceTrotterVibronic` have been added to perform resource 
  estimation for trotterization of vibrational and vibronic Hamiltonians, respectively.
  [(#7720)](https://github.com/PennyLaneAI/pennylane/pull/7720)

* Several new templates for various algorithms required for supporting compact Hamiltonian 
  development and resource estimation have been added: `qml.ResourceOutOfPlaceSquare`, 
  `qml.ResourcePhaseGradient`, `qml.ResourceOutMultiplier`, `qml.ResourceSemiAdder`, `qml.ResourceBasisRotation`, `qml.ResourceSelect`, and 
  `qml.ResourceQROM`.
  [(#7725)](https://github.com/PennyLaneAI/pennylane/pull/7725)

* A new module called `qml.labs.zxopt` has been added to provide access 
  to the basic optimization passes from [pyzx](https://pyzx.readthedocs.io/en/latest/) for PennyLane 
  circuits.
  [(#7471)](https://github.com/PennyLaneAI/pennylane/pull/7471)
  
    * `basic_optimization` performs peephole 
      optimizations on the circuit and is a useful subroutine for other optimization passes.
    * `full_optimize` optimizes 
      [(Clifford + T)](https://pennylane.ai/compilation/clifford-t-gate-set) circuits.
    * `full_reduce` can optimize arbitrary PennyLane 
      circuits and follows the pipeline described in the 
      [pyzx docs](https://pyzx.readthedocs.io/en/latest/simplify.html).
    * `todd` performs Third Order Duplicate and Destroy 
      (`TODD <https://arxiv.org/abs/1712.01557>`__) via phase polynomials and reduces T gate counts.

* New functionality has been added to create and manipulate product formulas in the `trotter_error` 
  module.
  [(#7224)](https://github.com/PennyLaneAI/pennylane/pull/7224)
 
    * :class:`ProductFormula <pennylane.labs.trotter_error.ProductFormula` allows users to create 
      custom product formulas.
    * :func:`bch_expansion <pennylane.labs.trotter_error.bch_expansion` computes the 
      Baker-Campbell-Hausdorff  expansion of a product formula.
    * :func:`effective_hamiltonian <pennylane.labs.trotter_error.effective_hamiltonian` computes the 
      effective Hamiltonian of a product formula.

* The :func:`perturbation_error <pennylane.labs.trotter_error.perturbation_error>` has been 
  optimized for better performance by grouping commutators by linearity and by using a task-based 
  executor to parallelize the computationally heavy parts of the algorithm.
  [(#7681)](https://github.com/PennyLaneAI/pennylane/pull/7681)
  [(#7790)](https://github.com/PennyLaneAI/pennylane/pull/7790)

* Missing table descriptions for :class:`qml.FromBloq <pennylane.FromBloq>`,
  :func:`qml.qchem.two_particle <pennylane.qchem.two_particle>`,
  and :class:`qml.ParticleConservingU2 <pennylane.ParticleConservingU2>` have been fixed.
  [(#7628)](https://github.com/PennyLaneAI/pennylane/pull/7628)

<h3>Breaking changes 💔</h3>

* Support for gradient keyword arguments as QNode keyword arguments has been removed. Instead please 
  use the new `gradient_kwargs` keyword argument accordingly.
  [(#7648)](https://github.com/PennyLaneAI/pennylane/pull/7648)

* The default value of `cache` is now `"auto"` with `qml.execute`. Like `QNode`, `"auto"` only turns 
  on caching when `max_diff > 1`.
  [(#7644)](https://github.com/PennyLaneAI/pennylane/pull/7644)

* The `return_type` property of `MeasurementProcess` has been removed. Please use `isinstance` for 
  type checking instead.
  [(#7322)](https://github.com/PennyLaneAI/pennylane/pull/7322)

* The `KerasLayer` class in `qml.qnn.keras` has been removed because Keras 2 is no longer actively 
  maintained. Please consider using a different machine learning framework, like 
  `PyTorch <demos/tutorial_qnn_module_torch>`__ or 
  `JAX <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>`__.
  [(#7320)](https://github.com/PennyLaneAI/pennylane/pull/7320)

* The `qml.gradients.hamiltonian_grad` function has been removed because this gradient recipe is no
  longer required with the :doc:`new operator arithmetic system </news/new_opmath>`.
  [(#7302)](https://github.com/PennyLaneAI/pennylane/pull/7302)

* Accessing terms of a tensor product (e.g., `op = X(0) @ X(1)`) via `op.obs` has been removed.
  [(#7324)](https://github.com/PennyLaneAI/pennylane/pull/7324)

* The `mcm_method` keyword argument in `qml.execute` has been removed.
  [(#7301)](https://github.com/PennyLaneAI/pennylane/pull/7301)

* The `inner_transform` and `config` keyword arguments in `qml.execute` have been removed.
  [(#7300)](https://github.com/PennyLaneAI/pennylane/pull/7300)

* `Sum.ops`, `Sum.coeffs`, `Prod.ops` and `Prod.coeffs` have been removed.
  [(#7304)](https://github.com/PennyLaneAI/pennylane/pull/7304)

* Specifying `pipeline=None` with `qml.compile` has been removed.
  [(#7307)](https://github.com/PennyLaneAI/pennylane/pull/7307)

* The `control_wires` argument in `qml.ControlledQubitUnitary` has been removed.
  Furthermore, the `ControlledQubitUnitary` no longer accepts `QubitUnitary` objects as arguments as its `base`.
  [(#7305)](https://github.com/PennyLaneAI/pennylane/pull/7305)

* `qml.tape.TapeError` has been removed.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

<h3>Deprecations 👋</h3>

Here's a list of deprecations made this release. For a more detailed breakdown of deprecations and alternative code to use instead,
please consult the :doc:`deprecations and removals page </development/deprecations>`.

* Python 3.10 support is deprecated and support will be removed in v0.43. Please upgrade to Python 
  3.11 or newer.

* Support for Mac x86 has been removed. This includes Macs running on Intel processors.
  This is because 
  [JAX has also dropped support for it since 0.5.0](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md#jax-050-jan-17-2025),
  with the rationale being that such machines are becoming increasingly scarce. If support for Mac x86 
  platforms is still desired, please install Catalyst v0.11.0, PennyLane v0.41.0, PennyLane-Lightning 
  v0.41.0, and JAX v0.4.28.
  
* Top-level access to `DeviceError`, `PennyLaneDeprecationWarning`, `QuantumFunctionError` and `ExperimentalWarning` have been deprecated and will be removed in v0.43. Please import them from the new `exceptions` module.
  [(#7292)](https://github.com/PennyLaneAI/pennylane/pull/7292)
  [(#7477)](https://github.com/PennyLaneAI/pennylane/pull/7477)
  [(#7508)](https://github.com/PennyLaneAI/pennylane/pull/7508)
  [(#7603)](https://github.com/PennyLaneAI/pennylane/pull/7603)

* `qml.operation.Observable` and the corresponding `Observable.compare` have been deprecated, as
  PennyLane now depends on the more general `Operator` interface instead. The
  `Operator.is_hermitian` property can instead be used to check whether or not it is highly likely
  that the operator instance is Hermitian.
  [(#7316)](https://github.com/PennyLaneAI/pennylane/pull/7316)

* The boolean functions provided in `qml.operation` are deprecated. See the 
  :doc:`deprecations page </development/deprecations>` for equivalent code to use instead. These 
  include `not_tape`, `has_gen`, `has_grad_method`, `has_multipar`, `has_nopar`, `has_unitary_gen`, 
  `is_measurement`, `defines_diagonalizing_gates`, and `gen_is_multi_term_hamiltonian`.
  [(#7319)](https://github.com/PennyLaneAI/pennylane/pull/7319)

* `qml.operation.WiresEnum`, `qml.operation.AllWires`, and `qml.operation.AnyWires` are deprecated. 
  To indicate that an operator can act on any number of wires, `Operator.num_wires = None` should be 
  used instead. This is the default and does not need to be overwritten unless the operator 
  developer wants to add wire number validation.
  [(#7313)](https://github.com/PennyLaneAI/pennylane/pull/7313)

* The :func:`qml.QNode.get_gradient_fn` method is now deprecated. Instead, use 
  :func:`~.workflow.get_best_diff_method` to obtain the differentiation method.
  [(#7323)](https://github.com/PennyLaneAI/pennylane/pull/7323)

<h3>Internal changes ⚙️</h3>

* Jits the `givens_matrix` computation from `BasisRotation` when it is within a jit context, which significantly reduces the program size and compilation time of workflows.
  [(#7823)](https://github.com/PennyLaneAI/pennylane/pull/7823)

* Private code in the `TransformProgram` has been moved to the `CotransformCache` class.
  [(#7750)](https://github.com/PennyLaneAI/pennylane/pull/7750)

* Type hinting in the `workflow` module has been improved.
  [(#7745)](https://github.com/PennyLaneAI/pennylane/pull/7745)

* `mitiq` has been unpinned in the CI.
  [(#7742)](https://github.com/PennyLaneAI/pennylane/pull/7742)

* The `qml.measurements.Shots` class can now handle abstract numbers of shots.
  [(#7729)](https://github.com/PennyLaneAI/pennylane/pull/7729)

* The `jax` and `tensorflow` dependencies for `doc` builds have been updated.
  [(#7667)](https://github.com/PennyLaneAI/pennylane/pull/7667)

* `Pennylane` has been renamed to `pennylane` in the `pyproject.toml` file  to match the expected 
  binary distribution format naming conventions.
  [(#7689)](https://github.com/PennyLaneAI/pennylane/pull/7689)

* The `qml.compiler.python_compiler` submodule has been restructured.
  [(#7645)](https://github.com/PennyLaneAI/pennylane/pull/7645)

* Program capture code has been moved closer to where it is used.
  [(#7608)](https://github.com/PennyLaneAI/pennylane/pull/7608)

* Tests using `OpenFermion` in `tests/qchem` no longer fail with NumPy>=2.0.0.
  [(#7626)](https://github.com/PennyLaneAI/pennylane/pull/7626)

* The `givens_decomposition` function and private helpers from `qchem` have been moved to the `math` 
  module.
  [(#7545)](https://github.com/PennyLaneAI/pennylane/pull/7545)

* Module dependencies in `pennylane` using `tach` have been enforced.
  [(#7185)](https://github.com/PennyLaneAI/pennylane/pull/7185)
  [(#7416)](https://github.com/PennyLaneAI/pennylane/pull/7416)
  [(#7418)](https://github.com/PennyLaneAI/pennylane/pull/7418)
  [(#7429)](https://github.com/PennyLaneAI/pennylane/pull/7429)
  [(#7430)](https://github.com/PennyLaneAI/pennylane/pull/7430)
  [(#7437)](https://github.com/PennyLaneAI/pennylane/pull/7437)
  [(#7504)](https://github.com/PennyLaneAI/pennylane/pull/7504)
  [(#7538)](https://github.com/PennyLaneAI/pennylane/pull/7538)
  [(#7542)](https://github.com/PennyLaneAI/pennylane/pull/7542)
  [(#7667)](https://github.com/PennyLaneAI/pennylane/pull/7667)
  [(#7743)](https://github.com/PennyLaneAI/pennylane/pull/7743)

* With program capture enabled, MCM method validation now happens on execution rather than setup.
  [(#7475)](https://github.com/PennyLaneAI/pennylane/pull/7475)

* A `.git-blame-ignore-revs` file has been added to the PennyLane repository. This file will allow 
  specifying commits that should be ignored in the output of `git blame`. For example, this can be 
  useful when a single commit includes bulk reformatting.
  [(#7507)](https://github.com/PennyLaneAI/pennylane/pull/7507)

* A `.gitattributes` file has been added to standardize LF as the end-of-line character for the 
  PennyLane repository.
  [(#7502)](https://github.com/PennyLaneAI/pennylane/pull/7502)

* `DefaultQubit` now implements `preprocess_transforms` and `setup_execution_config` instead of 
  `preprocess`.
  [(#7468)](https://github.com/PennyLaneAI/pennylane/pull/7468)

* A subset of `pylint` errors have been fixed in the `tests` folder.
  [(#7446)](https://github.com/PennyLaneAI/pennylane/pull/7446)

* Excessively expensive test cases that do not add value in `tests/templates/test_subroutines/` have 
  been reduced or removed.
  [(#7436)](https://github.com/PennyLaneAI/pennylane/pull/7436)

* `pytest-timeout` is no longer used in the PennyLane CI/CD pipeline.
  [(#7451)](https://github.com/PennyLaneAI/pennylane/pull/7451)

* A `RuntimeWarning` raised when using versions of JAX > 0.4.28 has been removed.
  [(#7398)](https://github.com/PennyLaneAI/pennylane/pull/7398)

* Wheel releases for PennyLane now follow the 
  `PyPA binary-distribution format <https://packaging.python.org/en/latest/specifications/binary-distribution-format/>_` 
  guidelines more closely.
  [(#7382)](https://github.com/PennyLaneAI/pennylane/pull/7382)

* `null.qubit` can now support an optional `track_resources` argument which allows it to record which gates are executed.
  [(#7226)](https://github.com/PennyLaneAI/pennylane/pull/7226)
  [(#7372)](https://github.com/PennyLaneAI/pennylane/pull/7372)
  [(#7392)](https://github.com/PennyLaneAI/pennylane/pull/7392)
  [(#7813)](https://github.com/PennyLaneAI/pennylane/pull/7813)

* A new internal module, `qml.concurrency`, is added to support internal use of multiprocess and multithreaded execution of workloads. This also migrates the use of `concurrent.futures` in `default.qubit` to this new design.
  [(#7303)](https://github.com/PennyLaneAI/pennylane/pull/7303)

* Test suites in `tests/transforms/test_defer_measurement.py` now use analytic mocker devices to 
  test numeric results.
  [(#7329)](https://github.com/PennyLaneAI/pennylane/pull/7329)

* A new `pennylane.exceptions` module has been added for custom errors and warnings.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)
  [(#7292)](https://github.com/PennyLaneAI/pennylane/pull/7292)

* Several `__init__.py` files in `math`, `ops`, `qaoa`, `tape` and `templates` have been cleaned up 
  to be explicit in what they import. 
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)
  
* The `Tracker` class has been moved into the `devices` module.
  [(#7281)](https://github.com/PennyLaneAI/pennylane/pull/7281)

* Functions that calculate rotation angles for unitary decompositions have been moved into an 
  internal module called `qml.math.decomposition`.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* Fixed a failing integration test for `qml.QDrift` which multiplied the operators of the 
  decomposition incorrectly to evolve the state.
  [(#7621)](https://github.com/PennyLaneAI/pennylane/pull/7621)

* The decomposition test in `assert_valid` no longer checks the matrix of the decomposition if the 
  operator does not define a matrix representation.
  [(#7655)](https://github.com/PennyLaneAI/pennylane/pull/7655)

<h3>Documentation 📝</h3>

* The documentation for mid-circuit measurements using the Tree Traversal algorithm has been updated
  to reflect supported devices and usage in analytic simulations (see the 
  :doc:`/introduction/dynamic_quantum_circuits` page).
  [(#7691)](https://github.com/PennyLaneAI/pennylane/pull/7691)

* The functions in `qml.qchem.vibrational` have been updated to include additional information about 
  the theory and input arguments.
  [(#6918)](https://github.com/PennyLaneAI/pennylane/pull/6918)

* The usage examples for `qml.decomposition.DecompositionGraph` have been updated.
  [(#7692)](https://github.com/PennyLaneAI/pennylane/pull/7692)

* The entry in the :doc:`/news/program_capture_sharp_bits` has been updated to include additional 
  supported lightning devices (`lightning.kokkos` and `lightning.gpu`).
  [(#7674)](https://github.com/PennyLaneAI/pennylane/pull/7674)

* The circuit drawings for `qml.Select` and `qml.SelectPauliRot` have been updated to include two 
  commonly used symbols for Select-applying, or -multiplexing, an operator. 
  [(#7464)](https://github.com/PennyLaneAI/pennylane/pull/7464)
  
* The entry in the :doc:`/news/program_capture_sharp_bits` page for transforms has been updated; 
  non-native transforms being applied to QNodes wherein operators have dynamic wires can lead to 
  incorrect results.
  [(#7426)](https://github.com/PennyLaneAI/pennylane/pull/7426)

* Fixed the wrong `theta` to `phi` in :class:`~pennylane.IsingXY`.
  [(#7427)](https://github.com/PennyLaneAI/pennylane/pull/7427)

* In the :doc:`/introduction/compiling_circuits` page, in the "Decomposition in stages" section,
  circuit drawings now render in a way that's easier to read.
  [(#7419)](https://github.com/PennyLaneAI/pennylane/pull/7419)

* The entry in the :doc:`/news/program_capture_sharp_bits` page for using program capture with 
  Catalyst has been updated. Instead of using ``qjit(experimental_capture=True)``, Catalyst is now 
  compatible with the global toggles ``qml.capture.enable()`` and ``qml.capture.disable()`` for 
  enabling and disabling program capture.
  [(#7298)](https://github.com/PennyLaneAI/pennylane/pull/7298)

* The simulation technique table in the :doc:`/introduction/dynamic_quantum_circuits` page has been 
  updated to correct an error regarding analytic mode support for the ``tree-traversal`` method; 
  ``tree-traversal`` supports analytic mode.
  [(#7490)](https://github.com/PennyLaneAI/pennylane/pull/7490)

* A warning has been added to the documentation for `qml.snapshots` and `qml.Snapshot`, clarifying 
  that compilation transforms may move operations across a `Snapshot`.
  [(#7746)](https://github.com/PennyLaneAI/pennylane/pull/7746)

* In the :doc:`/development/guide/documentation` page, references to the outdated Sphinx and 
  unsupported Python versions have been updated. This helps ensure contributors follow current 
  standards and avoid compatibility issues.
  [(#7479)](https://github.com/PennyLaneAI/pennylane/pull/7479)

* The documentation of `qml.pulse.drive` has been updated and corrected.
  [(#7459)](https://github.com/PennyLaneAI/pennylane/pull/7459)

* The API list in the documentation has been alphabetized to ensure consistent ordering. 
  [(#7792)](https://github.com/PennyLaneAI/pennylane/pull/7792)

<h3>Bug fixes 🐛</h3>

* Fixes `SelectPauliRot._flatten` and `TemporaryAND._primitve_bind_call`.
  [(#7843)](https://github.com/PennyLaneAI/pennylane/pull/7843)

* Fixes a bug where normalization in `qml.StatePrep` with `normalize=True` was skipped if
  `validate_norm` is set to `False`.
  [(#7835)](https://github.com/PennyLaneAI/pennylane/pull/7835) 

* The :func:`~.transforms.cancel_inverses` transform no longer changes the order of operations that 
  don't have shared wires, providing a deterministic output.
  [(#7328)](https://github.com/PennyLaneAI/pennylane/pull/7328)

* Fixed broken support of `qml.matrix` for a `QNode` when using mixed Torch GPU & CPU data for 
  parametric tensors.
  [(#7775)](https://github.com/PennyLaneAI/pennylane/pull/7775) 

* Fixed `CircuitGraph.iterate_parametrized_layers`, and thus `metric_tensor`, when the same 
  operation occurs multiple times in the circuit.
  [(#7757)](https://github.com/PennyLaneAI/pennylane/pull/7757)

* Fixed a bug with transforms that require the classical Jacobian applied to QNodes, where only
  some arguments are trainable and an intermediate transform does not preserve trainability 
  information.
  [(#7345)](https://github.com/PennyLaneAI/pennylane/pull/7345)

* The `qml.ftqc.ParametricMidMeasureMP` class was unable to accept data from `jax.numpy.array` 
  inputs when specifying the angle, due to the given hashing policy. The implementation was updated 
  to ensure correct hashing behavior for `float`, `numpy.array`, and `jax.numpy.array` inputs.
  [(#7693)](https://github.com/PennyLaneAI/pennylane/pull/7693)

* A bug in `qml.draw_mpl` for circuits with work wires has been fixed. The previously inconsistent 
  mapping for these wires has been resolved, ensuring accurate assignment during drawing.
  [(#7668)](https://github.com/PennyLaneAI/pennylane/pull/7668)

* A bug in `ops.op_math.Prod.simplify()` has been fixed that led to global phases being discarded
  in special cases. Concretely, this problem occurs when Pauli factors combine into the identity
  up to a global phase _and_ there is no Pauli representation of the product operator.
  [(#7671)](https://github.com/PennyLaneAI/pennylane/pull/7671)

* The behaviour of the `qml.FlipSign` operation has been fixed: passing an integer `m` as the wires 
  argument is now interpreted as a single wire (i.e. `wires=[m]`). This is different from the 
  previous interpretation of `wires=range(m)`. 
  Also, the `qml.FlipSign.wires` attribute is now returning the correct `Wires` object as for all other operations in PennyLane.
  [(#7647)](https://github.com/PennyLaneAI/pennylane/pull/7647)

* `qml.equal` now works with `qml.PauliError`s.
  [(#7618)](https://github.com/PennyLaneAI/pennylane/pull/7618)

* The `qml.transforms.cancel_inverses` transform can now be used with `jax.jit`.
  [(#7487)](https://github.com/PennyLaneAI/pennylane/pull/7487)

* `qml.StatePrep` no longer validates the norm of statevectors.
  [(#7615)](https://github.com/PennyLaneAI/pennylane/pull/7615)

* The `qml.PhaseShift` operation is now working correctly with a batch size of 1.
  [(#7622)](https://github.com/PennyLaneAI/pennylane/pull/7622)

* `qml.metric_tensor` can now be calculated with Catalyst present.
  [(#7528)](https://github.com/PennyLaneAI/pennylane/pull/7528)

* The mapping to standard wires (consecutive integers) of `qml.tape.QuantumScript` has been fixed
  to correctly consider work wires that are not used otherwise in the circuit.
  [(#7581)](https://github.com/PennyLaneAI/pennylane/pull/7581)

* Fixed a bug where certain transforms with a native program capture implementation give incorrect 
  results when dynamic wires were present in the circuit. The affected transforms were:
  * :func:`~.transforms.cancel_inverses`
  * :func:`~.transforms.merge_rotations`
  * :func:`~.transforms.single_qubit_fusion`
  * :func:`~.transforms.merge_amplitude_embedding`
  [(#7426)](https://github.com/PennyLaneAI/pennylane/pull/7426)

* The `Operator.pow` method has been fixed to raise to the power of 2 the qutrit operators 
  `~.TShift`, `~.TClock`, and `~.TAdd`.
  [(#7505)](https://github.com/PennyLaneAI/pennylane/pull/7505)

* The queuing behavior of the controlled of a controlled operation has been fixed.
  [(#7532)](https://github.com/PennyLaneAI/pennylane/pull/7532)

* A new decomposition was implemented for two-qubit `QubitUnitary` operators in 
  `two_qubit_decomposition` based on a type-AI Cartan decomposition. It fixes previously faulty edge 
  cases for unitaries that require 2 or 3 CNOT gates. Now, 3 CNOTs are used for both cases, using 
  one more CNOT than theoretically required in the former case.
  [(#7474)](https://github.com/PennyLaneAI/pennylane/pull/7474)

* Fixed a bug in `to_openfermion` where identity qubit-to-wires mapping was not obeyed.
  [(#7332)](https://github.com/PennyLaneAI/pennylane/pull/7332)

* Fixed a bug in the validation of :class:`~.SelectPauliRot` that prevents parameter broadcasting.
  [(#7377)](https://github.com/PennyLaneAI/pennylane/pull/7377)

* Usage of NumPy in `default.mixed` source code has been converted to `qml.math` to avoid
  unnecessary dependency on NumPy and to fix a bug that caused an error when using `default.mixed` 
  with PyTorch and GPUs.
  [(#7384)](https://github.com/PennyLaneAI/pennylane/pull/7384)

* With program capture enabled (`qml.capture.enable()`), `QSVT` no longer treats abstract values as 
  metadata.
  [(#7360)](https://github.com/PennyLaneAI/pennylane/pull/7360)

* A fix was made to `default.qubit` to allow for using `qml.Snapshot` with defer-measurements 
  (`mcm_method="deferred"`).
  [(#7335)](https://github.com/PennyLaneAI/pennylane/pull/7335)

* Fixed the repr for empty `Prod` and `Sum` instances to better communicate the existence of an 
  empty instance.
  [(#7346)](https://github.com/PennyLaneAI/pennylane/pull/7346)

* Fixed a bug where circuit execution fails with ``BlockEncode`` initialized with sparse matrices.
  [(#7285)](https://github.com/PennyLaneAI/pennylane/pull/7285)

* An informative error message has been added if `qml.cond` is used with an abstract condition with
  jitting on `default.qubit` when program capture is enabled.
  [(#7314)](https://github.com/PennyLaneAI/pennylane/pull/7314)

* Fixed a bug where using a ``StatePrep`` operation with `batch_size=1` did not work with 
  ``default.mixed``.
  [(#7280)](https://github.com/PennyLaneAI/pennylane/pull/7280)

* Gradient transforms can now be used in conjunction with batch transforms with all interfaces.
  [(#7287)](https://github.com/PennyLaneAI/pennylane/pull/7287)

* Fixed a bug where the global phase was not being added in the ``QubitUnitary`` decomposition.  
  [(#7244)](https://github.com/PennyLaneAI/pennylane/pull/7244)
  [(#7270)](https://github.com/PennyLaneAI/pennylane/pull/7270)

* Using finite differences with program capture without x64 mode enabled now raises a warning.
  [(#7282)](https://github.com/PennyLaneAI/pennylane/pull/7282)

* When the `mcm_method` is specified to the `"device"`, the `defer_measurements` transform will 
  no longer be applied. Instead, the device will be responsible for all MCM handling.
  [(#7243)](https://github.com/PennyLaneAI/pennylane/pull/7243)

* Fixed coverage of `qml.liealg.CII` and `qml.liealg.AIII`.
  [(#7291)](https://github.com/PennyLaneAI/pennylane/pull/7291)

* Fixed a bug where the phase is used as the wire label for a `qml.GlobalPhase` when capture is 
  enabled.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* Fixed a bug that caused `CountsMP.process_counts` to return results in the computational basis, 
  even if an observable was specified.
  [(#7342)](https://github.com/PennyLaneAI/pennylane/pull/7342)

* Fixed a bug that caused `SamplesMP.process_counts` used with an observable to return a list of 
  eigenvalues for each individual operation in the observable, instead of the overall result.
  [(#7342)](https://github.com/PennyLaneAI/pennylane/pull/7342)

* Fixed a bug where `two_qubit_decomposition` provides an incorrect decomposition for some special 
  matrices.
  [(#7340)](https://github.com/PennyLaneAI/pennylane/pull/7340)

* Fixed a bug where the powers of `qml.ISWAP` and `qml.SISWAP` were decomposed incorrectly.
  [(#7361)](https://github.com/PennyLaneAI/pennylane/pull/7361)

* Returning `MeasurementValue`s from the `ftqc` module's parametric mid-circuit measurements
  (`measure_arbitrary_basis`, `measure_x` and `measure_y`) no longer raises an error in circuits 
  using `diagonalize_mcms`.
  [(#7387)](https://github.com/PennyLaneAI/pennylane/pull/7387)

* Fixed a bug where the :func:`~.transforms.single_qubit_fusion` transform produces a tape that is
  off from the original tape by a global phase.
  [(#7619)](https://github.com/PennyLaneAI/pennylane/pull/7619)

* Fixed a bug where an error is raised from the decomposition graph when the resource params of an operator contains lists.
  [(#7722)](https://github.com/PennyLaneAI/pennylane/pull/7722)

* Fixed a bug with the new `Select` decomposition based on unary iteration. There was an erroneous `print` statement.
  [(#7842)](https://github.com/PennyLaneAI/pennylane/pull/7842)

* Fixes a bug where an operation wrapped in `partial_wires` does not get queued.
  [(#7830)](https://github.com/PennyLaneAI/pennylane/pull/7830)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Ali Asadi,
Utkarsh Azad,
Astral Cai,
Yushao Chen,
Diksha Dhawan,
Marcus Edwards,
Lillian Frederiksen,
Pietropaolo Frisoni,
Simone Gasperini,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Joseph Lee,
Austin Huang,
Anton Naim Ibrahim,
Erick Ochoa Lopez,
William Maxwell,
Luis Alfredo Nuñez Meneses,
Oumarou Oumarou,
Lee J. O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Justin Pickering,
Shuli Shu,
Jay Soni,
Kalman Szenes,
Marc Vandelle,
David Wierichs,
Jake Zaia
