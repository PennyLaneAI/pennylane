
# Release 0.32.0

<h3>New features since last release</h3>

<h4>Encode matrices using a linear combination of unitaries ⛓️️</h4>

* It is now possible to encode an operator `A` into a quantum circuit by decomposing it into
  a linear combination of unitaries using PREP 
  ([qml.StatePrep](https://docs.pennylane.ai/en/stable/code/api/pennylane.StatePrep.html)) and
  SELECT ([qml.Select](https://docs.pennylane.ai/en/stable/code/api/pennylane.Select.html)) 
  routines.
  [(#4431)](https://github.com/PennyLaneAI/pennylane/pull/4431)
  [(#4437)](https://github.com/PennyLaneAI/pennylane/pull/4437)
  [(#4444)](https://github.com/PennyLaneAI/pennylane/pull/4444)
  [(#4450)](https://github.com/PennyLaneAI/pennylane/pull/4450)
  [(#4506)](https://github.com/PennyLaneAI/pennylane/pull/4506)
  [(#4526)](https://github.com/PennyLaneAI/pennylane/pull/4526)

  Consider an operator `A` composed of a linear combination of Pauli terms:

  ```pycon
  >>> A = qml.PauliX(2) + 2 * qml.PauliY(2) + 3 * qml.PauliZ(2)
  ```

  A decomposable block-encoding circuit can be created:

  ```python
  def block_encode(A, control_wires):
      probs = A.coeffs / np.sum(A.coeffs)
      state = np.pad(np.sqrt(probs, dtype=complex), (0, 1))
      unitaries = A.ops
  
      qml.StatePrep(state, wires=control_wires)
      qml.Select(unitaries, control=control_wires)
      qml.adjoint(qml.StatePrep)(state, wires=control_wires)
  ```

  ```pycon
  >>> print(qml.draw(block_encode, show_matrices=False)(A, control_wires=[0, 1]))
  0: ─╭|Ψ⟩─╭Select─╭|Ψ⟩†─┤
  1: ─╰|Ψ⟩─├Select─╰|Ψ⟩†─┤
  2: ──────╰Select───────┤
  ```

  This circuit can be used as a building block within a larger QNode to perform algorithms such as
  [QSVT](https://docs.pennylane.ai/en/stable/code/api/pennylane.QSVT.html) and [Hamiltonian simulation](https://codebook.xanadu.ai/H.6).

* Decomposing a Hermitian matrix into a linear combination of Pauli words via `qml.pauli_decompose` is 
  now faster and differentiable.
  [(#4395)](https://github.com/PennyLaneAI/pennylane/pull/4395)
  [(#4479)](https://github.com/PennyLaneAI/pennylane/pull/4479)
  [(#4493)](https://github.com/PennyLaneAI/pennylane/pull/4493)

  ```python
  def find_coeffs(p):
      mat = np.array([[3, p], [p, 3]])
      A = qml.pauli_decompose(mat)
      return A.coeffs
  ```

  ```pycon
  >>> import jax
  >>> from jax import numpy as np
  >>> jax.jacobian(find_coeffs)(np.array(2.))
  Array([0., 1.], dtype=float32, weak_type=True)
  ```
  
<h4>Monitor PennyLane's inner workings with logging 📃</h4>

* Python-native logging can now be enabled with `qml.logging.enable_logging()`.
  [(#4377)](https://github.com/PennyLaneAI/pennylane/pull/4377)
  [(#4383)](https://github.com/PennyLaneAI/pennylane/pull/4383)

  Consider the following code that is contained in `my_code.py`:

  ```python
  import pennylane as qp
  qml.logging.enable_logging()  # enables logging

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def f(x):
      qml.RX(x, wires=0)
      return qml.state()

  f(0.5)
  ``` 

  Executing `my_code.py` with logging enabled will detail every step in PennyLane's
  pipeline that gets used to run your code.

  ```
  $ python my_code.py
  [1967-02-13 15:18:38,591][DEBUG][<PID 8881:MainProcess>] - pennylane.qnode.__init__()::"Creating QNode(func=<function f at 0x7faf2a6fbaf0>, device=<DefaultQubit device (wires=2, shots=None) at 0x7faf2a689b50>, interface=auto, diff_method=best, expansion_strategy=gradient, max_expansion=10, grad_on_execution=best, mode=None, cache=True, cachesize=10000, max_diff=1, gradient_kwargs={}"
  ...
  ```

  Additional logging configuration settings can be specified by modifying the contents of the logging 
  configuration file, which can be located by running `qml.logging.config_path()`. Follow our
  [logging docs page](https://docs.pennylane.ai/en/latest/introduction/logging.html) for more
  details!

<h4>More input states for quantum chemistry calculations ⚛️</h4>

* Input states obtained from advanced quantum chemistry calculations can be used in a circuit. 
  [(#4427)](https://github.com/PennyLaneAI/pennylane/pull/4427)
  [(#4433)](https://github.com/PennyLaneAI/pennylane/pull/4433)
  [(#4461)](https://github.com/PennyLaneAI/pennylane/pull/4461)
  [(#4476)](https://github.com/PennyLaneAI/pennylane/pull/4476)
  [(#4505)](https://github.com/PennyLaneAI/pennylane/pull/4505)


  Quantum chemistry calculations rely on an initial state that is typically selected to be the
  trivial Hartree-Fock state. For molecules with a complicated electronic structure, using initial
  states obtained from affordable post-Hartree-Fock calculations helps to improve the efficiency of
  the quantum simulations. These calculations can be done with external quantum chemistry libraries
  such as PySCF.

  It is now possible to import a PySCF solver object in PennyLane and extract the corresponding
  wave function in the form of a state vector that can be directly used in a circuit. First, perform
  your classical quantum chemistry calculations and then use the
  [qml.qchem.import_state](https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.import_state.html)
  function to import the solver object and return a state vector.

  ```pycon
  >>> from pyscf import gto, scf, ci
  >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g')
  >>> myhf = scf.UHF(mol).run()
  >>> myci = ci.UCISD(myhf).run()
  >>> wf_cisd = qml.qchem.import_state(myci, tol=1e-1)
  >>> print(wf_cisd)
  [ 0.        +0.j  0.        +0.j  0.        +0.j  0.1066467 +0.j
    1.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
    2.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
   -0.99429698+0.j  0.        +0.j  0.        +0.j  0.        +0.j]
  ```

  The state vector can be implemented in a circuit using `qml.StatePrep`.

  ```pycon
  >>> dev = qml.device('default.qubit', wires=4)
  >>> @qml.qnode(dev)
  ... def circuit():
  ...     qml.StatePrep(wf_cisd, wires=range(4))
  ...     return qml.state()
  >>> print(circuit())
  [ 0.        +0.j  0.        +0.j  0.        +0.j  0.1066467 +0.j
    1.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
    2.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
   -0.99429698+0.j  0.        +0.j  0.        +0.j  0.        +0.j]
  ```

  The currently supported post-Hartree-Fock methods are RCISD, UCISD, RCCSD, and UCCSD which 
  denote restricted (R) and unrestricted (U) configuration interaction (CI) and coupled cluster (CC) 
  calculations with single and double (SD) excitations.

<h4>Reuse and reset qubits after mid-circuit measurements ♻️</h4>

* PennyLane now allows you to define circuits that reuse a qubit after a mid-circuit
  measurement has taken place. Optionally, the wire can also be reset to the :math:`|0\rangle`
  state.
  [(#4402)](https://github.com/PennyLaneAI/pennylane/pull/4402)
  [(#4432)](https://github.com/PennyLaneAI/pennylane/pull/4432)

  Post-measurement reset can be activated by setting `reset=True` when calling
  [qml.measure](https://docs.pennylane.ai/en/stable/code/api/pennylane.measure.html).
  In this version of PennyLane, executing circuits with qubit reuse will result in the
  [defer_measurements](https://docs.pennylane.ai/en/latest/code/api/pennylane.defer_measurements.html)
  transform being applied. This transform replaces each reused wire with an *additional* qubit.
  However, future releases of PennyLane will explore device-level support for qubit reuse without
  consuming additional qubits.

  Qubit reuse and reset is also fully differentiable:

  ```python
  dev = qml.device("default.qubit", wires=4)

  @qml.qnode(dev)
  def circuit(p):
      qml.RX(p, wires=0)
      m = qml.measure(0, reset=True) 
      qml.cond(m, qml.Hadamard)(1)

      qml.RX(p, wires=0)
      m = qml.measure(0)
      qml.cond(m, qml.Hadamard)(1)
      return qml.expval(qml.PauliZ(1))
  ```

  ```pycon
  >>> jax.grad(circuit)(0.4)
  Array(-0.35867804, dtype=float32, weak_type=True)
  ```
  
  You can read more about mid-circuit measurements
  [in the documentation](https://docs.pennylane.ai/en/latest/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations),
  and stay tuned for more mid-circuit measurement features in the next few releases!

<h3>Improvements 🛠</h3>

<h4>A new PennyLane drawing style</h4>

* Circuit drawings and plots can now be created following a PennyLane style.
  [(#3950)](https://github.com/PennyLaneAI/pennylane/pull/3950)

  The `qml.draw_mpl` function accepts a `style='pennylane'` argument to create PennyLane themed
  circuit diagrams:

  ```python
  def circuit(x, z):
      qml.QFT(wires=(0,1,2,3))
      qml.Toffoli(wires=(0,1,2))
      qml.CSWAP(wires=(0,2,3))
      qml.RX(x, wires=0)
      qml.CRZ(z, wires=(3,0))
      return qml.expval(qml.PauliZ(0))

  qml.draw_mpl(circuit, style="pennylane")(1, 1)
  ```
  
  <img src="https://docs.pennylane.ai/en/stable/_images/pennylane_style.png" width=50%/>
  
  PennyLane-styled plots can also be drawn by passing `"pennylane.drawer.plot"` to Matplotlib's
  `plt.style.use` function:

  ```python
  import matplotlib.pyplot as plt

  plt.style.use("pennylane.drawer.plot")
  for i in range(3):
      plt.plot(np.random.rand(10))
  ```

  If the font 
  [Quicksand Bold](https://fonts.google.com/specimen/Quicksand) isn't available, 
  an available default font is used instead. 

<h4>Making operators immutable and PyTrees</h4>

* Any class inheriting from `Operator` is now automatically registered as a pytree with JAX.
  This unlocks the ability to jit functions of `Operator`.
  [(#4458)](https://github.com/PennyLaneAI/pennylane/pull/4458/)

  ```pycon
  >>> op = qml.adjoint(qml.RX(1.0, wires=0))
  >>> jax.jit(qml.matrix)(op)
  Array([[0.87758255-0.j        , 0.        +0.47942555j],
       [0.        +0.47942555j, 0.87758255-0.j        ]],      dtype=complex64, weak_type=True)
  >>> jax.tree_util.tree_map(lambda x: x+1, op)
  Adjoint(RX(2.0, wires=[0]))
  ```

* All `Operator` objects now define `Operator._flatten` and `Operator._unflatten` methods that separate
  trainable from untrainable components. These methods will be used in serialization and pytree registration.
  Custom operations may need an update to ensure compatibility with new PennyLane features.
  [(#4483)](https://github.com/PennyLaneAI/pennylane/pull/4483)
  [(#4314)](https://github.com/PennyLaneAI/pennylane/pull/4314)

* The `QuantumScript` class now has a `bind_new_parameters` method that allows creation of
  new `QuantumScript` objects with the provided parameters.
  [(#4345)](https://github.com/PennyLaneAI/pennylane/pull/4345)

* The `qml.gradients` module no longer mutates operators in-place for any gradient transforms.
  Instead, operators that need to be mutated are copied with new parameters.
  [(#4220)](https://github.com/PennyLaneAI/pennylane/pull/4220)

* PennyLane no longer directly relies on `Operator.__eq__`.
  [(#4398)](https://github.com/PennyLaneAI/pennylane/pull/4398)

* `qml.equal` no longer raises errors when operators or measurements of different types are compared.
  Instead, it returns `False`.
  [(#4315)](https://github.com/PennyLaneAI/pennylane/pull/4315)

<h4>Transforms</h4>

* Transform programs are now integrated with the QNode.
  [(#4404)](https://github.com/PennyLaneAI/pennylane/pull/4404)

  ```python
  def null_postprocessing(results: qml.typing.ResultBatch) -> qml.typing.Result:
      return results[0]

  @qml.transforms.core.transform
  def scale_shots(tape: qml.tape.QuantumTape, shot_scaling) -> (Tuple[qml.tape.QuantumTape], Callable):
      new_shots = tape.shots.total_shots * shot_scaling
      new_tape = qml.tape.QuantumScript(tape.operations, tape.measurements, shots=new_shots)
      return (new_tape, ), null_postprocessing

  dev = qml.devices.experimental.DefaultQubit2()

  @partial(scale_shots, shot_scaling=2)
  @qml.qnode(dev, interface=None)
  def circuit():
      return qml.sample(wires=0)
  ```

  ```pycon
  >>> circuit(shots=1)
  array([False, False])
  ```

* Transform Programs, `qml.transforms.core.TransformProgram`, can now be called on a batch of circuits
  and return a new batch of circuits and a single post processing function.
  [(#4364)](https://github.com/PennyLaneAI/pennylane/pull/4364)

* `TransformDispatcher` now allows registration of custom QNode transforms.
  [(#4466)](https://github.com/PennyLaneAI/pennylane/pull/4466)

* QNode transforms in `qml.qinfo` now support custom wire labels.
  [#4331](https://github.com/PennyLaneAI/pennylane/pull/4331)

* `qml.transforms.adjoint_metric_tensor` now uses the simulation tools in `qml.devices.qubit` instead of
  private methods of `qml.devices.DefaultQubit`.
  [(#4456)](https://github.com/PennyLaneAI/pennylane/pull/4456)

* Auxiliary wires and device wires are now treated the same way in `qml.transforms.metric_tensor`
  as in `qml.gradients.hadamard_grad`. All valid wire input formats for `aux_wire` are supported.
  [(#4328)](https://github.com/PennyLaneAI/pennylane/pull/4328)

<h4>Next-generation device API</h4>

* The experimental device interface has been integrated with the QNode for JAX, JAX-JIT, TensorFlow and PyTorch.
  [(#4323)](https://github.com/PennyLaneAI/pennylane/pull/4323)
  [(#4352)](https://github.com/PennyLaneAI/pennylane/pull/4352)
  [(#4392)](https://github.com/PennyLaneAI/pennylane/pull/4392)
  [(#4393)](https://github.com/PennyLaneAI/pennylane/pull/4393)

* The experimental `DefaultQubit2` device now supports computing VJPs and JVPs using the adjoint method.
  [(#4374)](https://github.com/PennyLaneAI/pennylane/pull/4374)

* New functions called `adjoint_jvp` and `adjoint_vjp` that compute the JVP and VJP of a tape using the adjoint method 
  have been added to `qml.devices.qubit.adjoint_jacobian` 
  [(#4358)](https://github.com/PennyLaneAI/pennylane/pull/4358)

* `DefaultQubit2` now accepts a `max_workers` argument which controls multiprocessing.
  A `ProcessPoolExecutor` executes tapes asynchronously
  using a pool of at most `max_workers` processes. If `max_workers` is `None`
  or not given, only the current process executes tapes. If you experience any
  issue, say using JAX, TensorFlow, Torch, try setting `max_workers` to `None`.
  [(#4319)](https://github.com/PennyLaneAI/pennylane/pull/4319)
  [(#4425)](https://github.com/PennyLaneAI/pennylane/pull/4425)

* `qml.devices.experimental.Device` now accepts a shots keyword argument and has a `shots`
  property. This property is only used to set defaults for a workflow, and does not directly
  influence the number of shots used in executions or derivatives.
  [(#4388)](https://github.com/PennyLaneAI/pennylane/pull/4388)

* `expand_fn()` for `DefaultQubit2` has been updated to decompose `StatePrep` operations present in the middle of a circuit.
  [(#4444)](https://github.com/PennyLaneAI/pennylane/pull/4444)

* If no seed is specified on initialization with `DefaultQubit2`, the local random number generator will be
  seeded from NumPy's global random number generator.
  [(#4394)](https://github.com/PennyLaneAI/pennylane/pull/4394)

<h4>Improvements to machine learning library interfaces</h4>

* `pennylane/interfaces` has been refactored. The `execute_fn` passed to the machine learning framework boundaries 
  is now responsible for converting parameters to NumPy. The gradients module can now handle TensorFlow parameters,
  but gradient tapes now retain the original `dtype` instead of converting to `float64`.  This may cause instability 
  with finite-difference differentiation and `float32` parameters. The machine learning boundary functions are now uncoupled from their legacy counterparts.
  [(#4415)](https://github.com/PennyLaneAI/pennylane/pull/4415)

* `qml.interfaces.set_shots` now accepts a `Shots` object as well as `int`'s and tuples of `int`'s.
  [(#4388)](https://github.com/PennyLaneAI/pennylane/pull/4388)

* Readability improvements and stylistic changes have been made to `pennylane/interfaces/jax_jit_tuple.py`
  [(#4379)](https://github.com/PennyLaneAI/pennylane/pull/4379/)

<h4>Pulses</h4>

* A `HardwareHamiltonian` can now be summed with `int` or `float` objects.
  A sequence of `HardwareHamiltonian`s can now be summed via the builtin `sum`.
  [(#4343)](https://github.com/PennyLaneAI/pennylane/pull/4343)

* `qml.pulse.transmon_drive` has been updated in accordance with [1904.06560](https://arxiv.org/abs/1904.06560).
  In particular, the functional form has been changed from
  :math:`\Omega(t)(\cos(\omega_d t + \phi) X - \sin(\omega_d t + \phi) Y)$ to $\Omega(t) \sin(\omega_d t + \phi) Y`.
  [(#4418)](https://github.com/PennyLaneAI/pennylane/pull/4418/)
  [(#4465)](https://github.com/PennyLaneAI/pennylane/pull/4465/)
  [(#4478)](https://github.com/PennyLaneAI/pennylane/pull/4478/)
  [(#4418)](https://github.com/PennyLaneAI/pennylane/pull/4418/)


<h4>Other improvements</h4>

* The `qchem` module has been upgraded to use the fermionic operators of the `fermi` module.
  [#4336](https://github.com/PennyLaneAI/pennylane/pull/4336)
  [#4521](https://github.com/PennyLaneAI/pennylane/pull/4521)

* The calculation of `Sum`, `Prod`, `SProd`, `PauliWord`, and `PauliSentence` sparse matrices
  are orders of magnitude faster.
  [(#4475)](https://github.com/PennyLaneAI/pennylane/pull/4475)
  [(#4272)](https://github.com/PennyLaneAI/pennylane/pull/4272)
  [(#4411)](https://github.com/PennyLaneAI/pennylane/pull/4411)

* A function called `qml.math.fidelity_statevector` that computes the fidelity between two state vectors has been added.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

* `qml.ctrl(qml.PauliX)` returns a `CNOT`, `Toffoli`, or `MultiControlledX` operation instead of `Controlled(PauliX)`.
  [(#4339)](https://github.com/PennyLaneAI/pennylane/pull/4339)

* When given a callable, `qml.ctrl` now does its custom pre-processing on all queued operators from the callable.
  [(#4370)](https://github.com/PennyLaneAI/pennylane/pull/4370)

* The `qchem` functions `primitive_norm` and `contracted_norm` have been modified to be compatible with
  higher versions of SciPy. The private function `_fac2` for computing double factorials has also been added.
  [#4321](https://github.com/PennyLaneAI/pennylane/pull/4321)

* `tape_expand` now uses `Operator.decomposition` instead of `Operator.expand` in order to make
  more performant choices.
  [(#4355)](https://github.com/PennyLaneAI/pennylane/pull/4355)

* CI now runs tests with TensorFlow 2.13.0
  [(#4472)](https://github.com/PennyLaneAI/pennylane/pull/4472)

* All tests in CI and pre-commit hooks now enable linting.
  [(#4335)](https://github.com/PennyLaneAI/pennylane/pull/4335)

* The default label for a `StatePrepBase` operator is now `|Ψ⟩`.
  [(#4340)](https://github.com/PennyLaneAI/pennylane/pull/4340)

* `Device.default_expand_fn()` has been updated to decompose `qml.StatePrep` operations present in the middle of a provided circuit.
  [(#4437)](https://github.com/PennyLaneAI/pennylane/pull/4437)

* `QNode.construct` has been updated to only apply the `qml.defer_measurements` transform if the device
  does not natively support mid-circuit measurements.
  [(#4516)](https://github.com/PennyLaneAI/pennylane/pull/4516)

* The application of the `qml.defer_measurements` transform has been moved from 
  `QNode.construct` to `qml.Device.batch_transform` to allow more fine-grain 
  control over when `defer_measurements` should be used.
  [(#4432)](https://github.com/PennyLaneAI/pennylane/pull/4432)

* The label for `ParametrizedEvolution` can display parameters with the requested format as set by the 
  kwarg `decimals`. Array-like parameters are displayed in the same format as matrices and stored in the 
  cache.
  [(#4151)](https://github.com/PennyLaneAI/pennylane/pull/4151)


<h3>Breaking changes 💔</h3>

* Applying gradient transforms to broadcasted/batched tapes has been deactivated until it is consistently
  supported for QNodes as well.
  [(#4480)](https://github.com/PennyLaneAI/pennylane/pull/4480)

* Gradient transforms no longer implicitly cast `float32` parameters to `float64`. Finite difference differentiation
  with `float32` parameters may no longer give accurate results.
  [(#4415)](https://github.com/PennyLaneAI/pennylane/pull/4415)

* The `do_queue` keyword argument in `qml.operation.Operator` has been removed. Instead of
  setting `do_queue=False`, use the `qml.QueuingManager.stop_recording()` context.
  [(#4317)](https://github.com/PennyLaneAI/pennylane/pull/4317)

* `Operator.expand` now uses the output of `Operator.decomposition` instead of what it queues.
  [(#4355)](https://github.com/PennyLaneAI/pennylane/pull/4355)

* The gradients module no longer needs shot information passed to it explicitly, as the shots are on the tapes.
  [(#4448)](https://github.com/PennyLaneAI/pennylane/pull/4448)

* `qml.StatePrep` has been renamed to `qml.StatePrepBase` and `qml.QubitStateVector` has been renamed to `qml.StatePrep`.
  `qml.operation.StatePrep` and `qml.QubitStateVector` are still accessible.
  [(#4450)](https://github.com/PennyLaneAI/pennylane/pull/4450)

* Support for Python 3.8 has been dropped.
  [(#4453)](https://github.com/PennyLaneAI/pennylane/pull/4453)

* `MeasurementValue`'s signature has been updated to accept a list of `MidMeasureMP`'s rather than a list of
  their IDs.
  [(#4446)](https://github.com/PennyLaneAI/pennylane/pull/4446)

* The `grouping_type` and `grouping_method` keyword arguments have been removed from `qchem.molecular_hamiltonian`.
  [(#4301)](https://github.com/PennyLaneAI/pennylane/pull/4301)

* `zyz_decomposition` and `xyx_decomposition` have been removed. Use `one_qubit_decomposition` instead.
  [(#4301)](https://github.com/PennyLaneAI/pennylane/pull/4301)

* `LieAlgebraOptimizer` has been removed. Use `RiemannianGradientOptimizer` instead.
  [(#4301)](https://github.com/PennyLaneAI/pennylane/pull/4301)

* `Operation.base_name` has been removed.
  [(#4301)](https://github.com/PennyLaneAI/pennylane/pull/4301)

* `QuantumScript.name` has been removed.
  [(#4301)](https://github.com/PennyLaneAI/pennylane/pull/4301)

* `qml.math.reduced_dm` has been removed. Use `qml.math.reduce_dm` or `qml.math.reduce_statevector` instead.
  [(#4301)](https://github.com/PennyLaneAI/pennylane/pull/4301)

* The `qml.specs` dictionary no longer supports direct key access to certain keys. 
  [(#4301)](https://github.com/PennyLaneAI/pennylane/pull/4301)

  Instead, these quantities can be accessed as fields of the new `Resources` object saved under
  `specs_dict["resources"]`:

  - `num_operations` is no longer supported, use `specs_dict["resources"].num_gates`
  - `num_used_wires` is no longer supported, use `specs_dict["resources"].num_wires`
  - `gate_types` is no longer supported, use `specs_dict["resources"].gate_types`
  - `gate_sizes` is no longer supported, use `specs_dict["resources"].gate_sizes`
  - `depth` is no longer supported, use `specs_dict["resources"].depth`

* `qml.math.purity`, `qml.math.vn_entropy`, `qml.math.mutual_info`, `qml.math.fidelity`,
  `qml.math.relative_entropy`, and `qml.math.max_entropy` no longer support state vectors as
  input.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

* The private `QuantumScript._prep` list has been removed, and prep operations now go into the `_ops` list.
  [(#4485)](https://github.com/PennyLaneAI/pennylane/pull/4485)

<h3>Deprecations 👋</h3>

* `qml.enable_return` and `qml.disable_return` have been deprecated. Please avoid calling
  `disable_return`, as the old return system has been deprecated along with these switch functions.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* `qml.qchem.jordan_wigner` has been deprecated. Use `qml.jordan_wigner` instead.
  List input to define the fermionic operator has also been deprecated; the fermionic
  operators in the `qml.fermi` module should be used instead.
  [(#4332)](https://github.com/PennyLaneAI/pennylane/pull/4332)

* The `qml.RandomLayers.compute_decomposition` keyword argument `ratio_imprimitive` will be changed to `ratio_imprim` to
  match the call signature of the operation.
  [(#4314)](https://github.com/PennyLaneAI/pennylane/pull/4314)

* The CV observables `qml.X` and `qml.P` have been deprecated. Use `qml.QuadX`
  and `qml.QuadP` instead.
  [(#4330)](https://github.com/PennyLaneAI/pennylane/pull/4330)

* The method `tape.unwrap()` and corresponding `UnwrapTape` and `Unwrap` classes
  have been deprecated. Use `convert_to_numpy_parameters` instead.
  [(#4344)](https://github.com/PennyLaneAI/pennylane/pull/4344)

* The `mode` keyword argument in QNode has been deprecated, as it was only used in the
  old return system (which has also been deprecated). Please use `grad_on_execution` instead.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The `QuantumScript.set_parameters` method and the `QuantumScript.data` setter have
  been deprecated. Please use `QuantumScript.bind_new_parameters` instead.
  [(#4346)](https://github.com/PennyLaneAI/pennylane/pull/4346)

* The `__eq__` and `__hash__` dunder methods of `Operator` and `MeasurementProcess` will now raise
  warnings to reflect upcoming changes to operator and measurement process equality and hashing.
  [(#4144)](https://github.com/PennyLaneAI/pennylane/pull/4144)
  [(#4454)](https://github.com/PennyLaneAI/pennylane/pull/4454)
  [(#4489)](https://github.com/PennyLaneAI/pennylane/pull/4489)
  [(#4498)](https://github.com/PennyLaneAI/pennylane/pull/4498)

* The `sampler_seed` argument of `qml.gradients.spsa_grad` has been deprecated, along with a bug
  fix of the seed-setting behaviour.
  Instead, the `sampler_rng` argument should be set, either to an integer value, which will be used
  to create a PRNG internally or to a NumPy pseudo-random number generator created via
  `np.random.default_rng(seed)`.
  [(4165)](https://github.com/PennyLaneAI/pennylane/pull/4165)

<h3>Documentation 📝</h3>

* The `qml.pulse.transmon_interaction` and `qml.pulse.transmon_drive` documentation has been updated.
  [#4327](https://github.com/PennyLaneAI/pennylane/pull/4327)

* `qml.ApproxTimeEvolution.compute_decomposition()` now has a code example.
  [(#4354)](https://github.com/PennyLaneAI/pennylane/pull/4354)

* The documentation for `qml.devices.experimental.Device` has been improved to clarify
  some aspects of its use.
  [(#4391)](https://github.com/PennyLaneAI/pennylane/pull/4391)

* Input types and sources for operators in `qml.import_operator` are specified.
  [(#4476)](https://github.com/PennyLaneAI/pennylane/pull/4476)

<h3>Bug fixes 🐛</h3>

* `qml.Projector` is pickle-able again.
  [(#4452)](https://github.com/PennyLaneAI/pennylane/pull/4452)

* `_copy_and_shift_params` does not cast or convert integral types, just relying on `+` and `*`'s casting rules in this case.
  [(#4477)](https://github.com/PennyLaneAI/pennylane/pull/4477)

* Sparse matrix calculations of `SProd`s containing a `Tensor` are now allowed. When using
  `Tensor.sparse_matrix()`, it is recommended to use the `wire_order` keyword argument over `wires`. 
  [(#4424)](https://github.com/PennyLaneAI/pennylane/pull/4424)
  
* `op.adjoint` has been replaced with `qml.adjoint` in `QNSPSAOptimizer`.
  [(#4421)](https://github.com/PennyLaneAI/pennylane/pull/4421)

* `jax.ad` (deprecated) has been replaced by `jax.interpreters.ad`.
  [(#4403)](https://github.com/PennyLaneAI/pennylane/pull/4403)

* `metric_tensor` stops accidentally catching errors that stem from
  flawed wires assignments in the original circuit, leading to recursion errors.
  [(#4328)](https://github.com/PennyLaneAI/pennylane/pull/4328)

* A warning is now raised if control indicators are hidden when calling `qml.draw_mpl`
  [(#4295)](https://github.com/PennyLaneAI/pennylane/pull/4295)

* `qml.qinfo.purity` now produces correct results with custom wire labels.
  [(#4331)](https://github.com/PennyLaneAI/pennylane/pull/4331)

* `default.qutrit` now supports all qutrit operations used with `qml.adjoint`.
  [(#4348)](https://github.com/PennyLaneAI/pennylane/pull/4348)

* The observable data of `qml.GellMann` now includes its index, allowing correct comparison
  between instances of `qml.GellMann`, as well as Hamiltonians and Tensors
  containing `qml.GellMann`.
  [(#4366)](https://github.com/PennyLaneAI/pennylane/pull/4366)

* `qml.transforms.merge_amplitude_embedding` now works correctly when the `AmplitudeEmbedding`s
  have a batch dimension.
  [(#4353)](https://github.com/PennyLaneAI/pennylane/pull/4353)

* The `jordan_wigner` function has been modified to work with Hamiltonians built with an active space.
  [(#4372)](https://github.com/PennyLaneAI/pennylane/pull/4372)

* When a `style` option is not provided, `qml.draw_mpl` uses the current style set from
  `qml.drawer.use_style` instead of `black_white`.
  [(#4357)](https://github.com/PennyLaneAI/pennylane/pull/4357)

* `qml.devices.qubit.preprocess.validate_and_expand_adjoint` no longer sets the
  trainable parameters of the expanded tape.
  [(#4365)](https://github.com/PennyLaneAI/pennylane/pull/4365)

* `qml.default_expand_fn` now selectively expands operations or measurements allowing more 
  operations to be executed in circuits when measuring non-qwc Hamiltonians.
  [(#4401)](https://github.com/PennyLaneAI/pennylane/pull/4401)

* `qml.ControlledQubitUnitary` no longer reports `has_decomposition` as `True` when it does
  not really have a decomposition.
  [(#4407)](https://github.com/PennyLaneAI/pennylane/pull/4407)

* `qml.transforms.split_non_commuting` now correctly works on tapes containing both `expval`
  and `var` measurements.
  [(#4426)](https://github.com/PennyLaneAI/pennylane/pull/4426)

* Subtracting a `Prod` from another operator now works as expected.
  [(#4441)](https://github.com/PennyLaneAI/pennylane/pull/4441)

* The `sampler_seed` argument of `qml.gradients.spsa_grad` has been changed to `sampler_rng`. One can either provide
  an integer, which will be used to create a PRNG internally. Previously, this lead to the same direction
  being sampled, when `num_directions` is greater than 1. Alternatively, one can provide a NumPy PRNG,
  which allows reproducibly calling `spsa_grad` without getting the same results every time.
  [(4165)](https://github.com/PennyLaneAI/pennylane/pull/4165)
  [(4482)](https://github.com/PennyLaneAI/pennylane/pull/4482)

* `qml.math.get_dtype_name` now works with autograd array boxes.
  [(#4494)](https://github.com/PennyLaneAI/pennylane/pull/4494)

* The backprop gradient of `qml.math.fidelity` is now correct.
  [(#4380)](https://github.com/PennyLaneAI/pennylane/pull/4380)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Thomas Bromley,
Isaac De Vlugt,
Amintor Dusko,
Stepan Fomichev,
Lillian M. A. Frederiksen,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Ivana Kurečić,
Christina Lee,
Vincent Michaud-Rioux,
Romain Moyard,
Lee James O'Riordan,
Mudit Pandey,
Borja Requena,
Matthew Silverman,
Jay Soni,
David Wierichs,
Frederik Wilde.
