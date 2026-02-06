
# Release 0.11.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* Added a new device, `default.qubit.autograd`, a pure-state qubit simulator written using Autograd.
  This device supports classical backpropagation (`diff_method="backprop"`); this can
  be faster than the parameter-shift rule for computing quantum gradients
  when the number of parameters to be optimized is large.
  [(#721)](https://github.com/XanaduAI/pennylane/pull/721)

  ```pycon
  >>> dev = qp.device("default.qubit.autograd", wires=1)
  >>> @qp.qnode(dev, diff_method="backprop")
  ... def circuit(x):
  ...     qp.RX(x[1], wires=0)
  ...     qp.Rot(x[0], x[1], x[2], wires=0)
  ...     return qp.expval(qp.PauliZ(0))
  >>> weights = np.array([0.2, 0.5, 0.1])
  >>> grad_fn = qp.grad(circuit)
  >>> print(grad_fn(weights))
  array([-2.25267173e-01, -1.00864546e+00,  6.93889390e-18])
  ```

  See the [device documentation](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.devices.default_qubit_autograd.DefaultQubitAutograd.html) for more details.

* A new experimental C++ state-vector simulator device is now available, `lightning.qubit`. It
  uses the C++ Eigen library to perform fast linear algebra calculations for simulating quantum
  state-vector evolution.

  `lightning.qubit` is currently in beta; it can be installed via `pip`:

  ```console
  $ pip install pennylane-lightning
  ```

  Once installed, it can be used as a PennyLane device:

  ```pycon
  >>> dev = qp.device("lightning.qubit", wires=2)
  ```

  For more details, please see the [lightning qubit documentation](https://pennylane-lightning.readthedocs.io).

<h4>New algorithms and templates</h4>

* Added built-in QAOA functionality via the new `qp.qaoa` module.
  [(#712)](https://github.com/PennyLaneAI/pennylane/pull/712)
  [(#718)](https://github.com/PennyLaneAI/pennylane/pull/718)
  [(#741)](https://github.com/PennyLaneAI/pennylane/pull/741)
  [(#720)](https://github.com/PennyLaneAI/pennylane/pull/720)

  This includes the following features:

  * New `qp.qaoa.x_mixer` and `qp.qaoa.xy_mixer` functions for defining Pauli-X and XY
    mixer Hamiltonians.

  * MaxCut: The `qp.qaoa.maxcut` function allows easy construction of the cost Hamiltonian
    and recommended mixer Hamiltonian for solving the MaxCut problem for a supplied graph.

  * Layers: `qp.qaoa.cost_layer` and `qp.qaoa.mixer_layer` take cost and mixer
    Hamiltonians, respectively, and apply the corresponding QAOA cost and mixer layers
    to the quantum circuit

  For example, using PennyLane to construct and solve a MaxCut problem with QAOA:

  ```python
  wires = range(3)
  graph = Graph([(0, 1), (1, 2), (2, 0)])
  cost_h, mixer_h = qaoa.maxcut(graph)

  def qaoa_layer(gamma, alpha):
      qaoa.cost_layer(gamma, cost_h)
      qaoa.mixer_layer(alpha, mixer_h)

  def antatz(params, **kwargs):

      for w in wires:
          qp.Hadamard(wires=w)

      # repeat the QAOA layer two times
      qp.layer(qaoa_layer, 2, params[0], params[1])

  dev = qp.device('default.qubit', wires=len(wires))
  cost_function = qp.VQECost(ansatz, cost_h, dev)
  ```

* Added an `ApproxTimeEvolution` template to the PennyLane templates module, which
  can be used to implement Trotterized time-evolution under a Hamiltonian.
  [(#710)](https://github.com/XanaduAI/pennylane/pull/710)

  <img src="https://pennylane.readthedocs.io/en/latest/_static/templates/subroutines/approx_time_evolution.png" width=50%/>

* Added a `qp.layer` template-constructing function, which takes a unitary, and
  repeatedly applies it on a set of wires to a given depth.
  [(#723)](https://github.com/PennyLaneAI/pennylane/pull/723)

  ```python
  def subroutine():
      qp.Hadamard(wires=[0])
      qp.CNOT(wires=[0, 1])
      qp.PauliX(wires=[1])

  dev = qp.device('default.qubit', wires=3)

  @qp.qnode(dev)
  def circuit():
      qp.layer(subroutine, 3)
      return [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(1))]
  ```

  This creates the following circuit:
  ```pycon
  >>> circuit()
  >>> print(circuit.draw())
  0: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
  1: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩
  ```

* Added the `qp.utils.decompose_hamiltonian` function. This function can be used to
  decompose a Hamiltonian into a linear combination of Pauli operators.
  [(#671)](https://github.com/XanaduAI/pennylane/pull/671)

  ```pycon
  >>> A = np.array(
  ... [[-2, -2+1j, -2, -2],
  ... [-2-1j,  0,  0, -1],
  ... [-2,  0, -2, -1],
  ... [-2, -1, -1,  0]])
  >>> coeffs, obs_list = decompose_hamiltonian(A)
  ```

<h4>New device features</h4>

* It is now possible to specify custom wire labels, such as `['anc1', 'anc2', 0, 1, 3]`, where the labels
  can be strings or numbers.
  [(#666)](https://github.com/XanaduAI/pennylane/pull/666)

  Custom wire labels are defined by passing a list to the `wires` argument when creating the device:

  ```pycon
  >>> dev = qp.device("default.qubit", wires=['anc1', 'anc2', 0, 1, 3])
  ```

  Quantum operations should then be invoked with these custom wire labels:

  ``` pycon
  >>> @qp.qnode(dev)
  >>> def circuit():
  ...    qp.Hadamard(wires='anc2')
  ...    qp.CNOT(wires=['anc1', 3])
  ...    ...
  ```

  The existing behaviour, in which the number of wires is specified on device initialization,
  continues to work as usual. This gives a default behaviour where wires are labelled
  by consecutive integers.

  ```pycon
  >>> dev = qp.device("default.qubit", wires=5)
  ```

* An integrated device test suite has been added, which can be used
  to run basic integration tests on core or external devices.
  [(#695)](https://github.com/PennyLaneAI/pennylane/pull/695)
  [(#724)](https://github.com/PennyLaneAI/pennylane/pull/724)
  [(#733)](https://github.com/PennyLaneAI/pennylane/pull/733)

  The test can be invoked against a particular device by calling the `pl-device-test`
  command line program:

  ```console
  $ pl-device-test --device=default.qubit --shots=1234 --analytic=False
  ```

  If the tests are run on external devices, the device and its dependencies must be
  installed locally. For more details, please see the
  [plugin test documentation](http://pennylane.readthedocs.io/en/latest/code/api/pennylane.devices.tests.html).

<h3>Improvements</h3>

* The functions implementing the quantum circuits building the Unitary Coupled-Cluster
  (UCCSD) VQE ansatz have been improved, with a more consistent naming convention and
  improved docstrings.
  [(#748)](https://github.com/PennyLaneAI/pennylane/pull/748)

  The changes include:

  - The terms *1particle-1hole (ph)* and *2particle-2hole (pphh)* excitations
    were replaced with the names *single* and *double* excitations, respectively.

  - The non-differentiable arguments in the `UCCSD` template were renamed accordingly:
    `ph` → `s_wires`, `pphh` → `d_wires`

  - The term *virtual*, previously used to refer the *unoccupied* orbitals, was discarded.

  - The Usage Details sections were updated and improved.

* Added support for TensorFlow 2.3 and PyTorch 1.6.
  [(#725)](https://github.com/PennyLaneAI/pennylane/pull/725)

* Returning probabilities is now supported from photonic QNodes.
  As with qubit QNodes, photonic QNodes returning probabilities are
  end-to-end differentiable.
  [(#699)](https://github.com/XanaduAI/pennylane/pull/699/)

  ```pycon
  >>> dev = qp.device("strawberryfields.fock", wires=2, cutoff_dim=5)
  >>> @qp.qnode(dev)
  ... def circuit(a):
  ...     qp.Displacement(a, 0, wires=0)
  ...     return qp.probs(wires=0)
  >>> print(circuit(0.5))
  [7.78800783e-01 1.94700196e-01 2.43375245e-02 2.02812704e-03 1.26757940e-04]
  ```

<h3>Breaking changes</h3>

* The `pennylane.plugins` and `pennylane.beta.plugins` folders have been renamed to
  `pennylane.devices` and `pennylane.beta.devices`, to reflect their content better.
  [(#726)](https://github.com/XanaduAI/pennylane/pull/726)

<h3>Bug fixes</h3>

* The PennyLane interface conversion functions can now convert QNodes with
  pre-existing interfaces.
  [(#707)](https://github.com/XanaduAI/pennylane/pull/707)

<h3>Documentation</h3>

* The interfaces section of the documentation has been renamed to 'Interfaces and training',
  and updated with the latest variable handling details.
  [(#753)](https://github.com/PennyLaneAI/pennylane/pull/753)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Shadab Hussain, Theodor
Isacsson, Josh Izaac, Nathan Killoran, Maria Schuld, Antal Száva, Nicola Vitucci.
