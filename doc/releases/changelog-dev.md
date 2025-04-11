:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

<<<<<<< HEAD
* Device preprocessing is now being performed in the execution pipeline for program capture.
  [(#7057)](https://github.com/PennyLaneAI/pennylane/pull/7057)
  [(#7089)](https://github.com/PennyLaneAI/pennylane/pull/7089)
  [(#7131)](https://github.com/PennyLaneAI/pennylane/pull/7131)
  [(#7135)](https://github.com/PennyLaneAI/pennylane/pull/7135)

* Added method `qml.math.sqrt_matrix_sparse` to compute the square root of a sparse Hermitian matrix.
  [(#6976)](https://github.com/PennyLaneAI/pennylane/pull/6976)

* Added a class `qml.capture.transforms.MergeRotationsInterpreter` that merges rotation operators
  following the same API as `qml.transforms.optimization.merge_rotations` when experimental program capture is enabled.
  [(#6957)](https://github.com/PennyLaneAI/pennylane/pull/6957)

* `qml.defer_measurements` can now be used with program capture enabled. Programs transformed by
  `qml.defer_measurements` can be executed on `default.qubit`.
  [(#6838)](https://github.com/PennyLaneAI/pennylane/pull/6838)
  [(#6937)](https://github.com/PennyLaneAI/pennylane/pull/6937)
  [(#6961)](https://github.com/PennyLaneAI/pennylane/pull/6961)

  Using `qml.defer_measurements` with program capture enables many new features, including:
  * Significantly richer variety of classical processing on mid-circuit measurement values.
  * Using mid-circuit measurement values as gate parameters.

  Functions such as the following can now be captured:

  ```python
  import jax.numpy as jnp

  qml.capture.enable()

  def f(x):
      m0 = qml.measure(0)
      m1 = qml.measure(0)
      a = jnp.sin(0.5 * jnp.pi * m0)
      phi = a - (m1 + 1) ** 4

      qml.s_prod(x, qml.RZ(phi, 0))

      return qml.expval(qml.Z(0))
  ```

* Added class `qml.capture.transforms.UnitaryToRotInterpreter` that decomposes `qml.QubitUnitary` operators
  following the same API as `qml.transforms.unitary_to_rot` when experimental program capture is enabled.
  [(#6916)](https://github.com/PennyLaneAI/pennylane/pull/6916)
  [(#6977)](https://github.com/PennyLaneAI/pennylane/pull/6977)

* Created a new `qml.liealg` module for Lie algebra functionality.

  `qml.liealg.cartan_decomp` allows to perform Cartan decompositions `g = k + m` using _involution_ functions that return a boolean value.
  A variety of typically encountered involution functions are included in the module, in particular the following:

  ```
  even_odd_involution
  concurrence_involution
  A
  AI
  AII
  AIII
  BD
  BDI
  DIII
  C
  CI
  CII
  ```

  ```pycon
  >>> g = qml.lie_closure([X(0) @ X(1), Y(0), Y(1)])
  >>> k, m = qml.liealg.cartan_decomp(g, qml.liealg.even_odd_involution)
  >>> g, k, m
  ([X(0) @ X(1), Y(0), Y(1), Z(0) @ X(1), X(0) @ Z(1), Z(0) @ Z(1)],
   [Y(0), Y(1)],
   [X(0) @ X(1), Z(0) @ X(1), X(0) @ Z(1), Z(0) @ Z(1)])
  ```

  The vertical subspace `k` and `m` fulfil the commutation relations `[k, m] ⊆ m`, `[k, k] ⊆ k` and `[m, m] ⊆ k` that make them a proper Cartan decomposition. These can be checked using the function `qml.liealg.check_cartan_decomp`.

  ```pycon
  >>> qml.liealg.check_cartan_decomp(k, m) # check Cartan commutation relations
  True
  ```

  `qml.liealg.horizontal_cartan_subalgebra` computes a horizontal Cartan subalgebra `a` of `m`.

  ```pycon
  >>> newg, k, mtilde, a, new_adj = qml.liealg.horizontal_cartan_subalgebra(k, m)
  ```

  `newg` is ordered such that the elements are `newg = k + mtilde + a`, where `mtilde` is the remainder of `m` without `a`. A Cartan subalgebra is an Abelian subalgebra of `m`, and we can confirm that indeed all elements in `a` are mutually commuting via `qml.liealg.check_abelian`.

  ```pycon
  >>> qml.liealg.check_abelian(a)
  True
  ```

  The following functions have also been added:
  * `qml.liealg.check_commutation_relation(A, B, C)` checks if all commutators between `A` and `B`
  map to a subspace of `C`, i.e. `[A, B] ⊆ C`.

  * `qml.liealg.adjvec_to_op` and `qml.liealg.op_to_adjvec` allow transforming operators within a Lie algebra to their adjoint vector representations and back.

  * `qml.liealg.change_basis_ad_rep` allows the transformation of an adjoint representation tensor according to a basis transformation on the underlying Lie algebra, without re-computing the representation.

  [(#6935)](https://github.com/PennyLaneAI/pennylane/pull/6935)
  [(#7026)](https://github.com/PennyLaneAI/pennylane/pull/7026)
  [(#7054)](https://github.com/PennyLaneAI/pennylane/pull/7054)

* ``qml.lie_closure`` now accepts and outputs matrix inputs using the ``matrix`` keyword.
  Also added ``qml.pauli.trace_inner_product`` that can handle batches of dense matrices.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

* Added class ``qml.FromBloq`` that takes Qualtran bloqs and translates them into equivalent PennyLane operators. For example, we can now import Bloqs and use them in a way similar to how we use PennyLane templates:
  ```python
  from qualtran.bloqs.basic_gates import CNOT
  
  # Execute on device
  dev = qml.device("default.qubit")
  @qml.qnode(dev)
  def circuit():
      qml.FromBloq(CNOT(), wires=[0, 1])
      return qml.state()
  
  >>> circuit()
  array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
  ```
  [(#6921)](https://github.com/PennyLaneAI/pennylane/pull/6921)

* Added template `qml.QROMStatePreparation` that prepares arbitrary states using `qml.QROM`.
  [(#6974)](https://github.com/PennyLaneAI/pennylane/pull/6974)

* ``qml.structure_constants`` now accepts and outputs matrix inputs using the ``matrix`` keyword.
  [(#6861)](https://github.com/PennyLaneAI/pennylane/pull/6861)

<h4>Gate-set targeted decompositions</h4>

* A new module called `qml.decomposition` that contains PennyLane's new experimental graph-based 
  gate-set-targeted decomposition system has been added.
  [(#6950)](https://github.com/PennyLaneAI/pennylane/pull/6950)

  * A decomposition rule in the new system is defined as a quantum function that declares its own
    resource requirements using `qml.register_resources`.
    ```python
    import pennylane as qml

    @qml.register_resources({qml.H: 2, qml.CZ: 1})
    def my_cnot(wires):
        qml.H(wires=wires[1])
        qml.CZ(wires=wires)
        qml.H(wires=wires[1])
    ```

  * Operators with dynamic resource requirements must be declared in a resource estimate using `qml.resource_rep`.
    ```python
    import pennylane as qml

    def _resource_fn(num_wires):
        return {
            qml.resource_rep(qml.MultiRZ, num_wires=num_wires - 1): 1,
            qml.resource_rep(qml.MultiRZ, num_wires=3): 2
        }
    
    @qml.register_resources(_resource_fn)
    def my_decomp(thata, wires):
        qml.MultiRZ(theta, wires=wires[:3])
        qml.MultiRZ(theta, wires=wires[1:])
        qml.MultiRZ(theta, wires=wires[:3])
    ```

  * The new system allows multiple decomposition rules to be registered for the same operator. 
    Use `qml.add_decomps` to register a decomposition with an operator; use `qml.list_decomps`
    to inspect all known decompositions for an operator.
    ```pycon
    >>> import pennylane as qml
    >>> qml.list_decomps(qml.CRX)
    [<pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9de0>,
     <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9db0>,
     <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9f00>]
    >>> print(qml.list_decomps(qml.CRX)[0])
    @register_resources(_crx_to_rx_cz_resources)
    def _crx_to_rx_cz(phi, wires, **__):
        qml.RX(phi / 2, wires=wires[1])
        qml.CZ(wires=wires)
        qml.RX(-phi / 2, wires=wires[1])
        qml.CZ(wires=wires)
    >>> qml.list_decomps(qml.CRX)[0].compute_resources()
    {qml.RX: 2, qml.CZ: 2}
    >>> print(qml.draw(qml.list_decomps(qml.CRX)[0])(0.5, wires=[0, 1]))
    0: ───────────╭●────────────╭●─┤
    1: ──RX(0.25)─╰Z──RX(-0.25)─╰Z─┤
    ```
* Decomposition rules are implemented using the new infrastructure for most PennyLane operators excluding templates.
  [(#6951)](https://github.com/PennyLaneAI/pennylane/pull/6951)

* A graph-based decomposition solver has been implemented that solves for the optimal decomposition
  rule to use for an operator towards a target gate set.
  [(#6952)](https://github.com/PennyLaneAI/pennylane/pull/6952)
  [(#7045)](https://github.com/PennyLaneAI/pennylane/pull/7045)
  [(#7058)](https://github.com/PennyLaneAI/pennylane/pull/7058)
  [(#7064)](https://github.com/PennyLaneAI/pennylane/pull/7064)

* Integrate the graph-based decomposition solver with `qml.transforms.decompose`.
  [(#6966)](https://github.com/PennyLaneAI/pennylane/pull/6966)

=======
>>>>>>> master
<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
