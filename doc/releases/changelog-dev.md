# Release 0.46.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Instances of `C(Prod)` now have a significantly more efficient decomposition in terms of `TemporaryAND` operators when work wires are provided.

  For example, a controlled multi-target-``X`` operation previously decomposed as

  ```
  c1: ─╭●─╭●─╭●─╭●─┤  State
  c2: ─├●─├●─├●─├●─┤  State
  c3: ─├●─├●─├●─├●─┤  State
   3: ─╰X─│──│──│──┤  State
   2: ────╰X─│──│──┤  State
   1: ───────╰X─│──┤  State
   0: ──────────╰X─┤  State
  ```

  With this upgrade, it decomposes into a ``TemporaryAND`` ladder and individual ``CNOT`` gates when work wires are available:

  ```python
  @qp.transforms.decompose(
      gate_set={"TemporaryAND":4, "Adjoint(TemporaryAND)":1, "MultiControlledX":7, "CNOT":1}
  )
  @qp.qnode(qp.device("default.qubit"))
  def qnode():
      qp.ctrl(qp.X(0) @ qp.X(1) @ qp.X(2) @ qp.X(3), control=["c1", "c2", "c3"], work_wires=["w1", "w2"], work_wire_type="zeroed")
      return qp.state()

  print(qp.draw(qnode)())
  ```

  ```
  c1: ─╭●─────────────────────●╮─┤  State
  c2: ─├●─────────────────────●┤─┤  State
  w1: ─╰⊕─╭●──────────────●╮──⊕╯─┤  State
  c3: ────├●──────────────●┤─────┤  State
  w2: ────╰⊕─╭●─╭●─╭●─╭●──⊕╯─────┤  State
   3: ───────╰X─│──│──│──────────┤  State
   2: ──────────╰X─│──│──────────┤  State
   1: ─────────────╰X─│──────────┤  State
   0: ────────────────╰X─────────┤  State
  ```
  [(#9368)](https://github.com/PennyLaneAI/pennylane/pull/9368)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>
