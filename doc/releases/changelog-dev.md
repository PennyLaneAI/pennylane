# Release 0.46.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Added three decompositions of :class:`~.OutMultiplier` that use significantly fewer costly gates
  than the existing QFT-based decomposition, at the cost of more auxiliary wires.
  In addition added a new argument ``output_wires_zeroed`` to ``OutMultiplier`` that can be
  used to indicate ``output_wires`` to be in the :math:`|0\rangle` state, leading to cheaper
  decompositions.
  [(#8900)](https://github.com/PennyLaneAI/pennylane/pull/8900)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Created a new ``~.labs.templates.LeftQuantumComparator`` template for performing inequality test of two quantum registers.
  [(#9277)](https://github.com/PennyLaneAI/pennylane/pull/9277)

  ```python
  import pennylane as qp
  from pennylane.labs.templates import LeftQuantumComparator
  
  dev = qp.device("lightning.qubit")

  @qp.qnode(dev, shots=1)
  def circuit(a, comparator, b):
    x_wires = [0, 3, 6, 9]
    y_wires = [1, 4, 7, 10]
    work_wires = [2, 5, 8]
    qp.BasisState(a, wires=x_wires)
    qp.BasisState(b, wires=y_wires)
    LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator)
    qp.CNOT(wires=[11, 12])
    qp.adjoint(LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator))

    return qp.sample(wires=[12])
  ```
  
  ```pycon
    >>> output = circuit(3, ">=", 2)
    >>> print(bool(output))
    True
  
  ```
  
<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

* The ``qp.transforms.create_expand_fn`` has been removed.
  Instead, please use the :func:`qp.transforms.decompose <.transforms.decompose>` function for decomposing circuits.
  [(#9472)](https://github.com/PennyLaneAI/pennylane/pull/9472)

<h3>Internal changes ⚙️</h3>

* Added usage of the `strict` keyword argument for `zip` throughout the codebase.
  [(#9393)](https://github.com/PennyLaneAI/pennylane/pull/9393)
  

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Marcus Edwards,
David Wierichs
