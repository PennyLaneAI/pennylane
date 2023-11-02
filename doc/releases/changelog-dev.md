:orphan:

# Release 0.34.0-dev (development release)

<h3>New features since last release</h3>

* Approximate Quantum Fourier Transform (AQFT) is now available from `qml.AQFT`.
  [(#4656)](https://github.com/PennyLaneAI/pennylane/pull/4656)

  * The `CosineWindow` template has been added to prepare an initial state based on a cosine wave function.
    [(#4683)](https://github.com/PennyLaneAI/pennylane/pull/4683)

    ```python
    import pennylane as qml
    import matplotlib.pyplot as plt
  
    dev = qml.device('default.qubit', wires=4)
  
    @qml.qnode(dev)
    def example_circuit():
          qml.CosineWindow(wires=range(4))
          return qml.state()
    output = example_circuit()
  
    # Graph showing state amplitudes
    plt.bar(range(len(output)), output)
    plt.show()
    ```
  
    We can show how this operator is built:
  
    ```python
    import pennylane as qml

    dev = qml.device("default.qubit", wires=5)
  
    op = qml.CosineWindow(wires=range(5))
    
    @qml.qnode(dev)
    def circuit():
        op.decomposition()
        return qml.state()
  
    print(qml.draw(circuit)())

    ```
    
    ```pycon
  
    0: ──────────────╭QFT†──Rϕ(1.57)─┤  State
    1: ──────────────├QFT†──Rϕ(0.79)─┤  State
    2: ──────────────├QFT†──Rϕ(0.39)─┤  State
    3: ──────────────├QFT†──Rϕ(0.20)─┤  State
    4: ──H──RZ(3.14)─╰QFT†──Rϕ(0.10)─┤  State
    
    ```  

* `qml.transforms.decomposition.sk_decomposition` method implements the Solovay-Kitaev algorithm for
  approximately decomposing any single-qubit operation to Clifford+T basis.
  [(#4687)](https://github.com/PennyLaneAI/pennylane/pull/4687)

<h3>Improvements 🛠</h3>

* Updates to some relevant Pytests to enable its use as a suite of benchmarks.
  [(#4703)](https://github.com/PennyLaneAI/pennylane/pull/4703)

* Added `__iadd__` method to PauliSentence, which enables inplace-addition using `+=`, we no longer need to perform a copy, leading to performance improvements.
[(#4662)](https://github.com/PennyLaneAI/pennylane/pull/4662) 

* `qml.ArbitraryUnitary` now supports batching.
  [(#4745)](https://github.com/PennyLaneAI/pennylane/pull/4745)

* `qml.draw` and `qml.draw_mpl` now render operator ids.
  [(#4749)](https://github.com/PennyLaneAI/pennylane/pull/4749)

<h3>Breaking changes 💔</h3>

* The `prep` keyword argument has been removed from `QuantumScript` and `QuantumTape`.
  `StatePrepBase` operations should be placed at the beginning of the `ops` list instead.
  [(#4756)](https://github.com/PennyLaneAI/pennylane/pull/4756)

* `qml.gradients.pulse_generator` has become `qml.gradients.pulse_odegen` to adhere to paper naming conventions.
  [(#4769)](https://github.com/PennyLaneAI/pennylane/pull/4769)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* Documentation page for `qml.measurements` now links top-level accessible functions (e.g. `qml.expval`) 
  to their top-level pages (rather than their module-level pages, eg. `qml.measurements.expval`).
  [(#4750)](https://github.com/PennyLaneAI/pennylane/pull/4750)

<h3>Bug fixes 🐛</h3>

* Jax jit now works with shot vectors.
  [(#4772)](https://github.com/PennyLaneAI/pennylane/pull/4772/)

* Any `ScalarSymbolicOp`, like `Evolution`, now states that it has a matrix if the target
  is a `Hamiltonian`.
  [(#4768)](https://github.com/PennyLaneAI/pennylane/pull/4768)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Amintor Dusko,
Lillian Frederiksen,
Ankit Khandelwal,
Christina Lee,
Anurav Modak,
Matthew Silverman,
David Wierichs,
Justin Woodring,
