:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>
* The `qml.fourier.reconstruct` function is added. It can be used to
  reconstruct one-dimensional Fourier series with a minimal number of calls
  to the original function.
  [(#1864)](https://github.com/PennyLaneAI/pennylane/pull/1864)

  The used reconstruction technique differs for functions with equidistant frequencies
  that are reconstructed using the function value at equidistant sampling points and
  for functions with arbitrary frequencies reconstructed using arbitrary sampling points.

  As an example, consider the following QNode:

  ```python
  dev = qml.device("default.qubit", wires=2)
  
  @qml.qnode(dev)
  def circuit(x, Y, f=1.0):
      qml.RX(f*x, wires=0)
      qml.RY(Y[0], wires=0)
      qml.RY(Y[1], wires=1)
      qml.CNOT(wires=[0, 1])
      qml.RY(3*Y[1], wires=1)
      return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
  ```

  It has three variational parameters overall: A scalar input `x`
  and an array-valued input `Y` with two entries. Additionally, we can
  tune the dependence on `x` with the frequency `f`.
  We then can reconstruct the QNode output function with respect to `x` via

  ```pycon
  >>> x = 0.3
  >>> Y = np.array([0.1, -0.9])
  >>> rec = qml.fourier.reconstruct(circuit, ids="x", nums_frequency={"x": {0: 1}})(x, Y)
  >>> rec
  {'x': {0: <function pennylane.fourier.reconstruct._reconstruct_equ.<locals>._reconstruction(x)>}}
  ```

  As we can see, we get a nested dictionary in the format of the input `nums_frequency`
  with functions as values. These functions are simple float-to-float callables:

  ```pycon
  >>> univariate = rec["x"][0]
  >>> univariate(x)
  -0.880208251507
  ```

  For more details on usage, reconstruction cost and differentiability support, please see the
  [fourier.reconstruct docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.fourier.reconstruct.html)

* A thermal relaxation channel is added to the Noisy channels. The channel description can be 
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

<h3>Improvements</h3>

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP` 
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order): 

Guillermo Alonso-Linaje, Jalani Kanem, Christina Lee, Alejandro Montanez, David Wierichs

