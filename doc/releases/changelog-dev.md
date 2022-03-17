:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* A new task-based device, `task.qubit` has been released. This allows offloading of multiple quantum circuits and workflows over all available cores. The following example demonstrates how to create task-based circuit evaluations using `default.qubit` as a compute backend, submitting them to the available queue:

  ```python 
    import dask.distributed as dist

    if __name__ == '__main__':
        cluster = dist.LocalCluster(n_workers=4, threads_per_worker=1)
        client = dist.Client(cluster)

        dev = qml.device("task.qubit", wires=6, backend="default.qubit")
        @qml.qnode(dev, cache=False, interface="tf", diff_method="parameter-shift") # caching must be disabled due to proxy interface
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.RZ(x[2], wires=0)
            qml.RZ(x[0], wires=1)
            qml.RX(x[1], wires=1)
            qml.RY(x[2], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        weights = tf.Variable(np.random.rand(3))
        def f_submit(weights):
            with tf.GradientTape() as tape:
                # Use the circuit to calculate the loss value
                loss = tf.abs(circuit(weights)-0.5)**2
            return tape.gradient(loss, weights)
  ```

  We can now submit the task on the running backend client:

  ```pycon
    >>> qml.taskify(f_submit)(weights)
    tf.Tensor([0.01776833 0.05199685 0.03689981], shape=(3,), dtype=float64)
  ```

  We can also spawn tasks asynchronously, and gather the final results after completion. For the above example, we will asynchronously submit the execution with different weights, and await on the results:

  Here, we submit 3 separate tasks with the weights scaled, and collect returned futures:

  ```pycon
    >>> results = [qml.taskify(f_submit, futures=True)(tf.Variable(weights*i)) for i in range(3)]
    >>> qml.untaskify(results)()
    [<tf.Tensor: shape=(3,), dtype=float64, numpy=array([0., 0., 0.])>, <tf.Tensor: shape=(3,), dtype=float64, numpy=array([-0.16661672, -0.07170375, -0.00387164])>, <tf.Tensor: shape=(3,), dtype=float64, numpy=array([ 0.94292007, -0.14209482, -0.0072056 ])>]
  ```

* Development of a circuit-cutting compiler extension to circuits with sampling
  measurements has begun:

  - The existing `qcut.tape_to_graph()` method has been extended to convert a
    sample measurement without an observable specified to multiple single-qubit sample
    nodes.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)

<h3>Improvements</h3>

* The function `qml.ctrl` was given the optional argument `control_values=None`.
  If overridden, `control_values` takes an integer or a list of integers corresponding to
  the binary value that each control value should take. The same change is reflected in
  `ControlledOperation`. Control values of `0` are implemented by `qml.PauliX` applied
  before and after the controlled operation
  [(#2288)](https://github.com/PennyLaneAI/pennylane/pull/2288)

* Circuit cutting now performs expansion to search for wire cuts in contained operations or tapes.
  [(#2340)](https://github.com/PennyLaneAI/pennylane/pull/2340)

<h3>Deprecations</h3>

<h3>Breaking changes</h3>

* The `ObservableReturnTypes` `Sample`, `Variance`, `Expectation`, `Probability`, `State`, and `MidMeasure`
  have been moved to `measurements` from `operation`.
  [(#2329)](https://github.com/PennyLaneAI/pennylane/pull/2329)

* The deprecated QNode, available via `qml.qnode_old.QNode`, has been removed. Please
  transition to using the standard `qml.QNode`.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated, non-batch compatible interfaces, have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated tape subclasses `QubitParamShiftTape`, `JacobianTape`, `CVParamShiftTape`, and
  `ReversibleTape` have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

<h3>Bug fixes</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Thomas Bromley, Anthony Hayes, Josh Izaac, Christina Lee.
