.. role:: html(raw)
   :format: html

.. _intro_interfaces:

Gradients and training
======================

PennyLane offers seamless integration between classical and quantum computations. Code up quantum
circuits in PennyLane, compute :doc:`quantum gradients <glossary/quantum_gradient>`, and
connect them easily to the top scientific computing and machine learning libraries.

Gradients
---------

When creating a QNode, you can specify the :doc:`differentiation method
<glossary/quantum_differentiable_programming>` that PennyLane should use whenever the gradient of
that QNode is requested.

.. code-block:: python

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.probs(wires=0)

PennyLane currently provides the following differentiation methods for QNodes, divided into
the following three categories.

.. note::

    If not specified, the default differentiation method is ``diff_method="best"``. PennyLane
    will attempt to determine the *best* differentiation method given the device and interface.
    Typically, PennyLane will prioritize device-provided gradients, backpropagation, parameter-shift
    rule, and finally finite-differences, in that order.

Device gradients
~~~~~~~~~~~~~~~~

* ``"device"``: Queries the device directly for the gradient.
  Only allowed on devices that provide their own gradient computation.

Reverse-mode differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods only support state-vector simulators, they are **not** hardware compatible;
they are only supported on *statevector* simulator devices such as :class:`default.qubit
<~.DefaultQubit>`.

However, for rapid prototyping on simulators, these methods typically out-perform forward-mode
accumulators such as the parameter-shift rule and finite-differences. For more details, see the
:doc:`quantum backpropagation <demos/tutorial_backprop>` demonstration.

* ``"backprop"``: Use standard backpropagation.

  This differentiation method is only allowed on simulator
  devices that are classically end-to-end differentiable, for example
  :class:`default.qubit <~.DefaultQubit>`. This method does *not* work on devices
  that estimate measurement statistics using a finite number of shots; please use
  the ``parameter-shift`` rule instead.

* ``"adjoint"``: A form of backpropagation that takes advantage of the unitary or reversible
  nature of quantum computation.

  The `adjoint method <https://arxiv.org/abs/2009.02823>`__  reverses through the circuit after a
  forward pass by iteratively applying the inverse (adjoint) gate. This method is similar to
  ``"backprop"``, but trades off increased runtime with significantly lower memory usage.

* ``"reversible"``: A form of backpropagation that takes advantage of the unitary or reversible
  nature of quantum computation.

  This method is similar to the ``adjoint`` method, but has a lower time overhead and a similar
  memory overhead.

  Compared to the parameter-shift rule, the reversible method can be faster or slower, depending on
  the density and location of parametrized gates in a circuit.

Forward-mode differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods support both quantum hardware and simulators. However, when using a simulator,
you may notice that the computational time required to compute the gradients scales
with the number of trainable circuit parameters.

* ``"parameter-shift"``: Use the analytic :doc:`parameter-shift rule
  <glossary/parameter_shift>` for all supported quantum operation arguments, with
  finite-difference as a fallback.

* ``"finite-diff"``: Uses numerical finite-differences for all quantum operation arguments.


Training and interfaces
-----------------------

The bridge between the quantum and classical worlds is provided in PennyLane via *interfaces*.
Currently, there are four built-in interfaces: NumPy, PyTorch, JAX, and TensorFlow. These interfaces
make each of these libraries quantum-aware, allowing quantum circuits to be treated just like any
other operation.

In PennyLane, an interface is declared when creating a :class:`~.QNode`, e.g.,

.. code-block:: python

    @qml.qnode(dev, interface="tf")
    def my_quantum_circuit(...):
        ...

.. note::
    If no interface is specified, PennyLane will default to the NumPy interface (powered by the
    `autograd <https://github.com/HIPS/autograd>`_ library).

.. note::

    QNodes with an interface will always incur a small overhead on evaluation. If you do not
    need to compute quantum gradients of a QNode, specifying ``interface=None`` will remove
    this overhead and result in a slightly faster evaluation. However, gradients will no
    longer be available.

This will allow native numerical objects of the specified library (NumPy arrays, Torch Tensors,
or TensorFlow Tensors) to be passed as parameters to the quantum circuit. It also makes
the gradients of the quantum circuit accessible to the classical library, enabling the
optimization of arbitrary hybrid circuits.

See the links below for walkthroughs of each specific interface:

.. raw:: html

    <style>
        #interfaces .card {
            box-shadow: none!important;
        }
        #interfaces .card:hover {
            box-shadow: none!important;
        }
    </style>
    <div id="interfaces" class="container mt-2 mb-2">
        <div class="row mt-3">
            <div class="col-lg-3 mb-2 align-items-stretch">
                <a href="interfaces/numpy.html">
                    <div class="card rounded-lg py-2" style="height:100%;">
                        <div class="d-flex justify-content-center align-items-center" style="height:100%;">
                            <img src="../_static/numpy.png" class="card-img-top" style="width:80%;"></img>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-3 mb-2 align-items-stretch">
                <a href="interfaces/torch.html">
                    <div class="card rounded-lg py-2" style="height:100%;">
                        <div class="d-flex justify-content-center align-items-center" style="height:100%;">
                          <img src="../_static/pytorch.png" class="card-img-top"></img>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-3 mb-2 align-items-stretch">
                <a href="interfaces/tf.html">
                    <div class="card rounded-lg py-2" style="height:100%;">
                        <div class="d-flex justify-content-center align-items-center" style="height:100%;">
                            <img src="../_static/tensorflow.png" class="card-img-top" style="width:90%;"></img>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-3 mb-2 align-items-stretch">
                <a href="interfaces/jax.html">
                    <div class="card rounded-lg py-2" style="height:100%;">
                        <div class="d-flex justify-content-center align-items-center" style="height:100%;">
                            <img src="../_static/jax.png" class="card-img-top" style="max-width:60%;"></img>
                        </div>
                    </div>
                </a>
            </div>
        </div>
    </div>

In addition to the core interfaces discussed above, PennyLane also provides higher-level classes for
converting QNodes into both Keras and ``torch.nn`` layers:


:html:`<div class="summary-table">`

.. autosummary::

    pennylane.qnn.KerasLayer
    pennylane.qnn.TorchLayer

:html:`</div>`

.. toctree::
    :hidden:

    interfaces/numpy
    interfaces/torch
    interfaces/tf
    interfaces/jax
