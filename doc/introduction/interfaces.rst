.. role:: html(raw)
   :format: html

.. _intro_interfaces:

Gradients and training
======================

PennyLane offers seamless integration between classical and quantum computations. Code up quantum
circuits in PennyLane, compute :doc:`gradients of quantum circuits <glossary/quantum_gradient>`, and
connect them easily to the top scientific computing and machine learning libraries.

Training and interfaces
-----------------------

The bridge between the quantum and classical worlds is provided in PennyLane via interfaces to
automatic differentiation libraries.
Currently, four libraries are supported: :doc:`NumPy <interfaces/numpy>`, :doc:`PyTorch
<interfaces/torch>`, :doc:`JAX <interfaces/jax>`, and :doc:`TensorFlow <interfaces/tf>`. PennyLane makes
each of these libraries quantum-aware, allowing quantum circuits to be treated just
like any other operation. Any automatic differentiation framework can be chosen with any device.

In PennyLane, an automatic differentiation framework is declared using the ``interface`` argument when creating
a :class:`QNode <pennylane.QNode>`, e.g.,

.. code-block:: python

    @qml.qnode(dev, interface="tf")
    def my_quantum_circuit(...):
        ...

.. note::
    If no interface is specified, PennyLane will default to the NumPy interface (powered by the
    `autograd <https://github.com/HIPS/autograd>`_ library).

This will allow native numerical objects of the specified library (NumPy arrays, JAX arrays, Torch Tensors,
or TensorFlow Tensors) to be passed as parameters to the quantum circuit. It also makes
the gradients of the quantum circuit accessible to the classical library, enabling the
optimization of arbitrary hybrid circuits by making use of the library's native optimizers.

When specifying an interface, the objects of the chosen framework are converted
into NumPy objects and are passed to a device in most cases. Exceptions include
cases when the devices support end-to-end computations in a framework. Such
devices may be referred to as backpropagation or passthru devices.

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

In addition to the core automatic differentiation frameworks discussed above,
PennyLane also provides higher-level classes for converting QNodes into both Keras and ``torch.nn`` layers:


:html:`<div class="summary-table">`

.. autosummary::

    pennylane.qnn.KerasLayer
    pennylane.qnn.TorchLayer


.. note::

    QNodes that allow for automatic differentiation will always incur a small overhead on evaluation.
    If you do not need to compute quantum gradients of a QNode, specifying ``interface=None`` will remove
    this overhead and result in a slightly faster evaluation. However, gradients will no
    longer be available.

.. _intro_ref_opt:

Optimizers
----------

Optimizers are objects which can be used to automatically update the parameters of a quantum
or hybrid machine learning model. The optimizers you should use are dependent on your choice
of the classical autodifferentiation library, and are available from different access
points.

NumPy
~~~~~

When using the standard NumPy framework, PennyLane offers some built-in optimizers.
Some of these are specific to quantum optimization, such as the :class:`~.QNGOptimizer`, :class:`~.LieAlgebraOptimizer`
:class:`~.RotosolveOptimizer`, :class:`~.RotoselectOptimizer`, :class:`~.ShotAdaptiveOptimizer`, and :class:`~.QNSPSAOptimizer`.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.AdagradOptimizer
    ~pennylane.AdamOptimizer
    ~pennylane.GradientDescentOptimizer
    ~pennylane.LieAlgebraOptimizer
    ~pennylane.MomentumOptimizer
    ~pennylane.NesterovMomentumOptimizer
    ~pennylane.QNGOptimizer
    ~pennylane.RMSPropOptimizer
    ~pennylane.RotosolveOptimizer
    ~pennylane.RotoselectOptimizer
    ~pennylane.ShotAdaptiveOptimizer
    ~pennylane.SPSAOptimizer
    ~pennylane.QNSPSAOptimizer

:html:`</div>`

PyTorch
~~~~~~~

If you are using the :ref:`PennyLane PyTorch framework <torch_interf>`, you should import one of the native
`PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ (found in ``torch.optim``).

TensorFlow
~~~~~~~~~~

When using the :ref:`PennyLane TensorFlow framework <tf_interf>`, you will need to leverage one of
the `TensorFlow optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer>`_
(found in ``tf.keras.optimizers``).

JAX
~~~

Check out the `JAXopt <https://github.com/google/jaxopt>`_ and the `Optax
<https://optax.readthedocs.io/en/latest/>`_ packages to find optimizers for the
:ref:`PennyLane JAX framework <jax_interf>`.

Gradients
---------

The interface between PennyLane and automatic differentiation libraries relies on PennyLane's ability
to compute or estimate gradients of quantum circuits. There are different strategies to do so, and they may
depend on the device used.

When creating a QNode, you can specify the :doc:`differentiation method
<glossary/quantum_differentiable_programming>` like this:

.. code-block:: python

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.probs(wires=0)

PennyLane currently provides the following differentiation methods for QNodes:

Simulation-based differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods use `reverse accumulation
<https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation>`__ to compute
gradients; a well-known example of this approach is backpropagation. These methods are **not** hardware compatible; they are only supported on
*statevector* simulator devices such as :class:`default.qubit <~.DefaultQubit>`.

However, for rapid prototyping on simulators, these methods typically out-perform forward-mode
accumulators such as the parameter-shift rule and finite-differences. For more details, see the
:doc:`quantum backpropagation <demos/tutorial_backprop>` demonstration.

* ``"backprop"``: Use standard backpropagation.

  This differentiation method is only allowed on simulator
  devices that are classically end-to-end differentiable, for example
  :class:`default.qubit <~.DefaultQubit>`. This method does *not* work on devices
  that estimate measurement statistics using a finite number of shots; please use
  the ``parameter-shift`` rule instead.

* ``"adjoint"``: Use a form of backpropagation that takes advantage of the unitary or reversible
  nature of quantum computation.

  The `adjoint method <https://arxiv.org/abs/2009.02823>`__  reverses through the circuit after a
  forward pass by iteratively applying the inverse (adjoint) gate. This method is similar to
  ``"backprop"``, but has significantly lower memory usage and a similar runtime.

Hardware-compatible differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods support both quantum hardware and simulators, and are examples of `forward
accumulation <https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation>`__.
However, when using a simulator, you may notice that the time required to compute the gradients
with these methods :doc:`scales linearly <demos/tutorial_backprop>` with the number of trainable circuit
parameters.

* ``"parameter-shift"``: Use the analytic :doc:`parameter-shift rule
  <glossary/parameter_shift>` for all supported quantum operation arguments, with
  finite-difference as a fallback.

* ``"finite-diff"``: Use numerical finite-differences for all quantum operation arguments.


Device gradients
~~~~~~~~~~~~~~~~

* ``"device"``: Queries the device directly for the gradient.
  Only allowed on devices that provide their own gradient computation.


.. note::

    If not specified, the default differentiation method is ``diff_method="best"``. PennyLane
    will attempt to determine the *best* differentiation method given the device and interface.
    Typically, PennyLane will prioritize device-provided gradients, backpropagation, parameter-shift
    rule, and finally finite differences, in that order.


Gradient transforms
-------------------

In addition to registering the differentiation method of QNodes to be used with autodifferentiation
frameworks, PennyLane also provides a library of **gradient transforms** via the
:mod:`qml.gradients <pennylane.gradients>` module.

Quantum gradient transforms are strategies for computing the gradient of a quantum
circuit that work by **transforming** the quantum circuit into one or more gradient circuits.
They accompany these circuits with a function that **post-processes** their output.
These gradient circuits, once executed and post-processed, return the gradient
of the original circuit.

Examples of quantum gradient transforms include finite-difference rules and parameter-shift
rules; these can be applied *directly* to QNodes:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.probs(wires=1)

>>> weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
>>> circuit(weights)
tensor([0.9658079, 0.0341921], requires_grad=True)
>>> qml.gradients.param_shift(circuit)(weights)
tensor([[-0.04673668, -0.09442394, -0.14409127],
        [ 0.04673668,  0.09442394,  0.14409127]], requires_grad=True)

Note that, while gradient transforms allow quantum gradient rules to be applied directly to QNodes,
this is not a replacement --- and should not be used instead of --- standard training workflows (for example,
``qml.grad()`` if using Autograd, ``loss.backward()`` for PyTorch, or ``tape.gradient()`` for TensorFlow).
This is because gradient transforms do not take into account classical computation nodes, and only
support gradients of QNodes.
For more details on available gradient transforms, as well as learning how to define your own
gradient transform, please see the :mod:`qml.gradients <pennylane.gradients>` documentation.


Differentiating gradient transforms and higher-order derivatives
----------------------------------------------------------------

Gradient transforms are themselves differentiable, allowing higher-order
gradients to be computed:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(qml.PauliZ(1))

>>> weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
>>> circuit(weights)
tensor(0.9316158, requires_grad=True)
>>> qml.gradients.param_shift(circuit)(weights)  # gradient
array([[-0.09347337, -0.18884787, -0.28818254]])
>>> qml.jacobian(qml.gradients.param_shift(circuit))(weights)  # hessian
array([[[-0.9316158 ,  0.01894799,  0.0289147 ],
        [ 0.01894799, -0.9316158 ,  0.05841749],
        [ 0.0289147 ,  0.05841749, -0.9316158 ]]])

Another way to compute higher-order derivatives is by passing the ``max_diff`` and
``diff_method`` arguments to the QNode and by successive differentiation:

.. code-block:: python

    @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(qml.PauliZ(1))

>>> weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
>>> qml.jacobian(qml.jacobian(circuit))(weights)  # hessian
array([[-0.9316158 ,  0.01894799,  0.0289147 ],
       [ 0.01894799, -0.9316158 ,  0.05841749],
       [ 0.0289147 ,  0.05841749, -0.9316158 ]])

Note that the ``max_diff`` argument only applies to gradient transforms and that its default value is ``1``; failing to
set its value correctly may yield incorrect results for higher-order derivatives. Also, passing
``diff_method="parameter-shift"`` is equivalent to passing ``diff_method=qml.gradients.param_shift``.

Supported configurations
------------------------

.. role:: gr
.. role:: rd

.. raw:: html

   <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
   <script>
     $(document).ready(function() {
       $('.gr').parent().parent().addClass('gr-parent');
       $('.rd').parent().parent().addClass('rd-parent');
     });
   </script>
   <style>
       .gr-parent {background-color:#bbffbb}
       .rd-parent {background-color:#ffbbbb}
   </style>

The table below show all the currently supported functionality for the ``"default.qubit"`` device.
At the moment, it takes into account the following parameters:

* The interface, e.g. ``"jax"``
* The differentiation method, e.g. ``"parameter-shift"``
* The return value of the QNode, e.g. ``qml.expval()`` or ``qml.probs()``
* The number of shots, either None or an integer > 0

.. raw:: html

   <style>
      .tb { border-collapse: collapse; }
      .tb th, .tb td { padding: 1px; border: solid 1px black; }
   </style>

.. rst-class:: tb

+-------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                 | **Return type**                                                                                                                                        |
+==================+==============================+==============+===============+==============+==============+===============+================+================+=============+=============+=============+
| **Interface**    |**Differentiation method**    | state        |density matrix |  probs       | sample       |expval (obs)   | expval (herm)  | expval (proj)  | var         | vn entropy  | mutual info |
+------------------+------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
| ``None``         | ``"device"``                 |    :rd:`1`   |      :rd:`1`  |   :rd:`1`    | :rd:`1`      |  :rd:`1`      | :rd:`1`        | :rd:`1`        | :rd:`1`     | :rd:`1`     | :rd:`1`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"backprop"``               |    :rd:`1`   |      :rd:`1`  |   :rd:`1`    | :rd:`1`      |  :rd:`1`      | :rd:`1`        | :rd:`1`        | :rd:`1`     | :rd:`1`     | :rd:`1`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"adjoint"``                |    :rd:`2`   |     :rd:`2`   |    :rd:`2`   | :rd:`2`      | :rd:`2`       |   :rd:`2`      | :rd:`2`        |:rd:`2`      |:rd:`2`      |:rd:`2`      |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"parameter-shift"``        |    :rd:`2`   |     :rd:`2`   |    :rd:`2`   | :rd:`2`      | :rd:`2`       |   :rd:`2`      | :rd:`2`        |:rd:`2`      |:rd:`2`      |:rd:`2`      |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"finite-diff"``            |    :rd:`2`   |     :rd:`2`   |    :rd:`2`   | :rd:`2`      | :rd:`2`       |   :rd:`2`      | :rd:`2`        |:rd:`2`      |:rd:`2`      |:rd:`2`      |
+------------------+------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
| ``"autograd"``   | ``"device"``                 |  :rd:`3`     |      :rd:`3`  |    :rd:`3`   | :rd:`3`      |   :rd:`3`     |  :rd:`3`       |   :rd:`3`      | :rd:`3`     | :rd:`3`     | :rd:`3`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"backprop"``               |     :gr:`4`  |   :gr:`4`     |     :gr:`5`  |     :rd:`9`  |   :gr:`5`     |    :gr:`5`     |   :gr:`5`      | :gr:`5`     | :gr:`5`     | :gr:`5`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"adjoint"``                |      :rd:`6` |     :rd:`6`   |  :rd:`6`     | :rd:`6`      |      :gr:`7`  |  :gr:`7`       |   :gr:`7`      | :rd:`6`     | :rd:`6`     | :rd:`6`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"parameter-shift"``        |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :rd:`10`  |   :rd:`10`  |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"finite-diff"``            |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :gr:`8`   |   :gr:`8`   |
+------------------+------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
| ``"jax"``        | ``"device"``                 |  :rd:`3`     |      :rd:`3`  |    :rd:`3`   | :rd:`3`      |   :rd:`3`     |  :rd:`3`       |   :rd:`3`      | :rd:`3`     | :rd:`3`     | :rd:`3`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"backprop"``               |     :gr:`5`  |   :gr:`5`     |     :gr:`5`  |     :rd:`9`  |   :gr:`5`     |    :gr:`5`     |   :gr:`5`      | :gr:`5`     | :gr:`5`     | :gr:`5`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"adjoint"``                |      :rd:`6` |     :rd:`6`   |  :rd:`6`     | :rd:`6`      |      :gr:`7`  |  :gr:`7`       |   :gr:`7`      | :rd:`6`     | :rd:`6`     | :rd:`6`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"parameter-shift"``        |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :rd:`10`  |   :rd:`10`  |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"finite-diff"``            |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :gr:`8`   |   :gr:`8`   |
+------------------+------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
| ``"tf"``         | ``"device"``                 |  :rd:`3`     |      :rd:`3`  |    :rd:`3`   | :rd:`3`      |   :rd:`3`     |  :rd:`3`       |   :rd:`3`      | :rd:`3`     | :rd:`3`     | :rd:`3`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"backprop"``               |     :gr:`5`  |   :gr:`5`     |     :gr:`5`  |     :rd:`9`  |   :gr:`5`     |    :gr:`5`     |   :gr:`5`      | :gr:`5`     | :gr:`5`     | :gr:`5`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"adjoint"``                |      :rd:`6` |     :rd:`6`   |  :rd:`6`     | :rd:`6`      |      :gr:`7`  |  :gr:`7`       |   :gr:`7`      | :rd:`6`     | :rd:`6`     | :rd:`6`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"parameter-shift"``        |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :rd:`10`  |   :rd:`10`  |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"finite-diff"``            |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :gr:`8`   |   :gr:`8`   |
+------------------+------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
| ``"torch"``      | ``"device"``                 |  :rd:`3`     |      :rd:`3`  |    :rd:`3`   | :rd:`3`      |   :rd:`3`     |  :rd:`3`       |   :rd:`3`      | :rd:`3`     | :rd:`3`     | :rd:`3`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"backprop"``               |     :gr:`5`  |   :gr:`5`     |     :gr:`5`  |     :rd:`9`  |   :gr:`5`     |    :gr:`5`     |   :gr:`5`      | :gr:`5`     | :gr:`5`     | :gr:`5`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"adjoint"``                |      :rd:`6` |     :rd:`6`   |  :rd:`6`     | :rd:`6`      |      :gr:`7`  |  :gr:`7`       |   :gr:`7`      | :rd:`6`     | :rd:`6`     | :rd:`6`     |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"parameter-shift"``        |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :rd:`10`  |   :rd:`10`  |
+                  +------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+
|                  | ``"finite-diff"``            |   :rd:`10`   |    :rd:`10`   |   :gr:`8`    |  :rd:`9`     |   :gr:`8`     |   :gr:`8`      | :gr:`8`        |   :gr:`8`   |   :gr:`8`   |   :gr:`8`   |
+------------------+------------------------------+--------------+---------------+--------------+--------------+---------------+----------------+----------------+-------------+-------------+-------------+

1. Not supported. Gradients are not computed even though ``diff_method`` is provided. Fails with error.
2. Not supported. Gradients are not computed even though ``diff_method`` is provided. Warns that no auto-differentiation framework is being used, but does not fail.
   Forward pass is still supported.
3. Not supported. The ``default.qubit`` device does not provide a native way to compute gradients. See
   :ref:`Device jacobian <Device jacobian>` for details.
4. Supported, but only when ``shots=None``. See :ref:`Backpropagation <Analytic backpropagation>` for details.

   If the circuit returns a state, then the circuit itself is not differentiable
   directly. However, any real scalar-valued post-processing done to the output of the
   circuit will be differentiable. See :ref:`State gradients <State gradients>` for details.
5. Supported, but only when ``shots=None``. See :ref:`Backpropagation <Analytic backpropagation>` for details.
6. Not supported. The adjoint differentiation algorithm is only implemented for computing the expectation values of observables. See
   :ref:`Adjoint differentation <Adjoint differentation>` for details.
7. Supported. Raises warning when ``shots>0`` since the gradient is always computed analytically. See
   :ref:`Adjoint differentation <Adjoint differentation>` for details.
8. Supported.
9. Not supported. The discretization of the output caused by wave function collapse is
   not differentiable. The forward pass is still supported. See :ref:`Sample gradients <Sample gradients>` for details.
10. Not supported. "We just don't have the theory yet."

:html:`</div>`

.. toctree::
    :hidden:

    interfaces/numpy
    interfaces/torch
    interfaces/tf
    interfaces/jax
    unsupported_gradients
