.. role:: html(raw)
   :format: html

.. _intro_interfaces:

Interfaces and training
=======================

PennyLane offers seamless integration between classical and quantum computations. Code up quantum
circuits in PennyLane, and connect them easily to the top scientific computing and machine learning
libraries.

The bridge between the quantum and classical worlds is provided in PennyLane via *interfaces*.
Currently, there are four built-in interfaces: NumPy, PyTorch, JAX, and TensorFlow.
These interfaces make each of these libraries quantum-aware, allowing quantum circuits to be
treated just like any other operation.

In PennyLane, an interface is declared when creating a :class:`~.QNode`, e.g.,

.. code-block:: python

    @qml.qnode(dev, interface="tf")
    def my_quantum_circuit(...):
        ...

.. note::
    If no interface is specified, PennyLane will default to the NumPy interface (powered by the
    `autograd <https://github.com/HIPS/autograd>`_ library).

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
