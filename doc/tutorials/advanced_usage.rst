.. role:: html(raw)
   :format: html

.. _advanced_features:

Advanced Usage
==============

In the previous three introductory tutorials (:ref:`qubit rotation <qubit_rotation>`, :ref:`Gaussian transformation <gaussian_transformation>`, and :ref:`plugins & hybrid computation <plugins_hybrid>`) we explored the basic concepts of PennyLane, including qubit- and CV-model quantum computations, gradient-based optimization, and the construction of hybrid classical-quantum computations. 

In this tutorial, we will highlight some of the more advanced features of Pennylane. 

Multiple expectation values
***************************

All the previous examples we considered utilized quantum functions with only single expectation values. In fact, PennyLane supports the return of multiple expectation values, up to one per wire. 

As usual, we begin by importing PennyLane and the PennyLane-provided version of NumPy, and set up a 2-wire qubit device for computations:

.. code::

    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device('default.qubit', wires=2)

We will start with a simple example circuit, which generates a two-qubit entangled state, then evaluates the expectation value of the Pauli Z operator on each wire.

.. code::

    @qml.qnode(dev)
    def circuit1(param):
        qml.RX(param, wires=0)
        qml.CNOT(wires=[0,1])
        return qml.expval.PauliZ(0), qml.expval.PauliZ(1)

The degree of entanglement of the qubits is determined by the value of ``param``. For a value of :math:`\frac{\pi}{2}`, they are maximally entangled. In this case, the reduced states on each subsystem are completely mixed, and local expectation values — like those we are measuring — will average to zero.

>>> circuit1(np.pi / 2)
>>> array([4.4408921e-16, 4.4408921e-16])

Notice that the output of the circuit is a NumPy array with ``shape=(2,)``, i.e., a two-dimensional vector. These two dimensions match the number of expectation values returned in our quantum function ``circuit1``.

.. note::
    It is important to emphasize that the expectation values in ``circuit`` are both **local**, i.e., this circuit is evaluating :math:`\langle \sigma_z \rangle_0` and :math:`\langle \sigma_z \rangle_1`, not :math:`\langle \sigma_z\otimes \sigma_z \rangle_{01}` (where the subscript denotes which wires the observable is located on).


Grad and Jacobian
*****************

How does automatic differentiation work in the case where the QNode returns multiple expectation values? If we were to naively try computing the gradient

.. code::

    g1 = qml.grad(circuit1, argnum=0)
    g1(np.pi / 2)
    
we would get an error message. The reason for this is that the `gradient <https://en.wikipedia.org/wiki/Gradient>`_ is only defined for scalar functions, i.e., functions which return a single value. In the case where the QNode returns multiple expectation values, this is obviously not the case. The correct differential operator to use in that case is the `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_. This can be accessed in PennyLane as ``jacobian``:

>>> j1 = qml.jacobian(circuit1, argnum=0)
>>> j1(np.pi / 2)
>>> array([-1., -1.])

The output of ``qml.jacobian`` is a two-dimensional vector, with the first/second element being the partial derivative of the first/second expectation value with respect to the input parameter. The Jacobian function has the same signature as the gradient function, requiring the user to specify which argument should be differentiated.

If you want to compute the Jacobian matrix for a function with multiple input parameters and multiple expectation values, the recommended way to do this is to combine the parameters into a single list/array and index into this inside your qfunc. Consider the following circuit:

.. code::

    @qml.qnode(dev)
    def circuit2(params):
        qml.RX(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        qml.RX(params[2], wires=1)
        qml.CNOT(wires=[0,1])
        return qml.expval.PauliZ(0), qml.expval.PauliZ(1)
        
It has a full Jacobian with two rows and three columns:

>>> j2 = qml.jacobian(circuit2, argnum=0)
>>> j2(np.pi / 3, 0.25, np.pi / 2)
>>> array([[-8.66025404e-01, -5.55111512e-17,  0.00000000e+00],
           [-4.71844785e-16, -1.38777878e-17, -5.00000000e-01]])

.. note:: Currently, ``qml.jacobian`` only the case supports when ``argnum`` is a single integer. For quantum functions with multiple arguments, use the above method to get the full Jacobian matrix.

    
Keyword arguments
*****************

While automatic differentiation is a handy feature, sometimes we want certain parts of our computational pipeline (e.g., the inputs :math:`x` to a parameterized quantum function :math:`f(x;\bf{\theta})` or the training data for a machine learning model) to not be differentiated. 

PennyLane uses the pattern that *all positional arguments to quantum functions are available to be differentiated*, while *keyword arguments are never differentiated*. Thus, when using the gradient-descent-based :mod:`optimizers <pennylane.optimize>` included in PennyLane, all numerical parameters appearing in non-keyword arguments will be updated, while all numerical values included as keyword arguments will not be updated.
