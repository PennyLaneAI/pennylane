.. role:: html(raw)
   :format: html

.. _advanced_usage:

Advanced Usage
==============

In the previous tutorials, we explored the basic concepts of PennyLane including qubit and CV model quantum computations, gradient-based optimization and the construction of hybrid classical-quantum computations.

PennyLane offers many exciting features such as constructing multiple QNodes on a single device, applying any arbitrary Unitary matrix to the states and measuring any custom-defined Hermitian operator. In this tutorial, we will highlight other advanced features of Pennylane.

Multiple expectation values
---------------------------

As we saw in some of the examples in the tutorials, PennyLane supports the return of multiple expectation values; up to one per wire.

As usual, we begin by importing PennyLane and the PennyLane-provided version of NumPy and set up a 2-wire qubit device for computations:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device('default.qubit', wires=2)

We will start with a simple example circuit, which generates a two-qubit entangled state, then evaluates the expectation value of the Pauli Z operator on each wire.

.. code-block:: python

    @qml.qnode(dev)
    def circuit1(param):
        qml.RX(param, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval.PauliZ(0), qml.expval.PauliZ(1)

The degree of entanglement of the qubits is determined by the value of ``param``. For a value of :math:`\frac{\pi}{2}`, they are maximally entangled. In this case, the reduced states on each subsystem are completely mixed, and local expectation values — like those we are measuring — will average to zero.

>>> circuit1(np.pi / 2)
array([4.4408921e-16, 4.4408921e-16])

Notice that the output of the circuit is a NumPy array with ``shape=(2,)``, i.e., a two-dimensional vector. These two dimensions match the number of expectation values returned in our quantum function ``circuit1``.

.. note::
    It is important to emphasize that the expectation values in ``circuit`` are both **local**, i.e., this circuit is evaluating :math:`\braket{\sigma_z}_0` and :math:`\braket{\sigma_z}_1`, not :math:`\braket{\sigma_z\otimes \sigma_z}_{01}` (where the subscript denotes which wires the observable is located on).


Keyword arguments
-----------------

While automatic differentiation is a handy feature, sometimes we want certain parts of our computational pipeline (e.g., the inputs :math:`x` to a parameterized quantum function :math:`f(x;\bf{\theta})` or the training data for a machine learning model) to not be differentiated.

PennyLane uses the pattern that *all positional arguments to quantum functions are available to be differentiated*, while *keyword arguments are never differentiated*. Thus, when using the gradient-descent-based :ref:`optimizers <optimization_methods>` included in PennyLane, all numerical parameters appearing in non-keyword arguments will be updated, while all numerical values included as keyword arguments will not be updated.

.. note:: When constructing the circuit, keyword arguments are defined by providing a **default value** in the function signature. If you would prefer that the keyword argument value be passed every time the quantum circuit function is called, the default value can be set to ``None``.

For example, let's create a quantum node that accepts two arguments; a differentiable circuit parameter ``param``, and a fixed circuit parameter ``fixed``:

.. code-block:: python

    @qml.qnode(dev)
    def circuit3(param, fixed=None):
        qml.RX(fixed, wires=0)
        qml.RX(param, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval.PauliZ(0), qml.expval.PauliZ(1)

Calling the circuit, we can feed values to the keyword argument ``fixed``:

>>> circuit3(0.1, fixed=-0.2)
0.9800665778412417
>>> circuit3(0.1, fixed=1.2)
0.36235775447667345

Since keyword arguments do not get considered when computing gradients, the Jacobian will still be a 2-dimensional vector.

>>> j3 = qml.jacobian(circuit3, argnum=0)
>>> j3(2.5, fixed=3.2)
[1.11022302e-16 5.97451615e-01]

.. important::

    Once defined, keyword arguments must *always* be passed as keyword arguments. PennyLane does not support passing keyword argument values as positional arguments.

    For example, the following circuit evaluation will correctly update the value of the fixed parameter:

    >>> circuit3(0.1, fixed=0.4)
    array([ 0.92106099,  0.91645953])

    However, attempting to pass the fixed parameter as a positional argument will not work, and PennyLane will attempt to use the default value (``None``) instead:

    >>> circuit3(0.1, 0.4)
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-6-949e31911afa> in <module>()
    ----> 1 circuit3(0.1, 0.4)
    ~/pennylane/variable.py in val(self)
        134
        135         # The variable is a placeholder for a keyword argument
    --> 136         value = self.kwarg_values[self.name][self.idx] * self.mult
        137         return value
    TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'

QNodes from different interfaces on one Device
-----------------------------------------------

PennyLane does not only provide the flexibility of having multiple nodes on one device, it also allows these nodes to have different interfaces. Let's look at the following simple example:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=1)
    def circuit(phi):
        qml.RX(phi, wires=0)
        return qml.expval.PauliZ(0)

Now, we contruct multiple QNodes on the same device and change the interface of one of them from NumPy to PyTorch: 

.. code-block:: python

    qnode1 = qml.QNode(circuit, dev1)
    qnode2 = qml.QNode(circuit, dev1)
    qnode1_torch = qnode1.to_torch()

Let's define the cost function. Notice that we can pass the QNode as an argument too. This avoids duplication of code. 

.. code-block:: python
   
    def cost(qnode, phi):
        return qnode(phi)

Now we can call the cost function as follows:

>>> print(cost(qnode1, np.pi))
-1.0
>>> print(cost(qnode2, np.pi))
-1.0
>>> print(cost(qnode1_torch, np.pi))
tensor(-1., dtype=torch.float64)


Ready-to-use Templates
------------------------
PennyLane provides a growing library of ready-to-use templates of common quantum machine learning circuit architectures and embedding functions. These can be used to easily embed classical data and build, evaluate and train more complex quantum machine learning models. They are provided as functions that can be called with the arguments; for details see :ref:`QML Templates <template>`.

