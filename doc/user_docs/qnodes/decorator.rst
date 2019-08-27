.. _decorator:

The QNode decorator
-------------------

**Function name:** :func:`pennylane.decorator.qnode`

.. currentmodule:: pennylane.decorator

The most convenient way for creating 'quantum nodes' or QNodes is the provided
`qnode` decorator. This decorator converts a quantum circuit function containing PennyLane quantum
operations to a :mod:`QNode <pennylane.qnode>` that will run on a quantum device.

The `interface` argument specified which type of QNode to create.

.. note:: 
    The decorator completely replaces the Python-defined function with 
    a :mod:`QNode <pennylane.qnode>` of the same name - as such, the original 
    function is no longer accessible (but is accessible via the :attr:`~.QNode.func` attribute).


.. automodule:: pennylane.decorator
   :members:
   :private-members:
   :inherited-members:

Examples
^^^^^^^^
.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)

    @qml.qnode(dev1)
    def qfunc1(x):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(x, wires=1)
        return qml.expval(qml.PauliZ(0))

    result = qfunc1(0.543)


.. code-block:: python

    dev2 = qml.device('default.gaussian', wires=2)

    @qml.qnode(dev2)
    def qfunc2(x, y):
        qml.Displacement(x, 0, wires=0)
        qml.Beamsplitter(y, 0, wires=[0, 1])
        return qml.expval(qml.NumberOperator(0))

    def hybrid_computation(x, y):
        return np.sin(qfunc1(y))*np.exp(-qfunc2(x+y, x)**2)

The :ref:`QNode decorator <qnode_decorator>` is the recommended way for creating
QNodes for interfaces like PyTorch and TensorFlow.

.. code-block:: python

      
    dev3 = qml.device('default.qubit', wires=2)

    # To construct a TensorFlow-capable QNode, specify the 
    # `interface='tfe'` keyword argument:
    
    @qml.qnode(dev3, interface='tfe')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

>>> phi = tfe.Variable([0.5, 0.1])
>>> theta = tfe.Variable(0.2)
>>> circuit(phi, theta)
<tf.Tensor: id=22, shape=(2,), dtype=float64, numpy=array([ 0.87758256,  0.68803733])>

