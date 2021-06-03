Analytic Circuit Repository
===========================

When possible, we recommend checking circuits against analytic results in ``pytest`` instead of
results computed via a different route in PennyLane.  So you don't need to calculate anything out
on pen and paper, we provide circuits here.

Circuit 1
---------

Operations

.. code-block:: python

    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    qml.CNOT(wires=(0,1))

State

.. code-block:: python

    state = np.array( [[np.cos(x/2)*np.cos(y/2), np.cos(x/2)*np.sin(y/2)],
                    [-1j*np.sin(x/2)*np.sin(y/2), 1j*np.sin(x/2)*np.cos(y/2)]])

    prob = state**2

Probs

.. code-block:: python

================================================== ========================
Measurement                                              Value
================================================== ========================
``qml.expval(qml.PauliZ(0))``                       ``np.cos(x)``
``qml.expval(qml.PauliX(1))``                       ``np.sin(y)``
``qml.expval(qml.PauliZ(0) @ qml.PauliX(1))``       ``np.cos(x)*np.sin(y)``
================================================== ========================




Single Input, Single Output
---------------------------

.. code-block:: python

    def qfunc(x):
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    def expected_res(x):
        return np.cos(x)

    def expected_grad(x):
        return -np.sin(x)

    def expected_hess(x):
        return -np.cos(x)


Single Input, Multiple Output
-----------------------------

.. code-block:: python

    def circuit(x):
        qml.RY(x, wires=0)
        return qml.probs(wires=[0])

    def expected_res(x):
        return np.array([np.cos(x/2.0)**2, np.sin(x/2.0)**2])

    def expected_jacobian(x):
        return np.array([-np.sin(x)/2.0, np.sin(x)/2.0])

    def expected_hess(x):
        return np.array([-np.cos(x)/2.0, np.cos(x)/2.0])

Single Input, State Output
--------------------------

.. code-block:: python

    def circuit(x):
        qml.RX(x, wires=0)
        return qml.state()

    def expected_res(x):
        return np.array([np.cos(x/2.0), -1j * np.sin(x/2.0)])

Vector Input, Single Output
---------------------------

.. code-block:: python

    def circuit(x):
        qml.RY(x[0], wires=0)
        qml.RX(x[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    def expected_res(x):
        return np.cos(x[0]) * np.cos(x[1])

    def expected_grad(x):
        return np.array([-np.sin(x[0]) * np.cos(x[1]), -np.cos(x[0]) * np.sin(x[1])])

    def expected_hess(x):
        return np.array([[-np.cos(x[0]) * np.cos(x[1]),  np.sin(x[0]) * np.sin(x[1])],
                         [ np.sin(x[0]) * np.sin(x[1]), -np.cos(x[0]) * np.cos(x[1])]])

Others
------

Found these being used in the testing

.. code-block:: python

    def qfunc(x, y):
        qml.RX(x, wires=[0])
        qml.RY(y, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
    
    def expected_res(x, y):
        return np.cos(x) * np.sin(y)

    def expected_grad(x, y):
        return (-np.sin(x) * np.sin(y), np.cos(x) * np.cos(y))


.. code-block:: python

    def qfunc(x, y):
        qml.RX(x, wires=[0])
        qml.RY(y, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

    def expected_res(x, y):
        return (np.cos(x), np.sin(y))
    
    def expected_jac(x, y):
        return [[-np.sin(x), 0], [0, np.cos(y)]

.. code-block:: python

    def qfunc(x, y):
        qml.RX(x, wires=[0])
        qml.RY(y, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

    def expected_res(x, y):
        return (np.cos(x), np.cos(y)**2)

    def expected_grad(x, y):
        return ([-np.sin(x), 0], [0, -2*np.cos(y)*np.sin(y)])
