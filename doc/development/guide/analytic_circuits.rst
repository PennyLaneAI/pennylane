Analytic Circuit Repository
===========================

When possible, we recommend checking circuits against analytic results in ``pytest`` instead of
results computed via a different route in PennyLane.  So you don't need to calculate anything out
on pen and paper, we provide circuits here.

.. code-block:: python

    import pennylane as qml
    from pennylane import math

Circuit 1
---------

Operations

.. code-block:: python

    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    qml.CNOT(wires=(0,1))

State and probabilities

.. code-block:: python

    state = math.array( [[math.cos(x/2)*math.cos(y/2), math.cos(x/2)*math.sin(y/2)],
                    [-1j*math.sin(x/2)*math.sin(y/2), 1j*math.sin(x/2)*math.cos(y/2)]])

    prob = state**2


.. list-table:: Measurements
    :widths: 50 50
    :header-row: 0

    * - ..code-block:: python
            qml.expval(qml.PauliZ(0))
      - ..code-block:: python
            math.cos(x)
    * - ..code-block:: python
            qml.expval(qml.PauliX(1))
      - ..code-block:: python
            math.sin(y)

================================================== ==========================
Measurement                                              Value
================================================== ==========================
``qml.expval(qml.PauliZ(0))``                       ``qml.math.cos(x)``
``qml.expval(qml.PauliX(1))``                       ``np.sin(y)``
``qml.expval(qml.PauliZ(0) @ qml.PauliX(1))``       ``np.cos(x)*np.sin(y)``
================================================== ==========================

Gate Testing Circuits
---------------------

IsingXX
^^^^^^^

.. code-block:: python

    psi_0 = 0.1
    psi_1 = 0.2
    psi_2 = 0.3
    psi_3 = 0.4
    init_state = np.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
    norm = np.linalg.norm(init_state)
    init_state /= norm

    @qml.qnode(dev)
    def circuit(phi):
        qml.QubitStateVector(init_state, wires=[0, 1])
        qml.IsingXX(phi, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    phi = np.array(0.1, requires_grad=True)

    expected_res =     psi_0 = 0.1
    psi_1 = 0.2
    psi_2 = 0.3
    psi_3 = 0.4
    init_state = np.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
    norm = np.linalg.norm(init_state)
    init_state /= norm

    @qml.qnode(dev)
    def circuit(phi):
        qml.QubitStateVector(init_state, wires=[0, 1])
        qml.IsingXX(phi, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    phi = np.array(0.1, requires_grad=True)

    expected_res = (
        (1 / norm ** 2)
        * (
            np.cos(phi) * (psi_0 ** 2 + psi_1 ** 2 - psi_2 ** 2 - psi_3 ** 2)
            + np.sin(phi / 2)**2
            * (-(psi_0 ** 2) - psi_1 ** 2 + psi_2 ** 2 + psi_3 ** 2)
        )
    )

    expected_grad = (
        0.5
        * (1 / norm ** 2)
        * (
            -np.sin(phi) * (psi_0 ** 2 + psi_1 ** 2 - psi_2 ** 2 - psi_3 ** 2)
            + 2
            * np.sin(phi / 2)
            * np.cos(phi / 2)
            * (-(psi_0 ** 2) - psi_1 ** 2 + psi_2 ** 2 + psi_3 ** 2)
        )
    )


IsingZZ
^^^^^^^

.. code-block:: python

    psi_0 = 0.1
    psi_1 = 0.2
    psi_2 = 0.3
    psi_3 = 0.4

    init_state = np.array([psi_0, psi_1, psi_2, psi_3], requires_grad=False)
    norm = np.linalg.norm(init_state)
    init_state /= norm
    phi = np.array(0.1, requires_grad=True)

    @qml.qnode(dev)
    def circuit(phi):
        qml.QubitStateVector(init_state, wires=[0, 1])
        qml.IsingZZ(phi, wires=[0, 1])
        return qml.expval(qml.PauliX(0))
    
    expected_result = (1 / norm ** 2) * (2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.cos(phi))
    expected_grad = (1 / norm ** 2) * (-2 * (psi_0 * psi_2 + psi_1 * psi_3) * np.sin(phi))