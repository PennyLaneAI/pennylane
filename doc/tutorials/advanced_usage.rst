.. role:: html(raw)
   :format: html

.. _advanced_features:

Advanced Usage
==============

In the previous three introductory tutorials (:ref:`qubit rotation <qubit_rotation>`, :ref:`Gaussian transformation <gaussian_transformation>`, and :ref:`plugins & hybrid computation <plugins_hybrid>`) we explored the basic concepts of PennyLane, including qubit- and CV-model quantum computations, gradient-based optimization, and the construction of hybrid classical-quantum computations.

In this tutorial, we will highlight some of the more advanced features of Pennylane.

Multiple expectation values
---------------------------

In all the previous examples, we considered quantum functions with only single expectation values. In fact, PennyLane supports the return of multiple expectation values, up to one per wire.

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
        qml.CNOT(wires=[0, 1])
        return qml.expval.PauliZ(0), qml.expval.PauliZ(1)

The degree of entanglement of the qubits is determined by the value of ``param``. For a value of :math:`\frac{\pi}{2}`, they are maximally entangled. In this case, the reduced states on each subsystem are completely mixed, and local expectation values — like those we are measuring — will average to zero.

>>> circuit1(np.pi / 2)
array([4.4408921e-16, 4.4408921e-16])

Notice that the output of the circuit is a NumPy array with ``shape=(2,)``, i.e., a two-dimensional vector. These two dimensions match the number of expectation values returned in our quantum function ``circuit1``.

.. note::
    It is important to emphasize that the expectation values in ``circuit`` are both **local**, i.e., this circuit is evaluating :math:`\braket{\sigma_z}_0` and :math:`\braket{\sigma_z}_1`, not :math:`\braket{\sigma_z\otimes \sigma_z}_{01}` (where the subscript denotes which wires the observable is located on).

