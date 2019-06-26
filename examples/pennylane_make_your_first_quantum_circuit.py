r"""
.. _make_your_first_quantum_circuit:

Make your First Quantum Circuit
=================================

This tutorial is the first step towards learning the features of PennyLane.

Importing PennyLane and NumPy
------------------------------

First, we import PennyLane itself, as well as a wrapped version of
*NumPy* which is provided via PennyLane. This allows us to use the
familiar NumPy functions along with *quantum functions*.
"""

import pennylane as qml
from pennylane import numpy as np

###############################################################################
# .. important::
#
#     When constructing a hybrid quantum/classical computational model with
#     PennyLane, it is important to always import NumPy from PennyLane, not
#     the standard NumPy! This way, PennyLane can pass gradients through classical and quantum
#     computations.
#
# Creating a device
# -------------------
#
# A **Quantum Circuit** comprises of quantum **operations** and **wires**
# that encode quantum bits that the operations can act on.
#
# .. admonition:: Definition
#     :class: defn
#
#     In PennyLane, the term quantum ‘device’ refers to a computational
#     object that can apply quantum operations and return an expectation
#     value.
#
# A device could be a hardware device (such as the IBM QX4, via the
# PennyLane-PQ plugin) or a software simulator (such as Strawberry Fields,
# via the PennyLane-SF plugin).
#
# PennyLane supports two built-in devices for **discrete variable** and 
# **continuous variable** quantum computation:
#
# * :mod:`default.qubit <pennylane.plugins.default_qubit>`: pure state qubit simulator
#
# * :mod:`default.gaussian <pennylane.plugins.default_gaussian>`: Gaussian states simulator
#
# Additional devices are supported through plugins - see :ref:`plugin ecosystem <plugins>`
# for more details.
#
# Let’s use the qubit simulator device provided by PennyLane. Devices are
# loaded in PennyLane via the function :func:`~.pennylane.device`:

dev = qml.device('default.qubit', wires = 1)

##############################################################################
# We call this function with two arguments:
#
# * ``name``: the name of the device to be loaded
#
# * ``wires``: the number of subsystems to initialize the device with
#
# Here, the argument ``wires = 1`` means that a single qubit is initiated. All qubits are initialized in the 
# state :math:`|0\rangle`.
#
# Constructing the QNode
# ----------------------
#
# Now that we have initialized our device, we can begin to construct a
# :class:`pennylane.QNode <pennylane.qnode.QNode>`.
#
# .. admonition:: Definition
#     :class: defn
#
#     QNodes are an abstract encapsulation of a quantum function, described
#     by a quantum circuit. QNodes are bound to a particular quantum
#     device, which is used to evaluate expectation values of this circuit.
#
# First, we need to define the quantum function that will be evaluated in
# the QNode:

# let's use PennyLane's RX gate - Rotation about the x-axis by angle param
def circuit(param):
    qml.RX(param, wires=0)
    return qml.expval.PauliZ(0)

##############################################################################
# **NOTE:** The function ``circuit()`` is constructed as if it were any
# other Python function using Python notation (``def fn(...)``); it
# accepts a positional argument ``param`` which may be a list, a tuple or
# an array.
#
# For a Python function to also be a valid quantum function, there are
# some important restrictions:
#
# *  Quantum functions must only contain quantum operations, one operation
#    per line, in the order in which they are to be applied. For a full
#    list of quantum operations, see :mod:`supported operations <pennylane.ops>`.
#
# *  Quantum functions must return either a single or a tuple of
#    expectation values ``expval`` or variances ``var``. As a result, the quantum function
#    always returns a classical quantity, allowing the QNode to interface
#    with other classical functions (and also other QNodes). For a full
#    list of available measurements, see :mod:`supported measurements <pennylane.measure>`.
#
# *  Quantum functions must not contain any classical processing of
#    circuit parameters.
#
# Once we have written the quantum function, we convert it into a :class:`~.QNode` 
# running on device ``dev`` by applying the :mod:`qnode decorator <pennylane.decorator>`
# directly above the function definition:

@qml.qnode(dev)
def circuit(param):
    qml.RX(param, wires = 0)
    return qml.expval.PauliZ(0)

################################################################################
#
# .. note::
#
#     We use the ``QNode`` decorator to indicate that this is not a typical
#     Python function. This prevents the function from being run as usual by
#     the Python interpretor. Instead, the function is evaluated on the device
#     ``dev`` (which may be quantum hardware).
#
# Thus, our ``circuit()`` quantum function is now a ``QNode``, which will
# run on device ``dev`` every time it is evaluated. To evaluate, we simply
# call the function with some appropriate numerical inputs. Let's rotate the state by :math:`\pi`:

print(circuit(np.pi))

################################################################################
#
# This is the expected outcome, as the qubit is rotated to the state :math:`\mid1\rangle`
# under the rotation :math:`RX(\pi)` as depicted in the figure below.
#
# .. figure:: ../../examples/figures/bloch.png
#     :align: center
#     :width: 70%
#     :target: javascript:void(0);
#
#
# Examples
# ---------
#
# Let's look at a couple of simple examples:

# using default.qubit device with one subsystem
dev1 = qml.device('default.qubit', wires = 1)
@qml.qnode(dev1)
def qfunc1():
    return qml.expval.Identity(0)
################################################################################

print(qfunc1())

################################################################################
# This is the expected outcome, as :math:`\langle0\mid\hat{I}\mid0\rangle=1`

# using default.gaussian device with one subsystem
dev2 = qml.device('default.gaussian', wires = 1)
@qml.qnode(dev2)
def qfunc2():
    return qml.expval.MeanPhoton(0)

################################################################################

print(qfunc2())

################################################################################
#
# This is expected, as the Gaussian state is initialized in the vaccum
# state (the lowest energy Gaussian state with no displacement or
# squeezing in phase space) that has zero number of Photons.
#
# .. note::
#     1. The *expectation values*, :math:`\langle \cdots \rangle`, returned by QNodes 
#     are averages, not single-shot results. As a result, these are deterministic,
#     whereas single-shot measurements are stochastic. This is what allows us to do
#     machine learning on the circuit (Note: the same principle holds for deep
#     learning models).
#
#     2. Since circuits are meant to be run on quantum hardware, there is
#     limited support for classical computation *inside* the circuit
#     function. On the other hand, classical processing of circuit
#     inputs/outputs is fully supported.
