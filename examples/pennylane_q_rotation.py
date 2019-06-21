r"""
.. _q_rotation:

Qubit Rotation
==============

To see how PennyLane allows the easy construction and optimization of
quantum functions, let’s consider the simple case of qubit rotation —
the PennyLane version of the ‘Hello, world!’ example.

 **GOAL:** To optimize two qubit rotation gates in order to flip a single
 qubit from state :math:`|0\rangle` to state :math:`|1\rangle`.

The Quantum Circuit
~~~~~~~~~~~~~~~~~~~

In the basic example, we wish to implement the following quantum
circuit:

.. figure:: ../../examples/figures/rotation_circuit.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

Breaking this down step-by-step:

We first start with a qubit in the ground state :math:`|0\rangle` and
rotate it around the x-axis by applying the gate:

.. math::  R_x(\phi_1)=e^{-i \phi_1\frac{\sigma_x}{2}}=\begin{pmatrix} \cos(\frac{\phi_1}{2}) &  -i\sin(\frac{\phi_1}{2}) \\ -i\sin(\frac{\phi_1}{2}) & \cos(\frac{\phi_1}{2}) \end{pmatrix}

and then around the y-axis via the gate:

.. math::  R_y(\phi_2)=e^{-i \phi_2\frac{\sigma_y}{2}}=\begin{pmatrix} \cos(\frac{\phi_2}{2}) &  -\sin(\frac{\phi_2}{2}) \\ \sin(\frac{\phi_2}{2}) & \cos(\frac{\phi_2}{2}) \end{pmatrix}

After these operations, the qubit is in the state:

.. math:: |\psi\rangle=R_y(\phi_2)R_x(\phi_1)|0\rangle

Finally, we measure the expectation value of the Pauli-Z operator
:math:`\langle\psi|\hat{\sigma}_z|\psi\rangle` where:

.. math:: \hat{\sigma}_z=\begin{pmatrix} 1 &  0 \\ 0 & -1 \end{pmatrix}

Using the above to calculate the exact expectation value, we find that:

.. math:: \langle\psi|\hat{\sigma}_z|\psi\rangle = \langle0| R_x(\phi_1)^{\dagger}R_y(\phi_2)^{\dagger}\hat{\sigma}_zR_y(\phi_2)R_x(\phi_1)|0\rangle = \cos(\phi_1)\cos(\phi_2)

Depending on the circuit parameters :math:`\phi_1` and :math:`\phi_2`,
the output expectation lies between ``1`` (if
:math:`|\psi\rangle = |0\rangle`) and ``-1`` (if
:math:`|\psi\rangle = |1\rangle`)

Let’s see how we can easily implement and optimize this circuit using
PennyLane.

PennyLane Execution
^^^^^^^^^^^^^^^^^^^
"""

# lets first import the essentials
import pennylane as qml
from pennylane import numpy as np

##############################################################################

# create a device
dev1 = qml.device('default.qubit', wires=1)

# construct a QNode
@qml.qnode(dev1)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval.PauliZ(0)

##############################################################################
#
# Let’s make use of PennyLane’s built-in optimizers to optimize the two
# circuit parameters :math:`\phi_1` and :math:`\phi_2` such that the
# qubit, originally in state :math:`|0\rangle`, is rotated to be in
# state :math:`|1\rangle`. This is equivalent to measuring a Pauli-Z
# expectation value of ``-1`` since the state :math:`|1\rangle` is an
# eigenvector of the Pauli-Z matrix with eigenvalue :math:`\lambda = −1`.
#
# In other words, the optimization procedure will find the weights
# :math:`\phi_1` and :math:`\phi_2` that result in the following rotation
# on the Bloch sphere:
#
# .. figure:: ../../examples/figures/bloch.png
#     :align: center
#     :width: 70%
#     :target: javascript:void(0);
#
# As our desired outcome is a Pauli-Z expectation value of ``−1``, we can
# define our cost directly as the output of the ``QNode``:

def cost(var):
    return circuit(var)

##############################################################################
#
# To begin our optimization, let’s choose small initial values of
# :math:`\phi_1` and :math:`\phi_2`:

init_params = np.array([0.011, 0.012])
print(cost(init_params))

##############################################################################
# We can see that, for these initial parameter values, the cost function
# is close to 1 as expected.
#
# Finally, we use the PennyLane optimizer to update the circuit parameters
# for 100 steps:

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 100

# set the initial parameter values
params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)
    # print cost after every 5 steps
    if (i+1) % 5 == 0:
        print('Cost after step {:5d}: {: .7f}'.format(i+1, cost(params)))

##############################################################################
# Try this yourself — the optimization should converge after approximately
# 40 steps, giving the following numerically optimum values of
# :math:`\phi_1` and :math:`\phi_2`:

# print final result
print('Optimized rotation angles: {}'.format(params))

##############################################################################
#
# Substituting this into the theoretical result
# :math:`\langle\psi|\hat{\sigma}_z|\psi\rangle=\cos(\phi_1)\cos(\phi_2)`,
# we can verify that this is indeed one possible set of circuit
# parameters that produces
# :math:`\langle\psi|\hat{\sigma}_z|\psi\rangle=-1`, resulting in
# the qubit being rotated to the state :math:`|1\rangle`.
#
# .. note::
#     Some optimizers, such as ``AdagradOptimizer``, have internal
#     hyperparameters that are stored in the optimizer instance. These can be
#     reset using the ``reset()`` method.
#
# Continue on to the next tutorial, Gaussian transformation, to see a
# similar example using continuous variable (CV) quantum nodes.


