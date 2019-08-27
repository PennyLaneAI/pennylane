r"""
.. _gaussian_transformation:

Gaussian Transformation
=======================

This tutorial demonstrates the basic working principles of PennyLane for
continuous variable (CV) photonic devices. For more details about
photonic quantum computing, the `Strawberry
Fields <https://strawberryfields.readthedocs.io/en/latest/>`__
documentation is a great starting point.

 **GOAL:** To optimize displacement gate parameters in order to displace
 a *vacuum mode* in phase space to get one *qumode*.

The Quantum Circuit
~~~~~~~~~~~~~~~~~~~

For this basic example, we will consider a special subset of CV
operations: *Gaussian transformations*. We work with the following
simple Gaussian circuit:

.. figure:: ../../examples/figures/gaussian_transformation.svg
    :align: center
    :width: 50%
    :target: javascript:void(0);

What is this circuit doing?

1. **We begin with one wire (qumode) in the vacuum state.** Note that we
   use the same notation :math:`|0\rangle` for the initial state as
   in the :ref:`qubit_rotation` example. In a photonic CV system, this state is
   the vacuum state i.e., the average photon number in the wire is zero.

2. **We displace the qumode.** The displacement gate linearly shifts the
   state of the qumode in phase space. The vacuum state is centered at
   the origin in phase space while the displaced state will be centered
   at the point :math:`\alpha`.

3. **We rotate the qumode.** This is another linear transformation in
   phase space, albeit a rotation (by angle :math:`\phi`) instead of a
   displacement.

4. **Finally, we measure the mean photon number**
   :math:`\langle\hat{n}\rangle=\langle\hat{a}^{\dagger}\hat{a}\rangle`.
   This quantity, which tells us the average number of photons in the
   final state, is proportional to the energy of the photonic system.

We want to optimize the circuit parameters :math:`(\alpha,\phi)` such
that the mean photon number is equal to one. The rotation gate is
actually a *passive transformation* meaning that it does not change the
energy of the system. The displacement gate is an *active
transformation* which modifies the energy of the photonic system.

PennyLane Execution
~~~~~~~~~~~~~~~~~~~
"""
# let's first import the essentials
import pennylane as qml
from pennylane import numpy as np

##############################################################################

# create a device
dev_gaussian = qml.device('default.gaussian', wires=1)

# construct a QNode
@qml.qnode(dev_gaussian)
def mean_photon_gaussian(mag_alpha, phase_alpha, phi):
    qml.Displacement(mag_alpha, phase_alpha, wires=0)
    qml.Rotation(phi, wires=0)
    return qml.expval(qml.NumberOperator(0))

##############################################################################
#
# As we want the mean photon number to be exactly one, we will use a
# squared-difference cost function:

def cost(params):
    return (mean_photon_gaussian(params[0], params[1], params[2]) - 1.) ** 2

##############################################################################
#
# At the beginning of the optimization, we choose arbitrary small initial
# parameters:

init_params = [0.015, 0.02, 0.005]
print(cost(init_params))

##############################################################################
#
# When the gate parameters are close to zero, the gates are close to the
# *identity transformation*, which leaves the initial state largely
# unchanged. Since the initial state contains no photons, the mean photon
# number of the circuit output is approximately zero and the cost is close
# to one.
#
# .. note::
#     We avoided initial parameters which are exactly zero because that
#     corresponds to a critical point with zero gradient.
#
# Now, let’s use the ``GradientDescentOptimizer`` and update the circuit
# parameters over 20 optimization steps:

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# set the number of steps
steps = 20

# set the initial parameter values
params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)
    # print cost after every step
    print('Cost after step {:5d}: {:8f}'.format(i+1, cost(params)))

##############################################################################
#
# Try this yourself — the optimization should converge after about 20
# steps to a cost function value of zero; corresponding to the following
# final values for the parameters:

print('Optimized mag_alpha:{:8f}'.format(params[0]))
print('Optimized phase_alpha:{:8f}'.format(params[1]))
print('Optimized phi:{:8f}'.format(params[2]))

##############################################################################
# We observe that the two angular parameters ``phase_alpha`` and ``phi``
# do not change during the optimization. This makes sense as only the
# magnitude of the complex displacement ``mag_alpha`` affects the mean
# photon number of the circuit.
#
# Continue on to the next tutorial, :ref:`plugins_hybrid`, to learn how to
# utilize the extensive plugin ecosystem of PennyLane,
# build continuous-variable (CV) quantum nodes, and to see an example of a
# hybrid qubit-CV-classical computation using PennyLane.
