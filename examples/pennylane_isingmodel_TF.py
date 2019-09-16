r"""
.. _isingmodel_TF:

Optimization of 3-qubit Ising model in TensorFlow
==================================================
*Author: Aroosa Ijaz (aroosa@xanadu.ai)*

Qubit Optimization for 3-qubit Ising model using the Pennylane ``default.qubit`` Plugin.
The Ising model is an example of a system whose optimization landscape is
non-convex and hence using gradient descent may not be the best strategy
since the optimization can get stuck in local minima.
Ising models are used in Quantum Annealing (for example on a D-wave system) to solve Quadratic 
Unconstrained Binary Optimization (QUBO) problems with non-convex cost functions. This example demonstrates how
gradient descent optimizers can get stuck in local minima when using non-convex cost functions.

The energy for this system is given by:

.. math::  H=-\sum_{<i,j>} J_{ij} \sigma_i \sigma_{j}

where each spin can be in +1 or -1 spin state and :math:`J_{ij}` are
nearest neighbour coupling strengths.

For simplicity, we will assume that the first spin is always in up state
(+1 eigenstate of Pauli-Z operator). Moreover, we will be solving this problem with 
fixed couplings between the three qubits using the coupling matrix J = [1,-1].

We will then optimize the rotation angles for the other two spins
so that the energy of the system is minimized for the given couplings.
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe

###############################################################################

# Pennylane interfaces with the eager execution of TensorFlow
tf.enable_eager_execution()

###############################################################################
# You can check if the eager execution is working by using the following command
# ``print(tf.executing_eagerly())``.

import pennylane as qml
from pennylane import numpy as np

###############################################################################

dev2 = qml.device("default.qubit", wires = 3)

@qml.qnode(dev2, interface = "tfe") # note the use of argument 'interface'
def circuit2(p1, p2):
    # We use the general Rot(phi,theta,omega,wires) single-qubit operation
    qml.Rot(p1[0], p1[1], p1[2], wires=1)
    qml.Rot(p2[0], p2[1], p2[2], wires=2)
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2)

###############################################################################
# Our cost function is the energy of the spin configuration which we will
# optimize using gradient descent.

def cost(var):
    # circuit2 function returns a numpy array of Pauli-Z exp values
    spins = circuit2(var[0], var[1])
    # the expectation value of Pauli-Z is +1 for spin up and -1 for spin down
    energy = -(1 * spins[0] * spins[1]) - (-1 * spins[1] * spins[2])
    return energy

###############################################################################
#
# Let’s test these functions for the [1,-1,-1] spin configuration.
# Total energy for this Ising model should be:
#
# .. math:: H = -1(J_1 s_1 \otimes s_2 + J_2 s_2 \otimes s3) = -1 [1 \times 1 \times (-1) + (-1) \times (-1) \times (-1)] = 2 
#

test1 = tfe.Variable([0, np.pi, 0], dtype=tf.float64)
test2 = tfe.Variable([0, np.pi, 0], dtype=tf.float64)

cost_check = cost([test1, test2])
print("Energy for [1,-1,-1] spin configuration:",cost_check)

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Energy for [1,-1,-1] spin configuration: tf.Tensor(2.0, shape=(), dtype=float64)
#

# Random initialization in TensorFlow

t1 = tfe.Variable(tf.random_uniform([3], 0, np.pi, dtype=tf.float64))
t2 = tfe.Variable(tf.random_uniform([3], 0, np.pi, dtype=tf.float64))

var_init = [t1, t2]
cost_init = cost(var_init)

print("Randomly initialized angles:",var_init)
print("Corresponding cost before initialization:",cost_init)

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Randomly initialized angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 2.44976174, 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 2.15003498, 2.09835371])>]
#     Corresponding cost before initialization: tf.Tensor(1.1916106188488236, shape=(), dtype=float64)
#
# Now we use the TensorFlow gradient descent optimizer to minimize the cost.

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

steps = 100

var = var_init
var_tf = [var]
cost_tf = [cost_init]

for i in range(steps):
    with tf.GradientTape() as tape:
        loss = cost([t1, t2])
        grads = tape.gradient(loss, [t1, t2])

    opt.apply_gradients(zip(grads, [t1, t2]), global_step=tf.train.get_or_create_global_step())
    if (i + 1) % 5 == 0:
        var_tf.append([t1, t2])
        cost_tf.append(cost([t1, t2]))
        print("Energy after step {:5d}: {: .7f} | Angles: {}".format(i + 1, cost([t1, t2]), [t1, t2]),"\n")

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Energy after step     5:  0.4140368 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 1.89414018, 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 1.87871132, 2.09835371])>] 
#
#     Energy after step    10: -0.3733069 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 1.272362  , 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 1.84382325, 2.09835371])>] 
#
#     Energy after step    15: -1.1314230 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 0.70508118, 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 2.07792232, 2.09835371])>] 
#
#     Energy after step    20: -1.6664062 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 0.30877062, 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 2.41753821, 2.09835371])>] 
#
#     Energy after step    25: -1.8889418 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 0.11415994, 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 2.69359866, 2.09835371])>] 
#
#     Energy after step    30: -1.9626913 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 0.03909354, 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 2.87317628, 2.09835371])>] 
#
#     Energy after step    35: -1.9871771 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 0.013011  , 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 2.982335  , 2.09835371])>] 
#
#     Energy after step    40: -1.9955487 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696, 0.00428673, 1.76898127])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.04739885, 2.09835371])>] 
#
#     Energy after step    45: -1.9984498 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 1.40735020e-03, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.08594067, 2.09835371])>] 
#
#     Energy after step    50: -1.9994597 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 4.61466676e-04, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.10872425, 2.09835371])>] 
#
#     Energy after step    55: -1.9998116 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 1.51248404e-04, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.12218286, 2.09835371])>] 
#
#     Energy after step    60: -1.9999343 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 4.95650767e-05, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.13013109, 2.09835371])>] 
#
#     Energy after step    65: -1.9999771 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 1.62419412e-05, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.13482466, 2.09835371])>] 
#
#     Energy after step    70: -1.9999920 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 5.32221141e-06, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.13759621, 2.09835371])>] 
#
#     Energy after step    75: -1.9999972 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 1.74398817e-06, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.13923279, 2.09835371])>] 
#
#     Energy after step    80: -1.9999990 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 5.71470715e-07, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.14019918, 2.09835371])>] 
#
#     Energy after step    85: -1.9999997 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 1.87259598e-07, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.14076982, 2.09835371])>] 
#
#     Energy after step    90: -1.9999999 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 6.13612329e-08, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.14110678, 2.09835371])>] 
#
#     Energy after step    95: -2.0000000 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 2.01068495e-08, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.14130575, 2.09835371])>] 
#
#     Energy after step   100: -2.0000000 | Angles: [<tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 6.58861252e-09, 1.76898127e+00])>, <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.14142324, 2.09835371])>]
#
# The minimum energy is -2  for the spin configuration [1,1,-1] which corresponds to
# :math:`(\phi, \theta, \omega) = (0, 0, 0)` for the second spin and
# :math:`(\phi, \theta, \omega) = (0, \pi, 0)` for 
# the third spin, respectively. We might not always see this value due to the non-convex cost function.

print("Optimized angles:",t1, t2)
print("Final cost after optimization:",cost([t1, t2]))

###############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Optimized angles: <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.51966696e+00, 6.58861252e-09, 1.76898127e+00])> <tf.Variable 'Variable:0' shape=(3,) dtype=float64, numpy=array([1.84730485, 3.14142324, 2.09835371])>
#     Final cost after optimization: tf.Tensor(-1.999999985649412, shape=(), dtype=float64)
#
# When we initialize close to zero, the optimizer is more likely to get
# stuck in a local minimum. Try it yourself! Download and run this file with different
# initialization parameters and see how the results change.
# The figure below shows outputs of two different runs from different optimizers.
#  
# .. figure:: ../../examples/figures/ising1.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0);
#
# .. figure:: ../../examples/figures/ising2.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0);


