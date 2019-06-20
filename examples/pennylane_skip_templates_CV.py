r"""
.. _templates_CV:

Single Layer CV QNN
===================

In this tutorial, we will look at one of the continuous variable circuit
architectures provided by PennyLane; :func:`~.CVNeuralNetLayer`.
This is the quantum analogue of a classical Neural Network.

.. important::

    For continuous variable computing, PennyLane provides a simplified
    circuit architecture template using the universal gate set containing
    Interferometers (made of single-qumode rotation and 2-qumode Beam
    Splitter gates), Squeezing gates, Displacement gates and non-Gaussian
    gates.

PennyLane’s :func:`~.CVNeuralNetLayers` template can be used to implement a
Quantum Neural Network (QNN) circuit. A single layer is implemented by
the template :func:`~.CVNeuralNetLayer`. It contains two Interferometers,
local squeezing gates, local displacement gates and local Kerr gates acting on all
qumodes. Note that this template uses the
:class:`~.Interferometer` template inside it for simplification and efficiency.

The ``CVNeuralNetLayer`` template function needs to be called with
eleven input arguments with all parameters initialized with the right
dimensions. This is why using the built-in templates to initialize
parameters can come in very handy. See :mod:`pennylane.init` for more details
on the various parameter initialization templates PennyLane offers.

Let’s implement the following single layer circuit on 4 qumodes to
understand how these templates works.

.. figure:: ../../examples/figures/cvqnn_example.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import CVNeuralNetLayer
from pennylane.init import cvqnn_layer_uniform

##############################################################################
#
# .. note::
#     For the Kerr operation, a suitable device should be used such as the
#     ``strawberryfields.fock`` device.

n_wires = 4
    
dev = qml.device('strawberryfields.fock', wires=n_wires,cutoff_dim=8)
    
# initialize parameters
pars = cvqnn_layer_uniform(n_wires)
print(pars)

##############################################################################
#
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     [array([2.61467562, 3.2371544 , 2.39636553, 3.3069315 , 1.71942329,
#             2.23407894]), array([3.08895824, 5.18375857, 2.65491782, 4.18073044, 5.54849448,
#             4.02551326]), array([3.9868442 , 2.72366934, 2.58366073, 2.97040392]), array([ 0.20247638,  0.04394124, -0.02263603, -0.00584289]), array([3.23375673, 6.08161587, 2.07426891, 6.22371632]), array([3.38820135, 2.24947009, 0.02053007, 2.93713089, 6.26890416,
#             0.81671074]), array([4.57224344, 5.15390558, 3.98982271, 3.86910136, 3.98449492,
#             2.68450659]), array([5.82997086, 6.26875476, 5.95598548, 5.9361312 ]), array([ 0.12620595,  0.01095255, -0.16871207, -0.02344182]), array([5.11870587, 0.43539138, 3.37922203, 5.66559641]), array([ 0.08173526, -0.07799196,  0.04952914,  0.01706851])]
#
# Note: For 4 qumodes, each Interferometer applies
# :math:`\frac{(4)(3)}{2}= 6` Beam Splitters and 4 Rotation gates.
#
# Let’s understand what ``cvqnn_layer_uniform`` function output ``pars``
# contains:
#
# - **array 1:** 6 theta for the 6 BS gates on first interferometer
# - **array 2:** 6 phi for the 6 BS gates on first interferometer
# - **array 3:** 4 varphi for the 4 Rotation gates on first interferometer
# - **array 4:** 4 squeezing magnitudes for the 4 squeezing gates
# - **array 5:** 4 squeezing phases for the 4 squeezing gates
# - **array 6:** 6 theta for the 6 BS gates on second interferometer
# - **array 7:** 6 phi for the 6 BS gates on second interferometer
# - **array 8:** 4 varphi for the 4 Rotation gates on second interferometer
# - **array 9:** 4 displacement magnitudes for the 4 displacement gates
# - **array 10:** 4 displacement phases for the 4 displacement gates
# - **array 11:** 4 Kerr parameters for the 4 Kerr gates
#
# .. note::
#     The displacement amplitude, squeezing amplitude and kerr parameter
#     values are initialized close to zero since they influence the mean
#     photon number (or energy) of the quantum system.

@qml.qnode(dev)
def circuit(pars):
    CVNeuralNetLayer(*pars, wires=range(n_wires))
    return [qml.expval.MeanPhoton(wires=w) for w in range(n_wires)]

##############################################################################

print(circuit(pars))

##############################################################################
#
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     array([0.05652615, 0.00162837, 0.03006308, 0.00087608])
#
# Measuring zero photons on average in each qumode is in line with how the
# parameters were initialized.

