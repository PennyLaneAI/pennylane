.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_pennylane_run_gaussian_transformation.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_pennylane_run_gaussian_transformation.py:


.. _gaussian_transformation:

Gaussian transformation
=======================

This tutorial demonstrates the basic working principles of PennyLane for
continuous-variable (CV) photonic devices. For more details about photonic
quantum computing, the
`Strawberry Fields documentation <https://strawberryfields.readthedocs.io/en/latest/>`_
is a great starting point.

The quantum circuit
-------------------

For this basic tutorial, we will consider a special subset of CV operations:
the *Gaussian transformations*. We work with the following simple Gaussian circuit:

.. figure:: ../../examples/figures/gaussian_transformation.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

What is this circuit doing?

1. **We begin with one wire (qumode) in the vacuum state**. Note that we use the same
   notation :math:`|0\rangle` for the initial state as the previous qubit tutorial.
   In a photonic CV system, this state is the *vacuum state*, i.e., the state with no
   photons in the wire.

2. **We displace the qumode**. The displacement gate linearly shifts the state of the
   qumode in phase space. The vacuum state is centered at the origin in phase space,
   while the displaced state will be centered at the point :math:`\alpha`.

3. **We rotate the qumode**. This is another linear transformation in phase space,
   albeit a rotation (by angle :math:`\phi`) instead of a displacement.

4. **Finally, we measure the mean photon number** :math:`\langle\hat{n}\rangle =
   \langle\hat{a}^\dagger \hat{a}\rangle`. This quantity, which tells us the average amount of photons in the final state, is proportional to the energy of the photonic system.

The aim of this tutorial is to optimize the circuit parameters :math:`(\alpha, \phi)`
such that the mean photon number is equal to one. The rotation gate is actually a
*passive transformation*, meaning that it does not change the energy of the system.
The displacement gate is an *active transformation*, which modifies the energy of the
photonic system.

Constructing the QNode
----------------------

As before, we import PennyLane, as well as the wrapped version of NumPy provided
by PennyLane:


.. code-block:: default


    import pennylane as qml
    from pennylane import numpy as np







Next, we instantiate a device which will be used to evaluate the circuit.
Because our circuit contains only Gaussian operations, we can make use of the
built-in ``default.gaussian`` device.


.. code-block:: default


    dev_gaussian = qml.device("default.gaussian", wires=1)







After initializing the device, we can construct our quantum node. As before, we use the
:mod:`qnode decorator <pennylane.decorator>` to convert our quantum function
(encoded by the circuit above) into a quantum node running on the ``default.gaussian``
device.


.. code-block:: default



    @qml.qnode(dev_gaussian)
    def mean_photon_gaussian(mag_alpha, phase_alpha, phi):
        qml.Displacement(mag_alpha, phase_alpha, wires=0)
        qml.Rotation(phi, wires=0)
        return qml.expval(qml.NumberOperator(0))








Notice that we have broken up the complex number :math:`\alpha` into two real
numbers ``mag_alpha`` and ``phase_alpha``, which form a polar representation of
:math:`\alpha`. This is so that the notion of a gradient is clear and well-defined.

Optimization
------------

As in the :ref:`qubit rotation <qubit_rotation>` tutorial, let's now use one
of the built-in PennyLane optimizers in order to optimize the quantum circuit
towards the desired output. We want the mean photon number to be exactly one,
so we will use a squared-difference cost function:


.. code-block:: default



    def cost(params):
        return (mean_photon_gaussian(params[0], params[1], params[2]) - 1.0) ** 2








At the beginning of the optimization, we choose arbitrary small initial parameters:


.. code-block:: default


    init_params = [0.015, 0.02, 0.005]
    print(cost(init_params))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9995500506249999


When the gate parameters are near to zero, the gates are close to the
identity transformation, which leaves the initial state largely unchanged.
Since the initial state contains no photons, the mean photon number of the
circuit output is approximately zero, and the cost is close to one.

.. note::

    We avoided initial parameters which are exactly zero because that
    corresponds to a critical point with zero gradient.

Now, let's use the :class:`~.GradientDescentOptimizer`, and update the circuit
parameters over 100 optimization steps.


.. code-block:: default


    # initialise the optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    # set the number of steps
    steps = 20
    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)

        print("Cost after step {:5d}: {:8f}".format(i + 1, cost(params)))

    print("Optimized mag_alpha:{:8f}".format(params[0]))
    print("Optimized phase_alpha:{:8f}".format(params[1]))
    print("Optimized phi:{:8f}".format(params[2]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Cost after step     1: 0.999118
    Cost after step     2: 0.998273
    Cost after step     3: 0.996618
    Cost after step     4: 0.993382
    Cost after step     5: 0.987074
    Cost after step     6: 0.974837
    Cost after step     7: 0.951332
    Cost after step     8: 0.907043
    Cost after step     9: 0.826649
    Cost after step    10: 0.690812
    Cost after step    11: 0.490303
    Cost after step    12: 0.258845
    Cost after step    13: 0.083224
    Cost after step    14: 0.013179
    Cost after step    15: 0.001001
    Cost after step    16: 0.000049
    Cost after step    17: 0.000002
    Cost after step    18: 0.000000
    Cost after step    19: 0.000000
    Cost after step    20: 0.000000
    Optimized mag_alpha:0.999994
    Optimized phase_alpha:0.020000
    Optimized phi:0.005000


The optimization converges after about 20 steps to a cost function value
of zero.

We observe that the two angular parameters ``phase_alpha`` and ``phi``
do not change during the optimization. Only the magnitude of the complex
displacement :math:`|\alpha|` affects the mean photon number of the circuit.

Continue on to the next tutorial, :ref:`plugins_hybrid`, to learn how to
utilize the extensive plugin ecosystem of PennyLane,
build continuous-variable (CV) quantum nodes, and to see an example of a
hybrid qubit-CV-classical computation using PennyLane.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.129 seconds)


.. _sphx_glr_download_tutorials_pennylane_run_gaussian_transformation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: pennylane_run_gaussian_transformation.py <pennylane_run_gaussian_transformation.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: pennylane_run_gaussian_transformation.ipynb <pennylane_run_gaussian_transformation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
