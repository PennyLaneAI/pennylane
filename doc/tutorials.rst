 .. role:: html(raw)
   :format: html

.. _New_Users:

Tutorials
=========

PennyLane basics
~~~~~~~~~~~~~~~~~~~

These step-by-step tutorials provide a gentle introduction to the basics of PennyLane. In the last two examples, we put everything together and consider PennyLane's version of the "Hello world!" example - qubit rotation and the Gaussian transformation of a qumode.

.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials/pennylane_run_make_your_first_quantum_circuit
    tutorials/pennylane_run_get_to_know_the_operations
    tutorials/pennylane_run_prepare_your_first_quantum_state
    tutorials/pennylane_run_quantum_gradients
    tutorials/pennylane_run_optimization_in_pennylane
    tutorials/pennylane_run_advanced_usage
    tutorials/pennylane_run_qubit_rotation
    tutorials/pennylane_run_gaussian_transformation

.. customgalleryitem::
    :tooltip: Learn how to make quantum circuits in PennyLane.
    :figure: ../examples/figures/firstcircuit.png
    :description: :ref:`make_your_first_quantum_circuit`

.. customgalleryitem::
    :tooltip: Get familiar with various operations PennyLane offers.
    :figure: ../examples/figures/squeeze1.png
    :description: :ref:`get_to_know_the_operations`

.. customgalleryitem::
    :tooltip: Prepare your first quantum state in PennyLane.
    :figure: ../examples/figures/NOON.png
    :description: :ref:`prepare_your_first_quantum_state`

.. customgalleryitem::
    :tooltip: See PennyLane's automatic differentiation of quantum circuits in action.
    :figure: ../examples/figures/descent.png
    :description: :ref:`quantum_gradients`

.. customgalleryitem::
    :tooltip: Get familiar with PennyLane's Optimizers.
    :figure: ../examples/figures/optimization.png
    :description: :ref:`optimization_in_pennylane`

.. customgalleryitem::
    :tooltip: Jacobians, keyword arguments and devices supporting multiple interfaces.
    :description: :ref:`advanced_features`

.. customgalleryitem::
    :tooltip: Use quantum machine learning to rotate a qubit.
    :figure: ../examples/figures/bloch.png
    :description: :ref:`qubit_rotation`

.. customgalleryitem::
    :tooltip: Use quantum machine learning to tune a beamsplitter.
    :figure: ../examples/figures/gaussiancircuit.png
    :description: :ref:`gaussian_transformation`

:html:`<div style='clear:both'></div>`

Classical interfaces
~~~~~~~~~~~~~~~~~~~~~

The following tutorials demonstrate how to optimise the energy of a simple Ising model using different interfaces offered by PennyLane. 

.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials/pennylane_isingmodel_NumPy 
    tutorials/pennylane_isingmodel_TF
    tutorials/pennylane_isingmodel_PyTorch

.. customgalleryitem::
    :tooltip: Ising model example with PennyLane NumPy interface.
    :figure: ../examples/figures/pennylane_xanadu.png
    :description: :ref:`isingmodel_NumPy`

.. customgalleryitem::
    :tooltip: Ising model example with PennyLane TensorFlow interface.
    :figure: ../examples/figures/tensorflow.png
    :description: :ref:`isingmodel_TF`

.. customgalleryitem::
    :tooltip: Ising model example with PennyLane PyTorch interface.
    :figure: ../examples/figures/pytorch.png
    :description: :ref:`isingmodel_PyTorch`

:html:`<div style='clear:both'></div>`


Plugins and hybrid computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plugins allow PennyLane to interact with a variety of quantum simulators and hardware. It also allows the user to run hybrid computations that combine different plugins and interfaces. The following tutorials showcase some examples in this regard.


.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials/pennylane_plugins_hybrid
    tutorials/pennylane_pytorch_noise


.. customgalleryitem::
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.
    :figure: ../examples/figures/photon_redirection.png
    :description: :ref:`plugins_hybrid`

.. customgalleryitem::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: ../examples/figures/bloch.gif
    :description: :ref:`pytorch_noise`

:html:`<div style='clear:both'></div>`


Ready-to-use templates
~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane provides a growing library of ready-to-use templates of common quantum machine learning circuit architectures and embedding functions. These can be used to easily embed classical data and build (and train) complex quantum machine learning models. They are provided as functions that can be called inside QNodes; for details see :mod:`pennylane.templates`. In the tutorials below, we will go through a couple of examples that use these templates in detail.


.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials/pennylane_run_templates_DV
    tutorials/pennylane_templates_CV

.. customgalleryitem::
    :tooltip: Make a strongly entangling circuit in a single line of code.
    :figure: ../examples/figures/slsec_example.png
    :description: :ref:`templates_DV`

.. customgalleryitem::
    :tooltip: Make a Quantum Neural Network in a single line of code.
    :figure: ../examples/figures/cvqnn_example.png
    :description: :ref:`templates_CV`

:html:`<div style='clear:both'></div>`


Quantum machine learning with PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One goal of PennyLane is to enable machine learning with quantum computers. These examples show PennyLane's implementation of some of the well-known algorithms from the recent research in quantum machine learning. This is, however, just to showcase how quantum machine learning with PennyLane *could* look like. In principle, *any* technique where a quantum circuit is optimized and potentially run on a real device can be implemented in PennyLane.  


.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials/pennylane_run_state_preparation
    tutorials/pennylane_run_QGAN
    tutorials/pennylane_run_variational_classifier
    tutorials/pennylane_quantum_neural_net
    tutorials/pennylane_run_variational_quantum_eigensolver

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: ../examples/figures/state_prep.png
    :description: :ref:`state_preparation`

.. customgalleryitem::
    :tooltip: Use PennyLane to create a simple QGAN.
    :figure: ../examples/figures/gan.png
    :description: :ref:`quantum_GAN`

.. customgalleryitem::
    :tooltip: A quantum variational classifier
    :figure: ../examples/figures/classifier_output_59_0.png
    :description: :ref:`variational_classifier`

.. customgalleryitem::
    :tooltip: Fit one dimensional noisy data with a quantum neural network.
    :figure: ../examples/figures/qnn_output_28_0.png
    :description: :ref:`quantum_neural_net`

.. customgalleryitem::
    :tooltip: Find the ground state of a Hamiltonian.
    :figure: ../examples/figures/vqe_output_22_0.png
    :description: :ref:`vqe`

:html:`<div style='clear:both'></div>`

