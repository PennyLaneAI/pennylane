 .. role:: html(raw)
   :format: html

.. _New_Users:

Tutorials
=========

:html:`<h3>Learn PennyLane</h3>`


The following tutorials introduce the core PennyLane concepts, including QNodes,
plugins, and devices, via simple and easy-to-follow examples.


.. customgalleryitem::
    :tooltip: Use quantum machine learning to rotate a qubit.
    :figure: ../examples/figures/bloch.png
    :description: :ref:`qubit_rotation`

.. customgalleryitem::
    :tooltip: Use quantum machine learning to tune a beamsplitter.
    :figure: ../examples/figures/gauss-circuit.png
    :description: :ref:`gaussian_transformation`

.. customgalleryitem::
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.
    :figure: ../examples/figures/photon_redirection.png
    :description: :ref:`plugins_hybrid`

.. customgalleryitem::
    :tooltip: Multiple expectation values, Jacobians, and keyword arguments.
    :description: :ref:`advanced_features`

.. customgalleryitem::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: ../examples/figures/bloch.gif
    :description: :ref:`pytorch_noise`

:html:`<div style='clear:both'></div>`


:html:`<h3>Quantum machine learning with PennyLane</h3>`

Take a deeper dive into quantum machine learning by exploring cutting-edge
algorithms using PennyLane and near-term quantum hardware.

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: ../examples/figures/NOON.png
    :description: :ref:`state_preparation`

.. customgalleryitem::
    :tooltip: Use PennyLane to create a simple QGAN
    :figure: ../examples/figures/qgan3.png
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

.. customgalleryitem::
    :tooltip: Universal Quantum Classifier with data-reuploading
    :figure: ../examples/figures/universal_dnn.png
    :description: :ref:`data_reuploading_classifier`

.. customgalleryitem::
    :tooltip: Faster optimization convergence using quantum natural gradient
    :figure: ../examples/figures/quantum_natural_gradient/qng_optimization.png
    :description: :ref:`quantum_natural_gradient`

.. customgalleryitem::
    :tooltip: Perform QAOA for MaxCut
    :figure: ../examples/figures/qaoa_maxcut_partition.png
    :description: :ref:`qaoa_maxcut`

.. customgalleryitem::
    :tooltip: Barren plateaus in quantum neural networks
    :figure: ../examples/figures/barren_plateaus/surface.png
    :description: :ref:`barren_plateaus`

:html:`<div style='clear:both'></div>`


.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials/pennylane_run_qubit_rotation
    tutorials/pennylane_run_gaussian_transformation
    tutorials/pennylane_run_plugins_hybrid
    tutorials/pennylane_run_advanced_usage
    tutorials/pennylane_pytorch_noise
    tutorials/pennylane_run_state_preparation
    tutorials/pennylane_run_QGAN
    tutorials/pennylane_run_variational_classifier
    tutorials/pennylane_quantum_neural_net
    tutorials/pennylane_run_variational_quantum_eigensolver
    tutorials/pennylane_run_data_reuploading_classifier
    tutorials/pennylane_run_quantum_natural_gradient
    tutorials/pennylane_run_qaoa_maxcut
    tutorials/pennylane_run_barren_plateaus
