 .. role:: html(raw)
   :format: html

.. _New_Users:

Tutorials
=========

These step-by-step tutorials provided a gentle introduction to the basics of PennyLane. Once you have
worked your way through the QuickStart tutorials, you will be ready to implement your own
interesting QML problems in PennyLane!


:html:`<h3>Learn the PennyLane basics</h3>`

.. customgalleryitem::
    :tooltip: Use quantum machine learning to rotate a qubit.
    :figure: ../examples/figures/bloch.png
    :description: :ref:`qubit_rotation`

.. customgalleryitem::
    :tooltip: Use quantum machine learning to tune a beamsplitter.
    :figure: ../examples/figures/gaussian_transformation.png
    :description: :ref:`gaussian_transformation`

.. customgalleryitem::
    :tooltip: Multiple expectation values, Jacobians, and keyword arguments.
    :description: :ref:`advanced_features`


.. customgalleryitem::
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.
    :figure: ../examples/figures/photon_redirection.png
    :description: :ref:`plugins_hybrid`

.. customgalleryitem::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: ../examples/figures/bloch.gif
    :description: :ref:`pytorch_noise`

:html:`<div style='clear:both'></div>`


:html:`<h3>Quantum machine learning algorithms</h3>`

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :description: :ref:`state_preparation`

.. customgalleryitem::
    :tooltip: Use PennyLane to create a simple QGAN
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
    :description: :ref:`universal_classifier`

:html:`<div style='clear:both'></div>`


.. toctree::
    :hidden:
    :maxdepth: 2

    tutorials/pennylane_qubit_rotation
    tutorials/pennylane_gaussian_transformation
    tutorials/pennylane_advanced_usage
    tutorials/pennylane_skip_plugins_hybrid
    tutorials/pennylane_skip_pytorch_noise
    tutorials/pennylane_skip_state_preparation
    tutorials/pennylane_QGAN
    tutorials/pennylane_skip_variational_classifier
    tutorials/pennylane_skip_quantum_neural_net
    tutorials/pennylane_variational_quantum_eigensolver
    tutorials/pennylane_universal_classifier
