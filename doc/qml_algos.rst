 .. role:: html(raw)
   :format: html

.. _qml_algos:

Quantum machine learning algorithms
-----------------------------------

PennyLane is desgined to do machine learning with quantum computers. These examples show PennyLane's implementation of some of the well-known algorithms from the recent research in quantum machine learning. This is, however, just to showcase how quantum machine learning with PennyLane *could* look like. In principle *any* technique where a quantum circuit is optimized and potentially run on a real device can be implemented in PennyLane.  


.. toctree::
    :hidden:
    :maxdepth: 1

    tutorials/pennylane_skip_QGAN
    tutorials/pennylane_skip_variational_classifier
    tutorials/pennylane_skip_quantum_neural_net
    tutorials/pennylane_variational_quantum_eigensolver


.. customgalleryitem::
    :tooltip: Use PennyLane to create a simple QGAN.
    :figure: ../examples/figures/gan.png
    :description: :ref:`QGAN`

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
    :description: :ref:`variational_quantum_eigensolver`

:html:`<div style='clear:both'></div>`
