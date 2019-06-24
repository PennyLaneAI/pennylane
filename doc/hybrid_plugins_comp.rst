.. role:: html(raw)
   :format: html

.. _hybrid_plugins_comp:

Using plugins and Quantum Hardware
===================================

The most powerful design feature of PennyLane is that it allows the user to run hybrid computations on a variety of different simulators and real quantum hardware. The framework that executes a quantum node is called the *device*. *Plugins* are the interfaces that allow PennyLane to communicate with different devices. They define whether and how PennyLane's standard operations are implemented, and how expectation values are computed.

.. note:: 
            Plugins have to be separately installed; see :ref:`plugins` for more details. 

The following tutorials introduce the notion of hybrid computation by combining several PennyLane plugins.


.. toctree::
    :hidden:
    :maxdepth: 1

    tutorials/pennylane_skip_plugins_hybrid
    tutorials/pennylane_skip_pytorch_noise
    tutorials/pennylane_skip_state_preparation

.. customgalleryitem::
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.
    :figure: ../examples/figures/photon_redirection.png
    :description: :ref:`plugins_hybrid`

.. customgalleryitem::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: ../examples/figures/bloch.gif
    :description: :ref:`pytorch_noise`

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: ../examples/figures/state_prep.png
    :description: :ref:`state_preparation`


:html:`<div style='clear:both'></div>`
