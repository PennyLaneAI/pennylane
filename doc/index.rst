.. role:: html(raw)
   :format: html

:html:`<br>`


.. image:: _static/pennylane_thin.png
    :align: center
    :width: 100%
    :target: javascript:void(0);
    :alt: PennyLane

-----------------------------------------------


.. raw:: html

    <style>
    h1 {
        display: none;
    }
    </style>


PennyLane
=========


:Release: |release|
:Date: |today|

PennyLane is a cross-platform Python library for quantum machine learning,
automatic differentiation, and optimization of hybrid quantum-classical computations.


:html:`<h2>Features</h2>`


.. image:: _static/code.png
    :align: right
    :width: 320px
    :target: javascript:void(0);


- *Follow the gradient*.
  Built-in **automatic differentiation** of quantum circuits

..

- *Best of both worlds*.
  Support for **hybrid quantum and classical** models

..

- *Batteries included*.
  Provides **optimization and machine learning** tools

..

- *Device independent*.
  The same quantum circuit model can be **run on different backends**

..

- *Compatible with existing machine learning libraries*.
  Quantum circuits can be set up to interface with either **NumPy**, **PyTorch**, or **TensorFlow**,
  allowing hybrid CPU-GPU-QPU computations.

..

- *Large plugin ecosystem*.
  Install plugins to run your computational circuits on more devices, including **Strawberry Fields**, **Rigetti Forest**, **ProjectQ**, **Microsoft QDK**, and **IBM Q**

:html:`<h4>Available plugins</h4>`

* `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`__: Supports integration with `Strawberry Fields <https://github.com/XanaduAI/strawberryfields>`__, a full-stack Python library for simulating continuous variable (CV) quantum optical circuits.

..

* `PennyLane-Forest <https://github.com/rigetti/pennylane-forest>`_: Supports integration with `PyQuil <https://github.com/rigetti/pyquil>`__, the `Rigetti Forest SDK <https://www.rigetti.com/forest>`__, and the `Rigetti QCS <https://www.rigetti.com/qcs>`__, an open-source quantum computation framework by Rigetti. Provides device support for the Quantum Virtual Machine (QVM) and Quantum Processing Units (QPUs) hardware devices.

..

* `PennyLane-qiskit <https://github.com/carstenblank/pennylane-qiskit>`__: Supports integration with `Qiskit Terra <https://qiskit.org/terra>`__, an open-source quantum computation framework by IBM. Provides device support for the Qiskit Aer quantum simulators, and IBM QX hardware devices.

..

* `PennyLane-PQ <https://github.com/XanaduAI/pennylane-pq>`__: Supports integration with `ProjectQ <https://github.com/ProjectQ-Framework/ProjectQ>`__, an open-source quantum computation framework that supports the IBM quantum experience.

..

* `PennyLane-Qsharp <https://github.com/XanaduAI/pennylane-qsharp>`_: Supports integration with the `Microsoft Quantum Development Kit <https://www.microsoft.com/en-us/quantum/development-kit>`__, a quantum computation framework that uses the Q# quantum programming language.


:html:`<h2>Getting started</h2>`

To get PennyLane installed and running on your system, begin at the :ref:`download and installation guide <installation>`. Then, familiarize yourself with the PennyLane's :ref:`key concepts <introduction>` for machine learning on quantum circuits.

For getting started with PennyLane, check out our basic :ref:`qubit rotation <qubit_rotation>`, and :ref:`Gaussian transformation <gaussian_transformation>` tutorials, before continuing on to explore :ref:`hybrid quantum optimization <plugins_hybrid>`, and :ref:`hybrid GPU-QPU optimization via PyTorch <pytorch_noise>`. More advanced tutorials include supervised learning, building quantum GANs (QGANs), and quantum classifiers.

Next, play around with the numerous devices and :ref:`plugins <plugins>` available for running your hybrid models — these include Strawberry Fields, provided by the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin, the Rigetti Aspen-1 QPU, provided by the `PennyLane-Forest <https://github.com/rigetti/pennylane-forest>`_ plugin, and the IBM QX4 quantum chip, provided by the `PennyLane-PQ <https://github.com/XanaduAI/pennylane-pq>`_ and `PennyLane-qiskit <https://github.com/carstenblank/pennylane-qiskit>`_ plugins.

Finally, detailed documentation on the PennyLane :ref:`interface <library_overview>` and API is provided. Look there for full details on available quantum operations and expectations, and detailed guides on :ref:`how to write your own <developer_overview>` PennyLane compatible quantum device.

:html:`<h2>How to cite</h2>`

If you are doing research using PennyLane, please cite

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Carsten Blank, Keri McKiernan, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

:html:`<h2>Support</h2>`

- **Source Code:** https://github.com/XanaduAI/PennyLane
- **Issue Tracker:** https://github.com/XanaduAI/PennyLane/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `PennyLane discussion forum <https://discuss.pennylane.ai>`_ - come join the discussion and chat with our PennyLane team.

For more details on contributing or performing research with PennyLane, please see
:ref:`research`.

:html:`<h2>License</h2>`

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.


.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   installing
   plugins
   research
   Get Help<https://discuss.pennylane.ai/>

.. toctree::
   :maxdepth: 1
   :caption: Key concepts
   :hidden:

   concepts/introduction
   concepts/hybrid_computation
   concepts/quantum_nodes
   concepts/concept_embeddings
   concepts/varcirc
   concepts/autograd_quantum

   zreferences

.. toctree::
   :maxdepth: 1
   :caption: Quickstart
   :hidden:

   tutorials


.. toctree::
   :maxdepth: 1
   :caption: User documentation
   :hidden:

   code/init
   code/decorator
   code/ops
   code/measure
   code/templates
   code/optimize
   code/configuration

.. toctree::
   :maxdepth: 1
   :caption: Plugin API
   :hidden:

   API/overview
   API/device
   API/reference_plugins

.. toctree::
   :maxdepth: 1
   :caption: Developer API
   :hidden:

   API/qnode
   API/utils
   API/operation
   API/variable
   API/circuit_graph


:html:`<h2>Indices and tables</h2>`

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
