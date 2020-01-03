PennyLane Documentation
=======================

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

    <style>
        #right-column.card {
            box-shadow: none!important;
        }
        #right-column.card:hover {
            box-shadow: none!important;
        }
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        .footer-relations {
            border-top: 0px;
        }
    </style>
    <div class="container mt-2 mb-2">
        <p class="lead grey-text">
            PennyLane is a cross-platform Python library for quantum machine learning,
            automatic differentiation, and optimization of hybrid quantum-classical computations.
        </p>
        <div class="row mt-3">
            <div class="col-lg-4 mb-2 adlign-items-stretch">
                <a href="introduction/pennylane.html">
                    <div class="card rounded-lg" style="height:100%;">
                        <div class="d-flex">
                            <div>
                                <h3 class="card-title pl-3 mt-4">
                                Using PennyLane
                                </h3>
                                <p class="mb-3 grey-text px-3">
                                    A guided tour of the core features of PennyLane <i class="fas fa-angle-double-right"></i>
                                </p>
                            </div>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="development/guide.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            Developing
                            </h3>
                            <p class="mb-3 grey-text px-3">How you can contribute to the development of PennyLane <i class="fas fa-angle-double-right"></i></p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="code/qml.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            API
                            </h3>
                            <p class="mb-3 grey-text px-3">Explore the PennyLane API <i class="fas fa-angle-double-right"></i></p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
        </div>
    </div>

Features
--------

.. image:: _static/intro.png
    :align: right
    :width: 400px
    :target: javascript:void(0);


- *Follow the gradient*.
  Built-in **automatic differentiation** of quantum circuits.

..

- *Best of both worlds*.
  Support for **hybrid quantum and classical** models; connect quantum
  hardware with PyTorch, TensorFlow, and NumPy.

..

- *Batteries included*.
  Provides **optimization and machine learning** tools.

..

- *Device independent*.
  The same quantum circuit model can be **run on different backends**. Install
  `plugins <https://pennylane.ai/plugins.html>`_ to access even more
  devices, including **Strawberry Fields**, **IBM Q**, **Google Cirq**, **Rigetti Forest**,
  **Microsoft QDK**, and **ProjectQ**.


Getting started
---------------

For getting started with PennyLane, check out some of the
`key concepts <https://pennylane.ai/qml/concepts.html>`_ behind quantum machine
learning, before moving on to some `introductory tutorials <https://pennylane.ai/qml/beginner.html>`_.

Then, take a deeper dive into quantum machine learning by
exploring cutting-edge algorithms using PennyLane and near-term quantum hardware,
with our collection of
`QML tutorials <https://pennylane.ai/qml/implementations.html>`_.

You can also check out the :doc:`Using PennyLane <introduction/pennylane>` section for
more details on the :doc:`quantum operations <introduction/operations>`, and to explore
the available :doc:`optimization tools <introduction/optimizers>` provided by PennyLane.
We also have a detailed guide on :doc:`how to write your own <development/plugins>`
PennyLane-compatible quantum device.

Finally, play around with the numerous `devices and plugins <https://pennylane.ai/plugins.html>`_
available for running your hybrid optimizations—these include
IBM Q, provided by the `PennyLane-Qiskit <https://pennylane-qiskit.rtfd.io>`__ plugin,
as well as the Rigetti Aspen-1 QPU provided by `PennyLane-Forest <https://pennylane-forest.rtfd.io>`__.

How to cite
-----------

If you are doing research using PennyLane, please cite

.. rst-class:: admonition warning

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Carsten Blank, Keri McKiernan,
    and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

Support
-------

- **Source Code:** https://github.com/XanaduAI/PennyLane
- **Issue Tracker:** https://github.com/XanaduAI/PennyLane/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

We also have a `PennyLane discussion forum <https://discuss.pennylane.ai>`_—come join the
discussion and chat with our PennyLane team.

License
-------

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.

.. toctree::
   :maxdepth: 1
   :caption: Using PennyLane
   :hidden:

   introduction/pennylane
   introduction/circuits
   introduction/interfaces
   introduction/operations
   introduction/measurements
   introduction/templates
   introduction/optimizers
   introduction/chemistry
   introduction/configuration

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   development/guide
   development/plugins
   development/research

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/qml
   code/qml_init
   code/qml_interfaces
   code/qml_operation
   code/qml_plugins
   code/qml_qchem
   code/qml_qnodes
   code/qml_templates
   code/qml_utils
   code/qml_variable
   code/qml_beta
