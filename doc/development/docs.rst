Writing documentation
=====================

This guide lists all doc source files to change when adding functionality to PennyLane.

Adding a quantum operation, measurement, template or optimizer
--------------------------------------------------------------

* Add a stub documentaion file to ``doc/code/``. The target label at the top of the file is the file name, but with
  underscores instead of dots.
  Example: ``docs/code/pennylane.ops.qubit.NewGate.rst`` has label ``_pennylane_ops_qubit_NewGate``

* Add an entry in the API file in ``docs/code/`` documenting the module.
  Example: ``docs/code/qml.rst`` gets an addition ``NewGate`` to the autosummary in the section
  'Operations - Qubits'.

* Add an entry in the quick references section.
  Example: The 'Qubit gates' section in ``docs/introduction/reference.rst`` gets a new item
  ``:ref:`NewGate <pennylane_ops_qubit_NewGate>```.


Adding a module
---------------

* Include the module in the API toctree.
  Example:

  .. code-block:: rest

      .. toctree::
         :maxdepth: 1
         :caption: API
         :hidden:

         code/qml
         code/qml_init
         code/qml_templates
         code/qml_newmodule

* Include submodules as a toc in the parent module.
  Example:

  .. code-block:: rest

    This module contains the following sub-modules:

    .. toctree::
        :maxdepth: 1

        sub1
        sub2

* Include the new module in the list of submodules of the top-level module (qml).

Adding an interface
-------------------

* Add the introduction to the new interface as ``doc/introduction/interfaces/new_interf.rst``
* Link to the interface description in ``doc/introduction/interfaces.rst``
* Update potential mentions on the landing page ``index.rst``
* Update the figure in ``doc/introduction/pennylane.rst``

Adding a plugin
---------------

* List the new plugin in ``doc/introduction/plugins.rst``
* Update potential mentions on the landing page ``index.rst``
* Update the figure in ``doc/introduction/pennylane.rst``
