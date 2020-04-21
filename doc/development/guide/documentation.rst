Documentation
=============

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ == 1.8.5
* `pygments-github-lexers <https://github.com/liluo/pygments-github-lexers>`_
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ == 0.4.2
* `sphinx-automodapi <https://github.com/astropy/sphinx-automodapi>`_

These can all be installed via ``pip``.

To build the HTML documentation, go to the top-level directory and run

.. code-block:: bash

    make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.

.. note::

    To build the interfaces documentation, PyTorch and TensorFlow will need to
    be installed, see :ref:`install_interfaces`.

Adding a new module to the docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several steps to adding a new module to the documentation:

1. Make sure your module has a one-to-two line module docstring, that summarizes
   what the module purpose is, and what it contains.

2. Add a file ``doc/code/qml_module_name.rst``, that contains the following:

   .. literalinclude:: example_module_rst.txt
       :language: rest

3. Add ``code/qml_module_name`` to the table of contents at the bottom of ``doc/index.rst``.


Adding a new package to the docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new subpackage to the documentation requires a slightly different process than
a module:

1. Make sure your package ``__init__.py`` file has a one-to-two line module docstring,
   that summarizes what the package purpose is, and what it contains.

2. At the bottom of the ``__init__.py`` docstring, add an autosummary table that contains
   all modules in your package:

   .. literalinclude:: example_module_autosummary.txt
       :language: rest

   All modules should also contain a module docstring that summarizes the module.

3. Add a file ``doc/code/qml_package_name.rst``, that contains the following:

   .. literalinclude:: example_package_rst.txt
       :language: rest

4. Add ``code/qml_package_name`` to the table of contents at the bottom of ``doc/index.rst``.
