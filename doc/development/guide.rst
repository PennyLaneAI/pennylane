Developers guide
================

Dependencies
------------

PennyLane requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.5

as well as the following Python packages:

* `numpy <http://numpy.org/>`_ >=1.13.3
* `scipy <http://scipy.org/>`_ >=1.0.0
* `NetworkX <https://networkx.github.io/>`_ >=1.0.0
* `autograd <https://github.com/HIPS/autograd>`_
* `toml <https://github.com/uiri/toml>`_
* `appdirs <https://github.com/ActiveState/appdirs>`_
* `semantic-version <https://github.com/rbarrois/python-semanticversion>`_ == 2.6

If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version
of Python packaged for scientific computation.

.. _install_interfaces:

Interface dependencies
~~~~~~~~~~~~~~~~~~~~~~

For development of the TensorFlow and PyTorch interfaces, additional
requirements are required.

* **PyTorch interface**: ``pytorch>=1.1``

* **TensorFlow interface**: ``tensorflow>=1.12``

  Note that any version of TensorFlow supporting eager execution mode
  is supported, however there are slight differences between the eager
  API in TensorFlow 1.X versus 2.X.

  Make sure that all modifications and tests involving the TensorFlow
  interface work for both TensorFlow 1.X and 2.X!

  This includes:

  - If ``tf.__version__[0] == "1"``, running ``tf.enable_eager_execution()``
    before execution, and importing ``Variable`` from ``tensorflow.contrib.eager``.

  - If ``tf.__version__[0] == "2"``, importing ``Variable`` from ``tensorflow``.

  - Only using the ``tf.GradientTape`` context for gradient computation.

QChem dependencies
~~~~~~~~~~~~~~~~~~

Finally, for development of the QChem package, the following dependencies
are required:

* `OpenFermion <https://github.com/quantumlib/OpenFermion>`__ >= 0.10

* `pySCF <https://sunqm.github.io/pyscf>`__
  and `OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-pyscf>`__ >= 0.4

* `Psi4 <http://www.psicode.org/>`__ and
  `OpenFermion-Psi4 <https://github.com/quantumlib/OpenFermion-Psi4>`__ >= 0.4
  (optional but recommended to run the full test suite)

  The easiest way to install Psi4 is via Ananconda:

  .. code-block:: bash

    conda install psi4 psi4-rt -c psi4

* `Open Babel <https://openbabel.org>`__ (optional but recommended to run the full test suite)

  Open Babel can be installed using ``apt`` if on Ubuntu/Debian:

  .. code-block:: bash

      sudo apt install openbabel

  or using Anaconda:

  .. code-block:: bash

      conda install -c conda-forge openbabel

Installation
------------

For development purposes, it is recommended to install PennyLane source code
using development mode:

.. code-block:: bash

    git clone https://github.com/XanaduAI/pennylane
    cd pennylane
    pip install -e .

If also developing for the QChem package, this will need to installed as well:

.. code-block:: bash

    pip install -e qchem

The ``-e`` flag ensures that edits to the source code will be reflected when
importing PennyLane in Python.


.. note::

    Due to the use of :ref:`entry points <installing_plugin>` to install
    plugins, changes to PennyLane device class locations or shortnames
    requires ``pip install -e .`` to be re-run in the plugin repository
    for the changes to take effect.

Software tests
--------------

The PennyLane test suite requires the Python pytest package, as well as pytest-cov
for test coverage; these can be installed via ``pip``:

.. code-block:: bash

    pip install pytest pytest-cov

To ensure that PennyLane is working correctly, the test suite can then be run by
navigating to the source code folder and running

.. code-block:: bash

    make test

while the test coverage can be checked by running

.. code-block:: bash

    make coverage

The output of the above command will show the coverage percentage of each
file, as well as the line numbers of any lines missing test coverage.


Documentation
-------------

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


Submitting a pull request
-------------------------

Before submitting a pull request, please make sure the following is done:

* **All new features must include a unit test.** If you've fixed a bug or added
  code that should be tested, add a test to the ``tests`` directory.

  PennyLane uses pytest for testing; common fixtures can be found in the ``tests/conftest.py``
  file.

* **All new functions and code must be clearly commented and documented.**

  Have a look through the source code at some of the existing function docstrings---
  the easiest approach is to simply copy an existing docstring and modify it as appropriate.

  If you do make documentation changes, make sure that the docs build and render correctly by
  running ``make docs``.

* **Ensure that the test suite passes**, by running ``make test``.

* **Make sure the modified code in the pull request conforms to the PEP8 coding standard.**

  The PennyLane source code conforms to `PEP8 standards <https://www.python.org/dev/peps/pep-0008/>`_.
  Before submitting the PR, you can autoformat your code changes using the
  `Black <https://github.com/psf/black>`_ Python autoformatter, with max-line length set to 100:

  .. code-block:: bash

      black -l 100 pennylane/path/to/modified/file.py

  We check all of our code against `Pylint <https://www.pylint.org/>`_ for errors.
  To lint modified files, simply ``pip install pylint``, and then from the source code
  directory, run

  .. code-block:: bash

      pylint pennylane/path/to/modified/file.py


When ready, submit your fork as a `pull request <https://help.github.com/articles/about-pull-requests>`_
to the PennyLane repository, filling out the pull request template. This template is added
automatically to the comment box when you create a new issue.

* When describing the pull request, please include as much detail as possible
  regarding the changes made/new features added/performance improvements. If including any
  bug fixes, mention the issue numbers associated with the bugs.

* Once you have submitted the pull request, three things will automatically occur:

  - The **test suite** will automatically run on `Travis CI <https://travis-ci.org/XanaduAI/pennylane>`_
    to ensure that all tests continue to pass.

  - Once the test suite is finished, a **code coverage report** will be generated on
    `Codecov <https://codecov.io/gh/XanaduAI/pennylane>`_. This will calculate the percentage
    of PennyLane covered by the test suite, to ensure that all new code additions
    are adequately tested.

  - Finally, the **code quality** is calculated by
    `Codefactor <https://app.codacy.com/app/XanaduAI/pennylane/dashboard>`_,
    to ensure all new code additions adhere to our code quality standards.

Based on these reports, we may ask you to make small changes to your branch before
merging the pull request into the master branch. Alternatively, you can also
`grant us permission to make changes to your pull request branch
<https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/>`_.

.. _adding_new_templates:

.. include:: adding_templates.rst
