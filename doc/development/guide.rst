Developers guide
================

Dependencies
------------

.. highlight:: bash

PennyLane requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.5

as well as the following Python packages:

* `numpy <http://numpy.org/>`_  >=1.13.3
* `scipy <http://scipy.org/>`_  >=1.0.0
* `NetworkX <https://networkx.github.io/>`_  >=1.0.0
* `autograd <https://github.com/HIPS/autograd>`_
* `toml <https://github.com/uiri/toml>`_
* `appdirs <https://github.com/ActiveState/appdirs>`_
* `semantic-version <https://github.com/rbarrois/python-semanticversion>`_ ==2.6

If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version
of Python packaged for scientific computation.

Interfaces
~~~~~~~~~~

For development of the TensorFlow and PyTorch interfaces, additional
requirements are required.

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

* **PyTorch interface**: ``pytorch>=1.1``


Installation
------------

For development purposes, it is recommended to install PennyLane source code
using development mode:
::

	git clone https://github.com/XanaduAI/pennylane
    cd pennylane
    pip install -e .

The ``-e`` flag ensures that edits to the source code will be reflected when
importing PennyLane in Python.


Software tests
--------------

The PennyLane test suite requires the Python pytest package, as well as pytest-cov for test coverage; these can be installed via ``pip``:
::

	pip install pytest pytest-cov

To ensure that PennyLane is working correctly, the test suite can then be run by navigating to the source code folder and running
::

	make test

while the coverage can be checked by running
::

	make coverage


Documentation
-------------

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `Sphinx-Gallery <https://sphinx-gallery.github.io/>`_ >=0.3
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

These can both be installed via ``pip``:
::

	$ python3 -m pip install sphinx sphinx_gallery sphinxcontrib-bibtex

To build the HTML documentation, go to the top-level directory and run
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.


Submitting a pull request