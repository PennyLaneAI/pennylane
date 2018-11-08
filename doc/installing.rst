.. _installation:

Installation
############

Dependencies
============

.. highlight:: bash

PennyLane requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.5

as well as the following Python packages:

* `NumPy <http://numpy.org/>`_  >=1.13.3
* `SciPy <http://scipy.org/>`_  >=1.0.0
* `Autograd <https://github.com/HIPS/autograd>`_
* `toml <https://github.com/uiri/toml>`_
* `appdirs <https://github.com/ActiveState/appdirs>`_


If you currently do not have Python 3 installed, we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version of Python packaged for scientific computation.


Installation
============

Installation of PennyLane, as well as all required Python packages mentioned above, can be installed via ``pip``:
::

   	$ python -m pip install pennylane


Make sure you are using the Python 3 version of pip.

Alternatively, you can install PennyLane from the source code by navigating to the top directory and running
::

	$ python setup.py install


Software tests
==============

The PennyLane test suite requires the Python pytest package; this can be installed via ``pip``:
::

	$ python -m pip install pytest

To ensure that PennyLane is working correctly, the test suite can then be run by navigating to the source code folder and running
::

	$ make test


Documentation
=============

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

These can both be installed via ``pip``:
::

	$ python3 -m pip install sphinx sphinxcontrib-bibtex

To build the HTML documentation, go to the top-level directory and run
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
