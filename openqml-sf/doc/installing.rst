.. _installation:

Installation and downloads
#################################

.. .. include:: ../README.rst
   :start-line: 6

Dependencies
============

.. highlight:: bash

OpenQML requires the following libraries be installed: TODO FIXME

* `Python <http://python.org/>`_ >=3.5

as well as the following Python packages:

* `NumPy <http://numpy.org/>`_  >=1.13.3
* `SciPy <http://scipy.org/>`_  >=1.0.0
* `NetworkX <http://networkx.github.io/>`_ >=2.0
* `TensorFlow <https://www.tensorflow.org/>`_ >=1.3,<1.7


If you currently do not have Python 3 installed, we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version of Python packaged for scientific computation.


Installation
============

Installation of OpenQML, as well as all required Python packages mentioned above, can be installed via ``pip``:
::

   	$ python -m pip install openqml


Make sure you are using the Python 3 version of pip.

Alternatively, you can install OpenQML from the source code by navigating to the top directory and running
::

	$ python setup.py install


Software tests
==============

To ensure that OpenQML is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

	$ make test


Documentation
=============

To build the documentation, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `graphviz <http://graphviz.org/>`_ >=2.38
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

If using Ubuntu, they can be installed via a combination of ``apt`` and ``pip``:
::

	$ sudo apt install graphviz
	$ pip install sphinx --user
	$ pip install sphinxcontrib-bibtex --user

To build the HTML documentation, go to the top-level directory and run
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
