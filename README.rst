OpenQML
#######

OpenQML is a Python quantum machine learning library by Xanadu Inc.


Features
========


Dependencies
============

OpenQML depends on the following Python packages: TODO FIXME

* `Python <http://python.org/>`_ >=3.5
* `NumPy <http://numpy.org/>`_  >=1.13.3
* `SciPy <http://scipy.org/>`_  >=1.0.0
* `NetworkX <http://networkx.github.io/>`_ >=2.0
* `Tensorflow <https://www.tensorflow.org/>`_ >=1.3

These can be installed using pip, or, if on linux, using your package manager (i.e. ``apt`` if on a Debian-based system.)


Installation
============


Software tests
==============


Documentation
=============

To build the documentation locally, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.6
* `graphviz <http://graphviz.org/>`_ >=2.38
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

If using Ubuntu, they can be installed via a combination of ``apt`` and ``pip``:
::

    $ sudo apt install graphviz
    $ pip3 install sphinx --user
    $ pip3 install sphinxcontrib-bibtex --user

To build the HTML documentation, go to the top-level directory and run the command
::

  $ make html

The documentation can then be found in the ``doc/_build/html/`` directory.

Authors
=======

Ville Bergholm, Nathan Killoran and Maria Schuld.


Support
=======

- **Source Code:** https://github.com/XanaduAI/openqml
- **Issue Tracker:** https://github.com/XanaduAI/openqml/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

TODO
