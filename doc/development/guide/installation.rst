Installation and dependencies
=============================

Dependencies
------------

PennyLane requires `Python <http://python.org/>`_ >= 3.11 to be installed.

After installing Python, we recommend using any virtual environment manager to install and manage
dependencies. See the `Python documentation <https://docs.python.org/3/tutorial/venv.html>`_
for more an example.

The following Python packages are hard dependencies, and will automatically
be installed alongside PennyLane:

* `numpy <http://numpy.org/>`_
* `scipy <http://scipy.org/>`_
* `NetworkX <https://networkx.github.io/>`_
* `rustworkx <https://github.com/Qiskit/rustworkx>`_ >= 0.14.0
* `autograd <https://github.com/HIPS/autograd>`_
* `tomlkit <https://github.com/python-poetry/tomlkit>`_
* `appdirs <https://github.com/ActiveState/appdirs>`_
* `autoray <https://github.com/jcmgray/autoray>`__ == 0.8.2 
* `cachetools <https://github.com/tkem/cachetools>`_
* `pennylane-lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ >= 0.42
* `requests <https://github.com/psf/requests>`_
* `typing_extensions <https://github.com/python/typing_extensions>`_
* `packaging <https://github.com/pypa/packaging>`_
* `diastatic-malt <https://github.com/PennyLaneAI/diastatic-malt>`_

The following Python packages are optional:

* `openfermionpyscf <https://github.com/quantumlib/OpenFermion-PySCF>`_, for the non-differentiable backend of the ``qml.qchem`` module
* ``matplotlib``: for ``qml.draw_mpl`` and associated code
* ``quimb``: for the ``default.tensor`` device
* ``pyzx``: for ``qml.transforms.to_zx`` and ``qml.transforms.from_zx``
* ``stim``: for ``default.clifford``
* ``openqasm3`` and ``antlr3_python3_runtime``: for ``qml.from_qasm3``
* ``kahypar`` and ``opt_einsum`` for ``qcut``
* ``cvxopt``for ``qml.kernels.closest_psd_matrix``

.. _install_interfaces:

Interface dependencies
~~~~~~~~~~~~~~~~~~~~~~

For development of the PyTorch and JAX interfaces, there are additional
requirements which must be installed manually:

* **JAX interface**: ``jax`` and ``jaxlib`` ~= 0.6.0

* **PyTorch interface**: ``pytorch``

Installation
------------

For development purposes, it is recommended to install PennyLane source code
using development mode:

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane
    cd pennylane
    python -m pip install -e .

The ``-e`` flag ensures that edits to the source code will be reflected when
importing PennyLane in Python.


.. note::

    Due to the use of :ref:`entry points <installing_plugin>` to install
    plugins, changes to PennyLane device class locations or shortnames
    requires ``pip install -e .`` to be re-run in the plugin repository
    for the changes to take effect.

Apart from the core packages needed to run PennyLane, some extra packages need
to be installed for several development processes, such as linting, testing, and
pre-commit quality checks. Those can be installed easily via ``pip``:

.. code-block:: bash

    python -m pip install --group dev
