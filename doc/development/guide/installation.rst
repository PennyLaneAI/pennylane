Installation and dependencies
=============================

Dependencies
------------

PennyLane requires the following libraries be installed:

* `Python <http://python.org/>`_ >= 3.10

The following Python packages are hard dependencies, and will automatically
be installed alongside PennyLane:

* `numpy <http://numpy.org/>`_ < 2.1
* `scipy <http://scipy.org/>`_
* `NetworkX <https://networkx.github.io/>`_
* `rustworkx <https://github.com/Qiskit/rustworkx>`_ >= 0.14.0
* `autograd <https://github.com/HIPS/autograd>`_
* `toml <https://github.com/uiri/toml>`_
* `appdirs <https://github.com/ActiveState/appdirs>`_
* `autoray <https://github.com/jcmgray/autoray>`__ >= 0.6.11
* `cachetools <https://github.com/tkem/cachetools>`_
* `pennylane-lightning <https://github.com/PennyLaneAI/pennylane-lightning>`_ >= 0.38
* `requests <https://github.com/psf/requests>`_
* `typing_extensions <https://github.com/python/typing_extensions>`_
* `packaging <https://github.com/pypa/packaging>`_

The following Python packages are optional:

* `openfermionpyscf <https://github.com/quantumlib/OpenFermion-PySCF>`_, for the non-differentiable backend of the ``qml.qchem`` module

If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version
of Python packaged for scientific computation.

.. _install_interfaces:

Interface dependencies
~~~~~~~~~~~~~~~~~~~~~~

For development of the TensorFlow, PyTorch, and JAX interfaces, there are additional
requirements which must be installed manually:

* **JAX interface**: ``jax`` and ``jaxlib`` <= 0.4.28

* **PyTorch interface**: ``pytorch`` <= 2.3.0

* **TensorFlow interface**: ``tensorflow`` < 2.17


Installation
------------

For development purposes, it is recommended to install PennyLane source code
using development mode:

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane
    cd pennylane
    pip install -e .

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

    pip install -r requirements-dev.txt

Docker
------

Build a PennyLane Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Docker** support exists for building using **CPU** and **GPU** (Nvidia CUDA 11.1+) images.

.. note::

    Docker builds using "make" will work on Linux and MacOS only. For MS Windows
    you can use `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
    They are currently not supported on the Apple M1 chip (ARM64).


Build a basic PennyLane image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- To build a basic PennyLane image without any additional interfaces (Torch,
  TensorFlow, or Jax) or **plugins** (qiskit, amazon-braket, cirq, forest), run
  the following:

  .. code-block:: bash

    make -f docker/Makefile build-base

Build a PennyLane image with a specific interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- To build a PennyLane image using a specific **interface** (Torch, TensorFlow or Jax), run the following:

  .. code-block:: bash

    make -f docker/Makefile build-interface interface-name=tensorflow

- To build a PennyLane image using a specific interface (Torch, TensorFlow or
  Jax) with GPU support, run the following:

  .. code-block:: bash

    make -f docker/Makefile build-interface-gpu interface-name=tensorflow

Build a PennyLane image with a plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- To build a PennyLane image using a specific plugin (qiskit, amazon-braket,
  cirq, forest, etc), run the following:

  .. code-block:: bash

    make -f docker/Makefile build-plugin plugin-name=qiskit
