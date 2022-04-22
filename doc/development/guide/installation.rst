Installation and dependencies
=============================

Dependencies
------------

PennyLane requires the following libraries be installed:

* `Python <http://python.org/>`_ >= 3.7

as well as the following Python packages:

* `numpy <http://numpy.org/>`_ >= 1.13.3
* `scipy <http://scipy.org/>`_ >= 1.0.0
* `NetworkX <https://networkx.github.io/>`_ >= 1.0.0
* `autograd <https://github.com/HIPS/autograd>`_
* `toml <https://github.com/uiri/toml>`_
* `appdirs <https://github.com/ActiveState/appdirs>`_
* `semantic-version <https://github.com/rbarrois/python-semanticversion>`_ == 2.6
* `autoray <https://github.com/jcmgray/autoray>`__

The following Python packages are optional:

* `dask["parallel"] <https://dask.org/>`_, for parallel QNodeCollection execution
* `tensornetwork <https://github.com/google/TensorNetwork>`_ >= 0.3, for the ``default.tensor`` plugin

If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version
of Python packaged for scientific computation.

.. _install_interfaces:

Interface dependencies
~~~~~~~~~~~~~~~~~~~~~~

For development of the TensorFlow and PyTorch interfaces, there are additional
requirements:

* **PyTorch interface**: ``pytorch >= 1.1``

* **TensorFlow interface**: ``tensorflow >= 1.15``

  Note that any version of TensorFlow supporting eager execution mode
  is supported, however there are slight differences between the eager
  API in TensorFlow 1.X versus 2.X.

  Make sure that all modifications and tests involving the TensorFlow
  interface work for both TensorFlow 1.X and 2.X!

  This includes:

  - If ``tf.__version__[0] == "1"``, running ``tf.enable_eager_execution()``
    before execution.

  - Only using the ``tf.GradientTape`` context for gradient computation.

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
