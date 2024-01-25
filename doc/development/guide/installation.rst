Installation and dependencies
=============================

Dependencies
------------

PennyLane requires the following libraries be installed:

* `Python <http://python.org/>`_ >= 3.9

The following Python packages are hard dependencies, and will automatically
be installed alongside PennyLane:

* `numpy <http://numpy.org/>`_ >= 1.13.3
* `scipy <http://scipy.org/>`_ >= 1.0.0
* `NetworkX <https://networkx.github.io/>`_ >= 1.0.0
* `autograd <https://github.com/HIPS/autograd>`_
* `toml <https://github.com/uiri/toml>`_
* `appdirs <https://github.com/ActiveState/appdirs>`_
* `semantic-version <https://github.com/rbarrois/python-semanticversion>`_ >= 2.7
* `autoray <https://github.com/jcmgray/autoray>`__ >= 0.6.1

The following Python packages are optional:

* `dask["parallel"] <https://dask.org/>`_, for parallel QNodeCollection execution
* `tensornetwork <https://github.com/google/TensorNetwork>`_ >= 0.3, for the ``default.tensor`` plugin
* `openfermionpyscf <https://github.com/quantumlib/OpenFermion-PySCF>`_, for the non-differentiable backend of the ``qml.qchem`` module

If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version
of Python packaged for scientific computation.

.. _install_interfaces:

Interface dependencies
~~~~~~~~~~~~~~~~~~~~~~

For development of the TensorFlow, PyTorch, and JAX interfaces, there are additional
requirements which must be installed manually:

* **JAX interface**: ``jax > 0.2.0`` and ``jaxlib``

* **PyTorch interface**: ``pytorch >= 1.1``

* **TensorFlow interface**: ``tensorflow >= 2.3``


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

Poetry
------

PennyLane uses `Poetry <https://python-poetry.org/>`_ to manage dependencies internally. If you
are familiar with Poetry, you might prefer this tool over ``pip`` for installing and managing
PennyLane on your devices. To install an editable version of PennyLane (once it's cloned to your
machine), just run ``poetry install``.

Depenency Groups
~~~~~~~~~~~~~~~~

Optional dependencies are organized into groups for convenience. While the groups are treated as
optional for most purposes, Poetry will still lock versions of each package such that all groups
are compatible with each other, as well as with PennyLane itself. At the time of writing, the
groups are as follows:

* ``dev``: Recommended packages for developers that wish to contribute to PennyLane
* ``ci``: Packages used by PennyLane CI workflows
* ``doc``: Packages used by ReadTheDocs to build PennyLane documentation
* ``jax``: Supported versions of jax and dependent packages, for users who wish to use the jax interface
* ``torch``: Supported version of torch, for users who wish to use the torch interface
* ``tf``: Supported version of tensorflow and dependent packages, for users who wish to use the tensorflow interface
* ``external``: Various external dependencies that only certain modules require, such as ``PyZX``, ``matplotlib`` and ``stim``
* ``qcut``: Packages needed to use all features of the ``qcut`` module
* ``qchem``: Packages needed to use all features of the ``qchem`` module
* ``data``: Packages needed to use the ``data`` module

Many commands will exclude these groups by default, as they are specified as optional. If you wish
to include them, this can typically be done using the ``--with`` option. For example, to install
PennyLane along with the jax and torch, you can run ``poetry install --with jax,torch``.

Managing Dependencies
~~~~~~~~~~~~~~~~~~~~~

For core dependencies, use ``poetry add <the_dependency>``. For optional dependencies, specify the
group using ``poetry add --group <the_group> <the_dependency>``. You can also provide a version
specifier if you have one, otherwise Poetry will compute and provide one for you. Please note that
if you are adding a new dependency group, they are not optional by default in Poetry, so you must
update the group options in ``pyproject.yaml`` to ``optional = true`` manually.

To update a dependency (core or optional) in the lockfile, simply run ``poetry update
<the_dependency> --lock``. If you omit the ``--lock`` option, it will also update it in your
environment. The Poetry CLI cannot be used to update the version constraints on a dependency; if
you wish to do this, please update the constraints manually in ``pyproject.toml``.

.. note::

    Calling ``poetry update --only <group>`` will also update all core dependencies. If you wish
    to update all dependencies in a group (but not the core dependencies), you must list each
    package name explicitly.

If you make any manual changes to ``pyproject.toml``, be sure to run ``poetry lock --no-update``
to update the lockfile (``poetry.lock``). Note that this file should only be modified by running
this exact command. Manually updating it is not recommended by Poetry itself, and we prefer the
``--no-update`` option to continue using minimal supported versions of dependencies. See `the
documentation on version contraints <https://python-poetry.org/docs/dependency-specification/#version-constraints>`_
provided by Poetry on how to specify supported version ranges for dependencies.

.. note::

    ``poetry show --tree`` will display a tree of dependencies for PennyLane, along with all
    downstream dependencies. Used with the ``--with`` option as detailed above, this is a very
    powerful tool for managing and understanding dependencies.

Updating requirements.txt files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many external users will choose to stick to pip, or some installation tool other than poetry. To
continue supporting all users, we should export version contraints from Poetry to this format using
a command like the following:

.. code::

    poetry export -f requirements.txt -o <target-file-name>.txt --without-hashes --without-urls

Each requirement files requires additional options to ensure completeness. They are as follows:

* ``requirements.txt``: None
* ``requirements-dev.txt``: ``--only dev``
* ``requirements-ci.txt``: ``--only ci``
* ``doc/requirements.txt``: ``--with doc,torch,jax,tf``

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
