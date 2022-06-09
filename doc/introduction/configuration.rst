.. role:: html(raw)
   :format: html

:orphan:

.. _intro_ref_config:

Configuration
=============

Some important default settings for a device, such as your user credentials for quantum hardware
access, the number of shots, or the cutoff dimension for continuous-variable simulators, are
defined in a configuration file called ``config.toml``.

Behaviour
---------

On first import, PennyLane attempts to load the configuration file by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``PENNYLANE_CONF``
3. The default user configuration directory:

   * On Linux: ``~/.config/pennylane``
   * On Windows: ``~C:\Users\USERNAME\AppData\Local\Xanadu\pennylane``
   * On MacOS: ``~/Library/Preferences/pennylane``

If no configuration file is found, a warning message will be displayed in the logs,
and all device parameters will need to be passed as keyword arguments when
loading the device.

The loaded configuration can be accessed via ``pennylane.default_config``, view the
loaded configuration filepath, print the configurations options, access and modify
them via keys (i.e., ``pennylane.default_config["main.shots"]``), and save/load new configuration files.

For example:

>>> import pennylane as qml
>>> qml.default_config.path
'config.toml'
>>> print(qml.default_config)
{'main': {'shots': 1000},
 'default': {'gaussian': {'hbar': 2}},
 'strawberryfields': {'fock': {'cutoff_dim': 10, 'shots': 1000, 'hbar': 2}}
}

Format
------

The configuration file ``config.toml`` uses the `TOML standard <https://github.com/toml-lang/toml>`_.
See the following example configuration that configures some global options, as well as plugin
and plugin device-specific options.

.. code-block:: toml

    [main]
    # Global PennyLane options.
    # Affects every loaded plugin if applicable.
    shots = 1000

    [strawberryfields.global]
    # Options for the Strawberry Fields plugin
    # For more details, see the PennyLane-SF documentation:
    # https://pennylane-sf.readthedocs.io
    hbar = 2
    shots = 100

        [strawberryfields.fock]
        # Options for the strawberryfields.fock device
        cutoff_dim = 10
        hbar = 2

        [strawberryfields.gaussian]
        # Indentation doesn't matter in TOML files,
        # but helps provide clarity.

    [qiskit.global]
    # Global options for the Qiskit plugin.
    # For more details, see the PennyLane-Qiskit documentation:
    # https://pennylaneqiskit.readthedocs.io/en/latest/index.html

    backend = "qasm_simulator"

        [qiskit.aer]
        # Default options for Qiskit Aer

        # set the default backend options for the Qiskit Aer device
        # Note that, in TOML, dictionary key-value pairs are defined
        # using '=' rather than ':'.
        backend_options = {"validation_threshold" = 1e-6}

        [qiskit.ibmq]
        # Default options for IBMQ

        # IBM Quantum Experience authentication token
        ibmqx_token = "XXX"

        # hardware backend device
        backend = "ibmq_rome"

        # pass (optional) provider information
        hub = "MYHUB"
        group = "MYGROUP"
        project = "MYPROJECT"


Standard PennyLane options are provided under the ``[main]`` section. These apply to all loaded devices.
Alternatively, options can be specified on a per-plugin basis, by setting the options under
``[plugin.global]``.

For example, in the above configuration file, the Strawberry Fields
devices will be loaded with a default of ``shots = 100``, rather than ``shots = 1000``. Finally,
you can also specify settings on a device-by-device basis, by placing the options under the
``[plugin.device]`` settings.
