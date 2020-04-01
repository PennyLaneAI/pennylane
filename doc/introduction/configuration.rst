.. role:: html(raw)
   :format: html

.. _intro_ref_config:

Configuration
=============

Some important default settings for a device, such as your user credentials for quantum hardware
access, the number of shots, or the cutoff dimension for continuous-variable simulators, are
defined in a configuration file called `config.toml`.

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

The user can access the initialized configuration via `pennylane.config`, view the
loaded configuration filepath, print the configurations options, access and modify
them via keys (i.e., ``pennylane.config['main.shots']``), and save/load new configuration files.

Format
------

The configuration file `config.toml` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [main]
    # Global PennyLane options.
    # Affects every loaded plugin if applicable.
    shots = 1000

    [strawberryfields.global]
    # Options for the Strawberry Fields plugin
    hbar = 1
    shots = 100

      [strawberryfields.fock]
      # Options for the Strawberry Fields Fock plugin
      cutoff_dim = 10
      hbar = 0.5

      [strawberryfields.gaussian]
      # Indentation doesn't matter in TOML files,
      # but helps provide clarity.

    [projectq.global]
    # Options for the Project Q plugin

      [projectq.simulator]
      gate_fusion = true

      [projectq.ibm]
      user = "johnsmith"
      password = "secret123"
      use_hardware = true
      device = "ibmqx4"
      num_runs = 1024

Standard PennyLane options are provided under the ``[main]`` section. These apply to all loaded devices.
Alternatively, options can be specified on a per-plugin basis, by setting the options under
``[plugin.global]``.

For example, in the above configuration file, the Strawberry Fields
devices will be loaded with a default of ``shots = 100``, rather than ``shots = 1000``. Finally,
you can also specify settings on a device-by-device basis, by placing the options under the
``[plugin.device]`` settings.
