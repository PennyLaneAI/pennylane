Logging
=======

PennyLane has support for Python's logging framework. PennyLane defines all logging configuration in a static TOML file, ``log_config.toml``, which can be customized to support rules for a given user or system environment.

To see the default logging options you can explore the contexts of ``log_config.toml`` at the path given from :func:`pennylane.logging.config_path`.


Enabling logging
----------------

To enable logging support with the default options defined in ``log_config.toml`` simply call :func:`pennylane.logging.enable_logging` after importing PennyLane:

.. code:: python

   import pennylane as qp
   qml.logging.enable_logging()
   ...


This will ensure all levels of the execution pipeline logs function entries, and
outputs to the default configured handler, which is directed to the standard output stream. To also direct logging output to a file named ``qml_debug.log`` in the directory of execution, the ``qml_debug_file`` handler can be added to the defined loggers in ``log_config.toml`` as follows:

.. code:: toml

   # Control logging across pennylane
   [loggers.pennylane]
   handlers = ["qml_debug_stream", "qml_debug_file"]
   level = "DEBUG" # Set to TRACE for highest verbosity
   propagate = false


Viewing the existing logging configuration file is possible by calling the :func:`pennylane.logging.edit_system_config` function which will open the file in an existing browser or editor window.

For more info on the customization of the logging options, please see the logging development guide at :doc:`/development/guide/logging`, and the `Python logging documentation <https://docs.python.org/3/library/logging.html>`_.
