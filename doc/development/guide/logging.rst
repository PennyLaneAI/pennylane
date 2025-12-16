Logging development guidelines
==============================

Enabling logging
----------------

To enable logging support in your PennyLane work-flow simply run the following lines:

.. code:: python

   import pennylane as qml
   qml.logging.enable_logging()


This will ensure all levels of the execution pipeline log function entries and
outputs. These are logged to handlers pointing to the standard output. Note, since Python's logging framework configuration functions overwrite previous logger configurations, it is expected that :func:`~pennylane.logging.enable_logging()` is called only when not using external logging configurations.

If you are defining custom logging configurations, you can extend the logging configuration options as defined in the section :ref:`Customizing the logging configuration <logging-config>`, or avoid calling :func:`~pennylane.logging.enable_logging()` in favour of your custom configuration options.

Adding logging supports during development
------------------------------------------

To add logging support to components of PennyLane, we must define a module logger and ensure all scoped log statements use that logger. A logger can be imported and defined with:

.. code:: python

   import logging
   logger = logging.getLogger(__name__)

which will be used within the given module, and track directories,
filenames and function names, as we have defined the appropriate types
within the formatter configuration (see :class:`pennylane.logging.DefaultFormatter`). With the logger defined, we can selectively add to the logger via two methods: 

#. Using decorators on the required functions and methods in a given module:

   .. code:: python

      # debug_logger can be used to decorate any method or free function
      # debug_logger_init can be used to decorate class __init__ methods.
      from pennylane.logging import debug_logger, debug_logger_init

      @debug_logger
      def my_func(arg1, arg2):
         return arg1 + arg2

#. Explicitly by if-else statements, which compare the given module’s log-level to any log record message it receives. This step is not necessary, as the message will
   only output if the level is enabled, though if an expensive function
   call is required to build the string for the log-message, it can be
   faster to perform this check:

   .. code:: python

      if logger.isEnabledFor(logging.DEBUG):
         logger.debug(
            """Entry with args=(arg_name_1=%s, arg_name_2=%s, ..., arg_name_n=%s)""",
            arg_name_1, arg_name_2, ..., arg_name_n,
         )

Both versions provide similar functionality, though the explicit logger call allows more custom message-formatting, such as expanding functions as string representation, filtering of data, and other useful processing for a valid record.

These logging options allow us a way to track the inputs and calls through the stack in the order they are executed, as is the basis for following a trail of
execution as needed.

All debug log-statements currently added to the PennyLane execution
pipeline output log level messages with the originating function making
that call, as this can allow us to track hard-to-find bug origins in the
execution pipeline.

.. _logging-config:

Customizing the logging configuration
-------------------------------------

As with any package that targets many domains, Python's logging is as
extensible and flexible as it is a challenge to configure for all user needs 
– ideally we define some good defaults that meet our development goals, 
and only deviate from them if required. For further details and customization 
options, consider reading the Python logging documentation
`how-to <https://docs.python.org/3/howto/logging.html#python%20logging>`__
and `“advanced”
tutorial <https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial>`__
level. 

PennyLane logging defaults are contained in a configuration file in the logging module titled 
``log_config.toml``, which is imported when logging is enabled. 
The file path can be accessed with :func:`qml.logging.config_path()`. To change log-levels that are 
reporting on a package or module-wide basis, it is possible to do so by 
modifying the entries in the ``log_config.toml`` file, under the ``[loggers]``
section. In addition, if we want to send the logs elsewhere, we can
adjust the ``[handlers]`` section, which controls what happens to each
message. If we do not like the output format of the messages, we can
adjust these through the ``[formatters]`` section. If we want to filter
messages based on some criteria, we can add these to the respective
handlers. As an example, we can go through the configuration file and
explore the options.

For ease-of-development, the function :func:`pennylane.logging.edit_system_config` opens an editor (if the ``EDITOR`` environment variable is set), or a viewer of the existing file configuration, which can be used to modify the existing options.

Modifying the configuration options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To allow for good expressivity when requiring logging, we must often
adjust several parts of the ecosystem to ensure messages are formatted a
certain way, we control logging internally to PennyLane different to
external packages, messages are sent to somewhere we can make them
actionable, and we can remove messages that are not needed based on some
criteria of interest. Let's break the ``log_config.toml`` file into
sections to discuss how these can be adjusted to suit needs:

.. code:: toml

   ###############################################################################
   # Avoid interfering with existing loggers in dependent libraries
   ###############################################################################

   disable_existing_loggers = false
   version = 1

   ###############################################################################
   # Bind formatters defined locally and those defined in pennylane.logging
   ###############################################################################

   [formatters]
   [formatters.qml_default_formatter]
   "()" = "pennylane.logging.formatters.formatter.DefaultFormatter"

   [formatters.qml_alt_formatter]
   "()" = "pennylane.logging.formatters.formatter.AnotherLogFormatter"

   [formatters.local_detailed]
   format = "\u001B[38;2;45;145;210m[%(asctime)s][%(levelname)s][<PID %(process)d:%(processName)s>] - %(name)s.%(funcName)s()::\"%(message)s\"\u001B[0m"

   [formatters.local_standard]
   format = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"

The first sections of the configuration file tell the logging
infrastructure to avoid modification to existing log settings — this is
set to ``true`` by default for backwards compatibility, though can be
problematic if using external packages. It is recommended to keep this
as ``false`` unless required otherwise.

.. code:: toml


   ###############################################################################
   # Bind LogRecord filters defined in pennylane.logging module
   ###############################################################################

   [filters]
   # Filter to show messages from the same local process as the Python script
   [filters.qml_LocalProcessFilter]
   "()" = "pennylane.logging.filter.LocalProcessFilter"

   # Filter to show debug level messages only
   [filters.qml_DebugOnlyFilter]
   "()" = "pennylane.logging.filter.DebugOnlyFilter"

The above section defines how to filter log messages (known as
``LogRecords``), given some predicate. In this case, we have defined
some classes, ``LocalProcessFilter`` and ``DebugOnlyFilter`` to filter
based on process ID and on the severity of the incoming message. These
can be used in the next section.

.. code:: toml

   ###############################################################################
   # Bind handlers defined in the logging and in pennylane.logging modules
   ###############################################################################

   [handlers]
   [handlers.qml_debug_stream]
   class = "logging.StreamHandler"
   formatter = "qml_default_formatter"
   level = "DEBUG"
   stream = "ext://sys.stdout"

   [handlers.qml_debug_stream_alt]
   class = "logging.StreamHandler"
   formatter = "qml_alt_formatter"
   level = "DEBUG"
   stream = "ext://sys.stdout"

   [handlers.qml_debug_file]
   class = "logging.handlers.RotatingFileHandler"
   formatter = "local_standard"
   level = "DEBUG"
   filename ='qml_debug.log' # use `/tmp/filename.log` on Linux machines to avoid long-term persistence
   maxBytes = 16777216 # 16MB per file before splitting
   backupCount = 10 # Create 'qml_debug.log.1', ... 'qml_debug.log.backupCount' files and rollover when maxBytes is reached

   [handlers.local_filtered_detailed_stdout]
   class = "logging.StreamHandler"
   formatter = "local_standard"
   level = "DEBUG"
   stream = "ext://sys.stdout"
   filters = ["qml_LocalProcessFilter", "qml_DebugOnlyFilter"]

The above defines how ``LogRecord`` messages are handled, and directs
them to the appropriate sink. The logging framework supports many such
directions (see
`here <https://docs.python.org/3/library/logging.handlers.html>`__ for
more info), but for this example we have defined stream handlers
(printing to the screen via the standard output), and a file handler
with a size cap at 16MB. Each handler can be customized by filters and
formatters so that the consumed message fits the needs of the user.

.. code:: toml

   ###############################################################################
   # Define logger controls for internal and external packages
   ###############################################################################

   [loggers]

   # Control JAX logging 
   [loggers.jax]
   handlers = ["qml_debug_stream",]
   level = "WARN"
   propagate = false

   # Control logging across pennylane
   [loggers.pennylane]
   handlers = ["qml_debug_stream",]
   level = "DEBUG" # Set to TRACE for highest verbosity
   propagate = false

   # Control logging specifically in the pennylane.qnode module
   # Note the required quotes to overcome TOML nesting issues
   [loggers."pennylane.qnode"]
   handlers = ["qml_debug_stream_alt",]
   level = "DEBUG" # Set to TRACE for highest verbosity
   propagate = false

   ###############################################################################

Finally, the ``loggers`` section which controls the individual loggers
across the packages we are using. Python’s logging framework follows a
parent-child hierarchy, where a logging configuration set at a parent
level will set all child levels with the same features. In this
instance, we have configured JAX, PennyLane and our script to all log
into the ``qml_debug_stream`` handler we defined earlier, and modified
the child logger ``"pennylane.qnode"`` (quotes needed due to TOML
parsing limitations) to use a different logger, in this case
``qml_debug_stream_alt``. We are free to define the module/package
log-level here (we opt for ``DEBUG`` for all), and to also use multiple
handlers per logger (such as for logging to the standard output and
files through ``qml_debug_stream`` and ``qml_debug_file``
simultaneously). Given the complexity explosion with configuring these
options, the default features in ``log_config.toml`` all use the same
log-level, and handler, which can be adjusted based on developer needs.

Logging example with PennyLane and JAX’s JIT support
----------------------------------------------------

As mentioned above, we have added execution function entry logging
supports, including some supports for each target interface. We can
examine this support for both internal and external packages, where we
enable logs for JAX, which has support for Python-native log messages.
To enable logging specifically for JAX, we can modify the ``level``
parameter for the ``[loggers.jax]`` entry in the ``log_config.toml``
file as:

.. code:: toml

   [loggers.jax]
   handlers = ["qml_debug_stream"]
   level = "DEBUG"
   propagate = false

where ``handlers`` represents some arbitrary custom class we define to
deal with the message, ``level`` the associated level we want that
package to log at, and ``propagate`` tells the logger to keep the
message at the given handler level, or also to throw it up to the parent logger
interface — all of these are adhering to the logging API. We convert the
highest supported log level from warning (less verbose) to debug (more
verbose). We can at the same time change the PennyLane logging level to
warnings and more severe, by making the following change:

.. code:: toml

   [loggers.pennylane]
   handlers = ["qml_debug_stream"]
   level = "WARN"
   propagate = false

Running the following example will produce lots of output about the JIT
process, and surrounding operations:

.. code-block:: python

    import pennylane as qml
    import jax, jax.numpy as jnp
    from jax import jacfwd, jacrev
    import logging

    # Enable logging
    qml.logging.enable_logging()

    # Get logger for use by this script only.
    logger = logging.getLogger(__name__)
    dev_name = "default.qubit"
    num_wires = 2
    num_shots = None

    @jax.jit
    def circuit(key, param):
       logger.info(f"Creating {dev_name} device with {num_wires} wires with {key} PNRG")
       dev = qml.device(dev_name, wires=num_wires, prng_key=key)

       @qml.set_shots(shots=num_shots)
       @qml.qnode(dev, interface="jax", diff_method="backprop")
       def my_circuit():
           qml.RX(param, wires=0)
           qml.CNOT(wires=[0, 1])
           return qml.expval(qml.PauliZ(0))

       logger.info(f"Created QNODE={my_circuit}")
       res =  my_circuit()
       logger.info(f"Created QNODE evaluation={res}")

       return res

    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    logger.info(f"Running circuit with key={key1}")
    circuit(key1, jnp.pi/2)
    logger.info(f"Running circuit with key={key2}")
    circuit(key2, jnp.pi/2)
    logger.info(f"Calculating jacobian circuit with key={key1}")
    logger.info(f"Jacobian={jacfwd(lambda x: circuit(key1, x))(jnp.pi/3)}")

We can examine the output of the log-statements, which shows debug level
messages from JAX, and info-level messages for the given script. To modify the logger defined in the Python script, a new section can be added as:

.. code:: toml

   # Control logging in the executing Python script
   [loggers.__main__]
   handlers = ["qml_debug_stream",]
   level = "INFO"
   propagate = false


To see PennyLane-wide debug messages, we can revert the PennyLane log level to
debug, and rerun the script. There should be more output than previously
observed.

Adding log-statements to the interface execution pipelines
----------------------------------------------------------

Similarly, for autograd (TF and Torch also), we can run examples that
tie-into the execution pipeline for devices without backprop supports:

.. code-block:: python

    import pennylane as qml
    import logging

    qml.logging.enable_logging()

    logger = logging.getLogger(__name__)
    dev_name = "lightning.qubit"
    num_wires = 2
    num_shots = None

    def circuit(param):
       logger.info(f"Creating {dev_name} device with {num_wires} wires")
       dev = qml.device(dev_name, wires=num_wires)

       @qml.set_shots(shots=num_shots)
       @qml.qnode(dev, diff_method="adjoint")
       def my_circuit(param):
           qml.RX(param, wires=0)
           qml.CNOT(wires=[0, 1])
           return qml.expval(qml.PauliZ(0))

       logger.info(f"Created QNODE={my_circuit}")
       res =  my_circuit(param)
       logger.info(f"Created QNODE evaluation={res}")

       return res

    par = qml.numpy.array([0.1,0.2])

    logger.info(f"Running circuit with par={par[0]}")
    circuit(par[0])
    logger.info(f"Running circuit with par={par[1]}")
    circuit(par[1])
    logger.info(f"Calculating jacobian circuit with par={par}")
    logger.info(f"Jacobian={qml.jacobian(circuit)(par[0])}")

By using ``lightning.qubit`` we can now treat the execution environment
as a black-box, and see the log-level messages as they hit the custom
functions as part of the execution pipeline.

The above features have been added for PyTorch, JAX and
autograd, and should produce a sufficient level of detail in the
execution messages.


Log formatting
--------------

The logging-formatter ties-into the ANSI color-code system to improve
visibility of standard output and error logging during execution. The
ANSI codes accept RGB-coded code to change the text and background
colors, allowing messages to be color coded for ease of readability. For
example, to generate all such sequences in steps of 5 across each 8-bit
range per color, we can use the following bash command:

.. code:: bash

   for r in `seq 0 5 255`; do
       for g in `seq 0 5 255`; do
           for b in `seq 0 5 255`; do
               echo -e "\e[38;2;${r};${g};${b}m"'\\e[38;2;'"${r};${g};${b}"m" FOREGROUND\e[0m"
               echo -e "\e[48;2;${r};${g};${b}m"'\\e[48;2;'"${r};${g};${b}"m" BACKGROUND\e[0m"
           done
       done
   done

The strings in the log messages are prepended with the appropriate ANSI
codes to ensure different log-levels are highlighted in different ways
when outputing to the standard output stream (stdout/stderr). These are
defined in the ``pennylane.logging.formatter`` module, and can be
customized to suit any colors, or messaging structure.
