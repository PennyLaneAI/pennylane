# Logging support in PennyLane

PennyLanes's log support ties into the native Python logger, which allows extensive configurability and support for both developer and end-user centric log messages.

The current structure defines some useful utility classes to allow ease-of-control with visibility and reporting of logs across the package.


The package-wide logging controls are specific in the `log_config.toml` file (alternatively in the `log_config.yaml`), and control the logger levels, the log handlers, the formatters used, and even which parts of the package ecosystem should enable logging, including external packages such as `jax`.




The logging-formatter ties-into the ANSI color-code system to improve visibility of standard output and error logging during execution. The ANSI codes accept RGB-coded code to change the text and background colors, allowing messages to be color coded for ease of readability. For example, to generate all such sequences in steps of 5 across each 8-bit range per color, we can use the following bash command:

```bash
for r in `seq 0 5 255`; do
    for g in `seq 0 5 255`; do
        for b in `seq 0 5 255`; do
            echo -e "\e[38;2;${r};${g};${b}m"'\\e[38;2;'"${r};${g};${b}"m" FOREGROUND\e[0m"
            echo -e "\e[48;2;${r};${g};${b}m"'\\e[48;2;'"${r};${g};${b}"m" FOREGROUND\e[0m"
        done
    done
done
```
The strings in the log messages are prepended with the appropriate ANSI codes to ensure different log-levels are highlighted in different ways when outputing to the standard output stream (stdout/stderr).

## Setting up logging supports

To avail of the new functionality, ensure the branch "debugging/logging_support" is installed in editable mode as:

```bash
python -m venv pyenv
source ./pyenv/bin/activate
python -m pip install pytoml
python -m pip install -e . 
```

To get started with adding logging to components of PennyLane, we must first define a logger per module we intend to use with the logging infrastructure, and ensure all scoped log statements are using this logger. The first steps is to import and define a logger as:

```python
import logging
logger = logging.getLogger(__name__)
```
which will be used within the given module, and track directories, filenames and function names, as we have defined the appropriate types within the formatter configuration (see `pennylane.logging.formatters.formatter.py::DefaultFormatter`). With the logger defined, we can selectively add to the logger by if-else statements, which compare the given module's log-level to any log record message it receives. This step is not necessary, as the message will only output if the level is enabled, though if an expensive function call is required to build the string for the log-message, it can be faster to perform this check:

```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(
        """Entry with args=(arg_name_1=%s, arg_name_2=%s, ..., arg_name_n=%s)""",
        arg_name_1, arg_name_2, ..., arg_name_n,
    )
```

The above line can be added to below the function/method entry point, and the provided arguments can be used to populate the log message. This allows us a way to track the inputs and calls through the stack in the order they are executed, as is the basis for following a trail of execution as needed.

All debug log-statements currently added to the PennyLane execution pipeline output log level messages with the originating function making that call, as this can allow us to track hard-to-find bug origins in the execution pipeline.

## Logging example with PennyLane and JAX's JIT support

As mentioned above, we have added execution function entry logging supports, including some supports for each target interface. We can examine this support for both internal and external packages, where we enable logs for JAX, which has support for Python-native log messages. To enable logging specfically for JAX, we can modify the `level` parameter for the `[loggers.jax]` entry in the `log_config.toml` file as:

```toml
[loggers.jax]
handlers = ["console_custom"]
level = "DEBUG"
propagate = false
```
where we convert the highest supported log level from warning (less verbose) to debug (more verbose). We can at the same time change the PennyLane logging level to warnings and more severe, by making the following change:

```toml
[loggers.jax]
handlers = ["console_custom"]
level = "WARN"
propagate = false
```

Running the following example will produce lots of output about the JIT process, and surrounding operations:

```python
import pennylane as qml
import jax, jax.numpy as jnp
from jax import jacfwd, jacrev
import logging

# Enable logging
qml.logging.enable_logging()

# Get logger for use by this script only.
logger = logging.getLogger(__name__)
dev_name = "default.qubit.jax"
num_wires = 2
num_shots = None

# Let's create our circuit with randomness and compile it with jax.jit.
@jax.jit
def circuit(key, param):
    # Notice how the device construction now happens within the jitted method.
    # Also note the added '.jax' to the device path.
    logger.info(f"Creating {dev_name} device with {num_wires} wires and {num_shots} shots with {key} PNRG")
    dev = qml.device(dev_name, wires=num_wires, shots=num_shots, prng_key=key)

    # Now we can create our qnode within the circuit function.
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
```

We can examine the output of the log-statements, which shows debug level messages from JAX, and info-level messages for the given script (controlled by `[loggers.main]` in the config file). To see PennyLane-wide debug messages, we can revert the PennyLane log level to debug, and rerun the script. There should be more output than previously observed.

## Adding log-statements to the autograd execution pipeline

Similarly, for autograd, where we have explicitly added log-statements to the execution pipeline:
```python
import pennylane as qml
import logging

qml.logging.enable_logging()

# Get logger for use herein
logger = logging.getLogger(__name__)
dev_name = "lightning.qubit"
num_wires = 2
num_shots = None

# Let's create our circuit with randomness and compile it with jax.jit.
def circuit(param):
    # Notice how the device construction now happens within the jitted method.
    # Also note the added '.jax' to the device path.
    logger.info(f"Creating {dev_name} device with {num_wires} wires and {num_shots} shots")
    dev = qml.device(dev_name, wires=num_wires, shots=num_shots)

    # Now we can create our qnode within the circuit function.
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
```

By using `lightning.qubit` we can now treat the execution environment as a black-box, and see the log-level messages as they hit the custom VJP functions as part of the autograd execution pipeline.

Note that the above features have been added for Torch, Tensorflow, JAX and autograd.

## Customizing logs

As with any package that targets many domains, Python's loging is as extensible and flexible as it is hard to configure -- ideally we define some good defaults that meet our development goals, and only deviate from them if required. To change log-levels that are reporrting on a package or module-wide basis, it is possible to do so by modidyfing the entries in the `log_config.toml` file, under the `[loggers]` section. In addition, if we want to send the logs eslewhere, we can adjust the `[handlers]` section, which control what happens to each message. If we do not like the output format of the messages, we can adjust these through the `[formatters]` section. If we want to filter messages based on some criteria, we can add these to the respective handlers. Rather than provide extensive docs here, I will suggest reading the Python logging documentation [how-to](https://docs.python.org/3/howto/logging.html#python logging) and ["advanced" tutorial](https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial) level.
