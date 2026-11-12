.. _plugin_overview:

Building a device
=================

.. note::

    Devices written with PennyLane Classic (``pennylane<=0.45.1``) are not
    compatible with pennylane. For information on building a device with
    PennyLane Classic, please consult the
    `PennyLane Classic documentation <https://docs.pennylane.ai/en/v0.45.1/development/plugins.html>`__.

PennyLane2 workflows are compiled :doc:`Catalyst <catalyst:index>` via
:func:`~pennylane.qjit`. Compiled programs are then executed on devices defined
in the :doc:`Catalyst <catalyst:index>` runtime, which uses a C++ interface
called ``QuantumDevice``. To make your own simulator or hardware backend
available in PennyLane2, you need three things:

#. A C++ implementation of the device that uses the ``QuantumDevice`` interface,
   compiled into a shared library.

#. A TOML configuration file that declares which instructions (e.g., gates,
   observables and measurements) the device supports.

#. A Python class that tells PennyLane2 where to find the shared library and the
   TOML configuration file, so that the device can be loaded with
   :func:`qp.device() <pennylane.device>`.

This page walks through each of these necessities. For examples to reference,
see the `Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`__ simulators,
such as ``lightning.qubit``, and Catalyst's
`null.qubit runtime device <https://github.com/PennyLaneAI/catalyst/tree/main/runtime/lib/backend/null_qubit>`__.

.. warning::

    The ``QuantumDevice`` interface might change quickly in the near future.
    Breaking changes are announced in the release changelogs, so please check
    back regularly to make sure your device is compatible with the version of
    PennyLane2 you're using.

1. Implementing the device in C++
---------------------------------

A custom device is a C++ class that inherits from ``Catalyst::Runtime::QuantumDevice``
(defined in the header ``QuantumDevice.hpp``). Below is a very basic device in
which the quantum operations have been stubbed out. It implements only the
methods that every device must provide (optional functions are listed
in the ``QuantumDevice``
`API documentation <https://docs.pennylane.ai/projects/catalyst/en/stable/api/structCatalyst_1_1Runtime_1_1QuantumDevice.html>`):

.. code-block:: c++

    #include <cstdlib>
    #include <numeric>
    #include <optional>
    #include <string>
    #include <vector>

    #include <QuantumDevice.hpp>

    struct CustomDevice final : public Catalyst::Runtime::QuantumDevice {
        // Static device parameters can be passed from the Python device to the constructor.
        CustomDevice(const std::string &kwargs = "{}") {}
        ~CustomDevice() = default;

        auto GetNumQubits() const -> size_t override { return device_nqubits; }
        void SetDeviceShots(size_t shots) override { device_shots = shots; }
        auto GetDeviceShots() const -> size_t override { return device_shots; }

        // Allocations produce numeric qubit identifiers that are passed to other operations.
        auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override {
            std::vector<QubitIdType> ids(num_qubits);
            std::iota(ids.begin(), ids.end(), 0);
            device_nqubits = num_qubits;
            return ids;
        }
        void ReleaseQubits(const std::vector<QubitIdType> &qubits) override {}

        // Here we don't do anything yet.
        void NamedOperation(const std::string &name, const std::vector<double> &params,
                            const std::vector<QubitIdType> &wires, bool inverse,
                            const std::vector<QubitIdType> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::string> &optional_params) override {}
        auto Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result override {
            return static_cast<Result>(malloc(sizeof(bool)));
        }

      private:
        size_t device_nqubits = 0;
        size_t device_shots = 0;
    };

The other methods of ``QuantumDevice`` are optional. By default they raise an
error at runtime, so implement the ones that match what your device declares in
its :ref:`configuration file <device_capabilities>`. For example:

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - To support...
      - Implement...
    * - Gates applied by name, such as :class:`~pennylane.RX`
      - ``NamedOperation``
    * - :class:`~pennylane.QubitUnitary`
      - ``MatrixOperation``
    * - Observables
      - ``Observable``, ``TensorObservable`` and ``HamiltonianObservable``
    * - :func:`~pennylane.expval` and :func:`~pennylane.var`
      - ``Expval`` and ``Var``
    * - :func:`~pennylane.probs`
      - ``Probs`` and ``PartialProbs``
    * - :func:`~pennylane.sample` and :func:`~pennylane.counts`
      - ``Sample``, ``PartialSample``, ``Counts`` and ``PartialCounts``
    * - :func:`~pennylane.state`
      - ``State``
    * - Mid-circuit measurements (:func:`~pennylane.measure`)
      - ``Measure``
    * - The adjoint differentiation method
      - ``Gradient``, ``StartTapeRecording`` and ``StopTapeRecording``

For the full interface, see the ``QuantumDevice``
`API documentation <https://docs.pennylane.ai/projects/catalyst/en/stable/api/structCatalyst_1_1Runtime_1_1QuantumDevice.html>`__.

.. _device_factory:

The device factory
^^^^^^^^^^^^^^^^^^

Your library must also provide an entry point for the device library: a factory
function named ``<DeviceIdentifier>Factory``, where ``<DeviceIdentifier>``
uniquely identifies the device. Using the identifier ``CustomDevice``:

.. code-block:: c++

    extern "C" Catalyst::Runtime::QuantumDevice *CustomDeviceFactory(const char *kwargs) {
        return new CustomDevice(std::string(kwargs));
    }

For simplicity, you can use the ``GENERATE_DEVICE_FACTORY(IDENTIFIER, CONSTRUCTOR)``
macro from ``QuantumDevice.hpp`` to define this function, where ``IDENTIFIER``
is the device identifier and ``CONSTRUCTOR`` is the C++ device constructor,
including its namespace. Here, both are the same:

.. code-block:: c++

    GENERATE_DEVICE_FACTORY(CustomDevice, CustomDevice);

A library can provide several factory functions if it packages several devices.
Choose identifiers that are as unique as possible to avoid clashes with other
devices. The identifier must match exactly the name returned by
``get_c_interface`` in the :ref:`Python device class <python_device_class>`.

The runtime deletes the device when it's no longer needed, which calls the
destructor of ``QuantumDevice``.

Device parameters
^^^^^^^^^^^^^^^^^

The constructor receives ``kwargs``, a string of device options in Python
dictionary format (e.g., a noise parameter: ``"{'noise': 0.5}"``). These
options are set in Python through the ``device_kwargs`` attribute of the
corresponding :ref:`Python device class <python_device_class>`, and are passed
to the ``CustomDevice`` constructor when the compiled program starts the device.
The ``CustomDevice`` constructor is responsible for parsing the string.

.. _compile_to_shared_lib:

Compiling the device into a shared library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compile your device into a shared library (``.so`` on Linux, ``.dylib`` on
macOS) following Catalyst's
`third-party device example <https://github.com/PennyLaneAI/catalyst/blob/26b412b298f22565fea529d2019554e7ad9b9624/runtime/tests/third_party/CMakeLists.txt>`__:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.20)

    project(third_party_device)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    add_library(custom_device SHARED custom_device.cpp)
    target_include_directories(custom_device PUBLIC ${runtime_includes})
    set_property(TARGET custom_device PROPERTY POSITION_INDEPENDENT_CODE ON)

.. _device_capabilities:

2. Configuration file for declaring supported instructions
----------------------------------------------------------

The configuration file is a `TOML file <https://toml.io/en/>`__ that tells the
compiler what a ``QuantumDevice`` supports. When compiling a program,
PennyLane2 decomposes operations and transforms instructions into ones the
device declares in the TOML file, and raises an error for anything that remains
unsupported.

Below is an example configuration file, with inline descriptions of each field.
All headers and fields are required unless stated otherwise.

.. code-block:: toml

    schema = 3

    # The gates supported by the device. The gate definitions should have the
    # following format:
    #
    #   GATE = { properties = [ PROPS ], conditions = [ CONDS ] }
    #
    # where PROPS and CONDS are zero or more comma-separated quoted strings.
    #
    # PROPS: additional support provided for each gate.
    #        - "controllable": if a controlled version of this gate is supported.
    #        - "invertible": if the adjoint of this operation is supported.
    #        - "differentiable": if device gradient is supported for this gate.
    # CONDS: constraints on the support for each gate.
    #        - "analytic" or "finiteshots": if this operation is only supported in
    #          either analytic execution or with shots, respectively.
    #
    [operators.gates]

    PauliX = { properties = ["controllable", "invertible"] }
    PauliY = { properties = ["controllable", "invertible"] }
    PauliZ = { properties = ["controllable", "invertible"] }
    RY = { properties = ["controllable", "invertible", "differentiable"] }
    RZ = { properties = ["controllable", "invertible", "differentiable"] }
    CRY = { properties = ["invertible", "differentiable"] }
    CRZ = { properties = ["invertible", "differentiable"] }
    CNOT = { properties = ["invertible"] }

    # Observables supported by the device for measurements. The observables defined
    # in this section should have the following format:
    #
    #   OBSERVABLE = { conditions = [ CONDS ] }
    #
    # where CONDS is zero or more comma-separated quoted strings, same as above.
    #
    # CONDS: constraints on the support for each observable.
    #        - "analytic" or "finiteshots": if this observable is only supported in
    #          either analytic execution or with shots, respectively.
    #        - "terms-commute": if a composite operator is only supported under the
    #          condition that its terms commute.
    #
    [operators.observables]

    PauliX = { }
    PauliY = { }
    PauliZ = { }
    Sum = { conditions = [ "terms-commute" ] }
    SProd = { }
    Prod = { }

    # Types of measurement processes supported on the device. The measurements in
    # this section should have the following format:
    #
    #   MEASUREMENT_PROCESS = { conditions = [ CONDS ] }
    #
    # where CONDS is zero or more comma-separated quoted strings, same as above.
    #
    # CONDS: constraints on the support for each measurement process.
    #        - "analytic" or "finiteshots": if this measurement is only supported
    #          in either analytic execution or with shots, respectively.
    #
    [measurement_processes]

    ExpectationMP = { }
    SampleMP = { }
    CountsMP = { conditions = ["finiteshots"] }
    StateMP = { conditions = ["analytic"] }

    # Additional support that the device may provide that informs the compilation
    # process. All accepted fields and their default values are listed below.
    [compilation]

    # Whether the device is compatible with qjit. Set this to true.
    qjit_compatible = true

    # Whether the device requires run time generation of the quantum circuit.
    runtime_code_generation = false

    # Whether the device supports allocating and releasing qubits during execution.
    dynamic_qubit_management = false

    # Whether simultaneous measurements on overlapping wires is supported.
    overlapping_observables = true

    # Whether simultaneous measurements of non-commuting observables is supported.
    # If false, a circuit with multiple non-commuting measurements will have to be
    # split into multiple executions for each subset of commuting measurements.
    non_commuting_observables = false

    # Whether the device supports initial state preparation.
    initial_state_prep = false

    # The methods of handling mid-circuit measurements that the device supports,
    # e.g., "one-shot", "tree-traversal", "device", etc. An empty list indicates
    # that the device does not support mid-circuit measurements.
    supported_mcm_methods = [ ]

    # Observables supported by the QuantumDevice.
    [qjit.operators.observables]

    Sum = { }




The configuration file is also loaded into the
:attr:`~pennylane.devices.Device.capabilities` attribute of your
:ref:`Python device class <python_device_class>`.

.. _python_device_class:

3. The Python device class
--------------------------

To use a custom ``QuantumDevice`` with PennyLane2 in Python, a Python class
that inherits from :class:`~pennylane.devices.Device` and points PennyLane2 to
your shared library and configuration file is required. It needs:

* A ``config_filepath`` class variable with the path to your configuration
  file.

* A static ``get_c_interface`` method that takes no arguments and returns a
  tuple of the device identifier (``<DeviceIdentifier>`` from
  :ref:`the device factory <device_factory>`) and the absolute path to the
  :ref:`shared library <compile_to_shared_lib>`.

* A ``device_kwargs`` dictionary of options to pass to the ``QuantumDevice``
  constructor, if applicable.

* Optionally, override :attr:`~pennylane.devices.Device.name` with
  the name that the device is registered under (see :ref:`registering_device`).

* An :meth:`~pennylane.devices.Device.execute` method. It's required by the
  :class:`~pennylane.devices.Device` class, but compiled programs run on your
  ``QuantumDevice`` and never call :meth:`~pennylane.devices.Device.execute`, so
  it can simply raise an error as shown in the code example below.

.. code-block:: python

    import pathlib
    import sys

    import pennylane as qp

    PACKAGE_DIR = pathlib.Path(__file__).parent


    class CustomDevice(qp.devices.Device):
        """A custom device, executed by the Catalyst runtime."""

        config_filepath = PACKAGE_DIR / "custom_device.toml"

        def __init__(self, wires=None, noise=0.0):
            super().__init__(wires=wires)
            # Passed to the QuantumDevice constructor as "{'noise': <value>}".
            self.device_kwargs = {"noise": noise}

        @property
        def name(self):
            return "custom.device"

        @staticmethod
        def get_c_interface():
            """Returns the device identifier and the path to the shared library."""
            ext = ".dylib" if sys.platform == "darwin" else ".so"
            return "CustomDevice", str(PACKAGE_DIR / f"libcustom_device{ext}")

        def execute(self, circuits, execution_config=None):
            raise NotImplementedError("CustomDevice can only run programs compiled with qp.qjit.")

You can then use the device in PennyLane2 as follows:

.. code-block:: python

    dev = CustomDevice(wires=2, noise=0.1)

    @qp.qjit
    @qp.qnode(dev)
    def circuit(x):
        qp.RY(x, wires=0)
        qp.CNOT(wires=[0, 1])
        return qp.expval(qp.Z(1))

Taking the TOML configuration from the
:ref:`"Configuration file for declaring supported instructions" <device_capabilities>`
section, it declares ``ExpectationMP``. Therefore, running ``circuit`` requires
the implementation of ``Expval`` and ``Observable`` in the ``CustomDevice``.

When initializing a custom device in PennyLane2, the device's ``wires`` argument
must be an integer or consecutive integers starting at
``0``, such as ``wires=2`` or ``wires=[0, 1]``. Shots are set with
:func:`~pennylane.set_shots` and reach the device through ``SetDeviceShots``.

.. _registering_device:

Registering and packaging the device
------------------------------------

To load your device by name with :func:`qp.device() <pennylane.device>`,
register the Python class as an
`entry point <https://packaging.python.org/specifications/entry-points/>`__ in
the ``pennylane.plugins`` group. For example, in your package's
``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."pennylane.plugins"]
    "custom.device" = "my_package.device:CustomDevice"

The name should have the form ``pluginname.devicename``, using periods for
delimitation. Once your package is installed, for example in developer mode
with ``pip install -e .``, the device can be loaded with:

.. code-block:: python

    dev = qp.device("custom.device", wires=2, noise=0.1)

Make sure that both the configuration file and the shared library are installed
with your package (e.g., as
`package data <https://setuptools.pypa.io/en/stable/userguide/datafiles.html>`__),
so that ``config_filepath`` and ``get_c_interface`` point to files that exist.

Further reading
---------------

* :doc:`Custom devices <catalyst:dev/custom_devices>` and
  :doc:`supported devices <catalyst:dev/devices>` in the Catalyst documentation.
* The `QuantumDevice API documentation <https://docs.pennylane.ai/projects/catalyst/en/stable/api/structCatalyst_1_1Runtime_1_1QuantumDevice.html>`__.
