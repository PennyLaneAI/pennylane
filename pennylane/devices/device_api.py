# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the Abstract Base Class for the next generation of devices.
"""

import abc
import warnings
from collections.abc import Iterable
from dataclasses import replace
from numbers import Number
from typing import overload

import pennylane as qml
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.measurements import Shots
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.tape.qscript import QuantumScriptBatch
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch, TensorLike
from pennylane.wires import Wires

from .capabilities import (
    DeviceCapabilities,
    observable_stopping_condition_factory,
    validate_mcm_method,
)
from .execution_config import ExecutionConfig
from .preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from .tracker import Tracker


# pylint: disable=unused-argument, no-self-use
class Device(abc.ABC):
    """A device driver that can control one or more backends. A backend can be either a physical
    Quantum Processing Unit or a virtual one such as a simulator.

    Only the ``execute`` method must be defined to construct a device driver.

    .. details::
        :title: Design Motivation

        **Streamlined interface:** Only methods that are required to interact with the rest of PennyLane will be placed in the
        interface. Developers will be able to clearly see what they can change while still having a fully functional device.

        **Reduction of duplicate methods:** Methods that solve similar problems are combined. Only one place will have
        to solve each individual problem.

        **Support for dynamic execution configurations:** Properties such as shots belong to specific executions.

        **Greater coverage for differentiation methods:** Devices can define any order of derivative, the vector jacobian product,
        or the jacobian vector product.  Calculation of derivatives can be done at the same time as execution to allow reuse of intermediate
        results.

    .. details::
        :title: Porting from the old interface

        :meth:`pennylane.devices.LegacyDevice.batch_execute` and :meth:`~pennylane.devices.LegacyDevice.execute` are now a single method, :meth:`~.Device.execute`

        :meth:`~.Device.batch_transform` and :meth:`~.Device.expand_fn` are now a single method, :meth:`~.Device.preprocess`

        Shot information is no longer stored on the device, but instead specified on individual input :class:`~.QuantumTape`.

        The old devices defined a :meth:`~.Device.capabilities` dictionary that defined characteristics of the devices and controlled various
        preprocessing and validation steps, such as ``"supports_broadcasting"``.  These capabilities should now be handled by the
        :meth:`~.Device.preprocess` method. For example, if a device does not support broadcasting, ``preprocess`` should
        split a quantum script with broadcasted parameters into a batch of quantum scripts. If the device does not support mid circuit
        measurements, then ``preprocess`` should apply :func:`~.defer_measurements`.  A set of default preprocessing steps will be available
        to make a seamless transition to the new interface.

        A class will be provided to easily construct default preprocessing steps from supported operations, supported observables,
        supported measurement processes, and various capabilities.

        Utility functions will be added to the ``devices`` module to query whether or not the device driver can do certain things, such
        as ``devices.supports_operator(op, dev, native=True)``. These functions will work by checking the behaviour of :meth:`~.Device.preprocess`
        to certain inputs.

        Versioning should be specified by the package containing the device. If an external package includes a PennyLane device,
        then the package requirements should specify the minimum PennyLane version required to work with the device.

    .. details::
        :title: The relationship between preprocessing and execution

        The :meth:`~.preprocess` method is assumed to be run before any :meth:`~.execute` or differentiation method.
        If an arbitrary, non-preprocessed circuit is provided, :meth:`~.execute` has no responsibility to perform any
        validation or provide clearer error messages.

        >>> op = qml.Permute(["c", 3,"a",2,0], wires=[3,2,"a",0,"c"])
        >>> circuit = qml.tape.QuantumScript([op], [qml.state()])
        >>> dev = DefaultQubit()
        >>> dev.execute(circuit)
        MatrixUndefinedError
        >>> circuit = qml.tape.QuantumScript([qml.Rot(1.2, 2.3, 3.4, 0)], [qml.expval(qml.Z(0))])
        >>> config = ExecutionConfig(gradient_method="adjoint")
        >>> dev.compute_derivatives(circuit, config)
        ValueError: Operation Rot is not written in terms of a single parameter
        >>> new_circuit, postprocessing, new_config = dev.preprocess(circuit, config)
        >>> dev.compute_derivatives(new_circuit, new_config)
        ((array(0.), array(-0.74570521), array(0.)),)

        Any validation checks or error messages should occur in :meth:`~.preprocess` to avoid failures after expending
        computation resources.

    .. details::
        :title: Execution Configuration

        Execution config properties related to configuring a device include:

        * ``device_options``: A dictionary of device specific options. For example, the python device may have ``multiprocessing_mode``
          as a key. These should be documented in the class docstring.

        * ``gradient_method``: A device can choose to have native support for any type of gradient method. If the method
          :meth:`~.supports_derivatives` returns ``True`` for a particular gradient method, it will be treated as a device
          derivative and not handled by pennylane core code.

        * ``gradient_keyword_arguments``: Options for the gradient method.

        * ``derivative_order``: Relevant for requested device derivatives.

        * ``mcm_config``: Options for methods of handling mid-circuit-measures.

    """

    config_filepath: str | None = None
    """A device can use a `toml` file to specify the capabilities of the backend device. If this
    is provided, the file will be loaded into a :class:`~.DeviceCapabilities` object assigned to
    the :attr:`capabilities` attribute."""

    capabilities: DeviceCapabilities | None = None
    """A :class:`~.DeviceCapabilities` object describing the capabilities of the backend device."""

    def __init_subclass__(cls, **kwargs):
        if cls.config_filepath is not None:
            cls.capabilities = DeviceCapabilities.from_toml_file(cls.config_filepath)
        if cls.preprocess is not Device.preprocess and (
            cls.setup_execution_config is not Device.setup_execution_config
            or cls.preprocess_transforms is not Device.preprocess_transforms
        ):
            raise ValueError(
                "A device should implement either `preprocess` or `setup_execution_config` "
                "and `preprocess_transforms`, but not both."
            )
        super().__init_subclass__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the device or set of devices.

        This property can either be the name of the class, or an alias to be used in the :func:`~.device` constructor,
        such as ``"default.qubit"`` or ``"lightning.qubit"``.

        """
        return type(self).__name__

    tracker: Tracker = Tracker()
    """A :class:`~pennylane.Tracker` that can store information about device executions, shots, batches,
    intermediate results, or any additional device dependent information.

    A plugin developer can store information in the tracker by:

    .. code-block:: python

        # querying if the tracker is active
        if self.tracker.active:

            # store any keyword: value pairs of information
            self.tracker.update(executions=1, shots=self._shots, results=results)

            # Calling a user-provided callback function
            self.tracker.record()
    """

    def __init__(self, wires=None, shots=None) -> None:
        # each instance should have its own Tracker.
        self.tracker = Tracker()
        if shots is not None and shots != Shots():
            warnings.warn(
                "Setting shots on device is deprecated. Please use the `set_shots` transform on the respective QNode instead.",
                PennyLaneDeprecationWarning,
            )
        self._shots = Shots(shots)

        if wires is not None:
            if not isinstance(wires, Iterable):
                # interpret wires as the number of consecutive wires
                wires = range(wires)
            wires = Wires(wires)

        self._wires = wires

    def __repr__(self):
        """String representation."""
        details = []
        if self.wires:
            details.append(f"wires={len(self.wires)}")
        if self.shots:
            details.append(f"shots={self.shots.total_shots}")
        details = f"({', '.join(details)}) " if details else ""
        return f"<{self.name} device {details}at {hex(id(self))}>"

    def __getattr__(self, key):
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{key}'."
            " You may be looking for a property or method present in the legacy device interface."
            f" Please consult the {type(self).__name__} documentation for an updated list of public"
            " properties and methods."
        )

    @property
    def shots(self) -> Shots:
        """Default shots for execution workflows containing this device.

        Note that the device itself should **always** pull shots from the provided :class:`~.QuantumTape` and its
        :attr:`~.QuantumTape.shots`, not from this property. This property is used to provide a default at the start of a workflow.

        """
        return self._shots

    @shots.setter
    def shots(self, _):
        raise AttributeError(
            "Shots can no longer be set on a device instance. "
            "You can set shots on a call to a QNode, on individual tapes, or "
            "create a new device instance instead."
        )

    @property
    def wires(self) -> Wires:
        """The device wires.

        Note that wires are optional, and the default value of None means any wires can be used.
        If a device has wires defined, they will only be used for certain features. This includes:

        * Validation of tapes being executed on the device
        * Defining the wires used when evaluating a :func:`~pennylane.state` measurement

        """
        return self._wires

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        """Device preprocessing function.

        .. warning::

            This function is tracked by machine learning interfaces and should be fully differentiable.
            The ``pennylane.math`` module can be used to construct fully differentiable transformations.

            Additional preprocessing independent of machine learning interfaces can be done inside of
            the :meth:`~.execute` method.

        Args:
            execution_config (ExecutionConfig): A datastructure describing the parameters needed to fully describe
                the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that is called before execution, and a configuration
                with unset specifications filled in.

        Raises:
            Exception: An exception can be raised if the input cannot be converted into a form supported by the device.

        Preprocessing program may include:

        * expansion to :class:`~.Operator`'s and :class:`~.MeasurementProcess` objects supported by the device.
        * splitting a circuit with the measurement of non-commuting observables or Hamiltonians into multiple executions
        * splitting circuits with batched parameters into multiple executions
        * gradient specific preprocessing, such as making sure trainable operators have generators
        * validation of configuration parameters
        * choosing a best gradient method and ``grad_on_execution`` value.

        **Example**

        All the transforms that are part of the preprocessing need to respect the transform contract defined in
        :func:`pennylane.transform`.

        .. code-block:: python

                from pennylane.tape import TapeBatch
                from pennylane.typing import PostprocessingFn

                @transform
                def my_preprocessing_transform(tape: qml.tape.QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
                    # e.g. valid the measurements, expand the tape for the hardware execution, ...

                    def blank_processing_fn(results):
                        return results[0]

                    return [tape], processing_fn

        Then we can define the preprocess method on the custom device. The program can accept an arbitrary number of
        transforms.

        .. code-block:: python

                def preprocess(config):
                    program = TransformProgram()
                    program.add_transform(my_preprocessing_transform)
                    return program, config

        .. seealso:: :func:`~.pennylane.transform.core.transform` and :class:`~.pennylane.transform.core.TransformProgram`

        .. details::
            :title: Post processing function and derivatives

            Derivatives and jacobian products will be bound to the machine learning library before the postprocessing
            function is called on results. Therefore the machine learning library will be responsible for combining the
            device provided derivatives and post processing derivatives.

            .. code-block:: python

                from pennylane.interfaces.jax import execute as jax_boundary

                def f(x):
                    circuit = qml.tape.QuantumScript([qml.Rot(*x, wires=0)], [qml.expval(qml.Z(0))])
                    config = ExecutionConfig(gradient_method="adjoint")
                    program, config = dev.preprocess(config)
                    circuit_batch, postprocessing = program((circuit, ))

                    def execute_fn(tapes):
                        return dev.execute_and_compute_derivatives(tapes, config)

                    results = jax_boundary(circuit_batch, dev, execute_fn, None, {})
                    return postprocessing(results)

                x = jax.numpy.array([1.0, 2.0, 3.0])
                jax.grad(f)(x)


            In the above code, the quantum derivatives are registered with jax in the ``jax_boundary`` function.
            Only then is the classical postprocessing called on the result object.

        """
        if execution_config is None:
            execution_config = ExecutionConfig()
        execution_config = self.setup_execution_config(execution_config)
        transform_program = self.preprocess_transforms(execution_config)
        return transform_program, execution_config

    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
    ) -> ExecutionConfig:
        """Sets up an ``ExecutionConfig`` that configures the execution behaviour.

        The execution config stores information on how the device should perform the execution,
        as well as how PennyLane should interact with the device. See :class:`ExecutionConfig`
        for all available options and what they mean.

        An ``ExecutionConfig`` is constructed from arguments passed to the ``QNode``, and this
        method allows the device to update the config object based on device-specific requirements
        or preferences. See :ref:`execution_config` for more details.

        Args:
            config (ExecutionConfig): The initial ExecutionConfig object that describes the
                parameters needed to configure the execution behaviour.
            circuit (QuantumScript): The quantum circuit to customize the execution config for.

        Returns:
            ExecutionConfig: The updated ExecutionConfig object

        """

        if self.__class__.preprocess is not Device.preprocess:
            if config:
                return self.preprocess(config)[1]
            return self.preprocess()[1]

        if config is None:
            config = ExecutionConfig()

        if self.supports_derivatives(config) and config.gradient_method in ("best", None):
            return replace(config, gradient_method="device")

        shots_present = circuit and bool(circuit.shots)
        validate_mcm_method(self.capabilities, config.mcm_config.mcm_method, shots_present)
        if config.mcm_config.mcm_method is None and self.capabilities is not None:
            # This is a sensible default strategy for resolving the MCM method based on declared
            # capabilities of a device, but if a device wishes to do this differently, it should
            # override the ``setup_execution_config`` method itself.
            default_mcm_method = _default_mcm_method(self.capabilities, shots_present)
            new_mcm_config = replace(config.mcm_config, mcm_method=default_mcm_method)
            config = replace(config, mcm_config=new_mcm_config)

        return config

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> TransformProgram:
        """Returns the transform program to preprocess a circuit for execution.

        Args:
            execution_config (ExecutionConfig): The execution configuration object

        Returns:
            TransformProgram: A transform program that is called before execution

        The transform program is composed of a list of individual transforms, which may include:

        * Decomposition of operations and measurements to what is supported by the device.
        * Splitting a circuit with measurements of non-commuting observables or Hamiltonians into multiple executions.
        * Splitting a circuit with batched parameters into multiple executions.
        * Validation of wires, measurements, and observables.
        * Gradient specific preprocessing, such as making sure trainable operators have generators.

        **Example**

        All transforms that are part of the preprocessing transform program need to respect the
        transform contract defined in :func:`pennylane.transform`.

        .. code-block:: python

            from pennylane.tape import QuantumScriptBatch
            from pennylane.typing import PostprocessingFn

            @qml.transform
            def my_preprocessing_transform(tape: qml.tape.QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
                # e.g. valid the measurements, expand the tape for the hardware execution, ...

                def blank_processing_fn(results):
                    return results[0]

                return [tape], processing_fn

        A transform program can hold an arbitrary number of individual transforms:

        .. code-block:: python

            def preprocess(self, config):
                program = TransformProgram()
                program.add_transform(my_preprocessing_transform)
                return program

        .. seealso:: :func:`~.pennylane.transform.core.transform` and :class:`~.pennylane.transform.core.TransformProgram`

        .. details::
            :title: Post processing function and derivatives

            Derivatives and Jacobian products will be bound to the machine learning library before
            the postprocessing function is called on the results. Therefore, the machine learning
            library will be responsible for combining and post-processing derivatives returned from
            the device.

            .. code-block:: python

                from pennylane.interfaces.jax import execute as jax_boundary

                def f(x):
                    circuit = qml.tape.QuantumScript([qml.Rot(*x, wires=0)], [qml.expval(qml.Z(0))])
                    config = ExecutionConfig(gradient_method="adjoint")
                    config = dev.setup_execution_config(config)
                    program = dev.preprocess_transforms(config)
                    circuit_batch, postprocessing = program((circuit, ))

                    def execute_fn(tapes):
                        return dev.execute_and_compute_derivatives(tapes, config)

                    results = jax_boundary(circuit_batch, dev, execute_fn, None, {})
                    return postprocessing(results)

                x = jax.numpy.array([1.0, 2.0, 3.0])
                jax.grad(f)(x)

            In the above code, the quantum derivatives are registered with jax in the ``jax_boundary``
            function. Only then is the classical postprocessing called on the result object.

        """

        # TODO: this is obviously not pretty but it's a temporary solution to ensure backwards
        #       compatibility. Basically there are three scenarios:
        #       1. The device does not override anything, then this method returns the default
        #          transform program, and preprocess calls this method, all good.
        #       2. The device overrides preprocess, but not this method, then this method will
        #          return what is returned from the overridden preprocess method.
        #       3. The device overrides this method and not preprocess (recommended and what we
        #          are ultimately aiming for), then preprocess calls the overridden method.
        if self.__class__.preprocess is not Device.preprocess:
            if execution_config:
                return self.preprocess(execution_config)[0]
            return self.preprocess()[0]

        if not self.capabilities:
            # The capabilities are required to construct a default transform program.
            return TransformProgram()

        if not execution_config:
            execution_config = ExecutionConfig()

        program = TransformProgram()

        # First handle mid-circuit measurements because it may add wires. At this point we
        # should assume that the mcm method is already validated and resolved.
        if execution_config.mcm_config.mcm_method == "deferred":
            program.add_transform(
                qml.transforms.defer_measurements,
                allow_postselect=self.capabilities.supports_operation("Projector"),
            )
        elif execution_config.mcm_config.mcm_method == "one-shot":
            program.add_transform(
                qml.transforms.dynamic_one_shot,
                postselect_mode=execution_config.mcm_config.postselect_mode,
            )

        capabilities_analytic = self.capabilities.filter(finite_shots=False)
        capabilities_shots = self.capabilities.filter(finite_shots=True)

        needs_diagonalization = False
        base_obs = {"PauliZ": qml.Z, "PauliX": qml.X, "PauliY": qml.Y, "Hadamard": qml.H}
        if (
            not all(obs in self.capabilities.observables for obs in base_obs)
            # This check is to confirm that `split_non_commuting` has been applied, since
            # `diagonalize_measurements` does not work with non-commuting measurements. If
            # a device is flexible enough to support non-commuting observables but for some
            # reason does not support all of `PauliZ`, `PauliX`, `PauliY`, and `Hadamard`,
            # we consider it enough of an edge case that the device should just implement
            # its own preprocessing transform.
            and not self.capabilities.non_commuting_observables
        ):
            needs_diagonalization = True
        else:
            # If the circuit does not need diagonalization, we decompose the circuit before
            # potentially applying `split_non_commuting` that produces multiple tapes with
            # duplicated operations. Otherwise, `decompose` has to be applied last because
            # `diagonalize_measurements` may add additional gates that are not supported.
            program.add_transform(
                decompose,
                stopping_condition=capabilities_analytic.supports_operation,
                stopping_condition_shots=capabilities_shots.supports_operation,
                name=self.name,
            )

        if not self.capabilities.overlapping_observables:
            program.add_transform(qml.transforms.split_non_commuting, grouping_strategy="wires")
        elif not self.capabilities.non_commuting_observables:
            program.add_transform(qml.transforms.split_non_commuting, grouping_strategy="qwc")
        elif not self.capabilities.supports_observable("Sum"):
            program.add_transform(qml.transforms.split_to_single_terms)

        if needs_diagonalization:
            obs_names = base_obs.keys() & self.capabilities.observables.keys()
            obs = {base_obs[obs] for obs in obs_names}
            program.add_transform(qml.transforms.diagonalize_measurements, supported_base_obs=obs)
            program.add_transform(
                decompose,
                stopping_condition=lambda o: capabilities_analytic.supports_operation(o.name),
                stopping_condition_shots=lambda o: capabilities_shots.supports_operation(o.name),
                name=self.name,
            )

        program.add_transform(qml.transforms.broadcast_expand)

        # Handle validations
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(
            validate_measurements,
            analytic_measurements=lambda mp: type(mp).__name__
            in capabilities_analytic.measurement_processes,
            sample_measurements=lambda mp: type(mp).__name__
            in capabilities_shots.measurement_processes,
            name=self.name,
        )
        program.add_transform(
            validate_observables,
            stopping_condition=observable_stopping_condition_factory(capabilities_analytic),
            stopping_condition_shots=observable_stopping_condition_factory(capabilities_shots),
            name=self.name,
        )

        return program

    @abc.abstractmethod
    @overload
    def execute(
        self, circuits: QuantumScript, execution_config: ExecutionConfig | None = None
    ) -> Result: ...

    @abc.abstractmethod
    @overload
    def execute(
        self,
        circuits: QuantumScriptBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> ResultBatch: ...

    @abc.abstractmethod
    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.

        **Interface parameters:**

        The provided ``circuits`` may contain interface specific data-types like ``torch.Tensor`` or ``jax.Array`` when
        :attr:`~.ExecutionConfig.gradient_method` of ``"backprop"`` is requested. If the gradient method is not backpropagation,
        then only vanilla numpy parameters or builtins will be present in the circuits.

        .. details::
            :title: Return Shape

            See :ref:`Return Type Specification <ReturnTypeSpec>` for more detailed information.

            The result for each :class:`~.QuantumTape` must match the shape specified by :class:`~.QuantumTape.shape`.

            The level of priority for dimensions from outer dimension to inner dimension is:

            1. Quantum Script in batch
            2. Shot choice in a shot vector
            3. Measurement in the quantum script
            4. Parameter broadcasting
            5. Measurement shape for array-valued measurements like probabilities

            For a batch of quantum scripts with multiple measurements, a shot vector, and parameter broadcasting:

            * ``result[0]``: the results for the first script
            * ``result[0][0]``: the first shot number in the shot vector
            * ``result[0][0][0]``: the first measurement in the quantum script
            * ``result[0][0][0][0]``: the first parameter broadcasting choice
            * ``result[0][0][0][0][0]``: the first value for an array-valued measurement

            With the exception of quantum script batches, dimensions with only a single component should be eliminated.

            For example:

            With a single script and a single measurement process, execute should return just the
            measurement value in a numpy array. ``shape`` currently accepts a device, as historically devices
            stored shot information. In the future, this method will accept an ``ExecutionConfig`` instead.

            >>> tape = qml.tape.QuantumTape(measurements=qml.expval(qml.Z(0))])
            >>> tape.shape(dev)
            ()
            >>> dev.execute(tape)
            array(1.0)

            If execute recieves a batch of scripts, then it should return a tuple of results:

            >>> dev.execute([tape, tape])
            (array(1.0), array(1.0))
            >>> dev.execute([tape])
            (array(1.0),)

            If the script has multiple measurements, then the device should return a tuple of measurements.

            >>> tape = qml.tape.QuantumTape(measurements=[qml.expval(qml.Z(0)), qml.probs(wires=(0,1))])
            >>> tape.shape(dev)
            ((), (4,))
            >>> dev.execute(tape)
            (array(1.0), array([1., 0., 0., 0.]))

        """
        raise NotImplementedError

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Determine whether or not a device provided derivative is potentially available.

        Default behaviour assumes first order device derivatives for all circuits exist if :meth:`~.compute_derivatives` is overriden.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.
            circuit (None, QuantumTape): A specific circuit to check differentation for.

        Returns:
            Bool

        The device can support multiple different types of "device derivatives", chosen via ``execution_config.gradient_method``.
        For example, a device can natively calculate ``"parameter-shift"`` derivatives, in which case :meth:`~.compute_derivatives`
        will be called for the derivative instead of :meth:`~.execute` with a batch of circuits.

        >>> config = ExecutionConfig(gradient_method="parameter-shift")
        >>> custom_device.supports_derivatives(config)
        True

        In this case, :meth:`~.compute_derivatives` or :meth:`~.execute_and_compute_derivatives` will be called instead of :meth:`~.execute` with
        a batch of circuits.

        If ``circuit`` is not provided, then the method should return whether or not device derivatives exist for **any** circuit.

        **Example:**

        For example, the Python device will support device differentiation via the adjoint differentiation algorithm
        if the order is ``1`` and the execution occurs with no shots (``shots=None``).

        >>> config = ExecutionConfig(derivative_order=1, gradient_method="adjoint")
        >>> dev.supports_derivatives(config)
        True
        >>> circuit_analytic = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))], shots=None)
        >>> dev.supports_derivatives(config, circuit=circuit_analytic)
        True
        >>> circuit_finite_shots = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))], shots=10)
        >>> dev.supports_derivatives(config, circuit = circuit_fintite_shots)
        False

        >>> config = ExecutionConfig(derivative_order=2, gradient_method="adjoint")
        >>> dev.supports_derivatives(config)
        False

        Adjoint differentiation will only be supported for circuits with expectation value measurements.
        If a circuit is provided and it cannot be converted to a form supported by differentiation method by
        :meth:`~.Device.preprocess`, then ``supports_derivatives`` should return False.

        >>> config = ExecutionConfig(derivative_order=1, shots=None, gradient_method="adjoint")
        >>> circuit = qml.tape.QuantumScript([qml.RX(2.0, wires=0)], [qml.probs(wires=(0,1))])
        >>> dev.supports_derivatives(config, circuit=circuit)
        False

        If the circuit is not natively supported by the differentiation method but can be converted into a form
        that is supported, it should still return ``True``.  For example, :class:`~.Rot` gates are not natively
        supported by adjoint differentation, as they do not have a generator, but they can be compiled into
        operations supported by adjoint differentiation. Therefore this method may reproduce compilation
        and validation steps performed by :meth:`~.Device.preprocess`.

        >>> config = ExecutionConfig(derivative_order=1, shots=None, gradient_method="adjoint")
        >>> circuit = qml.tape.QuantumScript([qml.Rot(1.2, 2.3, 3.4, wires=0)], [qml.expval(qml.Z(0))])
        >>> dev.supports_derivatives(config, circuit=circuit)
        True

        **Backpropagation:**

        This method is also used be to validate support for backpropagation derivatives. Backpropagation
        is only supported if the device is transparent to the machine learning framework from start to finish.

        >>> config = ExecutionConfig(gradient_method="backprop")
        >>> python_device.supports_derivatives(config)
        True
        >>> cpp_device.supports_derivatives(config)
        False

        """
        if execution_config is None:
            return type(self).compute_derivatives != Device.compute_derivatives

        if (
            execution_config.gradient_method not in {"device", "best"}
            or execution_config.derivative_order != 1
        ):
            return False

        return type(self).compute_derivatives != Device.compute_derivatives

    def compute_derivatives(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ):
        """Calculate the jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple: The jacobian for each trainable parameter

        .. seealso:: :meth:`~.supports_derivatives` and :meth:`~.execute_and_compute_derivatives`.

        **Execution Config:**

        The execution config has ``gradient_method`` and ``order`` property that describes the order of differentiation requested. If the requested
        method or order of gradient is not provided, the device should raise a ``NotImplementedError``. The :meth:`~.supports_derivatives`
        method can pre-validate supported orders and gradient methods.

        **Return Shape:**

        If a batch of quantum scripts is provided, this method should return a tuple with each entry being the gradient of
        each individual quantum script. If the batch is of length 1, then the return tuple should still be of length 1, not squeezed.

        """
        raise NotImplementedError(f"{self.name} does not support differentiable workflows.")

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ):
        """Compute the results and jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tuple: A numeric result of the computation and the gradient.

        See :meth:`~.execute` and :meth:`~.compute_derivatives` for more information about return shapes and behaviour.
        If :meth:`~.compute_derivatives` is defined, this method should be as well.

        This method can be used when the result and execution need to be computed at the same time, such as
        during a forward mode calculation of gradients. For certain gradient methods, such as adjoint
        diff gradients, calculating the result and gradient at the same can save computational work.

        """
        if execution_config is None:
            execution_config = ExecutionConfig()
        return self.execute(circuits, execution_config), self.compute_derivatives(
            circuits, execution_config
        )

    def compute_jvp(
        self,
        circuits: QuantumScriptOrBatch,
        tangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        r"""The jacobian vector product used in forward mode calculation of derivatives.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits
            tangents (tensor-like): Gradient vector for input parameters.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple: A numeric result of computing the jacobian vector product

        **Definition of jvp:**

        If we have a function with jacobian:

        .. math::

            \vec{y} = f(\vec{x}) \qquad J_{i,j} = \frac{\partial y_i}{\partial x_j}

        The Jacobian vector product is the inner product with the derivatives of :math:`x`, yielding
        only the derivatives of the output :math:`y`:

        .. math::

            \text{d}y_i = \Sigma_{j} J_{i,j} \text{d}x_j

        **Shape of tangents:**

        The ``tangents`` tuple should be the same length as ``circuit.get_parameters()`` and have a single number per
        parameter. If a number is zero, then the gradient with respect to that parameter does not need to be computed.

        """
        raise NotImplementedError

    def execute_and_compute_jvp(
        self,
        circuits: QuantumScriptOrBatch,
        tangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        """Execute a batch of circuits and compute their jacobian vector products.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): circuit or batch of circuits
            tangents (tensor-like): Gradient vector for input parameters.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple, Tuple: A numeric result of execution and of computing the jacobian vector product

        .. seealso:: :meth:`~pennylane.devices.Device.execute` and :meth:`~.Device.compute_jvp`
        """
        if execution_config is None:
            execution_config = ExecutionConfig()
        return self.execute(circuits, execution_config), self.compute_jvp(
            circuits, tangents, execution_config
        )

    def supports_jvp(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Whether or not a given device defines a custom jacobian vector product.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.
            circuit (None, QuantumTape): A specific circuit to check differentation for.

        Default behaviour assumes this to be ``True`` if :meth:`~.compute_jvp` is overridden.

        """
        return type(self).compute_jvp != Device.compute_jvp

    def compute_vjp(
        self,
        circuits: QuantumScriptOrBatch,
        cotangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        r"""The vector jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, `cotangents` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tensor-like: A numeric result of computing the vector jacobian product

        **Definition of vjp:**

        If we have a function with jacobian:

        .. math::

            \vec{y} = f(\vec{x}) \qquad J_{i,j} = \frac{\partial y_i}{\partial x_j}

        The vector jacobian product is the inner product of the derivatives of the output ``y`` with the
        Jacobian matrix. The derivatives of the output vector are sometimes called the **cotangents**.

        .. math::

            \text{d}x_i = \Sigma_{i} \text{d}y_i J_{i,j}

        **Shape of cotangents:**

        The value provided to ``cotangents`` should match the output of :meth:`~.execute`.

        """
        raise NotImplementedError

    def execute_and_compute_vjp(
        self,
        circuits: QuantumScriptOrBatch,
        cotangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        r"""Calculate both the results and the vector jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits to be executed
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, `cotangents` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector jacobian product

        .. seealso:: :meth:`~pennylane.devices.Device.execute` and :meth:`~.Device.compute_vjp`
        """
        if execution_config is None:
            execution_config = ExecutionConfig()
        return self.execute(circuits, execution_config), self.compute_vjp(
            circuits, cotangents, execution_config
        )

    def supports_vjp(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Whether or not a given device defines a custom vector jacobian product.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.
            circuit (None, QuantumTape): A specific circuit to check differentation for.

        Default behaviour assumes this to be ``True`` if :meth:`~.compute_vjp` is overridden.
        """
        return type(self).compute_vjp != Device.compute_vjp

    def eval_jaxpr(
        self,
        jaxpr: "jax.extend.core.Jaxpr",
        consts: list[TensorLike],
        *args,
        execution_config: ExecutionConfig | None = None,
        shots: Shots = Shots(None),
    ) -> list[TensorLike]:
        """An **experimental** method for natively evaluating PLXPR. See the ``capture`` module for more details.

        Args:
            jaxpr (jax.extend.core.Jaxpr): Pennylane variant jaxpr containing quantum operations and measurements
            consts (list[TensorLike]): the closure variables ``consts`` corresponding to the jaxpr
            *args (TensorLike): the variables to use with the jaxpr.

        Keyword Args:
            execution_config (Optional[ExecutionConfig]): a data structure with additional information required for execution
            shots (Shots): the number of shots to use for the evaluation

        Returns:
            list[TensorLike]: the result of evaluating the jaxpr with the given parameters.

        """
        raise NotImplementedError

    def jaxpr_jvp(
        self,
        jaxpr: "jax.extend.core.Jaxpr",
        args,
        tangents,
        execution_config: ExecutionConfig | None = None,
    ):
        """An **experimental** method for computing the results and jvp for PLXPR.
        See the ``capture`` module for more details.

        Args:
            jaxpr (jax.extend.core.Jaxpr): Pennylane variant jaxpr containing quantum operations
                and measurements
            args (Sequence[TensorLike]): the ``consts`` followed by the normal   arguments
            tangents (Sequence[TensorLike]): the tangents corresponding to ``args``.
                May contain ``jax.interpreters.ad.Zero``.

        Keyword Args:
            execution_config (Optional[ExecutionConfig]): a data structure with additional information required for execution

        Returns:
            Sequence[TensorLike], Sequence[TensorLike]: the results and jacobian vector products

        >>> qml.capture.enable()
        >>> import jax
        >>> closure_var = jax.numpy.array(0.5)
        >>> def f(x):
        ...     qml.RX(closure_var, 0)
        ...     qml.RX(x, 1)
        ...     return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))
        >>> jaxpr = jax.make_jaxpr(f)(1.2)
        >>> args = (closure_var, 1.2)
        >>> zero = jax.interpreters.ad.Zero(jax.core.ShapedArray((), float))
        >>> tangents = (zero, 1.0)
        >>> config = qml.devices.ExecutionConfig(gradient_method="adjoint")
        >>> dev = qml.device('default.qubit', wires=2)
        >>> res, jvps = dev.jaxpr_jvp(jaxpr.jaxpr, args, tangents, execution_config=config)
        >>> res
        [Array(0.87758255, dtype=float32), Array(0.36235774, dtype=float32)]
        >>> jvps
        [Array(0., dtype=float32), Array(-0.932039, dtype=float32)]

        """
        raise NotImplementedError(f"device {self} does not yet support PLXPR jvps.")


def _default_mcm_method(capabilities: DeviceCapabilities, shots_present: bool) -> str:
    """Simple strategy to find the best match for the default mcm method."""

    supports_one_shot = "one-shot" in capabilities.supported_mcm_methods
    has_device_support = "device" in capabilities.supported_mcm_methods

    if has_device_support:
        return "device"

    if supports_one_shot and shots_present:
        return "one-shot"

    return "deferred"
