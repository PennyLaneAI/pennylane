# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the Abstract Base Class for the next generation of devices.

"""

import abc

from numbers import Number
from typing import Callable, Union, Sequence, Tuple

from pennylane.tape import QuantumTape
from pennylane import Tracker

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]

# pylint: disable=unused-argument, no-self-use
class QuantumDevice(abc.ABC):
    """An experimental PennyLane device.

    The child classes can define any number of class specific arguments and keyword arguments.

    Experimental devices should be configured to run under :func:`~.enable_return`, the newer
    return shape specification.

    Only the ``execute`` function must be defined to construct a device.

    .. warning::

        The ``QuantumTape`` objects passed to the device may inherit from :class:`~.AnnotatedQueue` and have a threading lock.
        To use these objects in a distributed manner, please convert them to a :class:`~.QuantumScript` internally to remove
        the threading lock before sending them to different threads.

    .. details::
        :title: Design Motivation

        **Streamlined interface:** Only methods that are required to interact with the rest of PennyLane will be placed in the
        interface. Developers will be able to clearly see what they can change while still having a fully functional device.

        **Reduction of duplicate methods:** Methods that solve similar problems are combined together. Only one place will have
        to solve each individual problem.

        **Support for dynamic execution configurations:** Properties such as shots belong to specific executions.

        **Greater coverage for differentiation methods:** Devices can define any order of derivative, the vector jacobian product,
        or the jacobian vector product.  Calculation of derivatives can be done at the same time as execution to allow reuse of intermediate
        results.

    .. details::
        :title: Porting from the old interface

        :meth:`~.Device.batch_execute` and :meth:`~.Device.execute` are now a single method, :meth:`~.QuantumDevice.execute`

        :meth:`~.Device.batch_transform` and :meth:`~.Device.expand_fn` are now a single method, :meth:`~.QuantumDevice.preprocess`

        Shot information is no longer stored on the device, but instead provided by a runtime argument, the ``ExecutionConfig``. Devices
        may define their own default values in the absence of specified values in the execution config.

        The old devices defined a :meth:`~.Device.capabilities` dictionary that defined characterisitics of the devices and controlled various
        preprocessing and validation steps, such as ``"supports_broadcasting"``.  These capabilites should now be handled by the
        :meth:`~.QuantumDevice.preprocess` method. For example, if a device does not support broadcasting, ``preprocess`` should
        split a quantum script with broadcasted parameters into a batch of quantum scripts. If the device does not support mid circuit
        measurements, then ``preprocess`` should apply :func:`~.defer_measurements`.  A set of default preprocessing steps will be available
        to make a seemless transition to the new interface.

        A class will be provided to easily construct default preprocessing steps from supported operations, supported observables,
        supported measurement processes, and various capabilities.

        Versioning should be specified by the package containing the device. If an external package includes a PennyLane device,
        then the package requirements should specify the minimium PennyLane version required to work with the device.


    .. details::
        :title: Execution Configuration

        Most methods in this device accept an ``ExecutionConfig`` argument,  The execution config can be:

        * ``None``: device defaults should be used

        * A single ``ExecutionConfig`` datastructure: All scripts in the batch use the same configuration

        * An Iterable of ``ExecutionConfig``'s of the same length as the the circuits input: Each quantum script
          has its own set of configuration parameters. For example, every script can be executed with a different
          number of shots.

        Execution config properties most relevant for devices are:

        * ``shots``: Either an integer or a tuple of integer. A tuple of integers represenents a shot vector.

        * ``device_options``: A dictionary of device specific options. For example, the python device may have `use_multiprocessing`
          as a key. These should be documented in the class docstring.

    """

    tracker: Tracker = Tracker()
    """A :class:`~.Tracker` that can store information about device executions, shots, batches,
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

    def __init__(self, *args, **kwargs) -> None:
        # each instance should have its own Tracker.
        self.tracker = Tracker()

    def preprocess(
        self, circuits: QuantumTape_or_Batch, execution_config=None
    ) -> Tuple[QuantumTapeBatch, Callable]:
        """Device preprocessing function.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): The circuit or a batch of circuits to preprocess
                before execution on the device
            execution_config (ExecutionConfig): A datastructure describing the parameters needed to fully describe
                the execution. Includes such information as shots.

        Returns:
            Sequence[QuantumTape], Callable: QuantumTapes that the device can natively execute without
            error and a postprocessing function to be called after execution.

        Raises:
            Exception: An exception is raised if the input cannot be converted into a form supported by the device.

        This function is tracked by machine learning interfaces and should be fully differentiable.

        Preprocessing steps may include:

        * expansion to :class:`~.Operator`'s and :class:`~.MeasurementProcess` objects supported by the device.
        * splitting a quantum script with the measurement of non-commuting observables or Hamiltonians into multiple executions
        * splitting quantum scripts with batched parameters into multiple executions

        This step may also validate parameters of the ``execution_config``. For example, it could raise an error if the
        device doesn't support finite shots but the ``execution_config`` requests it.

        Processing steps can depend on the requested gradient method. For example, when the device gradient method is requested,
        preprocessing can decompose the operations till they all have

        """

        def blank_postprocessing_fn(res):
            """Identity postprocessing function created in QuantumDevice preprocessing.

            Args:
                res (tensor-like): A result object

            Returns:
                tensor-like: The function input.

            """
            return res

        circuit_batch = [circuits] if isinstance(circuits, QuantumTape) else circuits
        return circuit_batch, blank_postprocessing_fn

    @abc.abstractmethod
    def execute(self, circuits: QuantumTape_or_Batch, execution_config=None):
        """Execute a quantum script or a batch of quantum scripts and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum script(s) to be executed
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            A numeric result of the computation.

        **Interface parameters:**

        Note that the parameters contained within the quantum script may contain interface-specific data types, such as
        ``torch.tensor`` or ``tensorflow.Variable``. If the device does not wish to handle interface-specific parameters, they
        can implement an optional "internal preprocessing" step that converts all parameters to vanilla numpy. A convenience
        transform implementing this will be provided.

        .. details::
            :title: Return Shape

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

            >>> qs = qml.tape.QuantumTape(measurements=qml.expval(qml.PauliZ(0))])
            >>> qs.shape(dev)
            ()
            >>> dev.execute(qs)
            array(1.0)

            If execute recieves a batch of scripts, then it should return a tuple of results:

            >>> dev.execute([qs, qs])
            (array(1.0), array(1.0))
            >>> dev.execute([qs])
            (array(1.0),)

            If the script has multiple measurments, then the device should return a tuple of measurements.

            >>> qs = qml.tape.QuantumTape(measurements=[qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
            >>> qs.shape(dev)
            ((), (4,))
            >>> dev.execute(qs)
            (array(1.0), array([1., 0., 0., 0.]))

        """
        raise NotImplementedError

    @classmethod
    def supports_differentiation(cls, execution_config=None) -> bool:
        """Determine whether or not a gradient is available with a given execution configuration.

        Default behaviour assumes first order derivatives exist if :meth:`~.differentiate` is overriden.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.

        Returns:
            Bool

        **Example:**

        For example, the Python device will a device differentiation via the adjoint differentiation algorithm
        if the order is ``1`` and the execution occurs with no shots ``shots=None``.

        >>> config = ExecutionConfig(derivative_order=1, shots=None, gradient_method="device")
        >>> dev.supports_differentiation(config)
        True
        >>> config = ExecutionConfig(derivative_order=1, shots=10, gradient_method="device")
        >>> dev.supports_differentiation(config)
        False
        >>> config = ExecutionConfig(derivative_order=2, shots=None, gradient_method="device")
        >>> dev.supports_differentiation(config)
        False

        **Backpropagation:**

        This method is also used be to validate support for backpropagation derivatives. Backpropagation
        is only supported if the devices is transparent to the machine learning framework from start to finish.

        >>> config = ExecutionConfig(gradient_method="backprop", framework="torch")
        >>> python_device.supports_differentiation(config)
        True
        >>> cpp_device.supports_differentiation(config)
        False

        """
        if getattr(execution_config, "gradient_method", None) != "device":
            return False
        order = getattr(execution_config, "order", 1)
        return cls.differentiate != QuantumDevice.differentiate if order == 1 else False

    def differentiate(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config=None,
    ):
        """Calculate the jacobian of either a single or a batch of Quantum Scripts.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the QuantumTape to be executed
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple: The jacobian for each trainable parameter

        See also :meth:`~.supports_differentiation` and :meth:`~.execute_and_differentiate`.

        **Execution Config:**

        The execution config has an ``order`` property that describes the order of gradient requested. If the requested order
        of gradient is not provided, the device should raise a ``NotImplementedError``. The :meth:`~.supports_jacobian_with_configuration`
        method can pre-validate supported orders.

        **Return Shape:**

        If a batch of quantum scripts is provided, this method should return a tuple with each entry being the gradient of
        each individual quantum script. If the batch is of length 1, then the return tuple should still be of length 1, not squeezed.

        """
        raise NotImplementedError

    def execute_and_differentiate(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config=None,
    ):
        """Compute the results and jacobians of QuantumTapes at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the QuantumTape or batch to be executed
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tuple, tuple: A numeric result of the computation and the gradient.

        See :meth:`~.execute` and :meth:`~.differentiate` for more information about return shapes and behaviour. If :meth:`~.differentiate`
        is defined, this method should be as well.

        This method can be used when the result and execution need to be computed at the same time, such as
        during a forward mode calculation of gradients. For certain gradient methods, such as adjoint
        diff gradients, calculating the result and gradient at the same can save computational work.

        """
        return self.execute(circuits, execution_config), self.differentiate(
            circuits, execution_config
        )

    def jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config=None,
    ):
        r"""The jacobian vector product used in forward mode calculation of derivatives.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the QuantumTape to be executed
            tangents (tensor-like): Gradient vector for input parameters.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple[tensor-like]: A numeric result of computing the jacobian vector product

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

    def execute_and_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config=None,
    ):
        """Execute a batch of quantum scripts and compute their jacobian vector products.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the QuantumTape to be executed
            tangents (tensor-like): Gradient vector for input parameters.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple[tensor-like], Tuple[tensor-like]: A numeric result of computing the jacobian vector product

        See :meth:`~.QuantumDevice.execute` and :meth:`~.QuantumDevice.jvp` for more information on each method.
        """
        return self.execute(circuits, execution_config), self.jvp(
            circuits, tangents, execution_config
        )

    @classmethod
    def supports_jvp(cls, execution_config=None) -> bool:
        """Whether or not a given device defines a custom jacobian vector product.

        Default behaviour assumes this to be ``True`` if :meth:`~.jvp` is overridden.

        """
        return cls.jvp != QuantumDevice.jvp

    def vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config=None,
    ):
        r"""The vector jacobian product used in reversed mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the QuantumTape to be executed
            cotangents (Tuple[Number]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit
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

        """
        raise NotImplementedError

    def execute_and_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config=None,
    ):
        r"""Calculate both the results and the vector jacobian product used in reversed mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the QuantumTape to be executed
            cotangents Tuple[Number]: Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector jacobian product

        See :meth:`~.QuantumDevice.execute` and :meth:`~.QuantumDevice.vjp` for more information.
        """
        return self.execute(circuits, execution_config), self.vjp(
            circuits, cotangents, execution_config
        )

    @classmethod
    def supports_vjp(cls, execution_config=None) -> bool:
        """Whether or not a given device defines a custom vector jacobian product.

        Default behaviour assumes this to be ``True`` if :meth:`~.vjp` is overridden.
        """
        return cls.vjp != QuantumDevice.vjp
