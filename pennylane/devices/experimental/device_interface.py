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
"""This module contains the Abstract Base Class for experimental pennylane devices.

"""

import abc

from typing import Callable, List, Union, Sequence, Tuple

# Runtime specific utilities, such as pre and postprocessing annotations
from pennylane.workflow import ExecutionConfig
from pennylane.tape import QuantumScript
from pennylane import Tracker

QScriptBatch = Sequence[QuantumScript]

# pylint: disable=unused-argument
class AbstractDevice(abc.ABC):
    """An experimental PennyLane device.

    The child classes can define any number of class specific arguments and keyword arguments.

    Experimental devicves should be configured to run under ``qml.enable_return()``, the newer
    return shape specification.

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
        self, qscript: Union[QuantumScript, QScriptBatch], execution_config: ExecutionConfig = None
    ) -> Tuple[List[QuantumScript], Callable]:
        """Device preprocessing function.

        Args:
            qscript (Union[QuantumScript, Sequence[QuantumScript]]): The circuit or a batch of circuits to preprocess
                before execution on the device
            execution_config (ExecutionConfig): A datastructure describing the parameters needed to fully describe
                the execution. Includes such information as shots.

        Returns:
            Union[QuantumScript, Sequence[QuantumScript]], Callable: QuantumScripts that the device can natively execute without
            error and a postprocessing function to be called after execution.

        Raises:
            Exception: An exception is raised if the input cannot be converted into a form supported by the device.

        Preprocessing steps may include:
        - expansion to :class:`~.Operator`'s and :class:`~.MeasurementProcess` objects supported by the device.
        - splitting a quantum script with the measurement of non-commuting observables or Hamiltonians into multiple executions
        - splitting quantum scripts with batched parameters into multiple executions

        This step may also validate parameters of the ``execution_config``. For example, it could raise an error if the
        device doesn't support finite shots but the ``execution_config`` requests it.

        """

        def blank_postprocessing_fn(res):
            """Identity postprocessing function created in AbstractDevice preprocessing.

            Args:
                res (tensor-like): A result object

            Returns:
                tensor-like: The function input.

            """
            return res

        return qscript, blank_postprocessing_fn

    @abc.abstractmethod
    def execute(
        self, qscripts: Union[QuantumScript, QScriptBatch], execution_config: ExecutionConfig = None
    ):
        """Execute a Quantum Script and turn it into a result.

        Args:
            qscripts (Union[QuantumScript, Sequence[QuantumScript]]): the QuantumScript to be executed
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            A numeric result of the computation.

        **Return shape:**

        The result for each :class:`~.QuantumScript` must match the shape specified by :class:`~.QuantumScript.shape`.

        For example:

        >>> from pennylane.devices import experimental as devices
        >>> qml.enable_return()
        >>> dev = devices.PythonDevice()
        >>> dev = devices.backward_patch_interface(dev)
        >>> qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
        >>> qs.shape(dev)
        ((), (4,))
        >>> dev.execute(qs)
        (1.0, array([1., 0., 0., 0.]))

        """
        raise NotImplementedError

    def execute_and_gradients(
        self,
        qscripts: Union[QuantumScript, QScriptBatch],
        execution_config: ExecutionConfig = None,
    ):
        """Compute the results and gradients of QuantumScripts at the same time.

        Args:
            qscripts (Union[QuantumScript, Sequence[QuantumScript]]): the QuantumScript to be executed
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tensor-like, tensor-like: A numeric result of the computation.

        This method can be used when the result and execution need to be computed at the same time, such as
        during a forward mode calculation of gradients. For certain gradient methods, such as adjoint
        diff gradients, calculating the result and gradient at the same can save computational work.

        """
        return self.execute(qscripts, execution_config), self.gradient(qscripts, execution_config)

    def gradient(
        self,
        qscript: Union[QuantumScript, QScriptBatch],
        execution_config: ExecutionConfig = None,
    ):
        """Calculate the gradient of either a single or a batch of Quantum Scripts.

        Args:
            qscripts (Union[QuantumScript, Sequence[QuantumScript]]): the QuantumScript to be executed
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tensor-like: A numeric result of the computation.
        """
        raise NotImplementedError

    def vjp(
        self,
        qscript: Union[QuantumScript, QScriptBatch],
        dy,
        execution_config: ExecutionConfig = None,
    ):
        """The vector jacobian product.

        Args:
            qscripts (Union[QuantumScript, Sequence[QuantumScript]]): the QuantumScript to be executed
            dy (tensor-like): Gradient-output vector. Must have shape matching the output shape of the
                corresponding qscript
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tensor-like: A numeric result of computing the vector jacobian product

        """
        raise NotImplementedError

    @classmethod
    def supports_gradient_with_configuration(cls, execution_config: ExecutionConfig) -> bool:
        """Determine whether or not a gradient is available with a given execution configuration.

        Default behaviour assumes first order derivatives exist if :meth:`~.gradient` is overriden.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.

        Returns:
            Bool

        **Example:**

        For example, the current Python device supports adjoint differentiation if the order is ``1`` and
        the execution occurs with no shots ``shots=None``.

        >>> from pennylane.workflow import ExecutionConfig
        >>> from pennylane.devices import experimental as devices
        >>> dev = devices.PythonDevice()
        >>> config = ExecutionConfig(order=1, shots=None)
        >>> dev.supports_gradient_with_configuration(config)
        True
        >>> config = ExecutionConfig(order=1, shots=10)
        >>> dev.supports_gradient_with_configuration(config)
        False
        >>> config = ExecutionConfig(order=2, shots=None)
        >>> dev.supports_gradient_with_configuration(config)
        False

        """
        return cls.gradient != AbstractDevice.gradient if execution_config.order == 1 else False

    @classmethod
    def supports_vjp(cls, execution_config: ExecutionConfig) -> bool:
        """Whether or not a given device defines a custom vector jacobian product.

        Default behaviour assumes this to be ``True`` if :meth:`~.vjp` is overridden.


        """
        return cls.vjp != AbstractDevice.vjp
