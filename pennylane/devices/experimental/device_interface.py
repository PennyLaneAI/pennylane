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
"""module docstring"""

import abc

from typing import Callable, List, Union, Sequence, Tuple

# Runtime specific utilities, such as pre and postprocessing annotations
from pennylane.workflow import ExecutionConfig
from pennylane.tape import QuantumScript
from pennylane import Tracker

QScriptBatch = Sequence[QuantumScript]


class AbstractDevice(abc.ABC):
    """
    This abstract device interface enables direct and dynamic function registration for pre-processing, post-processing, gradients, VJPs, and arbitrary functionality.
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

        The result for each :class:`~.QuantumScript` must match the shape specified by :class:`~.QuantumScript.shape`.

        """
        pass

    def execute_and_gradients(
        self,
        qscripts: Union[QuantumScript, QScriptBatch],
        execution_config: ExecutionConfig = None,
        order: int = 1,
    ):
        """Compute the results and gradients of QuantumScripts at the same time.

        Args:
            qscripts (Union[QuantumScript, Sequence[QuantumScript]]): the QuantumScript to be executed
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tensor-like, tensor-like: A numeric result of the computation.

        """
        return self.execute(qscripts, execution_config), self.gradient(
            qscripts, execution_config, order=order
        )

    def gradient(
        self,
        qscript: Union[QuantumScript, QScriptBatch],
        execution_config: ExecutionConfig = None,
        order: int = 1,
    ):
        """Main gradient method, contains validation and post-processing
        so that device developers do not need to replicate all the
        internal pieces. Contain 'user' facing details here."""
        pass

    def vjp(self, qscript: Union[QuantumScript, QScriptBatch], dy):
        """VJP method. Added through registration"""
        pass

    @classmethod
    def supports_gradient_of_order(cls, order: int = 1) -> bool:
        """

        Args:
            order (int): The order of gradient

        """
        return cls.gradient != AbstractDevice.gradient if order == 1 else False

    @classmethod
    def supports_vjp(cls) -> bool:
        return cls.vjp != AbstractDevice.vjp
