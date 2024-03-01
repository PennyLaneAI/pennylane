# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
"""
This module contains the qml.var measurement.
"""
import warnings
from typing import Sequence, Tuple

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import SampleMeasurement, StateMeasurement, Variance
from .mid_measure import MeasurementValue


def _var_op(op: Operator, argname=None):
    if argname is not None and argname != "op":
        warnings.warn(
            f"var got argument '{argname}' of type {type(op)}. Using argument as op", UserWarning
        )

    if not op.is_hermitian:
        warnings.warn(f"{op.name} might not be hermitian.")

    return VarianceMP(obs=op)


def _var_mv(mv: MeasurementValue, argname=None):
    if argname is not None and argname != "mv":
        warnings.warn(
            f"var got argument '{argname}' of type {type(mv)}. Using argument as mv", UserWarning
        )

    if isinstance(mv, Sequence):
        raise ValueError(
            "qml.var does not support measuring sequences of measurements or observables"
        )
    return VarianceMP(mv=mv)


def var(*args, **kwargs) -> "VarianceMP":
    r"""Variance of the supplied observable.

    Args:
        op (Operator): a quantum observable object.
        mv (MeasurementValue): ``MeasurementValue`` corresponding to mid-circuit
            measurements. To get the variance for more than one ``MeasurementValue``,
            they can be composed using arithmetic operators.

    Returns:
        VarianceMP: Measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Y(0))

    Executing this QNode:

    >>> circuit(0.5)
    0.7701511529340698
    """
    _args = [a for a in args if a is not None]
    _kwargs = {key: value for key, value in kwargs.items() if value is not None}

    if (n_args := len(_args) + len(_kwargs)) != 1:
        raise ValueError(f"var takes 1 argument but {n_args} were given")

    if _args:
        arg = args[0]
        argname = None

    elif _kwargs:
        argname, arg = next(iter(_kwargs.items()))

        if argname not in ("op", "mv"):
            raise TypeError(f"var got an unexpected keyword argument '{argname}'")

    if isinstance(arg, Operator):
        return _var_op(arg, argname=argname)

    return _var_mv(arg, argname=argname)


class VarianceMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the variance of the supplied observable.

    Please refer to :func:`var` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    @property
    def return_type(self):
        return Variance

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum(s.copies for s in shots.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        # estimate the variance
        op = self.mv if self.mv is not None else self.obs
        with qml.queuing.QueuingManager.stop_recording():
            samples = qml.sample(op).process_samples(
                samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size
            )

        # With broadcasting, we want to take the variance over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        # TODO: do we need to squeeze here? Maybe remove with new return types
        return qml.math.squeeze(qml.math.var(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")

        # we use ``wires`` instead of ``op`` because the observable was
        # already applied to the state
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(state=state, wire_order=wire_order)
        # In case of broadcasting, `prob` has two axes and these are a matrix-vector products
        return qml.math.dot(prob, (eigvals**2)) - qml.math.dot(prob, eigvals) ** 2
