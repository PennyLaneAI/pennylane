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
"""Code for the tape transform implementing the deferred measurement principle."""
from typing import Sequence, Callable
import pennylane as qml
from pennylane.measurements import MidMeasureMP, ProbabilityMP, SampleMP, CountsMP, MeasurementValue
from pennylane.ops.op_math import ctrl

from pennylane.tape import QuantumTape
from pennylane.transforms import transform

from pennylane.wires import Wires
from pennylane.queuing import QueuingManager

# pylint: disable=too-many-branches, protected-access, too-many-statements


def _check_tape_validity(tape: QuantumTape):
    """Helper function to check that the tape is valid."""
    cv_types = (qml.operation.CVOperation, qml.operation.CVObservable)
    ops_cv = any(isinstance(op, cv_types) and op.name != "Identity" for op in tape.operations)
    obs_cv = any(
        isinstance(getattr(op, "obs", None), cv_types)
        and not isinstance(getattr(op, "obs", None), qml.Identity)
        for op in tape.measurements
    )
    if ops_cv or obs_cv:
        raise ValueError("Continuous variable operations and observables are not supported.")

    for mp in tape.measurements:
        if isinstance(mp, (CountsMP, ProbabilityMP, SampleMP)) and not (
            mp.obs or mp._wires or mp.mv
        ):
            raise ValueError(
                f"Cannot use {mp.__class__.__name__} as a measurement without specifying wires "
                "when using qml.defer_measurements. Deferred measurements can occur "
                "automatically when using mid-circuit measurements on a device that does not "
                "support them."
            )

        if mp.__class__.__name__ == "StateMP":
            raise ValueError(
                "Cannot use StateMP as a measurement when using qml.defer_measurements. "
                "Deferred measurements can occur automatically when using mid-circuit "
                "measurements on a device that does not support them."
            )

    samples_present = any(isinstance(mp, SampleMP) for mp in tape.measurements)
    postselect_present = any(
        op.postselect is not None for op in tape.operations if isinstance(op, MidMeasureMP)
    )
    if postselect_present and samples_present and tape.batch_size is not None:
        raise ValueError(
            "Returning qml.sample is not supported when postselecting mid-circuit "
            "measurements with broadcasting"
        )


def _collect_mid_measure_info(tape: QuantumTape):
    """Helper function to collect information related to mid-circuit measurements in the tape."""

    # Find wires that are reused after measurement
    measured_wires = []
    reused_measurement_wires = set()
    any_repeated_measurements = False
    is_postselecting = False

    for op in tape:
        if isinstance(op, MidMeasureMP):
            if op.postselect is not None:
                is_postselecting = True
            if op.reset:
                reused_measurement_wires.add(op.wires[0])

            if op.wires[0] in measured_wires:
                any_repeated_measurements = True
            measured_wires.append(op.wires[0])

        else:
            reused_measurement_wires = reused_measurement_wires.union(
                set(measured_wires).intersection(op.wires.toset())
            )

    return measured_wires, reused_measurement_wires, any_repeated_measurements, is_postselecting


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def defer_measurements(tape: QuantumTape, **kwargs) -> (Sequence[QuantumTape], Callable):
    """Quantum function transform that substitutes operations conditioned on
    measurement outcomes to controlled operations.

    This transform uses the `deferred measurement principle
    <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`_ and
    applies to qubit-based quantum functions.

    Support for mid-circuit measurements is device-dependent. If a device
    doesn't support mid-circuit measurements natively, then the QNode will
    apply this transform.

    .. note::

        The transform uses the :func:`~.ctrl` transform to implement operations
        controlled on mid-circuit measurement outcomes. The set of operations
        that can be controlled as such depends on the set of operations
        supported by the chosen device.

    .. note::

        Devices that inherit from :class:`~pennylane.QubitDevice` **must** be initialized
        with an additional wire for each mid-circuit measurement after which the measured
        wire is reused or reset for ``defer_measurements`` to transform the quantum tape
        correctly.

    .. note::

        This transform does not change the list of terminal measurements returned by
        the quantum function.

    .. note::

        When applying the transform on a quantum function that contains the
        :class:`~.Snapshot` instruction, state information corresponding to
        simulating the transformed circuit will be obtained. No
        post-measurement states are considered.

    .. warning::

        :func:`~.pennylane.state` is not supported with the ``defer_measurements`` transform.
        Additionally, :func:`~.pennylane.probs`, :func:`~.pennylane.sample` and
        :func:`~.pennylane.counts` can only be used with ``defer_measurements`` if wires
        or an observable are explicitly specified.

    .. warning::

        ``defer_measurements`` does not support using custom wire labels if any measured
        wires are reused or reset.

    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ValueError: If custom wire labels are used with qubit reuse or reset
        ValueError: If any measurements with no wires or observable are present
        ValueError: If continuous variable operations or measurements are present
        ValueError: If using the transform with any device other than
            :class:`default.qubit <~pennylane.devices.DefaultQubit>` and postselection is used

    **Example**

    Suppose we have a quantum function with mid-circuit measurements and
    conditional operations:

    .. code-block:: python3

        def qfunc(par):
            qml.RY(0.123, wires=0)
            qml.Hadamard(wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, qml.RY)(par, wires=0)
            return qml.expval(qml.Z(0))

    The ``defer_measurements`` transform allows executing such quantum
    functions without having to perform mid-circuit measurements:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qfunc = qml.defer_measurements(qfunc)
    >>> qnode = qml.QNode(transformed_qfunc, dev)
    >>> par = np.array(np.pi/2, requires_grad=True)
    >>> qnode(par)
    tensor(0.43487747, requires_grad=True)

    We can also differentiate parameters passed to conditional operations:

    >>> qml.grad(qnode)(par)
    tensor(-0.49622252, requires_grad=True)

    Reusing and reseting measured wires will work as expected with the
    ``defer_measurements`` transform:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = qml.measure(1, reset=True)

            qml.cond(m_0, qml.RY)(y, wires=0)
            qml.RX(np.pi/4, wires=1)
            return qml.probs(wires=[0, 1])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.76960924, 0.13204407, 0.08394415, 0.01440254], requires_grad=True)
    """

    if not any(isinstance(o, MidMeasureMP) for o in tape.operations):
        return (tape,), null_postprocessing

    _check_tape_validity(tape)

    device = kwargs.get("device", None)

    device = kwargs.get("device", None)

    new_operations = []

    # Find wires that are reused after measurement
    (
        measured_wires,
        reused_measurement_wires,
        any_repeated_measurements,
        is_postselecting,
    ) = _collect_mid_measure_info(tape)

    if is_postselecting and device is not None and not isinstance(device, qml.devices.DefaultQubit):
        raise ValueError(f"Postselection is not supported on the {device} device.")

    if len(reused_measurement_wires) > 0 and not all(isinstance(w, int) for w in tape.wires):
        raise ValueError(
            "qml.defer_measurements does not support custom wire labels with qubit reuse/reset."
        )

    # Apply controlled operations to store measurement outcomes and replace
    # classically controlled operations
    control_wires = {}
    cur_wire = (
        max(tape.wires) + 1 if reused_measurement_wires or any_repeated_measurements else None
    )

    for op in tape.operations:
        if isinstance(op, MidMeasureMP):
            _ = measured_wires.pop(0)

            if op.postselect is not None:
                with QueuingManager.stop_recording():
                    new_operations.append(qml.Projector([op.postselect], wires=op.wires[0]))

            # Store measurement outcome in new wire if wire gets reused
            if op.wires[0] in reused_measurement_wires or op.wires[0] in measured_wires:
                control_wires[op.id] = cur_wire

                with QueuingManager.stop_recording():
                    new_operations.append(qml.CNOT([op.wires[0], cur_wire]))

                if op.reset:
                    with QueuingManager.stop_recording():
                        # No need to manually reset if postselecting on |0>
                        if op.postselect is None:
                            new_operations.append(qml.CNOT([cur_wire, op.wires[0]]))
                        elif op.postselect == 1:
                            # We know that the measured wire will be in the |1> state if
                            # postselected |1>. So we can just apply a PauliX instead of
                            # a CNOT to reset
                            new_operations.append(qml.X(op.wires[0]))

                cur_wire += 1
            else:
                control_wires[op.id] = op.wires[0]

        elif op.__class__.__name__ == "Conditional":
            with QueuingManager.stop_recording():
                new_operations.extend(_add_control_gate(op, control_wires))
        else:
            new_operations.append(op)

    new_measurements = []

    for mp in tape.measurements:
        if mp.mv is not None:
            # Update measurement value wires. We can't use `qml.map_wires` because the same
            # wire can map to different control wires when multiple mid-circuit measurements
            # are made on the same wire. This mapping is determined by the id of the
            # MidMeasureMPs. Thus, we need to manually map wires for each MidMeasureMP.
            if isinstance(mp.mv, MeasurementValue):
                new_ms = [
                    qml.map_wires(m, {m.wires[0]: control_wires[m.id]}) for m in mp.mv.measurements
                ]
                new_m = MeasurementValue(new_ms, mp.mv.processing_fn)
            else:
                new_m = []
                for val in mp.mv:
                    new_ms = [
                        qml.map_wires(m, {m.wires[0]: control_wires[m.id]})
                        for m in val.measurements
                    ]
                    new_m.append(MeasurementValue(new_ms, val.processing_fn))

            with QueuingManager.stop_recording():
                new_mp = (
                    type(mp)(obs=new_m)
                    if not isinstance(mp, CountsMP)
                    else CountsMP(obs=new_m, all_outcomes=mp.all_outcomes)
                )
        else:
            new_mp = mp
        new_measurements.append(new_mp)

    new_tape = type(tape)(new_operations, new_measurements, shots=tape.shots)

    if is_postselecting and new_tape.batch_size is not None:
        # Split tapes if broadcasting with postselection
        return qml.transforms.broadcast_expand(new_tape)

    return [new_tape], null_postprocessing


@defer_measurements.custom_qnode_transform
def _defer_measurements_qnode(self, qnode, targs, tkwargs):
    """Custom qnode transform for ``defer_measurements``."""
    if tkwargs.get("device", None):
        raise ValueError(
            "Cannot provide a 'device' value directly to the defer_measurements decorator "
            "when transforming a QNode."
        )

    tkwargs.setdefault("device", qnode.device)
    return self.default_qnode_transform(qnode, targs, tkwargs)


def _add_control_gate(op, control_wires):
    """Helper function to add control gates"""
    control = [control_wires[m.id] for m in op.meas_val.measurements]
    new_ops = []

    for branch, value in op.meas_val._items():
        if value:
            qscript = qml.tape.make_qscript(
                ctrl(
                    lambda: qml.apply(op.then_op),  # pylint: disable=cell-var-from-loop
                    control=Wires(control),
                    control_values=branch,
                )
            )()
            new_ops.extend(qscript.circuit)
    return new_ops
