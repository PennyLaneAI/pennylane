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

from functools import partial

import pennylane as qp
from pennylane.core.qscript import QuantumScript, QuantumScriptBatch
from pennylane.core.queuing import QueuingManager
from pennylane.measurements import (
    CountsMP,
    ProbabilityMP,
    SampleMP,
)
from pennylane.ops.mid_measure import MeasurementValue, MidMeasure
from pennylane.ops.op_math import ctrl
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

# pylint: disable=too-many-branches, protected-access, too-many-statements


def _check_tape_validity(tape: QuantumScript):
    """Helper function to check that the tape is valid."""
    cv_types = (qp.operation.CVOperation, qp.operation.CVObservable)
    ops_cv = any(isinstance(op, cv_types) and op.name != "Identity" for op in tape.operations)
    obs_cv = any(
        isinstance(getattr(op, "obs", None), cv_types)
        and not isinstance(getattr(op, "obs", None), qp.Identity)
        for op in tape.measurements
    )
    if ops_cv or obs_cv:
        raise ValueError("Continuous variable operations and observables are not supported.")

    for mp in tape.measurements:
        if isinstance(mp, (CountsMP, ProbabilityMP, SampleMP)) and not (
            mp.obs or mp._wires or mp.mv is not None
        ):
            raise ValueError(
                f"Cannot use {mp.__class__.__name__} as a measurement without specifying wires "
                "when using qp.defer_measurements. Deferred measurements can occur "
                "automatically when using mid-circuit measurements on a device that does not "
                "support them."
            )

        if mp.__class__.__name__ == "StateMP":
            raise ValueError(
                "Cannot use StateMP as a measurement when using qp.defer_measurements. "
                "Deferred measurements can occur automatically when using mid-circuit "
                "measurements on a device that does not support them."
            )

    samples_present = any(isinstance(mp, SampleMP) for mp in tape.measurements)
    postselect_present = any(
        op.postselect is not None for op in tape.operations if isinstance(op, MidMeasure)
    )
    if postselect_present and samples_present and tape.batch_size is not None:
        raise ValueError(
            "Returning qp.sample is not supported when postselecting mid-circuit "
            "measurements with broadcasting"
        )


def _collect_mid_measure_info(tape: QuantumScript):
    """Helper function to collect information related to mid-circuit measurements in the tape."""

    # Find wires that are reused after measurement
    measured_wires = []
    reused_measurement_wires = set()
    any_repeated_measurements = False
    is_postselecting = False

    for op in tape:
        if isinstance(op, MidMeasure):
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


# pylint: disable=unused-argument
@partial(transform)
def defer_measurements(
    tape: QuantumScript,
    reduce_postselected: bool = True,
    allow_postselect: bool = True,
    num_wires: int | None = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
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

        Devices that inherit from :class:`~pennylane.devices.QubitDevice` **must** be initialized
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

    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit.
        reduce_postselected (bool): Whether to use postselection information to reduce the number
            of operations and control wires in the output tape. Active by default. This is currently
            ignored if program capture is enabled.
        allow_postselect (bool): Whether postselection is allowed. In order to perform postselection
            with ``defer_measurements``, the device must support the :class:`~.Projector` operation.
            Defaults to ``True``. This is currently ignored if program capture is enabled.
        num_wires (int): Optional argument to specify the total number of circuit wires. This is
            only used if program capture is enabled.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
            transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    Raises:
        ValueError: If any measurements with no wires or observable are present
        ValueError: If continuous variable operations or measurements are present
        ValueError: If using the transform with any device other than
            :class:`default.qubit <~pennylane.devices.DefaultQubit>` and postselection is used

    **Example**

    Suppose we have a quantum function with mid-circuit measurements and
    conditional operations:

    .. code-block:: python

        def qfunc(par):
            qp.RY(0.123, wires=0)
            qp.Hadamard(wires=1)
            m_0 = qp.measure(1)
            qp.cond(m_0, qp.RY)(par, wires=0)
            return qp.expval(qp.Z(0))

    The ``defer_measurements`` transform allows executing such quantum
    functions without having to perform mid-circuit measurements:

    >>> dev = qp.device('default.qubit', wires=2)
    >>> transformed_qfunc = qp.defer_measurements(qfunc)
    >>> qnode = qp.QNode(transformed_qfunc, dev)
    >>> par = pnp.array(np.pi/2, requires_grad=True)
    >>> qnode(par)
    tensor(0.434..., requires_grad=True)

    We can also differentiate parameters passed to conditional operations:

    >>> qp.grad(qnode)(par)
    tensor(-0.496... requires_grad=True)

    Reusing and resetting measured wires will work as expected with the
    ``defer_measurements`` transform:

    .. code-block:: python

        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def func(x, y):
            qp.RY(x, wires=0)
            qp.CNOT(wires=[0, 1])
            m_0 = qp.measure(1, reset=True)

            qp.cond(m_0, qp.RY)(y, wires=0)
            qp.RX(np.pi/4, wires=1)
            return qp.probs(wires=[0, 1])

    Executing this QNode:

    >>> pars = pnp.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.769..., 0.132..., 0.0839..., 0.014...], requires_grad=True)

    .. details::
        :title: Usage Details

        By default, ``defer_measurements`` makes use of postselection information of
        mid-circuit measurements in the circuit in order to reduce the number of controlled
        operations and control wires. We can explicitly switch this feature off and compare
        the created circuits with and without this optimization. Consider the following circuit:

        .. code-block:: python

            @qp.qnode(qp.device("default.qubit"))
            def node(x):
                qp.RX(x, 0)
                qp.RX(x, 1)
                qp.RX(x, 2)

                mcm0 = qp.measure(0, postselect=0, reset=False)
                mcm1 = qp.measure(1, postselect=None, reset=True)
                mcm2 = qp.measure(2, postselect=1, reset=False)
                qp.cond(mcm0+mcm1+mcm2==1, qp.RX)(0.5, 3)
                return qp.expval(qp.Z(0) @ qp.Z(3))

        Without the optimization, we find three gates controlled on the three measured
        qubits. They correspond to the combinations of controls that satisfy the condition
        ``mcm0+mcm1+mcm2==1``.

        >>> print(qp.draw(qp.defer_measurements(node, reduce_postselected=False))(0.6)) # doctest: +SKIP
        0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────────────────────────────────┤ ╭<Z@Z>
        1: ──RX(0.60)─────────│──╭●─╭X───────────────────────────────────────┤ │
        2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|─╭○────────╭●────────╭○────────┤ │
        3: ───────────────────│──│──│──────────├RX(0.50)─├RX(0.50)─├RX(0.50)─┤ ╰<Z@Z>
        4: ───────────────────╰X─│──│──────────├○────────├○────────├●────────┤
        5: ──────────────────────╰X─╰●─────────╰●────────╰○────────╰○────────┤

        If we do not explicitly deactivate the optimization, we obtain a much simpler circuit:

        >>> print(qp.draw(qp.defer_measurements(node))(0.6))
        0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────┤ ╭<Z@Z>
        1: ──RX(0.60)─────────│──╭●─╭X───────────┤ │
        2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|───┤ │
        3: ───────────────────│──│──│──╭RX(0.50)─┤ ╰<Z@Z>
        4: ───────────────────╰X─│──│──│─────────┤
        5: ──────────────────────╰X─╰●─╰○────────┤

        There is only one controlled gate with only one control wire.

    """
    if not any(isinstance(o, MidMeasure) for o in tape.operations):
        return (tape,), null_postprocessing

    _check_tape_validity(tape)

    new_operations = []

    # Find wires that are reused after measurement
    (
        measured_wires,
        reused_measurement_wires,
        any_repeated_measurements,
        is_postselecting,
    ) = _collect_mid_measure_info(tape)

    if is_postselecting and not allow_postselect:
        raise ValueError(
            "Postselection is not allowed on the device with deferred measurements. The device "
            "must support the Projector gate to apply postselection."
        )

    integer_wires = [w for w in tape.wires if isinstance(w, int)]

    # Apply controlled operations to store measurement outcomes and replace
    # classically controlled operations
    control_wires = {}
    cur_wire = (
        (max(integer_wires) + 1 if integer_wires else 0)
        if reused_measurement_wires or any_repeated_measurements
        else None
    )

    for op in tape.operations:
        if isinstance(op, MidMeasure):
            _ = measured_wires.pop(0)

            if op.postselect is not None:
                with QueuingManager.stop_recording():
                    new_operations.append(qp.Projector([op.postselect], wires=op.wires[0]))

            # Store measurement outcome in new wire if wire gets reused
            if op.wires[0] in reused_measurement_wires or op.wires[0] in measured_wires:
                control_wires[op.meas_uid] = cur_wire

                with QueuingManager.stop_recording():
                    new_operations.append(qp.CNOT([op.wires[0], cur_wire]))

                if op.reset:
                    with QueuingManager.stop_recording():
                        # No need to manually reset if postselecting on |0>
                        if op.postselect is None:
                            new_operations.append(qp.CNOT([cur_wire, op.wires[0]]))
                        elif op.postselect == 1:
                            # We know that the measured wire will be in the |1> state if
                            # postselected |1>. So we can just apply a PauliX instead of
                            # a CNOT to reset
                            new_operations.append(qp.X(op.wires[0]))

                cur_wire += 1
            else:
                control_wires[op.meas_uid] = op.wires[0]

        elif op.__class__.__name__ == "Conditional":
            with QueuingManager.stop_recording():
                new_operations.extend(_add_control_gate(op, control_wires, reduce_postselected))
        else:
            new_operations.append(op)

    new_measurements = []

    for mp in tape.measurements:
        if mp.mv is not None:
            # Update measurement value wires. We can't use `qp.map_wires` because the same
            # wire can map to different control wires when multiple mid-circuit measurements
            # are made on the same wire. This mapping is determined by the id of the
            # MidMeasures. Thus, we need to manually map wires for each MidMeasure.
            if isinstance(mp.mv, MeasurementValue):
                new_ms = [
                    qp.map_wires(m, {m.wires[0]: control_wires[m.meas_uid]})
                    for m in mp.mv.measurements
                ]
                new_m = MeasurementValue(
                    new_ms, mp.mv.processing_fn if mp.mv.has_processing else None
                )
            else:
                new_m = []
                for val in mp.mv:
                    new_ms = [
                        qp.map_wires(m, {m.wires[0]: control_wires[m.meas_uid]})
                        for m in val.measurements
                    ]
                    new_m.append(
                        MeasurementValue(new_ms, val.processing_fn if val.has_processing else None)
                    )

            with QueuingManager.stop_recording():
                new_mp = (
                    type(mp)(obs=new_m)
                    if not isinstance(mp, CountsMP)
                    else CountsMP(obs=new_m, all_outcomes=mp.all_outcomes)
                )
        else:
            new_mp = mp
        new_measurements.append(new_mp)

    new_tape = tape.copy(operations=new_operations, measurements=new_measurements)

    if is_postselecting and new_tape.batch_size is not None:
        # Split tapes if broadcasting with postselection
        return qp.transforms.broadcast_expand(new_tape)

    return [new_tape], null_postprocessing


def _add_control_gate(op, control_wires, reduce_postselected):
    """Helper function to add control gates"""
    if reduce_postselected:
        control = [
            control_wires[m.meas_uid] for m in op.meas_val.measurements if m.postselect is None
        ]
        items = op.meas_val.postselected_items()
    else:
        control = [control_wires[m.meas_uid] for m in op.meas_val.measurements]
        items = op.meas_val.items()

    new_ops = []

    for branch, value in items:
        if value:
            # Empty sampling branches can occur when using _postselected_items
            new_op = (
                op.base
                if branch == ()
                else ctrl(op.base, control=Wires(control), control_values=branch)
            )
            new_ops.append(new_op)
    return new_ops
