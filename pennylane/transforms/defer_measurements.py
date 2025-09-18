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

from collections.abc import Callable, Sequence
from functools import lru_cache, partial
from numbers import Number
from warnings import warn

import pennylane as qml
from pennylane.exceptions import TransformError
from pennylane.measurements import (
    CountsMP,
    MeasurementValue,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    get_mcm_predicates,
)
from pennylane.ops.op_math import ctrl
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

# pylint: disable=too-many-branches, protected-access, too-many-statements


def _check_tape_validity(tape: QuantumScript):
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
            mp.obs or mp._wires or mp.mv is not None
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


def _collect_mid_measure_info(tape: QuantumScript):
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


@lru_cache
def _get_plxpr_defer_measurements():
    try:
        # pylint: disable=import-outside-toplevel
        import jax

        from pennylane.capture import PlxprInterpreter
        from pennylane.capture.primitives import cond_prim, ctrl_transform_prim, measure_prim
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name

    class DeferMeasurementsInterpreter(PlxprInterpreter):
        """Interpreter for applying the defer_measurements transform to plxpr."""

        # pylint: disable=attribute-defined-outside-init,no-self-use

        def __init__(self, num_wires):
            super().__init__()
            self._num_wires = num_wires

            # We use a dict here instead of a normal int variable because we want the state to mutate
            # when we interpret higher-order primitives
            # We store all used wires rather than just the max because if the wires are tracers, then
            # we can't do comparisons to find the max wire
            self.state = {"cur_target": num_wires - 1, "used_wires": set()}

        def cleanup(self) -> None:
            """Perform any final steps after iterating through all equations.

            This method resets the internal ``state``, specifically the ``cur_idx`` entry.
            ``cur_idx`` tracks the index within the auxiliary wires, determining which will
            be used for the target wire of the next mid-circuit measurement's replacement
            :class:`~pennylane.CNOT`.
            """
            self.state = {"cur_target": self._num_wires - 1, "used_wires": set()}

        def _update_used_wires(self, wires: qml.wires.Wires, cur_target: int):
            """Update the state with the number of wires that have been used and validate that
            there is no overlap between the used circuit wires and the mid-circuit measurement
            target wires.

            Args:
                wires (pennylane.wires.Wires): wires to add to the set of used wires
                cur_target (int): target wire to be used for a mid-circuit measurement

            Raises:
                TransformError: if there is an overlap between the used circuit wires and mid-circuit
                measurement target wires
            """
            self.state["used_wires"] |= wires.toset()
            if self.state["used_wires"].intersection(range(cur_target, self._num_wires)):
                raise TransformError(
                    "Too many mid-circuit measurements for the specified number of wires "
                    "with 'defer_measurements'."
                )

        def interpret_dynamic_operation(self, data, struct, inds):
            """Interpret an operation that uses mid-circuit measurement outcomes as parameters.

            * This will not work if mid-circuit measurement values are used to specify
              operator wires.
            * This will not work if more than one parameter uses mid-circuit measurement values.

            Args:
                data (TensorLike): Flattened data of the operator
                struct (PyTreeDef): Pytree structure of the operator
                inds (Sequence[int]): Indices of mid-circuit measurement values in ``data``

            Raises:
                TransformError: if there is an overlap between the used circuit wires and mid-circuit
                measurement target wires
            """
            if len(inds) > 1:
                raise TransformError(
                    "Cannot create operations with multiple parameters based on "
                    "mid-circuit measurements with 'defer_measurements'."
                )

            idx = inds[0]
            mv = data[idx]
            for branch, value in mv.items():
                data[idx] = value
                op = jax.tree_util.tree_unflatten(struct, data)
                qml.ctrl(op, mv.wires, control_values=branch)

        def interpret_operation(self, op: "qml.operation.Operator"):
            """Interpret a PennyLane operation instance.

            Args:
                op (Operator): a pennylane operator instance

            Returns:
                Any

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            See also: :meth:`~.interpret_operation_eqn`.

            """
            # Range for comparison is [cur_target + 1, num_wires) because cur_target
            # is the _next_ wire to be used for an MCM. We want to check if the used
            # wires overlap with the already applied MCMs.
            self._update_used_wires(op.wires, self.state["cur_target"] + 1)

            # We treat operators with operators based on mid-circuit measurement values
            # separately, and otherwise default to the standard behaviour
            data, struct = jax.tree_util.tree_flatten(op)
            mcm_data_inds = []
            for i, d in enumerate(data):
                if isinstance(d, MeasurementValue):
                    mcm_data_inds.append(i)

            if mcm_data_inds:
                return self.interpret_dynamic_operation(data, struct, mcm_data_inds)

            return jax.tree_util.tree_unflatten(struct, data)

        def interpret_measurement(self, measurement: "qml.measurement.MeasurementProcess"):
            """Interpret a measurement process instance.

            Args:
                measurement (MeasurementProcess): a measurement instance.

            See also :meth:`~.interpret_measurement_eqn`.

            """
            if measurement.mv is not None:
                kwargs = {"wires": measurement.wires}
                if isinstance(measurement.mv, MeasurementValue):
                    kwargs["eigvals"] = measurement.eigvals()
                if isinstance(measurement, CountsMP):
                    kwargs["all_outcomes"] = measurement.all_outcomes
                measurement = type(measurement)(**kwargs)

            else:
                # Range for comparison is [cur_target + 1, num_wires) because cur_target
                # is the _next_ wire to be used for an MCM. We want to check if the used
                # wires overlap with the already applied MCMs.
                self._update_used_wires(measurement.wires, self.state["cur_target"] + 1)

            return super().interpret_measurement(measurement)

        def resolve_mcm_values(
            self,
            primitive: "jax.extend.core.Primitive",
            subfuns: Sequence[Callable],
            invals: Sequence[MeasurementValue | Number],
            params: dict,
        ) -> MeasurementValue:
            """Create a ``MeasurementValue`` that captures all classical processing of the
            input ``eqn`` in its ``processing_fn``.

            Args:
                primitive (jax.extend.core.Primitive): Jax primitive
                subfuns (Sequence[Callable]): Callable positional arguments to the primitive.
                    These are created by pre-processing jaxpr equation parameters.
                invals (Sequence[Union[MeasurementValue, Number]]): Inputs to the primitive
                params (dict): Keyword arguments to the primitive

            Returns:
                MeasurementValue: ``MeasurementValue`` containing classical processing information
                for applying the input equation to mid-circuit measurement outcomes.
            """
            # pylint: disable=protected-access
            # MeasurementValue._apply is for applying a new operation to the current
            # MeasurementValue
            # It is used when either:
            # 1. A unary operation is performed on one MeasurementValue, e.g., ~m0
            # 2. A binary operation is performed on a MeasurementValue and a scalar,
            #    e.g., m0 + 1
            #
            # MeasurementValue._transform_bin_op is for applying a binary operation
            # to two MeasurementValues
            assert len(invals) <= 2
            processing_fn = partial(primitive.bind, *subfuns, **params)

            # One MeasurementValue
            if len(invals) == 1:
                m0 = invals[0]
                return m0._apply(processing_fn)

            # Two MeasurementValues
            if all(isinstance(inval, MeasurementValue) for inval in invals):
                m0, m1 = invals
                return m0._transform_bin_op(processing_fn, m1)

            # One MeasurementValue, one number
            [m0, other] = invals if isinstance(invals[0], MeasurementValue) else invals[::-1]
            return m0._apply(lambda x: processing_fn(x, other))

        def eval(self, jaxpr: "jax.extend.core.Jaxpr", consts: list, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.extend.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.

            """
            self._env = {}
            self.setup()

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env[constvar] = const

            for eqn in jaxpr.eqns:
                primitive = eqn.primitive
                custom_handler = self._primitive_registrations.get(primitive, None)

                if custom_handler:
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif getattr(primitive, "prim_type", "") == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif getattr(primitive, "prim_type", "") == "measurement":
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    subfuns, params = primitive.get_bind_params(eqn.params)
                    if any(isinstance(inval, MeasurementValue) for inval in invals):
                        outvals = self.resolve_mcm_values(primitive, subfuns, invals, params)
                    else:
                        outvals = primitive.bind(*subfuns, *invals, **params)

                if not primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                    self._env[outvar] = outval

            # Read the final result of the Jaxpr from the environment
            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, qml.operation.Operator):
                    outvals.append(self.interpret_operation(outval))
                else:
                    outvals.append(outval)
            self.cleanup()
            self._env = {}
            return outvals

    @DeferMeasurementsInterpreter.register_primitive(measure_prim)
    def _(self, wires, reset=False, postselect=None):
        cur_target = self.state["cur_target"]
        # Range for comparison is [cur_target, num_wires) because cur_target
        # is the _current_ wire to be used for an MCM.
        self._update_used_wires(Wires(wires), cur_target)

        # Using type.__call__ instead of normally constructing the class prevents
        # the primitive corresponding to the class to get binded. We do not want the
        # MidMeasureMP's primitive to get recorded.
        meas = type.__call__(
            MidMeasureMP, Wires(cur_target), reset=reset, postselect=postselect, id=str(cur_target)
        )

        cnot_wires = (wires, cur_target)
        if postselect is not None:
            qml.Projector(jax.numpy.array([postselect]), wires=wires)

        qml.CNOT(wires=cnot_wires)
        if reset:
            if postselect is None:
                qml.CNOT(wires=cnot_wires[::-1])
            elif postselect == 1:
                qml.PauliX(wires=wires)

        self.state["cur_target"] -= 1
        return MeasurementValue([meas])

    @DeferMeasurementsInterpreter.register_primitive(cond_prim)
    def _(self, *invals, jaxpr_branches, consts_slices, args_slice):
        n_branches = len(jaxpr_branches)
        conditions = invals[:n_branches]
        if not any(isinstance(c, MeasurementValue) for c in conditions):
            return PlxprInterpreter._primitive_registrations[cond_prim](
                self,
                *invals,
                jaxpr_branches=jaxpr_branches,
                consts_slices=consts_slices,
                args_slice=args_slice,
            )

        conditions = get_mcm_predicates(conditions[:-1])
        args = invals[args_slice]

        for i, (condition, jaxpr) in enumerate(zip(conditions, jaxpr_branches, strict=True)):

            if isinstance(condition, MeasurementValue):
                control_wires = Wires([m.wires[0] for m in condition.measurements])

                for branch, value in condition.items():
                    # When reduce_postselected is True, some branches can be ()
                    cur_consts = invals[consts_slices[i]]
                    qml.cond(value, ctrl_transform_prim.bind)(
                        *cur_consts,
                        *args,
                        *control_wires,
                        jaxpr=jaxpr,
                        n_control=len(control_wires),
                        control_values=branch,
                        work_wires=None,
                        n_consts=len(cur_consts),
                    )

        return [None] * len(jaxpr_branches[0].outvars)

    def defer_measurements_plxpr_to_plxpr(jaxpr, consts, targs, tkwargs, *args):
        """Function for applying the ``defer_measurements`` transform on plxpr."""

        if not tkwargs.get("num_wires", None):
            raise ValueError(
                "'num_wires' argument for qml.defer_measurements must be provided "
                "when qml.capture.enabled() is True."
            )
        if tkwargs.pop("reduce_postselected", False):
            warn(
                "Cannot set 'reduce_postselected=True' with qml.capture.enabled() "
                "when using qml.defer_measurements. Argument will be ignored.",
                UserWarning,
            )
        if tkwargs.pop("allow_postselect", False):
            warn(
                "Cannot set 'allow_postselect=True' with qml.capture.enabled() "
                "when using qml.defer_measurements. Argument will be ignored.",
                UserWarning,
            )

        interpreter = DeferMeasurementsInterpreter(*targs, **tkwargs)

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return jax.make_jaxpr(wrapper)(*args)

    return DeferMeasurementsInterpreter, defer_measurements_plxpr_to_plxpr


DeferMeasurementsInterpreter, defer_measurements_plxpr_to_plxpr = _get_plxpr_defer_measurements()


# pylint: disable=unused-argument
@partial(transform, plxpr_transform=defer_measurements_plxpr_to_plxpr)
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
            transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
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

    Reusing and resetting measured wires will work as expected with the
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

    .. details::
        :title: Usage Details

        By default, ``defer_measurements`` makes use of postselection information of
        mid-circuit measurements in the circuit in order to reduce the number of controlled
        operations and control wires. We can explicitly switch this feature off and compare
        the created circuits with and without this optimization. Consider the following circuit:

        .. code-block:: python3

            @qml.qnode(qml.device("default.qubit"))
            def node(x):
                qml.RX(x, 0)
                qml.RX(x, 1)
                qml.RX(x, 2)

                mcm0 = qml.measure(0, postselect=0, reset=False)
                mcm1 = qml.measure(1, postselect=None, reset=True)
                mcm2 = qml.measure(2, postselect=1, reset=False)
                qml.cond(mcm0+mcm1+mcm2==1, qml.RX)(0.5, 3)
                return qml.expval(qml.Z(0) @ qml.Z(3))

        Without the optimization, we find three gates controlled on the three measured
        qubits. They correspond to the combinations of controls that satisfy the condition
        ``mcm0+mcm1+mcm2==1``.

        >>> print(qml.draw(qml.defer_measurements(node, reduce_postselected=False))(0.6))
        0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────────────────────────────────┤ ╭<Z@Z>
        1: ──RX(0.60)─────────│──╭●─╭X───────────────────────────────────────┤ │
        2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|─╭○────────╭○────────╭●────────┤ │
        3: ───────────────────│──│──│──────────├RX(0.50)─├RX(0.50)─├RX(0.50)─┤ ╰<Z@Z>
        4: ───────────────────╰X─│──│──────────├○────────├●────────├○────────┤
        5: ──────────────────────╰X─╰●─────────╰●────────╰○────────╰○────────┤

        If we do not explicitly deactivate the optimization, we obtain a much simpler circuit:

        >>> print(qml.draw(qml.defer_measurements(node))(0.6))
        0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────┤ ╭<Z@Z>
        1: ──RX(0.60)─────────│──╭●─╭X───────────┤ │
        2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|───┤ │
        3: ───────────────────│──│──│──╭RX(0.50)─┤ ╰<Z@Z>
        4: ───────────────────╰X─│──│──│─────────┤
        5: ──────────────────────╰X─╰●─╰○────────┤

        There is only one controlled gate with only one control wire.

    .. details::
        :title: Deferred measurements with program capture

        ``qml.defer_measurements`` can be applied to callables when program capture is enabled. To do so,
        the ``num_wires`` argument must be provided, which should be an integer corresponding to the total
        number of available wires. For ``m`` mid-circuit measurements, ``range(num_wires - m, num_wires)``
        will be the range of wires used to map mid-circuit measurements to ``CNOT`` gates.

        .. warning::

            While the transform includes validation to avoid overlap between wires of the original
            circuit and mid-circuit measurement target wires, if any wires of the original circuit
            are traced, i.e. dependent on dynamic arguments to the transformed workflow, the
            validation may not catch overlaps. Consider the following example:

            .. code-block:: python

                from functools import partial
                import jax

                qml.capture.enable()

                @qml.capture.expand_plxpr_transforms
                @partial(qml.defer_measurements, num_wires=1)
                def f(n):
                    qml.measure(n)

            >>> jax.make_jaxpr(f)(0)
            { lambda ; a:i64[]. let _:AbstractOperator() = CNOT[n_wires=2] a 0 in () }

            The circuit gets transformed without issue because the concrete value of the measured wire
            is unknown. However, execution with n = 0 would raise an error, as the CNOT wires would
            be (0, 0).

            Thus, users must be cautious when transforming a circuit. **For n total wires and
            c circuit wires, the number of mid-circuit measurements allowed is n - c.**

        Using ``defer_measurements`` with program capture enabled introduces new features and
        restrictions:

        **New features**

        * Arbitrary classical processing of mid-circuit measurement values is now possible. With
          program capture disabled, only limited classical processing, as detailed in the
          documentation for :func:`~pennylane.measure`. With program capture enabled, any unary
          or binary ``jax.numpy`` functions that can be applied to scalars can be used with mid-circuit
          measurements.

        * Using mid-circuit measurements as gate parameters is now possible. This feature currently
          has the following restrictions. First, mid-circuit measurement values cannot be used
          for multiple parameters of the same gate. Second, mid-circuit measurement values
          cannot be used as wires.

          .. code-block:: python

              from functools import partial
              import jax
              import jax.numpy as jnp

              qml.capture.enable()

              @qml.capture.expand_plxpr_transforms
              @partial(qml.defer_measurements, num_wires=10)
              def f():
                  m0 = qml.measure(0)

                  phi = jnp.sin(jnp.pi * m0)
                  qml.RX(phi, 0)
                  return qml.expval(qml.PauliZ(0))

          >>> jax.make_jaxpr(f)()
          { lambda ; . let
              _:AbstractOperator() = CNOT[n_wires=2] 0 9
              a:f64[] = mul 0.0 3.141592653589793
              b:f64[] = sin a
              c:AbstractOperator() = RX[n_wires=1] b 0
              _:AbstractOperator() = Controlled[
                control_values=(False,)
                work_wires=Wires([])
              ] c 9
              d:f64[] = mul 1.0 3.141592653589793
              e:f64[] = sin d
              f:AbstractOperator() = RX[n_wires=1] e 0
              _:AbstractOperator() = Controlled[
                control_values=(True,)
                work_wires=Wires([])
              ] f 9
              g:AbstractOperator() = PauliZ[n_wires=1] 0
              h:AbstractMeasurement(n_wires=None) = expval_obs g
            in (h,) }

        The above dummy example showcases how the transform is applied when the aforementioned
        features are used.

        **What doesn't work**

        * mid-circuit measurement values cannot be used in the condition for a
          :func:`~pennylane.while_loop`.
        * :func:`~pennylane.measure` cannot be used inside the body of loop primitives
          (:func:`~pennylane.while_loop`, :func:`~pennylane.for_loop`).
        * If a branch of :func:`~pennylane.cond` uses mid-circuit measurements as its
          predicate, then all other branches must also use mid-circuit measurement values
          as predicates.
        * For an ``n``-parameter gate, mid-circuit measurement values can only be used
          for 1 of the ``n`` parameters.
        * :func:`~pennylane.measure` can only be used in the bodies of branches of
          :func:`~pennylane.cond` if none of the branches use mid-circuit measurements
          as predicates
        * :func:`~pennylane.measure` cannot be used inside the body of functions
          being transformed with :func:`~pennylane.adjoint` or :func:`~pennylane.ctrl`.
    """
    if not any(isinstance(o, MidMeasureMP) for o in tape.operations):
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
                new_operations.extend(_add_control_gate(op, control_wires, reduce_postselected))
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
                new_m = MeasurementValue(
                    new_ms, mp.mv.processing_fn if mp.mv.has_processing else None
                )
            else:
                new_m = []
                for val in mp.mv:
                    new_ms = [
                        qml.map_wires(m, {m.wires[0]: control_wires[m.id]})
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
        return qml.transforms.broadcast_expand(new_tape)

    return [new_tape], null_postprocessing


def _add_control_gate(op, control_wires, reduce_postselected):
    """Helper function to add control gates"""
    if reduce_postselected:
        control = [control_wires[m.id] for m in op.meas_val.measurements if m.postselect is None]
        items = op.meas_val.postselected_items()
    else:
        control = [control_wires[m.id] for m in op.meas_val.measurements]
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
