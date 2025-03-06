def _consolidate_conditional_measurements(ops):
    # this should occur before general diagonalization, while the measurements are still grouped together

    new_operations = []
    mps_mapping = {}

    curr_idx = 0

    for i, op in enumerate(ops):

        if i != curr_idx:
            continue

        if isinstance(op, qml.ops.Conditional):

            # from MCM mapping, map any MCMs in the condition if needed
            processing_fn = op.meas_val.processing_fn
            mps = [mps_mapping.get(op, op) for op in op.meas_val.measurements]
            expr = MeasurementValue(mps, processing_fn=processing_fn)

            if isinstance(op.base, qml.measurements.MidMeasureMP):
                # in core PennyLane with tapes, we can assume that a conditional has only a
                # true_fn and false_fn, so we make that assumption in the PL tape transform
                # for Catalyst and ProgramCapture, it is also possible to have additional elif functions
                true_cond, false_cond = (op, ops[i + 1] if i < len(ops) else None)
                _validate_false_cond(true_cond, false_cond)
                curr_idx += 1

                # add conditional diagonalizing gates + conditional MCM to the tape
                with qml.QueuingManager.stop_recording():
                    for op in [true_cond, false_cond]:
                        diag_gates = [
                            qml.ops.Conditional(expr=expr, then_op=gate)
                            for gate in op.diagonalizing_gates()
                        ]

                        new_operations.extend(diag_gates)

                    new_mp = MidMeasureMP(
                        op.wires, reset=op.base.reset, postselect=op.base.postselect, id=op.base.id
                    )

                # track mapping from original to computational basis MCMs
                new_operations.append(new_mp)
                mps_mapping[op.base] = new_mp
                curr_idx += 1

            else:
                with qml.QueuingManager.stop_recording():
                    new_cond = qml.ops.Conditional(expr=expr, then_op=op.base)
                new_operations.append(new_cond)
                curr_idx += 1

        else:
            new_operations.append(op)
            curr_idx += 1

    return new_operations