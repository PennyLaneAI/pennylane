import pennylane as qp
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import _twos_complement_helper

def semi_signed_out_multiplier(x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=False):
    """Multiplier of an unsigned register and a signed register into an unsigned output register

    Args:
        x_wires (WiresLike): register storing the unsigned integer :math:`x` to be multiplied
        y_wires (WiresLike): register storing the signed integer :math:`y` to be multiplied
        output_wires (WiresLike): register storing the unsigned integer :math:`z` before the
            calculation and the output :math:`(z+xy)\mod 2^k` afterwards, where :math:`k` is the
            number of wires in ``output_wires``.
        work_wires (WiresLike): work wires to use for the calculation.
        output_wires_zeroed (bool): Whether the ``output_wires`` start out in the state :math:`z=0`.

    This is a very specific setup of a multiplier that is useful for the 
    :func:`~.pennylane.labs.templates.trotter_vibronic` function.

    """
    y_aux, work_wires = work_wires[0], work_wires[1:]

    # Sign extension
    qp.CNOT([y_wires[0], y_aux])

    # Take 2s complement
    _twos_complement_helper(y_wires, y_aux, work_wires)

    # at this point the sign is only kept in the auxiliary qubit's state

    # Multiply the magnitudes into the output register
    # If y was negative, flip all output qubits before and after (unsigned) multiplication onto
    # the output wires. This effects that we are subtracting the product if y was negative, and
    # add it otherwise.
    qp.ctrl(qp.BasisState([1] * len(output_wires), output_wires), control=y_aux)
    qp.OutMultiplier(
        x_wires,
        y_wires,
        output_wires,
        work_wires=work_wires,
        output_wires_zeroed=output_wires_zeroed,
    )
    qp.ctrl(qp.BasisState([1] * len(output_wires), output_wires), control=y_aux)

    # Return input y to original state
    _twos_complement_helper(y_wires, y_aux, work_wires)

    # Uncompute sign extension
    qp.CNOT([y_wires[0], y_aux])

