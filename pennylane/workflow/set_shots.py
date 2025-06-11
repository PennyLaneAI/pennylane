import functools
from typing import Any, Callable, Optional, Sequence, Union

from pennylane.measurements import Shots

from .qnode import QNode


def set_shots(qnode_or_function: QNode = None, *, shots=None):
    """Transform used to set or update a circuit's shots.

    Args:
        tape (QuantumScript): The quantum circuit to be modified.
        shots (None or int or Sequence[int] or Sequence[tuple[int, int]] or pennylane.shots.Shots): The
            number of shots (or a shots vector) that the transformed circuit will execute.
            This specification will override any shots value previously associated
            with the circuit or QNode during execution.

    Returns:
        tuple[List[QuantumScript], function]: The transformed circuit as a batch of tapes and a
        post-processing function, as described in :func:`qml.transform <pennylane.transform>`. The output
        tape(s) will have their ``shots`` attribute set to the value provided in the ``shots`` argument.

    There are three ways to specify shot values (see :func:`qml.measurements.Shots <pennylane.measurements.Shots>` for more details):

    * The value ``None``: analytic mode, no shots
    * A positive integer: a fixed number of shots
    * A sequence consisting of either positive integers or a tuple-pair of positive integers of the form ``(shots, copies)``

    **Examples**

    Set the number of shots as a decorator:

    .. code-block:: python

        from functools import partial

        @partial(qml.set_shots, shots=2)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.sample(qml.Z(0))

    Run the circuit:

    >>> circuit()
    array([1., -1.])

    Update the shots in-line for an existing circuit:

    >>> new_circ = qml.set_shots(circuit, shots=(4, 10)) # shot vector
    >>> new_circ()
    (array([-1.,  1., -1.,  1.]), array([ 1.,  1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1.]))

    """
    # When called directly without arguments
    if qnode_or_function is None:
        # Return a decorator that will apply shots when called
        def decorator(obj):
            # If applied to a QNode
            if hasattr(obj, "update_shots"):
                return obj.update_shots(shots)
            # If applied to a function that returns a QNode
            elif callable(obj):

                @functools.wraps(obj)
                def wrapper(*args, **kwargs):
                    result = obj(*args, **kwargs)
                    if hasattr(result, "update_shots"):
                        return result.update_shots(shots)
                    return result

                return wrapper
            else:
                raise ValueError(
                    "set_shots can only be applied to QNodes or functions that return QNodes"
                )

        return decorator

    # When called directly with a function/QNode
    if hasattr(qnode_or_function, "update_shots"):
        return qnode_or_function.update_shots(shots)
    elif callable(qnode_or_function):

        @functools.wraps(qnode_or_function)
        def wrapper(*args, **kwargs):
            result = qnode_or_function(*args, **kwargs)
            if hasattr(result, "update_shots"):
                return result.update_shots(shots)
            return result

        return wrapper
    else:
        raise ValueError("set_shots can only be applied to QNodes or functions that return QNodes")
