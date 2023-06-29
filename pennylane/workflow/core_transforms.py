from asyncio.trsock import TransportSocket
import pennylane as qml
from pennylane.typing import ResultBatch, Result
from pennylane.interfaces import INTERFACE_MAP
from pennylane.transforms.core import TransformContainer


def empty_post_processing_function(results: ResultBatch) -> Result:
    return results[0]


def no_counts_with_jitting(tape):
    if any(isinstance(m, qml.measurements.CountsMP) for m in tape.measurements) and any(
        qml.math.is_abstract(a) for a in tape.get_parameters()
    ):
        raise qml.QuantumFunctionError("Can't JIT a quantum function that returns counts.")
    return (tape,), empty_post_processing_function


def qfunc_output_validation(tape):
    if not tape.measurements:
        raise qml.QuantumFunctionError(
            "A quantum function must return either a single measurement, "
            "or a nonempty sequence of measurements."
        )

    return (tape,), empty_post_processing_function


def _convert_to_interface(res, interface):
    """
    Recursively convert res to the given interface.
    """
    interface = INTERFACE_MAP[interface]

    if interface in ["Numpy"]:
        return res

    if isinstance(res, (list, tuple)):
        return type(res)(_convert_to_interface(r, interface) for r in res)

    if isinstance(res, dict):
        return {k: _convert_to_interface(v, interface) for k, v in res.items()}

    return qml.math.asarray(res, like=interface if interface != "tf" else "tensorflow")


def make_sure_output_correct_interface(tape, ml_interface="auto"):
    if ml_interface == "auto":
        return (tape,), empty_post_processing_function

    def cast_results_to_proper_interface(results: ResultBatch) -> Result:
        result = results[0]
        return _convert_to_interface(result, ml_interface)

    return (tape,), cast_results_to_proper_interface


def get_default_core_transforms(ml_interface):
    return (
        TransformContainer(no_counts_with_jitting),
        TransformContainer(qfunc_output_validation),
        TransformContainer(
            make_sure_output_correct_interface, kwargs={"ml_interface": ml_interface}
        ),
    )
