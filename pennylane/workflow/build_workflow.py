from typing import Optional

from functools import lru_cache

import pennylane as qml

from pennylane.transforms.core import TransformProgram, TransformContainer

from .executor import DeviceExecutor, TransformProgramLayer
from .interfaces import get_interface_boundary
from .gradient_layers import TransformDerivatives, DeviceDerivatives


@lru_cache
def build_workflow(
    device,
    device_config,
    ml_interface,
    gradient_method,
    gradient_kwargs: Optional[dict] = None,
    inner_program: Optional[TransformProgram] = None,
    outer_program: Optional[TransformProgram] = None,
    cache=False,
):
    inner_program = TransformProgram(inner_program) or TransformProgram()
    outer_program = TransformProgram(outer_program) or TransformProgram()

    inner_program.append(TransformContainer(qml.transforms.convert_to_numpy_parameters))
    if cache is not False:
        inner_program.append(TransformContainer(qml.transforms.cache_results, kwargs={"cache": {}}))

    device_preprocess = TransformContainer(
        device.preprocess, kwargs={"execution_config": device_config}
    )
    device_preprocess_location = (
        outer_program if device_config.use_device_gradient else inner_program
    )
    device_preprocess_location.append(device_preprocess)

    outer_executor = DeviceExecutor(device_config, device)

    outer_executor = TransformProgramLayer(outer_executor, inner_program)

    if device_config.use_device_gradient:
        derivatives_executor = DeviceDerivatives(outer_executor, device, device_config)
    elif isinstance(gradient_method, qml.gradients.gradient_transform):
        par_shift = TransformContainer(qml.gradients.param_shift, kwargs=gradient_kwargs)
        derivatives_executor = TransformDerivatives(outer_executor, par_shift)
    else:
        raise ValueError(f"do not recognize {gradient_method}")

    ml_layer_type = get_interface_boundary(ml_interface)
    outer_executor = ml_layer_type(outer_executor, derivatives_executor)

    return TransformProgramLayer(outer_executor, outer_program)
