# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Core API for registering xDSL transforms for use with PennyLane and Catalyst."""

from collections.abc import Callable
from warnings import warn

from catalyst.from_plxpr import register_transform
from xdsl.passes import ModulePass

from pennylane.transforms.core.transform_dispatcher import TransformDispatcher

from .apply_transform_sequence import register_pass


def _create_null_transform(name: str) -> Callable:
    """Create a dummy tape transform. The tape transform raises an error if used."""

    def null_transform(_):
        raise RuntimeError(
            f"Cannot apply the {name} pass without '@qml.qjit'. Additionally, program capture "
            "must be enabled using 'qml.capture.enable()'. Otherwise, the pass must be applied "
            f"using the '@catalyst.passes.apply_pass(\"{name}\")' decorator."
        )

    return null_transform


class PassDispatcher(TransformDispatcher):
    """Wrapper class for applying passes to QJIT-ed workflows."""

    name: str
    module_pass: ModulePass

    def __init__(self, module_pass: ModulePass):
        self.module_pass = module_pass
        self.name = module_pass.name
        tape_transform = _create_null_transform(self.name)
        super().__init__(tape_transform)


def compiler_transform(module_pass: ModulePass) -> PassDispatcher:
    """Wrapper function to register xDSL passes to use with QJIT-ed workflows."""
    dispatcher = PassDispatcher(module_pass)

    # Make pass visible to the xDSL universe. Passes that are created in the
    # pennylane.compiler.python_compiler.transforms namespace should already
    # be loaded, so we don't add them to the universe here. This is needed so
    # that transforms that ship with PennyLane are loaded into the universe
    # eagerly. Other passes will be loaded lazily.
    if not module_pass.__module__.startswith("pennylane.compiler.python_compiler.transforms"):
        # pylint: disable=import-outside-toplevel
        from ..universe import XDSL_UNIVERSE

        if module_pass.name in XDSL_UNIVERSE.all_passes:
            warn(
                f"The '{module_pass.name}' pass already exists, please change the pass name "
                "to register as a compiler transform",
                UserWarning,
            )
        XDSL_UNIVERSE.all_passes[module_pass.name] = module_pass

    # Registration to map from plxpr primitive to pass
    register_transform(dispatcher, module_pass.name, False)

    # Registration for apply-transform-sequence interpreter
    def get_pass_cls():
        return module_pass

    register_pass(module_pass.name, get_pass_cls)
    return dispatcher
