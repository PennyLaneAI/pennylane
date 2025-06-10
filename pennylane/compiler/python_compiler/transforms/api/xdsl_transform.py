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

from typing import Callable

from catalyst.from_plxpr import register_transform
from xdsl.passes import ModulePass

from pennylane.transforms.core.transform_dispatcher import TransformDispatcher

from .apply_transform_sequence import register_pass


def _create_null_transform(name: str) -> Callable:
    """Create a dummy tape transform. The tape transform raises an error if used."""

    def null_transform(_):
        raise RuntimeError(f"Cannot apply the {name} pass without 'qml.qjit'.")

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

    # def __call__(self, *args, **kwargs) -> Callable:
    #     fn = None

    #     if args:
    #         fn = args[0]
    #         args = args[1:]

    #     if not enabled:
    #         raise RuntimeError(
    #             "Program capture must be enabled using 'qml.capture.enable()' to apply Python "
    #             "compiler passes."
    #         )
    #     if not callable(fn):
    #         raise RuntimeError(f"Cannot apply Python compiler passes to {type(fn)}.")

    #     wrapper = self._create_workflow_wrapper(fn, pass_args, pass_kwargs)
    #     return wrapper

    # def _create_workflow_wrapper(
    #     self, fn: Callable, pass_args: tuple[Any], pass_kwargs: dict[str, Any]
    # ) -> Callable:
    #     """Return a wrapper that applies the transform to the workflow."""

    #     @wraps(fn)
    #     def wrapper_fn(*args, **kwargs):
    #         import jax  # pylint: disable=import-outside-toplevel

    #         flat_fn = FlatFn(fn)
    #         jaxpr = jax.make_jaxpr(partial(flat_fn, **kwargs))(*args)
    #         flat_args = jax.tree_util.tree_leaves(args)

    #         n_args = len(flat_args)
    #         n_consts = len(jaxpr.consts)
    #         args_slice = slice(0, n_args)
    #         consts_slice = slice(n_args, n_args + n_consts)
    #         targs_slice = slice(n_args + n_consts, None)

    #         results = self._primitive.bind(
    #             *flat_args,
    #             *jaxpr.consts,
    #             *pass_args,
    #             inner_jaxpr=jaxpr.jaxpr,
    #             args_slice=args_slice,
    #             consts_slice=consts_slice,
    #             targs_slice=targs_slice,
    #             tkwargs=pass_kwargs,
    #         )

    #         assert flat_fn.out_tree is not None
    #         return jax.tree_util.tree_unflatten(flat_fn.out_tree, results)

    #     return wrapper_fn


def xdsl_transform(module_pass: ModulePass) -> PassDispatcher:
    """Wrapper function to register xDSL passes to use with QJIT-ed workflows."""
    dispatcher = PassDispatcher(module_pass)

    # Registration to map from plxpr primitive to pass
    register_transform(dispatcher, module_pass.name, False)

    # Registration for apply-transform-sequence interpreter
    def get_pass_cls():
        return module_pass

    register_pass(module_pass.name, get_pass_cls)
    return dispatcher
