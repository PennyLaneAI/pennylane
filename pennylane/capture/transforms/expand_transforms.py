# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Helper function for expanding transforms with program capture
"""
from functools import partial, wraps
from typing import Callable
from pennylane.capture.base_interpreter import PlxprInterpreter
from pennylane.capture.flatfn import FlatFn


class ExpandTransformsInterpreter(PlxprInterpreter):
    pass


def expand_plxpr_transforms(f: Callable) -> Callable:
    # pylint: disable=import-outside-toplevel
    from jax import make_jaxpr
    from jax.tree_util import tree_unflatten

    @wraps(f)
    def wrapper(*args, **kwargs):
        # f_partial = partial(f, **kwargs)
        # flat_f = FlatFn(f_partial)
        # jaxpr = make_jaxpr(flat_f)(*args)
        # assert flat_f.out_tree is not None, "Should be set when constructing the jaxpr"

        # interpreter = ExpandTransformsInterpreter()
        # res = interpreter.eval(jaxpr.jaxpr, jaxpr.consts, *args)
        # return tree_unflatten(flat_f.out_tree, res)
        transformed_f = ExpandTransformsInterpreter()(f)
        return transformed_f(*args, **kwargs)

    return wrapper
