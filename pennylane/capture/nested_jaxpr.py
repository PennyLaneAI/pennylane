# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from functools import partial

import jax


def bind_nested_jaxpr(fn):

    prim = jax.core.Primitive(fn.__name__)
    prim.multiple_results = True

    @prim.def_abstract_eval
    def _(*args, jaxpr, **_):
        return jaxpr.out_avals

    @prim.def_impl
    def _(*args, jaxpr, qfunc_kwargs=None, **kwargs):
        bound = partial(jax.core.eval_jaxpr, jaxpr.jaxpr)
        return [fn(bound)((), *args)]

    def new_version(qfunc, **qfunc_kwargs):
        def wrapper(*args, **kwargs):
            jaxpr = jax.make_jaxpr(qfunc)(*args, **kwargs)
            return prim.bind(*args, jaxpr=jaxpr, qfunc_kwargs=qfunc_kwargs, **kwargs)

        return wrapper

    return new_version
