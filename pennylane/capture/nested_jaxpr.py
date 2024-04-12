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

import jax


def bind_nested_jaxpr(fn):

    prim = jax.core.Primitive(fn.__name__)
    prim.multiple_results = True

    @prim.def_abstract_eval
    def _(jaxpr):
        return jaxpr.out_avals

    def new_version(qfunc):
        def wrapper(*args, **kwargs):
            jaxpr = jax.make_jaxpr(qfunc)(*args, **kwargs)
            return prim.bind(jaxpr=jaxpr)

        return wrapper

    return new_version
