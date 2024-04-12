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

import jax


nested_adjoint_prim = jax.core.Primitive("AdjointJaxpr")
nested_adjoint_prim.multiple_results=True

@nested_adjoint_prim.def_abstract_eval
def _(inner_jaxpr):
    return inner_jaxpr.out_avals

def adjoint_qfunc(qfunc):
    def wrapper(*args, **kwargs):
        jaxpr = jax.make_jaxpr(qfunc)(*args, **kwargs)
        return nested_adjoint_prim.bind(inner_jaxpr=jaxpr)
    return wrapper