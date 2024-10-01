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

"""The make_plxpr function and helper methods"""

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


# ToDo: type hints
def make_plxpr(circuit, static_argnums, **kwargs):
    """Creates a function that produces its PLxPR given example args.

    This function relies on `jax.make_jaxpr` as part of creating the representation. Any
    keyword arguments passed to `make_plxpr` that are not directly used in the function will
    be passed to make_jaxpr.

        Args:
            circuit (~.tape.QuantumTape):  # is this a qnode or a tape or both?

        Kwargs:
            static_argnums (Union(int, Sequence[int], None)): optional, an int or
                collection of ints that specify which positional arguments to treat
                as static (trace- and compile-time constant).

        Returns:
            Callable: function that, when called, returns the PLxPR representation of `circuit` for the specified inputs.

    """
    return jax.make_jaxpr(circuit, static_argnums, **kwargs)
