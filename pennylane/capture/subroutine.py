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

import copy

has_jax = True
try:
    import jax
    from jax.experimental.pjit import pjit_p

    quantum_subroutine_p = copy.deepcopy(pjit_p)
    quantum_subroutine_p.name = "quantum_subroutine_p"

except ImportError:
    has_jax = False
    quantum_subroutine_p = None

from .autograph import wraps


class Patcher:
    """Patcher, a class to replace object attributes.

    Args:
        patch_data: List of triples. The first element in the triple corresponds to the object
        whose attribute is to be replaced. The second element is the attribute name. The third
        element is the new value assigned to the attribute.
    """

    def __init__(self, *patch_data):
        self.backup = {}
        self.patch_data = patch_data

        assert all(len(data) == 3 for data in patch_data)

    def __enter__(self):
        for obj, attr_name, fn in self.patch_data:
            self.backup[(obj, attr_name)] = getattr(obj, attr_name)
            setattr(obj, attr_name, fn)

    def __exit__(self, _type, _value, _traceback):
        for obj, attr_name, _ in self.patch_data:
            setattr(obj, attr_name, self.backup[(obj, attr_name)])


def subroutine(func):
    """
    Denotes the creation of a function in the intermediate representation.

    May be used to reduce compilation times. Instead of repeatedly compiling
    inlined versions of the function passed as a parameter, when functions
    are annotated with a subroutine, a single version of the function
    will be compiled and called from potentially multiple callsites.

    .. note::

        Subroutines are only available when using the PLxPR program capture
        interface.


    **Example**

    .. code-block:: python

        qml.capture.enable()

        @qml.capture.subroutine
        @qml.capture.run_autograph
        def f(x, wires):
            for i in range(len(x)):
                qml.RX(x[i], wires[i])

        def c(x):
            f(x, jnp.array([0,1,2]))
            return [qml.expval(qml.Z(i)) for i in range(3)]

        print(jax.make_jaxpr(c)(jnp.array([0.1, 0.2, 0.3])))

    .. code-block::

        { lambda a:i64[3]; b:f64[3]. let
            quantum_subroutine_p[
            compiler_options_kvs=()
            ctx_mesh=None
            donated_invars=(False, False)
            in_layouts=(None, None)
            in_shardings=(UnspecifiedValue, UnspecifiedValue)
            inline=False
            jaxpr={ lambda ; c:f64[3] d:i64[3]. let
                for_loop[
                    abstract_shapes_slice=slice(2, 2, None)
                    args_slice=slice(2, None, None)
                    consts_slice=slice(0, 2, None)
                    jaxpr_body_fn={
                        ...
                        _:AbstractOperator() = RX[n_wires=1] m s
                    in () }
                ] 0:i64[] 3:i64[] 1:i64[] c d
                in () }
            keep_unused=False
            name=ag__f
            out_layouts=()
            out_shardings=()
            ] b a
            t:AbstractOperator() = PauliZ[n_wires=1] 0:i64[]
            u:AbstractMeasurement(n_wires=None) = expval_obs t
            v:AbstractOperator() = PauliZ[n_wires=1] 1:i64[]
            w:AbstractMeasurement(n_wires=None) = expval_obs v
            x:AbstractOperator() = PauliZ[n_wires=1] 2:i64[]
            y:AbstractMeasurement(n_wires=None) = expval_obs x
        in (u, w, y) }


    """
    if not has_jax:
        raise ImportError("jax is required for use of subroutine")

    old_pjit = jax._src.pjit.pjit_p

    @wraps(func)
    def inside(*args, **kwargs):
        with Patcher(
            (
                jax._src.pjit,
                "pjit_p",
                old_pjit,
            ),
        ):
            return func(*args, **kwargs)

    @wraps(inside)
    def wrapper(*args, **kwargs):

        with Patcher(
            (
                jax._src.pjit,
                "pjit_p",
                quantum_subroutine_p,
            ),
        ):
            return jax.jit(inside)(*args, **kwargs)

    return wrapper
