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
Public/internal API for the AutoGraph module.
"""

from .transformer import (
    autograph_source,
    disable_autograph,
    run_autograph,
)

autograph_ignore_fallbacks = False
"""bool: Specify whether AutoGraph should avoid raising
warnings when conversion fails and control flow instead falls back
to being interpreted by Python at compile-time.

**Example**

In certain cases, AutoGraph will fail to convert control flow (for example,
when an object that can not be converted to a JAX array is indexed in a
loop), and will raise a warning informing of the failure.

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = ["0.1", "0.2", "0.3"]
...     for i in range(3):
...         qml.RX(float(x[i]), wires=i)
...     return qml.expval(qml.PauliZ(0))
Warning: Tracing of an AutoGraph converted for loop failed with an exception:
...
If you intended for the conversion to happen, make sure that the (now dynamic)
loop variable is not used in tracing-incompatible ways, for instance by indexing a
Python list with it. In that case, the list should be wrapped into an array.

Setting this variable to ``True`` will suppress warning messages:

>>> catalyst.autograph_strict_conversion = False
>>> catalyst.autograph_ignore_fallbacks = True
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = ["0.1", "0.2", "0.3"]
...     for i in range(3):
...         qml.RX(float(x[i]), wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> f()
array(0.99500417)
"""

autograph_strict_conversion = False
"""bool: Specify whether AutoGraph should raise exceptions
when conversion fails, rather than falling back to interpreting
control flow by Python at compile-time.

**Example**

In certain cases, AutoGraph will fail to convert control flow (for example,
when an object that cannot be converted to a JAX array is indexed in a
loop), and will automatically fallback to interpreting the control flow
logic at compile-time via Python:

>>> dev = qml.device("lightning.qubit", wires=1)
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     params = ["0", "1", "2"]
...     for x in params:
...         qml.RY(int(x) * jnp.pi / 4, wires=0)
...     return qml.expval(qml.PauliZ(0))
>>> f()
array(-0.70710678)

Setting this variable to ``True`` will cause AutoGraph
to error rather than fallback when conversion fails:

>>> catalyst.autograph_strict_conversion = True
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     params = ["0", "1", "2"]
...     for x in params:
...         qml.RY(int(x) * jnp.pi / 4, wires=0)
...     return qml.expval(qml.PauliZ(0))
AutoGraphError: Could not convert the iteration target ['0', '1', '2'] to array
while processing the following with AutoGraph:
  File "<ipython-input-44-dbae11e6d745>", line 7, in f
    for x in params:
"""


__all__ = (
    "autograph_source",
    "disable_autograph",
    "run_autograph",
)
