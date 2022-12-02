# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines classes and functions corresponding to the
new Device API design.

The tracker is a copy of the top-level tracker file. It's placed here
to have a more relevant location.

.. currentmodule:: pennylane.devices.experimental

.. autosummary::
    :toctree: api

    ~AbstractDevice
    ~backward_patch_interface
    ~PythonDevice
    ~Tracker
    ~simple_preprocessor
    ~adjoint_diff_gradient
    ~PlainNumpySimulator

**Examples:**

>>> from pennylane.workflow import ExecutionConfig
>>> from pennylane.devices import experimental as devices
>>> qml.enable_return()

Create the python device and patch the interface so it works with PennyLane's execution
workflow.

>>> dev = devices.PythonDevice()
>>> dev = devices.backward_patch_interface(dev)

Now construct an arbitrary batch of :class:`~.QuantumScript`:

.. code-block:: python

    n_layers = 5
    n_wires = 10
    num_qscripts = 5

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    rng = qml.numpy.random.default_rng(seed=42)

    qscripts = []
    for i in range(num_qscripts):
        params = rng.random(shape)
        op = qml.StronglyEntanglingLayers(params, wires=range(n_wires))
        qs = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])
        qscripts.append(qs)

To execute these qscripts, we first need to preprocess them so the device
natively executes them:

>>> processed_qscripts, post_processing_fn = dev.preprocess(qscripts)

And then we can execute the new Quantum Script batch on the device:

>>> dev.execute(processed_qscripts)
(-0.0006888975950537501,
 0.025576307134457577,
 -0.0038567269892757494,
 0.1339705146860149,
 -0.03780669772690448)

We can also use :func:`~.execute` or :class:`~.QNode` with ``cache=False``.

>>> qml.execute(qscripts, dev, cache=False)
[(-0.0006888975950537501,),
 (0.025576307134457577,),
 (-0.0038567269892757494,),
 (0.1339705146860149,),
 (-0.03780669772690448,)]
>>> @qml.qnode(dev, cache=False)
... def circuit(x):
...     qml.RX(x, 0)
...     return qml.probs(0)
>>> circuit(1.2)
(array([0.68117888, 0.31882112]),)

"""

from .tracker import Tracker
from .device_interface import AbstractDevice
from .python_device import PythonDevice
from .compatibility_patching import backward_patch_interface
from .python_preprocessor import simple_preprocessor
from .simulator import adjoint_diff_gradient, PlainNumpySimulator
