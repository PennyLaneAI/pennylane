# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Built-in decomposition rules for drawer.label."""

# pylint: disable=unused-argument

from pennylane.core.apply import apply
from pennylane.decomposition import add_decomps, register_resources
from pennylane.drawer.label import LabelledOp


def _resources(base):
    return {base: 1}


@register_resources(_resources)
def _custom_label_decomp(*params, wires, base, **_):
    apply(base)


add_decomps(LabelledOp, _custom_label_decomp)
