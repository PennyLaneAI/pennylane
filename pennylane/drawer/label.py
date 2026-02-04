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

from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operator
from pennylane.ops import SymbolicOp
from pennylane.queuing import apply


class CustomLabelOp(SymbolicOp):

    def __repr__(self):
        return f"<{self.base}, label={self.hyperparameters['custom_label']}>"

    resource_keys = {"base_class", "base_params"}

    @property
    def resource_params(self) -> dict:
        return {"base_class": type(self.base), "base_params": self.base.resource_params}

    def __init__(self, base, custom_label):
        super().__init__(base)
        self.hyperparameters["custom_label"] = custom_label

    def label(
        self, decimals: int | None = None, base_label: str | None = None, cache: dict | None = None
    ) -> str:
        return self.hyperparameters["custom_label"]


def _resources(base_class, base_params):
    return resource_rep(base_class, **base_params)


@register_resources(_resources)
def CustomLabelDecomp(*params, wires, base, **_):
    apply(base)


def label(op: Operator, new_label: str) -> CustomLabelOp:
    return CustomLabelOp(op, new_label)
