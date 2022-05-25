# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule defines the symbolic operation that indicates the power of an operator.
"""
from pennylane.operation import Operator, PowUndefinedError
from pennylane.queuing import QueuingContext, QueuingError

_superscript = str.maketrans("0123456789.+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⋅⁺⁻")


class Pow(Operator):
    """
    docstring
    """

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten becuase the base must be copied too.
        copied_op = object.__new__(type(self))
        # copied_op must maintain inheritance structure of self
        # For example, it must keep AdjointOperation if self has it
        # this way preserves inheritance structure

        copied_base = self.base.__copy__()
        copied_op._hyperparameters = {"base": copied_base}
        for attr, value in vars(self).items():
            if attr not in {"data", "base", "_hyperparameters"}:
                setattr(copied_op, attr, value)

        return copied_op

    # pylint: disable=super-init-not-called
    def __init__(self, base=None, z=None, do_queue=True, id=None):
        self.hyperparameters["base"] = base
        self.hyperparameters["z"] = z
        self._id = id
        self.queue_idx = None

        self._name = f"{self.base.name}**{z}"

        if do_queue:
            self.queue()

    @property
    def base(self):
        """The operator that is adjointed."""
        return self.hyperparameters["base"]

    @property
    def z(self):
        """The exponent."""
        return self.hyperparameters["z"]

    @property
    def data(self):
        """Trainable parameters that the operator depends on."""
        return self.base.data

    @data.setter
    def data(self, new_data):
        """Allows us to set base operation parameters."""
        self.base.data = new_data

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def wires(self):
        return self.base.wires

    @property
    def num_wires(self):
        return self.base.num_wires

    def queue(self, context=QueuingContext):
        try:
            context.update_info(self.base, owner=self)
        except QueuingError:
            self.base.queue(context=context)
            context.update_info(self.base, owner=self)

        context.append(self, owns=self.base)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        z_string = format(self.z).translate(_superscript)
        return self.base.label(decimals, base_label, cache=cache) + z_string

    # pylint: disable=raise-missing-from
    def decomposition(self):
        try:
            return self.base.pow(self.z)
        except PowUndefinedError:
            if isinstance(self.z, int) and self.z > 0:
                return [self.base.__copy__() for _ in range(self.z)]
            raise NotImplementedError
