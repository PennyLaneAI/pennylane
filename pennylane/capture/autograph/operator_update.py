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
"""Converter for array element operator assignment."""

# pylint: disable=no-member

import gast
from malt.core import converter
from malt.pyct import templates


# pylint: disable=too-few-public-methods
# TODO: The methods from this class should be moved to the SliceTransformer class in DiastaticMalt
class SingleIndexArrayOperatorUpdateTransformer(converter.Base):
    """Converts array element operator assignment statements into calls to update_item_with_{op},
    where op is one of the following:
    - `add` corresponding to `+=`
    - `sub` to `-=`
    - `mult` to `*=`
    - `div` to `/=`
    - `pow` to `**=`
    """

    def _process_single_update(self, target, op, value):

        if not isinstance(target, gast.Subscript):
            return None

        s = target.slice

        if isinstance(s, (gast.Tuple, gast.Call)):
            return None

        if not isinstance(op, (gast.Mult, gast.Add, gast.Sub, gast.Div, gast.Pow)):
            return None

        template = f"""
            target = ag__.update_item_with_op(target, index, x, "{type(op).__name__.lower()}")
        """
        lower, upper, step = None, None, None

        if isinstance(s, (gast.Slice)):
            # Replace unused arguments in template with "None" to preserve each arguments' position.
            # templates.replace ignores None and does not accept string so change is applied here.
            lower_str = "lower" if s.lower is not None else "None"
            upper_str = "upper" if s.upper is not None else "None"
            step_str = "step" if s.step is not None else "None"
            template = template.replace("index", f"slice({lower_str}, {upper_str}, {step_str})")

            lower, upper, step = s.lower, s.upper, s.step

        return templates.replace(
            template,
            target=target.value,
            index=target.slice,
            lower=lower,
            upper=upper,
            step=step,
            x=value,
        )

    def visit_AugAssign(self, node):
        """The AugAssign node is replaced with a call to ag__.update_item_with_{op}
        when its target is a single index array subscript and its op is an arithmetic
        operator (i.e. Add, Sub, Mult, Div, or Pow), otherwise the node is left as is.
        Example:
            `x[i] += y` is replaced with `x = ag__.update_item_with(x, i, y)`
            `x[i] ^= y` remains unchanged
        """
        node = self.generic_visit(node)
        replacement = self._process_single_update(node.target, node.op, node.value)
        if replacement is not None:
            return replacement
        return node


def transform(node, ctx):
    """Replace an AugAssign node with a call to ag__.update_item_with_{op}
    when the its target is a single index array subscript and its op is an arithmetic
    operator (i.e. Add, Sub, Mult, Div, or Pow), otherwise the node is left as is.
    Example:
        `x[i] += y` is replaced with `x = ag__.update_item_with(x, i, y)`
        `x[i] ^= y` remains unchanged
    """
    return SingleIndexArrayOperatorUpdateTransformer(ctx).visit(node)
