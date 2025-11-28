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
# pylint: disable=no-member  # False positives with xDSL region.block access

"""Implementation of the Tree-Traversal MCM simulation method as an xDSL transform in Catalyst."""

from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.printer import Printer

print_everthing = True
# print_everthing = False


def print_mlir(op, msg="", should_print: bool = True):
    """Print the MLIR of an operation with a message."""
    should_print = print_everthing
    if should_print:
        printer = Printer()
        print("-" * 100)
        print(f"// Start || {msg}")
        if isinstance(op, Region):
            printer.print_region(op)
        elif isinstance(op, Block):
            printer.print_block(op)
        elif isinstance(op, Operation):
            printer.print_op(op)
        print(f"\n// End {msg}")
        print("-" * 100)


def print_ssa_values(values, msg="SSA Values || ", should_print: bool = True):
    """Print SSA Values"""
    should_print = print_everthing
    if should_print:
        print(f"// {msg}")
        for val in values:
            print(f"  - {val}")
