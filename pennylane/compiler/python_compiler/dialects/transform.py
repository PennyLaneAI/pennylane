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
"""
This file contains an updated version of the transform dialect.
As of the time of writing, xDSL uses the MLIR released with LLVM's
version 20.1.7. However, https://github.com/PennyLaneAI/catalyst/pull/1916
will be updating MLIR where the transform dialect has the
`apply_registered_pass` operation re-defined.

See the following changelog on the above PR

    Things related to transform.apply_registered_pass op:

    It now takes in a dynamic_options

    [MLIR][Transform] Allow ApplyRegisteredPassOp to take options as
    a param llvm/llvm-project#142683. We don't need to use this as all our pass options are static.
    https://github.com/llvm/llvm-project/pull/142683

    The options it takes in are now dictionaries instead of strings
    [MLIR][Transform] apply_registered_pass op's options as a dict llvm/llvm-project#143159
    https://github.com/llvm/llvm-project/pull/143159

This file will re-define the apply_registered_pass operation in xDSL
and the transform dialect.

Once xDSL moves to a newer version of MLIR, these changes should
be contributed upstream.
"""

# pylint: disable=unused-wildcard-import,wildcard-import
from xdsl.dialects.transform import *
