#!/bin/bash
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

set -e

wget https://github.com/llvm/llvm-project/releases/download/llvmorg-${VERSION}/llvm-project-${VERSION}.src.tar.xz
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-${VERSION}/llvm-project-${VERSION}.src.tar.xz.sig

# Hard code signature key id?
gpg --recv-keys --keyserver hkps://keyserver.ubuntu.com 474E22316ABF4785A88C6E8EA2C794A986419D8A
# Check the signature.
gpg --verify llvm-project-${VERSION}.src.tar.xz.sig llvm-project-${VERSION}.src.tar.xz

tar -xf llvm-project-${VERSION}.src.tar.xz

mkdir llvm-project-${VERSION}.src/build
cd llvm-project-${VERSION}.src/build
pip install -r ../mlir/python/requirements.txt
pip install cmake

cmake -G "Ninja" ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=$(which python3) \
   -DLLVM_PARALLEL_COMPILE_JOBS=$(nproc)

cmake --build . --target check-mlir
cmake --build .
cmake -DCMAKE_INSTALL_PREFIX=/opt/llvm -P cmake_install.cmake
