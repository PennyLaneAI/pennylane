#!/bin/bash
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

INTERFACE_NAME=$1
# Initial Setup
apt-get update
apt-get -y install --no-install-recommends make
rm -rf /var/lib/apt/lists/*
case $INTERFACE_NAME in
# Build Jax GPU interface
  "jax")
  echo "##########-Installing" "$INTERFACE_NAME" "GPU INTERFACE-##########"
	pip3 install jax==0.4.10 jaxlib==0.4.10+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
  ;;
# Build Torch GPU interface
  "torch")
  echo "##########-Installing" "$INTERFACE_NAME" "GPU INTERFACE-##########"
  pip3 install torch==2.0.0
  ;;
# Build Tensorflow GPU interface
  "tensorflow")
  echo "##########-Installing" "$INTERFACE_NAME" "GPU INTERFACE-##########"
  pip3 install tensorflow==2.12.0
  ;;
	*)
  echo "##########-No-GPU-Interface-Installed-##########"
  ;;
  esac
