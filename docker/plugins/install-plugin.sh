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

PLUGIN_NAME=$1
# Initial Setup
apt-get update
apt-get -y install --no-install-recommends make
rm -rf /var/lib/apt/lists/*
case $PLUGIN_NAME in
# Build Qiskit Plugin
  "qiskit")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-qiskit
  ;;
# Build Amazon-Braket Plugin
  "amazon-braket")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install amazon-braket-pennylane-plugin
  ;;
 # Build QuantumInspire Plugin
  "quantuminspire")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-quantuminspire
  ;;
# Build Cirq Plugin
  "cirq")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-cirq
  ;;
# Build Qulacs Plugin
  "qulacs")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install qulacs pennylane-qulacs
  ;;
# Build AQT Plugin
  "aqt")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-aqt
  ;;
# Build Honeywell Plugin
  "honeywell")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-honeywell
  ;;
# Build PQ Plugin
  "pq")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane_pq
  ;;
# Build Qsharp Plugin
  "qsharp")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-qsharp
  ;;
# Build Forest Plugin
  "forest")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-forest
  ;;
# Build orquestra Plugin
  "orquestra")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-orquestra
  ;;
# Build Ionq Plugin
  "ionq")
  echo "##########-Installing" "$PLUGIN_NAME" "Plugin-##########"
  pip3 install pennylane-ionq
  ;;
	*)
  echo "##########-No-Plugin-Installed-##########"
  ;;
  esac
