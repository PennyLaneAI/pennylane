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

from functools import wraps

from pennylane.tape import QuantumScript
from pennylane.typing import Callable

from ..transforms.api.apply_transform_sequence import register_callback
from .collector import QMLCollector

# TODO: This caching mechanism should be improved,
# because now it relies on a mutable global state
_cache_store: dict[Callable, dict[int, tuple[str, str]]] = {}


def draw(qnode, *, level: None | int = None):
    "Draw the quantum circuit at the specified level."

    cache: dict[int, tuple[str, str]] = _cache_store.setdefault(qnode, {})

    def draw_callback(pass_instance, module, pass_level):
        collector = QMLCollector(module)
        ops, meas = collector.collect()
        # This is just a quick way to visualize the circuit
        # using PennyLane's built-in drawing capabilities of QuantumScript
        tape = QuantumScript(ops, meas)
        cache[pass_level] = (tape.draw(), pass_instance.name if pass_level else "No transforms")

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        register_callback(draw_callback)
        # TODO: this currently does not work if the QNode has no argument, because
        # `qjit` applies all transformations before reaching this point in that case
        qnode(*args, **kwargs)

        if level is not None:
            if level in cache:
                return cache[level][0]
            raise ValueError("No drawing available for the specified level.")

        for lvl in sorted(cache):
            print(f"\n[Level {lvl}]")
            print(f"Applying transform: {cache[lvl][1]}")
            print(cache[lvl][0])

    return wrapper
