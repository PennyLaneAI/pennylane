# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This tool checks the syntax of quantum device configuration files. It is a strict parser of TOML
format, narrowed down to match our requirements.  For the Lark's EBNF dialect syntax, see the
Lark grammar reference:
 * https://lark-parser.readthedocs.io/en/latest/grammar.html
"""

# pylint: disable=unused-import

from textwrap import dedent  # pragma: no cover

try:  # pragma: no cover
    from lark import Lark, LarkError, UnexpectedInput  # pragma: no cover
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "toml_check requires `lark` library. Consider `pip install lark`"
    ) from e  # pragma: no cover

parser = Lark(  # pragma: no cover
    dedent(
        """
        start: schema_body \
               gates_section \
               pennylane_gates_section? \
               qjit_gates_section? \
               observables_section \
               pennylane_observables_section? \
               qjit_observables_section? \
               measurement_processes_section \
               pennylane_measurement_processes_section? \
               qjit_measurement_processes_section? \
               compilation_section \
               pennylane_compilation_section? \
               qjit_compilation_section?
        schema_body: schema_decl
        gates_section: "[operators.gates]" operator_decl*
        pennylane_gates_section: "[pennylane.operators.gates]" operator_decl*
        qjit_gates_section: "[qjit.operators.gates]" operator_decl*
        observables_section: "[operators.observables]" operator_decl*
        pennylane_observables_section: "[pennylane.operators.observables]" operator_decl*
        qjit_observables_section: "[qjit.operators.observables]" operator_decl*
        measurement_processes_section: "[measurement_processes]" mp_decl*
        pennylane_measurement_processes_section: "[pennylane.measurement_processes]" mp_decl*
        qjit_measurement_processes_section: "[qjit.measurement_processes]" mp_decl*
        compilation_section: "[compilation]" compilation_option_decl*
        pennylane_compilation_section: "[pennylane.compilation]" compilation_option_decl*
        qjit_compilation_section: "[qjit.compilation]" compilation_option_decl*
        schema_decl: "schema" "=" "3"
        operator_decl: name "=" "{" (operator_trait ("," operator_trait)*)? "}"
        operator_trait: conditions | properties
        conditions: "conditions" "=" "[" condition ("," condition)* "]"
        properties: "properties" "=" "[" property ("," property)* "]"
        condition: "\\"finiteshots\\"" | "\\"analytic\\"" | "\\"terms-commute\\""
        property: "\\"controllable\\"" | "\\"invertible\\"" | "\\"differentiable\\""
        mp_decl: name "=" "{" (mp_trait)? "}"
        mp_trait: conditions
        compilation_option_decl: boolean_option | mcm_option
        boolean_option: ( \
            "qjit_compatible" | "runtime_code_generation" | "dynamic_qubit_management" | \
            "overlapping_observables" | "non_commuting_observables" | "initial_state_prep" | \
        ) "=" boolean
        mcm_option: "supported_mcm_methods" "=" "[" ("\\"" name "\\"" ("," "\\"" name "\\"" )*)? "]"
        name: /[a-zA-Z0-9_-]+/
        boolean: "true" | "false"
        COMMENT: "#" /./*
        %import common.WS
        %ignore WS
        %ignore COMMENT
        """
    )
)
