# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Tests for the templates utility functions.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import numpy as np
from pennylane.qnode import Variable
from pennylane.templates.utils import (_check_wires,
                                       _check_shape,
                                       _check_no_variable,
                                       _check_hyperp_is_in_options,
                                       _check_type)

#######################################
# Interfaces

INTERFACES = [('numpy', np.array),
              ('numpy', lambda x: x),  # identity
              ]

try:
    import torch
    INTERFACES.append(('torch', torch.tensor))
except ImportError as e:
    pass

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        print(tf.__version__)
        import tensorflow.contrib.eager as tfe
        tf.enable_eager_execution()
        TFVariable = tfe.Variable
    else:
        from tensorflow import Variable as TFVariable
    INTERFACES.append(('tf', TFVariable))
except ImportError as e:
    pass

#########################################
# Inputs

WIRES_PASS = [(0, ([0], 1)),
              ([4], ([4], 1)),
              ([1, 2], ([1, 2], 2))]
WIRES_FAIL = [[-1],
              ['a'],
              lambda x: x]

SHP_PASS = [(0.231, (), None),
            ([[1., 2.], [3., 4.]], (2, 2), None),
            ([-2.3], (1, ), None),
            ([-2.3, 3.4], (4,), 'max'),
            ([-2.3, 3.4], (1,), 'min'),
            ([-2.3], (1,), 'max'),
            ([-2.3], (1,), 'min'),
            ([[-2.3, 3.4], [1., 0.2]], (3, 3), 'max'),
            ([[-2.3, 3.4, 1.], [1., 0.2, 1.]], (1, 2), 'min'),
            ]

SHP_LST_PASS = [([0.231, 0.1], [(), ()], None),
                ([[1., 2.], [4.]], [(2, ), (1, )], None),
                ([[-2.3], -1.], [(1, ), ()], None),
                ([[-2.3, 0.1], -1.], [(1,), ()], 'min'),
                ([[-2.3, 0.1], -1.], [(3,), ()], 'max')
                ]

SHP_FAIL = [(0.231, (1,), None),
            ([[1., 2.], [3., 4.]], (2, ), None),
            ([-2.3], (4, 5), None),
            ([-2.3, 3.4], (4,), 'min'),
            ([-2.3, 3.4], (1,), 'max'),
            ([[-2.3, 3.4], [1., 0.2]], (3, 3), 'min'),
            ([[-2.3, 3.4, 1.], [1., 0.2, 1.]], (1, 2), 'max'),
            ]

SHP_LST_FAIL = [([0.231, 0.1], [(), (3, 4)], None),
                ([[1., 2.], [4.]], [(1, ), (1, )], None),
                ([[-2.3], -1.], [(1, 2), (1,)], None),
                ([[-2.3, 0.1], -1.], [(1,), ()], 'max'),
                ([[-2.3, 0.1], -1.], [(3,), ()], 'min')
                ]

NOVARS_PASS = [[[], np.array([1., 4.])],
               [1, 'a']]

NOVARS_FAIL = [[[Variable(0.1)], Variable([0.1])],
               np.array([Variable(0.3), Variable(4.)]),
               [Variable(-1.)]]

OPTIONS_PASS = [("a", ["a", "b"])]

OPTIONS_FAIL = [("c", ["a", "b"])]

TYPE_PASS = [(["a"], list, type(None)),
             (1, int, type(None)),
             ("a", int, str),
             (Variable(1.), list, Variable)
             ]

TYPE_FAIL = [("a", list, type(None)),
             (Variable(1.), int, list),
             (1., Variable, type(None))
             ]

##############################


class TestInputChecks:
    """Test private functions that check the input of templates."""

    @pytest.mark.parametrize("wires, targt", WIRES_PASS)
    def test_check_wires(self, wires, targt):
        res = _check_wires(wires=wires)
        assert res == targt

    @pytest.mark.parametrize("wires", WIRES_FAIL)
    def test_check_wires_exception(self, wires):
        with pytest.raises(ValueError):
            _check_wires(wires=wires)

    @pytest.mark.parametrize("inpt, target_shape, bound", SHP_PASS)
    @pytest.mark.parametrize("intrfc, to_var", INTERFACES)
    def test_check_shape_with_diff_interfaces(self, inpt, target_shape, bound, intrfc, to_var):
        inpt = to_var(inpt)
        _check_shape(inpt, target_shape, bound=bound)

    @pytest.mark.parametrize("inpt, target_shape, bound", SHP_LST_PASS)
    @pytest.mark.parametrize("intrfc, to_var", INTERFACES)
    def test_check_shape_list_of_inputs_with_diff_interfaces(self, inpt, target_shape, bound, intrfc, to_var):
        inpt = [to_var(i) for i in inpt]
        _check_shape(inpt, target_shape, bound=bound)

    @pytest.mark.parametrize("inpt, target_shape, bound", SHP_FAIL)
    @pytest.mark.parametrize("intrfc, to_var", INTERFACES)
    def test_check_shape_with_diff_interfaces_exception(self, inpt, target_shape, bound, intrfc, to_var):
        inpt = to_var(inpt)
        with pytest.raises(ValueError):
            _check_shape(inpt, target_shape, bound=bound)

    @pytest.mark.parametrize("inpt, target_shape, bound", SHP_LST_FAIL)
    @pytest.mark.parametrize("intrfc, to_var", INTERFACES)
    def test_check_shape_list_of_inputs_with_diff_interfaces_exception(self, inpt, target_shape, bound, intrfc, to_var):
        inpt = [to_var(i) for i in inpt]
        with pytest.raises(ValueError):
            _check_shape(inpt, target_shape, bound=bound)

    def test_check_shape_exception_message(self):
        with pytest.raises(ValueError) as excinfo:
            _check_shape([0.], (3,), msg="XXX")
        assert "XXX" in str(excinfo.value)

    @pytest.mark.parametrize("arg", NOVARS_PASS)
    def test_check_no_variables(self, arg):
        _check_no_variable(arg, "dummy")

    @pytest.mark.parametrize("arg", NOVARS_FAIL)
    def test_check_no_variables_exception(self, arg):
        with pytest.raises(ValueError):
            _check_no_variable(arg, "dummy")

    def test_check_no_variables_exception_message(self):
        with pytest.raises(ValueError) as excinfo:
            a = Variable(0)
            _check_no_variable([a], ["dummy"], msg="XXX")
        assert "XXX" in str(excinfo.value)

    @pytest.mark.parametrize("hp, opts", OPTIONS_PASS)
    def test_check_hyperp_options(self, hp, opts):
        _check_hyperp_is_in_options(hp, opts)

    @pytest.mark.parametrize("hp, opts", OPTIONS_FAIL)
    def test_check_hyperp_options_exception(self, hp, opts):
        with pytest.raises(ValueError):
            _check_hyperp_is_in_options(hp, opts)

    @pytest.mark.parametrize("hp, typ, alt", TYPE_PASS)
    def test_check_type(self, hp, typ, alt):
        _check_type(hp, [typ, alt])

    @pytest.mark.parametrize("hp, typ, alt", TYPE_FAIL)
    def test_check_type_exception(self, hp, typ, alt):
        with pytest.raises(ValueError):
            _check_type(hp, [typ, alt])

    @pytest.mark.parametrize("hp, typ, alt", TYPE_FAIL)
    def test_check_type_exception_message(self, hp, typ, alt):
        with pytest.raises(ValueError) as excinfo:
            _check_type(hp, [typ, alt], msg="XXX")
        assert "XXX" in str(excinfo.value)
