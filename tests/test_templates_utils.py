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
from pennylane.variable import Variable
from pennylane.templates.utils import (_check_wires,
                                       _check_shape,
                                       _check_shapes,
                                       _get_shape,
                                       _check_number_of_layers,
                                       _check_no_variable,
                                       _check_is_in_options,
                                       _check_type)


#########################################
# Inputs

WIRES_PASS = [(0, [0]),
              ([4], [4]),
              ([1, 2], [1, 2])]
WIRES_FAIL = [[-1],
              ['a'],
              lambda x: x]

SHAPE_PASS = [(0.231, (), None),
              ([[1., 2.], [3., 4.]], (2, 2), None),
              ([-2.3], (1, ), None),
              ([-2.3, 3.4], (4,), 'max'),
              ([-2.3, 3.4], (1,), 'min'),
              ([-2.3], (1,), 'max'),
              ([-2.3], (1,), 'min'),
              ([[-2.3, 3.4], [1., 0.2]], (3, 3), 'max'),
              ([[-2.3, 3.4, 1.], [1., 0.2, 1.]], (1, 2), 'min'),
              ]

SHAPE_LST_PASS = [([0.231, 0.1], [(), ()], None),
                  ([[1., 2.], [4.]], [(2, ), (1, )], None),
                  ([[-2.3], -1.], [(1, ), ()], None),
                  ([[-2.3, 0.1], -1.], [(1,), ()], 'min'),
                  ([[-2.3, 0.1], -1.], [(3,), ()], 'max')
                  ]

SHAPE_FAIL = [(0.231, (1,), None),
              ([[1., 2.], [3., 4.]], (2, ), None),
              ([-2.3], (4, 5), None),
              ([-2.3, 3.4], (4,), 'min'),
              ([-2.3, 3.4], (1,), 'max'),
              ([[-2.3, 3.4], [1., 0.2]], (3, 3), 'min'),
              ([[-2.3, 3.4, 1.], [1., 0.2, 1.]], (1, 2), 'max'),
              ]

GET_SHAPE_PASS = [(0.231, ()),
                  ([[1., 2.], [3., 4.]], (2, 2)),
                  ([-2.3], (1, )),
                  ([-2.3, 3.4], (2,)),
                  ([-2.3], (1,)),
                  ([[-2.3, 3.4, 1.], [1., 0.2, 1.]], (2, 3)),
                  ]

#TODO: Think of a data structure that CANNOT be converted to numpy array
GET_SHAPE_FAIL = []

SHAPE_LST_FAIL = [([0.231, 0.1], [(), (3, 4)], None),
                  ([[1., 2.], [4.]], [(1, ), (1, )], None),
                  ([[-2.3], -1.], [(1, 2), (1,)], None),
                  ([[-2.3, 0.1], -1.], [(1,), ()], 'max'),
                  ([[-2.3, 0.1], -1.], [(3,), ()], 'min')
                  ]

LAYERS_PASS = [([[1], [2], [3]], 1),
               ([[[1], [2], [3]], [['a'], ['b'], ['c']]], 3),
             ]

LAYERS_FAIL = [([1, 2, 3], None),
               ([[[1], [2], [3]], [['b'], ['c']]], 3),
              ]

NO_VARIABLES_PASS = [[[], np.array([1., 4.])],
                     [1, 'a']]

NO_VARIABLES_FAIL = [[[Variable(0.1)], Variable([0.1])],
                     np.array([Variable(0.3), Variable(4.)]),
                     Variable(-1.)]

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

    @pytest.mark.parametrize("arg", NO_VARIABLES_PASS)
    def test_check_no_variable(self, arg):
        """Tests that variable check succeeds for valid arguments."""
        _check_no_variable(arg, msg="XXX")

    @pytest.mark.parametrize("arg", NO_VARIABLES_FAIL)
    def test_check_no_variable_exception(self, arg):
        """Tests that variable check throws error for invalid arguments."""
        with pytest.raises(ValueError, match="XXX"):
            _check_no_variable(arg, msg="XXX")

    @pytest.mark.parametrize("wires, target", WIRES_PASS)
    def test_check_wires(self, wires, target):
        """Tests that wires check returns correct wires list and its length."""
        res = _check_wires(wires=wires)
        assert res == target

    @pytest.mark.parametrize("wires", WIRES_FAIL)
    def test_check_wires_exception(self, wires):
        """Tests that wires check fails if ``wires`` is not an integer or iterable."""
        with pytest.raises(ValueError, match="wires must be a positive integer"):
            _check_wires(wires=wires)

    @pytest.mark.parametrize("inpt, target_shape", GET_SHAPE_PASS)
    def test_get_shape(self, inpt, target_shape):
        """Tests that ``_get_shape`` returns correct shape."""
        shape = _get_shape(inpt)
        assert shape == target_shape

    @pytest.mark.parametrize("inpt, target_shape, bound", SHAPE_PASS)
    def test_check_shape(self, inpt, target_shape, bound):
        """Tests that shape check succeeds for valid arguments."""
        _check_shape(inpt, target_shape, bound=bound, msg="XXX")

    @pytest.mark.parametrize("inpt, target_shape, bound", SHAPE_LST_PASS)
    def test_check_shape_list_of_inputs(self, inpt, target_shape, bound):
        """Tests that list version of shape check succeeds for valid arguments."""
        _check_shapes(inpt, target_shape, bounds=[bound]*len(inpt), msg="XXX")

    @pytest.mark.parametrize("inpt, target_shape, bound", SHAPE_FAIL)
    def test_check_shape_exception(self, inpt, target_shape, bound):
        """Tests that shape check fails for invalid arguments."""
        with pytest.raises(ValueError, match="XXX"):
            _check_shape(inpt, target_shape, bound=bound, msg="XXX")

    @pytest.mark.parametrize("inpt, target_shape, bound", SHAPE_LST_FAIL)
    def test_check_shape_list_of_inputs_exception(self, inpt, target_shape, bound):
        """Tests that list version of shape check succeeds for valid arguments."""
        with pytest.raises(ValueError, match="XXX"):
            _check_shapes(inpt, target_shape, bounds=[bound]*len(inpt), msg="XXX")

    @pytest.mark.parametrize("hp, opts", OPTIONS_PASS)
    def test_check_options(self, hp, opts):
        """Tests that option check succeeds for valid arguments."""
        _check_is_in_options(hp, opts, msg="XXX")

    @pytest.mark.parametrize("hp, opts", OPTIONS_FAIL)
    def test_check_options_exception(self, hp, opts):
        """Tests that option check throws error for invalid arguments."""
        with pytest.raises(ValueError, match="XXX"):
            _check_is_in_options(hp, opts, msg="XXX")

    @pytest.mark.parametrize("hp, typ, alt", TYPE_PASS)
    def test_check_type(self, hp, typ, alt):
        """Tests that type check succeeds for valid arguments."""
        _check_type(hp, [typ, alt], msg="XXX")

    @pytest.mark.parametrize("hp, typ, alt", TYPE_FAIL)
    def test_check_type_exception(self, hp, typ, alt):
        """Tests that type check throws error for invalid arguments."""
        with pytest.raises(ValueError, match="XXX"):
            _check_type(hp, [typ, alt], msg="XXX")

    @pytest.mark.parametrize("inpt, repeat", LAYERS_PASS)
    def test_check_num_layers(self, inpt, repeat):
        """Tests that layer check returns correct number of layers."""
        n_layers = _check_number_of_layers(inpt)
        assert n_layers == repeat

    @pytest.mark.parametrize("inpt, repeat", LAYERS_FAIL)
    def test_check_num_layers_exception(self, inpt, repeat):
        """Tests that layer check throws exception for invalid arguments."""
        with pytest.raises(ValueError, match="the first dimension of the weight parameters"):
            _check_number_of_layers(inpt)

