# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit and integration tests for the compile pipeline."""
# pylint: disable=no-member


import pytest
import rustworkx as rx

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import (
    CompilePipeline,
    TransformContainer,
    TransformError,
    transform,
)
from pennylane.transforms.core.compile_pipeline import (
    CotransformCache,
    _apply_postprocessing_stack,
    _batch_postprocessing,
    null_postprocessing,
)
from pennylane.typing import PostprocessingFn, Result, ResultBatch


def first_valid_transform(
    tape: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid transform."""
    tape = tape.copy()
    tape._ops.pop(index)  # pylint:disable=protected-access
    return [tape], lambda x: x


def expand_transform(
    tape: QuantumScript, index: int  # pylint:disable=unused-argument
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid expand transform."""
    return [tape], lambda x: x


def second_valid_transform(
    tape: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid trasnform."""
    tape1 = tape.copy()
    tape2 = tape.copy()
    tape2 = tape._ops.pop(index)  # pylint:disable=protected-access

    def fn(results):
        return qml.math.sum(results)

    return [tape1, tape2], fn


def informative_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid informative transform"""

    def fn(results):
        return len(results[0].operations)

    return [tape], fn


class TestUtilityHelpers:
    """test the private functions used in post processing."""

    def test_batch_postprocessing(self):
        """Test the _batch_postprocessing helper function."""
        results = (1.0, 2.0, 3.0, 4.0)

        def postprocessing1(results):
            return results[0] + results[1]

        def postprocessing2(results):
            return results[0] + 1

        out = _batch_postprocessing(
            results, (postprocessing1, postprocessing2), [slice(0, 2), slice(2, 4)]
        )
        assert out == (3.0, 4.0)

    def test_postprocessing_stack(self):
        """Tests the _apply_postprocessing_stack helper function."""

        results = (1.0, 2.0, 3.0, 4.0)

        def postprocessing1(results):
            return (results[0] + results[1], results[2] + results[3])

        def postprocessing2(results):
            return (results[0] + 1, results[1] + 2)

        out1 = _apply_postprocessing_stack(results, [postprocessing1])
        assert out1 == (3.0, 7.0)

        out2 = _apply_postprocessing_stack(results, [postprocessing2, postprocessing1])
        assert out2 == (4.0, 9.0)


# pylint: disable=too-many-public-methods
class TestCompilePipelineDunders:
    """Test the dunder methods."""

    def test_bool(self):
        """Check that a compile pipeline is falsy if empty and truthy if not."""
        empty_prog = CompilePipeline()
        assert not empty_prog

        transform1 = TransformContainer(qml.transform(first_valid_transform))
        populated_prog = CompilePipeline((transform1,))
        assert populated_prog

    def test_iter_pipeline(self):
        """Test iteration over the compile pipeline."""
        compile_pipeline = CompilePipeline()
        transform1 = TransformContainer(qml.transform(first_valid_transform))

        for _ in range(10):
            compile_pipeline.push_back(transform1)

        assert len(compile_pipeline) == 10

        for elem in compile_pipeline:
            assert isinstance(elem, TransformContainer)
            assert elem.transform is first_valid_transform

    def test_getitem(self):
        """Tests for the getitem dunder."""

        t0 = TransformContainer(qml.transform(first_valid_transform))
        t1 = TransformContainer(transform=qml.transform(second_valid_transform))
        t2 = TransformContainer(transform=qml.transform(informative_transform))
        pipeline = CompilePipeline([t0, t1, t2])

        assert pipeline[0] == t0
        assert pipeline[1] == t1
        assert pipeline[2] == t2

        assert pipeline[:2] == CompilePipeline([t0, t1])
        assert pipeline[::-1] == CompilePipeline([t2, t1, t0])

    def test_contains(self):
        """Test that we can check whether a transform or transform container exists in a transform."""

        t0 = TransformContainer(transform=qml.transform(first_valid_transform))
        t1 = TransformContainer(transform=qml.transform(second_valid_transform))
        t2 = TransformContainer(transform=qml.transform(informative_transform))
        pipeline = CompilePipeline([t0, t1, t2])

        assert t0 in pipeline
        assert t1 in pipeline
        assert t2 in pipeline
        assert qml.compile not in pipeline

        assert t0 in pipeline
        assert t1 in pipeline
        assert t2 in pipeline

        t_not = TransformContainer(transform=qml.compile)
        assert t_not not in pipeline

        assert "a" not in pipeline

    # ============ Parametrized addition tests ============
    @pytest.mark.parametrize(
        "left, right, expected_first, expected_second",
        [
            # container + container -> pipeline with 1 then 2
            pytest.param(
                TransformContainer(transform=qml.transform(first_valid_transform)),
                TransformContainer(transform=qml.transform(second_valid_transform)),
                first_valid_transform,
                second_valid_transform,
                id="container+container",
            ),
            # dispatcher + dispatcher -> pipeline with 1 then 2
            pytest.param(
                qml.transform(first_valid_transform),
                qml.transform(second_valid_transform),
                first_valid_transform,
                second_valid_transform,
                id="dispatcher+dispatcher",
            ),
            # dispatcher + container -> pipeline with dispatcher then container
            pytest.param(
                qml.transform(first_valid_transform),
                TransformContainer(transform=qml.transform(second_valid_transform)),
                first_valid_transform,
                second_valid_transform,
                id="dispatcher+container",
            ),
            # container + dispatcher -> pipeline with container then dispatcher
            pytest.param(
                TransformContainer(transform=qml.transform(first_valid_transform)),
                qml.transform(second_valid_transform),
                first_valid_transform,
                second_valid_transform,
                id="container+dispatcher",
            ),
            # pipeline + container -> new pipeline with container at end
            pytest.param(
                CompilePipeline(
                    [TransformContainer(transform=qml.transform(first_valid_transform))]
                ),
                TransformContainer(transform=qml.transform(second_valid_transform)),
                first_valid_transform,
                second_valid_transform,
                id="pipeline+container",
            ),
            # pipeline + dispatcher -> new pipeline with dispatcher at end
            pytest.param(
                CompilePipeline(
                    [TransformContainer(transform=qml.transform(first_valid_transform))]
                ),
                qml.transform(second_valid_transform),
                first_valid_transform,
                second_valid_transform,
                id="pipeline+dispatcher",
            ),
            # dispatcher + pipeline -> pipeline with dispatcher first, then pipeline contents
            pytest.param(
                qml.transform(first_valid_transform),
                CompilePipeline(
                    [TransformContainer(transform=qml.transform(second_valid_transform))]
                ),
                first_valid_transform,
                second_valid_transform,
                id="dispatcher+pipeline",
            ),
            # container + pipeline -> pipeline with container first, then pipeline contents
            pytest.param(
                TransformContainer(transform=qml.transform(first_valid_transform)),
                CompilePipeline(
                    [TransformContainer(transform=qml.transform(second_valid_transform))]
                ),
                first_valid_transform,
                second_valid_transform,
                id="container+pipeline",
            ),
            # pipeline + pipeline -> new pipeline with one followed by two
            pytest.param(
                CompilePipeline(
                    [TransformContainer(transform=qml.transform(first_valid_transform))]
                ),
                CompilePipeline(
                    [TransformContainer(transform=qml.transform(second_valid_transform))]
                ),
                first_valid_transform,
                second_valid_transform,
                id="pipeline+pipeline",
            ),
        ],
    )
    def test_addition_operations(self, left, right, expected_first, expected_second):
        """Test all addition operations between dispatchers, containers, and pipelines."""
        result = left + right
        assert isinstance(result, CompilePipeline)
        assert len(result) == 2
        assert result[0].transform is expected_first
        assert result[1].transform is expected_second

    # ============ Parametrized multiplication tests ============
    @pytest.mark.parametrize(
        "obj",
        [
            TransformContainer(transform=qml.transform(first_valid_transform)),
            qml.transform(first_valid_transform),
            CompilePipeline([TransformContainer(transform=qml.transform(first_valid_transform))]),
        ],
        ids=["container", "dispatcher", "pipeline"],
    )
    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_multiplication_operations(self, obj, n):
        """Test all multiplication operations for dispatchers, containers, and pipelines."""
        # Test left multiplication (obj * n)
        result = obj * n
        assert isinstance(result, CompilePipeline)
        assert len(result) == n
        assert all(t.transform is first_valid_transform for t in result)

        # Test right multiplication (n * obj)
        result = n * obj
        assert isinstance(result, CompilePipeline)
        assert len(result) == n
        assert all(t.transform is first_valid_transform for t in result)

    # ============ Error tests for invalid types ============
    @pytest.mark.parametrize(
        "obj",
        [
            qml.transform(first_valid_transform),
            TransformContainer(transform=qml.transform(first_valid_transform)),
        ],
        ids=["dispatcher", "container"],
    )
    @pytest.mark.parametrize("invalid_value", ["invalid", 42, [1, 2]], ids=["str", "int", "list"])
    def test_add_invalid_type_raises_error(self, obj, invalid_value):
        """Test that adding invalid types raises TypeError."""
        with pytest.raises(TypeError):
            _ = obj + invalid_value

    @pytest.mark.parametrize(
        "obj",
        [
            qml.transform(first_valid_transform),
            TransformContainer(transform=qml.transform(first_valid_transform)),
        ],
        ids=["dispatcher", "container"],
    )
    @pytest.mark.parametrize("invalid_value", ["invalid", 42], ids=["str", "int"])
    def test_radd_invalid_type_raises_error(self, obj, invalid_value):
        """Test that right addition with invalid types raises TypeError."""
        with pytest.raises(TypeError):
            _ = invalid_value + obj

    @pytest.mark.parametrize(
        "obj",
        [
            qml.transform(first_valid_transform),
            TransformContainer(transform=qml.transform(first_valid_transform)),
            CompilePipeline([TransformContainer(transform=qml.transform(first_valid_transform))]),
        ],
        ids=["dispatcher", "container", "pipeline"],
    )
    @pytest.mark.parametrize(
        "invalid_value", ["invalid", 3.5, [1, 2]], ids=["str", "float", "list"]
    )
    def test_mul_invalid_type_raises_error(self, obj, invalid_value):
        """Test that multiplying with invalid types raises TypeError."""
        with pytest.raises(TypeError):
            _ = obj * invalid_value
        with pytest.raises(TypeError):
            _ = invalid_value * obj

    @pytest.mark.parametrize(
        "obj",
        [
            qml.transform(first_valid_transform),
            TransformContainer(transform=qml.transform(first_valid_transform)),
            CompilePipeline([TransformContainer(transform=qml.transform(first_valid_transform))]),
        ],
        ids=["dispatcher", "container", "pipeline"],
    )
    def test_mul_negative_raises_error(self, obj):
        """Test that negative multiplication raises ValueError."""
        with pytest.raises(ValueError, match=r"Cannot multiply (transform)|(compile)"):
            _ = obj * -1

    def test_pipeline_rmul_final_transform_error(self):
        """Test that multiplying a pipeline with a final transform raises an error."""
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        pipeline = CompilePipeline([transform1, transform2])

        with pytest.raises(
            TransformError,
            match="Cannot multiply a compile pipeline that has a terminal transform",
        ):
            _ = 2 * pipeline

    def test_add_two_pipelines(self):
        """Test adding two compile pipelines"""
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(transform=qml.transform(second_valid_transform))

        compile_pipeline1 = CompilePipeline()
        compile_pipeline1.push_back(transform1)
        compile_pipeline1.push_back(transform1)
        compile_pipeline1.push_back(transform1)

        compile_pipeline2 = CompilePipeline()
        compile_pipeline1.push_back(transform2)
        compile_pipeline1.push_back(transform2)

        compile_pipeline = compile_pipeline1 + compile_pipeline2

        assert len(compile_pipeline) == 5

        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is first_valid_transform

        assert isinstance(compile_pipeline[1], TransformContainer)
        assert compile_pipeline[1].transform is first_valid_transform

        assert isinstance(compile_pipeline[2], TransformContainer)
        assert compile_pipeline[2].transform is first_valid_transform

        assert isinstance(compile_pipeline[3], TransformContainer)
        assert compile_pipeline[3].transform is second_valid_transform

        assert isinstance(compile_pipeline[4], TransformContainer)
        assert compile_pipeline[4].transform is second_valid_transform

    def test_add_both_final_compile_pipelines(self):
        """Test that an error is raised if two pipelines are added when both have
        terminal transforms"""
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )

        compile_pipeline1 = CompilePipeline()
        compile_pipeline1.push_back(transform1)
        compile_pipeline1.push_back(transform2)

        compile_pipeline2 = CompilePipeline()
        compile_pipeline2.push_back(transform1)
        compile_pipeline2.push_back(transform2)

        with pytest.raises(
            TransformError, match="The compile pipeline already has a terminal transform"
        ):
            _ = compile_pipeline1 + compile_pipeline2

    def test_add_pipelines_with_one_final_transform(self):
        """Test that compile pipelines are added correctly when one of them has a terminal
        transform."""
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )

        compile_pipeline1 = CompilePipeline()
        compile_pipeline1.push_back(transform1)

        compile_pipeline2 = CompilePipeline()
        compile_pipeline2.push_back(transform1)
        compile_pipeline2.push_back(transform2)

        merged_pipeline1 = compile_pipeline1 + compile_pipeline2
        assert len(merged_pipeline1) == 3

        assert isinstance(merged_pipeline1[0], TransformContainer)
        assert merged_pipeline1[0].transform is first_valid_transform

        assert isinstance(merged_pipeline1[1], TransformContainer)
        assert merged_pipeline1[1].transform is first_valid_transform

        assert isinstance(merged_pipeline1[2], TransformContainer)
        assert merged_pipeline1[2].transform is second_valid_transform

        merged_pipeline2 = compile_pipeline2 + compile_pipeline1
        assert len(merged_pipeline2) == 3

        assert isinstance(merged_pipeline2[0], TransformContainer)
        assert merged_pipeline2[0].transform is first_valid_transform

        assert isinstance(merged_pipeline2[1], TransformContainer)
        assert merged_pipeline2[1].transform is first_valid_transform

        assert isinstance(merged_pipeline2[2], TransformContainer)
        assert merged_pipeline2[2].transform is second_valid_transform

    @pytest.mark.parametrize(
        "right",
        [
            pytest.param(
                TransformContainer(transform=qml.transform(second_valid_transform)),
                id="pipeline+container",
            ),
            pytest.param(qml.transform(second_valid_transform), id="pipeline+dispatcher"),
        ],
    )
    def test_pipeline_add_maintains_final_transform_at_end(self, right):
        """Test that adding to a pipeline with final_transform keeps final at end."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        pipeline = CompilePipeline([container1])

        result = pipeline + right
        assert isinstance(result, CompilePipeline)
        assert len(result) == 2
        # Final transform should be at the end
        assert result[0].transform is second_valid_transform
        assert result[1].transform is first_valid_transform
        assert result[1].final_transform

    @pytest.mark.parametrize(
        "right",
        [
            pytest.param(
                TransformContainer(
                    transform=qml.transform(second_valid_transform, final_transform=True)
                ),
                id="pipeline+container_final",
            ),
            pytest.param(
                qml.transform(second_valid_transform, final_transform=True),
                id="pipeline+dispatcher_final",
            ),
        ],
    )
    def test_pipeline_add_with_both_final_transform_error(self, right):
        """Test that adding with final_transform to a pipeline with final_transform raises error."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        pipeline = CompilePipeline([container1])

        with pytest.raises(TransformError, match="already has a terminal transform"):
            _ = pipeline + right

    def test_actual_final_transform_error(self):
        """qml.gradients.param_shift + qml.gradients.hadamard should raise an error since both are final transforms."""
        with pytest.raises(TransformError, match="are final transforms and cannot be combined."):
            _ = qml.gradients.param_shift + qml.gradients.hadamard_grad

    def test_dispatcher_add_container_both_final_error(self):
        """Test that adding a final container to a final dispatcher raises an error."""
        dispatcher = qml.transform(first_valid_transform, final_transform=True)
        container = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        with pytest.raises(TransformError, match="are final transforms and cannot be combined"):
            _ = dispatcher + container

    def test_dispatcher_mul_final_transform_error(self):
        """Test that multiplying a final dispatcher by n > 1 raises an error."""
        dispatcher = qml.transform(first_valid_transform, final_transform=True)
        with pytest.raises(
            TransformError, match="is a final transform and cannot be applied more than once"
        ):
            _ = dispatcher * 2

    def test_container_add_container_both_final_error(self):
        """Test that adding two final containers raises an error."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        container2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        with pytest.raises(TransformError, match="are final transforms and cannot be combined"):
            _ = container1 + container2

    def test_container_add_dispatcher_both_final_error(self):
        """Test that adding a final dispatcher to a final container raises an error."""
        container = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        dispatcher = qml.transform(second_valid_transform, final_transform=True)
        with pytest.raises(TransformError, match="are final transforms and cannot be combined"):
            _ = container + dispatcher

    def test_container_mul_final_transform_error(self):
        """Test that multiplying a final container by n > 1 raises an error."""
        container = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        with pytest.raises(
            TransformError, match="is a final transform and cannot be applied more than once"
        ):
            _ = container * 2

    # ============ __radd__ tests ============
    @pytest.mark.parametrize(
        "left",
        [
            pytest.param(
                TransformContainer(transform=qml.transform(first_valid_transform)),
                id="container+pipeline",
            ),
            pytest.param(qml.transform(first_valid_transform), id="dispatcher+pipeline"),
        ],
    )
    def test_pipeline_radd(self, left):
        """Test that __radd__ prepends a transform to a pipeline."""
        container2 = TransformContainer(transform=qml.transform(second_valid_transform))
        pipeline = CompilePipeline([container2])

        result = left + pipeline
        assert isinstance(result, CompilePipeline)
        assert len(result) == 2
        assert result[0].transform is first_valid_transform
        assert result[1].transform is second_valid_transform

    def test_pipeline_radd_with_final_transform_error(self):
        """Test that __radd__ raises error when adding final to pipeline with final."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        container2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        pipeline = CompilePipeline([container2])

        with pytest.raises(TransformError, match="already has a terminal transform"):
            _ = container1 + pipeline

    # ============ __iadd__ tests ============
    def test_pipeline_iadd_container(self):
        """Test that __iadd__ appends a container in place."""
        container1 = TransformContainer(transform=qml.transform(first_valid_transform))
        container2 = TransformContainer(transform=qml.transform(second_valid_transform))
        pipeline = CompilePipeline([container1])

        original_id = id(pipeline)
        pipeline += container2

        assert id(pipeline) == original_id  # same object
        assert len(pipeline) == 2
        assert pipeline[0].transform is first_valid_transform
        assert pipeline[1].transform is second_valid_transform

    def test_pipeline_iadd_dispatcher(self):
        """Test that __iadd__ appends a dispatcher in place."""
        container1 = TransformContainer(transform=qml.transform(first_valid_transform))
        dispatcher = qml.transform(second_valid_transform)
        pipeline = CompilePipeline([container1])

        original_id = id(pipeline)
        pipeline += dispatcher

        assert id(pipeline) == original_id
        assert len(pipeline) == 2
        assert pipeline[0].transform is first_valid_transform
        assert pipeline[1].transform is second_valid_transform

    def test_pipeline_iadd_pipeline(self):
        """Test that __iadd__ extends with another pipeline in place."""
        container1 = TransformContainer(transform=qml.transform(first_valid_transform))
        container2 = TransformContainer(transform=qml.transform(second_valid_transform))
        pipeline1 = CompilePipeline([container1])
        pipeline2 = CompilePipeline([container2])

        original_id = id(pipeline1)
        pipeline1 += pipeline2

        assert id(pipeline1) == original_id
        assert len(pipeline1) == 2
        assert pipeline1[0].transform is first_valid_transform
        assert pipeline1[1].transform is second_valid_transform

    def test_pipeline_iadd_maintains_final_transform_at_end(self):
        """Test that __iadd__ keeps final transform at the end."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        container2 = TransformContainer(transform=qml.transform(second_valid_transform))
        pipeline = CompilePipeline([container1])

        pipeline += container2

        assert len(pipeline) == 2
        assert pipeline[0].transform is second_valid_transform
        assert pipeline[1].transform is first_valid_transform
        assert pipeline[1].final_transform

    def test_pipeline_iadd_with_both_final_transform_error(self):
        """Test that __iadd__ raises error when adding final to pipeline with final."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        container2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        pipeline = CompilePipeline([container1])

        with pytest.raises(TransformError, match="already has a terminal transform"):
            pipeline += container2

    def test_pipeline_iadd_pipeline_with_both_final_transform_error(self):
        """Test that __iadd__ raises error when adding pipeline with final to pipeline with final."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        container2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        pipeline1 = CompilePipeline([container1])
        pipeline2 = CompilePipeline([container2])

        with pytest.raises(TransformError, match="already has a terminal transform"):
            pipeline1 += pipeline2

    def test_pipeline_iadd_pipeline_maintains_final_transform_at_end(self):
        """Test that __iadd__ with pipeline keeps final transform at the end."""
        container1 = TransformContainer(
            transform=qml.transform(first_valid_transform, final_transform=True)
        )
        container2 = TransformContainer(transform=qml.transform(second_valid_transform))
        pipeline1 = CompilePipeline([container1])
        pipeline2 = CompilePipeline([container2])

        pipeline1 += pipeline2

        assert len(pipeline1) == 2
        assert pipeline1[0].transform is second_valid_transform
        assert pipeline1[1].transform is first_valid_transform
        assert pipeline1[1].final_transform

    def test_pipeline_iadd_pipeline_with_cotransform_cache(self):
        """Test that __iadd__ correctly handles cotransform_cache when adding pipelines."""

        @qml.qnode(qml.device("default.qubit"))
        def f(*_, **__):
            return qml.state()

        new_t = qml.transform(
            qml.gradients.param_shift.transform, classical_cotransform=lambda *args: 0
        )
        hybrid_t = TransformContainer(new_t, (), {"hybrid": True})

        # pipeline1 has no cotransform_cache, pipeline2 has cotransform_cache
        pipeline1 = CompilePipeline()
        pipeline2 = CompilePipeline((hybrid_t,))
        pipeline2.set_classical_component(f, (1,), {"a": 2})

        pipeline1 += pipeline2
        assert pipeline1.cotransform_cache == CotransformCache(f, (1,), {"a": 2})

    def test_pipeline_iadd_pipeline_with_both_cotransform_cache_error(self):
        """Test that __iadd__ raises error when both pipelines have cotransform_cache."""

        @qml.qnode(qml.device("default.qubit"))
        def f(*_, **__):
            return qml.state()

        new_t = qml.transform(
            qml.gradients.param_shift.transform, classical_cotransform=lambda *args: 0
        )
        hybrid_t = TransformContainer(new_t, (), {"hybrid": True})

        pipeline1 = CompilePipeline((hybrid_t,))
        pipeline1.set_classical_component(f, (1,), {"a": 2})
        pipeline2 = CompilePipeline((hybrid_t,))
        pipeline2.set_classical_component(f, (2,), {"b": 3})

        with pytest.raises(
            ValueError, match="Cannot add two compile pipelines with cotransform caches"
        ):
            pipeline1 += pipeline2

    def test_pipeline_iadd_invalid_type_raises_error(self):
        """Test that __iadd__ with invalid type raises TypeError."""
        container = TransformContainer(transform=qml.transform(first_valid_transform))
        pipeline = CompilePipeline([container])

        with pytest.raises(TypeError):
            pipeline += "invalid"

        with pytest.raises(TypeError):
            pipeline += 42

    def test_pipeline_add_invalid_type_raises_error(self):
        """Test that __add__ with invalid type raises TypeError."""
        container = TransformContainer(transform=qml.transform(first_valid_transform))
        pipeline = CompilePipeline([container])

        with pytest.raises(TypeError):
            _ = pipeline + "invalid"

        with pytest.raises(TypeError):
            _ = pipeline + 42

    def test_pipeline_radd_invalid_type_raises_error(self):
        """Test that __radd__ with invalid type raises TypeError."""
        container = TransformContainer(transform=qml.transform(first_valid_transform))
        pipeline = CompilePipeline([container])

        with pytest.raises(TypeError):
            _ = "invalid" + pipeline

        with pytest.raises(TypeError):
            _ = 42 + pipeline

    def test_repr_pipeline(self):
        """Test the string representation of a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(transform=qml.transform(second_valid_transform))

        compile_pipeline.push_back(transform1)
        compile_pipeline.push_back(transform2)

        str_pipeline = repr(compile_pipeline)
        assert (
            str_pipeline
            == "CompilePipeline("
            + str(first_valid_transform.__name__)
            + ", "
            + str(second_valid_transform.__name__)
            + ")"
        )

    def test_equality(self):
        """Tests that we can compare CompilePipeline objects with the '==' and '!=' operators."""
        t1 = TransformContainer(qml.transforms.compile, kwargs={"num_passes": 2})
        t2 = TransformContainer(qml.transforms.compile, kwargs={"num_passes": 2})
        t3 = TransformContainer(qml.transforms.transpile, kwargs={"coupling_map": [(0, 1), (1, 2)]})

        p1 = CompilePipeline([t1, t3])
        p2 = CompilePipeline([t2, t3])
        p3 = CompilePipeline([t3, t2])

        # test for equality of identical objects
        assert p1 == p2
        # test for inequality of different objects
        assert p1 != p3
        assert p1 != t1

        # Test inequality with different transforms
        t4 = TransformContainer(qml.transforms.transpile, kwargs={"coupling_map": [(0, 1), (2, 3)]})
        p4 = CompilePipeline([t1, t4])
        assert p1 != p4


class TestCompilePipelineConstruction:
    """Tests the different ways to initialize a CompilePipeline."""

    def test_empty_pipeline(self):
        """Test an empty pipeline."""

        pipeline = CompilePipeline()
        assert pipeline.is_empty()
        assert pipeline.cotransform_cache is None
        assert len(pipeline) == 0

        with pytest.raises(TransformError, match="The compile pipeline is empty"):
            pipeline.get_last()

    def test_list_of_transforms(self):
        """Tests constructing a CompilePipeline with a list of transforms."""

        pipeline = CompilePipeline(
            [
                TransformContainer(qml.transforms.compile),
                TransformContainer(qml.transforms.decompose),
                TransformContainer(qml.transforms.cancel_inverses),
            ]
        )
        assert len(pipeline) == 3

    def test_variable_length_arguments(self):
        """Tests constructing a CompilePipeline with a mixed series of things."""

        another_pipeline = CompilePipeline(
            qml.transforms.cancel_inverses,
            qml.transforms.diagonalize_measurements,
        )
        pipeline = CompilePipeline(
            qml.transforms.cancel_inverses,
            another_pipeline,
            TransformContainer(qml.transforms.decompose, kwargs={"gate_set": {qml.Rot, qml.CNOT}}),
        )
        assert len(pipeline) == 4


class TestCompilePipeline:
    """Test the compile pipeline class and its method."""

    def test_get_last(self):
        """Tests the get_last method"""
        pipeline = CompilePipeline()
        pipeline.add_transform(transform(first_valid_transform))
        pipeline.add_transform(transform(second_valid_transform))
        assert pipeline.get_last() == TransformContainer(
            transform=transform(second_valid_transform)
        )

    def test_push_back(self):
        """Test to push back multiple transforms into a pipeline and also the different methods of a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = TransformContainer(transform=transform(first_valid_transform))
        compile_pipeline.push_back(transform1)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is first_valid_transform

        transform2 = TransformContainer(transform=transform(second_valid_transform))
        compile_pipeline.push_back(transform2)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[1], TransformContainer)
        assert compile_pipeline[1].transform is second_valid_transform

        compile_pipeline.push_back(transform1)
        compile_pipeline.push_back(transform2)

        sub_pipeline_transforms = compile_pipeline[2:]
        assert len(sub_pipeline_transforms) == 2
        assert sub_pipeline_transforms[0] is transform1
        assert sub_pipeline_transforms[1] is transform2

        with pytest.raises(
            TransformError,
            match="Only transform container can be added to the compile pipeline.",
        ):
            compile_pipeline.push_back(10.0)

    def test_add_transform(self):
        """Test to add multiple transforms into a pipeline and also the different methods of a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform)
        compile_pipeline.add_transform(transform1)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is first_valid_transform

        transform2 = transform(second_valid_transform)
        compile_pipeline.add_transform(transform2)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[1], TransformContainer)
        assert compile_pipeline[1].transform is second_valid_transform

        compile_pipeline.add_transform(transform1)
        compile_pipeline.add_transform(transform2)

        sub_pipeline_transforms = compile_pipeline[2:]
        assert len(sub_pipeline_transforms) == 2
        assert sub_pipeline_transforms[0].transform is first_valid_transform
        assert sub_pipeline_transforms[1].transform is second_valid_transform

        with pytest.raises(
            TransformError,
            match="Only transform dispatcher can be added to the compile pipeline.",
        ):
            compile_pipeline.add_transform(10.0)

    def test_add_transform_with_expand(self):
        """Test to add a transform with expand into a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform, expand_transform=expand_transform)
        compile_pipeline.add_transform(transform1)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is expand_transform

        assert isinstance(compile_pipeline[1], TransformContainer)
        assert compile_pipeline[1].transform is first_valid_transform

    def test_pop_front(self):
        """Test the pop front method of the compile pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = TransformContainer(transform=transform(first_valid_transform))
        compile_pipeline.push_back(transform1)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is first_valid_transform

        transform_container = compile_pipeline.pop_front()

        assert compile_pipeline.is_empty()
        assert transform_container is transform1

    def test_insert_front(self):
        """Test to insert a transform (container) at the beginning of a compile pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = TransformContainer(transform=transform(first_valid_transform))
        compile_pipeline.push_back(transform1)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is first_valid_transform

        transform2 = TransformContainer(transform=transform(second_valid_transform))
        compile_pipeline.insert_front(transform2)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0] is transform2
        assert isinstance(compile_pipeline[1], TransformContainer)
        assert compile_pipeline[1] is transform1

        transform3 = TransformContainer(
            transform=transform(second_valid_transform, is_informative=True)
        )

        with pytest.raises(
            TransformError,
            match="Informative transforms can only be added at the end of the program.",
        ):
            compile_pipeline.insert_front(transform3)

    def test_insert_transform(self):
        """Test to insert a transform (dispatcher) at the beginning of a compile pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform)
        compile_pipeline.insert_front_transform(transform1)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is first_valid_transform

        transform2 = transform(second_valid_transform)
        compile_pipeline.insert_front_transform(transform2)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is second_valid_transform
        assert isinstance(compile_pipeline[1], TransformContainer)
        assert compile_pipeline[1].transform is first_valid_transform

        transform3 = transform(second_valid_transform, is_informative=True)

        with pytest.raises(
            TransformError,
            match="Informative transforms can only be added at the end of the program.",
        ):
            compile_pipeline.insert_front_transform(transform3)

    def test_insert_transform_with_expand(self):
        """Test to insert front a transform with expand into a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform, expand_transform=expand_transform)
        compile_pipeline.insert_front_transform(transform1)

        assert not compile_pipeline.is_empty()
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], TransformContainer)
        assert compile_pipeline[0].transform is expand_transform

        assert isinstance(compile_pipeline[1], TransformContainer)
        assert compile_pipeline[1].transform is first_valid_transform

    def test_valid_transforms(self):
        """Test adding transforms to a pipeline with a terminal transform."""
        compile_pipeline = CompilePipeline()
        transform1 = TransformContainer(qml.transform(first_valid_transform, is_informative=True))
        compile_pipeline.push_back(transform1)

        t_normal = TransformContainer(qml.transform(second_valid_transform))
        compile_pipeline.push_back(t_normal)
        print(compile_pipeline)
        assert len(compile_pipeline) == 2
        assert compile_pipeline[0] is t_normal
        assert compile_pipeline[1] is transform1

        t_normal2 = TransformContainer(qml.transform(first_valid_transform))
        compile_pipeline.push_back(t_normal2)
        assert compile_pipeline[0] is t_normal
        assert compile_pipeline[1] is t_normal2
        assert compile_pipeline[2] is transform1

        with pytest.raises(
            TransformError, match="The compile pipeline already has a terminal transform."
        ):
            compile_pipeline.push_back(transform1)

        transform2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        with pytest.raises(
            TransformError, match="The compile pipeline already has a terminal transform."
        ):
            compile_pipeline.push_back(transform2)


class TestClassicalCotransfroms:
    """Test for handling the classical cotransform component."""

    def test_classical_cotransform_caching(self):
        """Tests for setting the classical cotransform."""

        @qml.qnode(qml.device("default.qubit"))
        def f(*_, **__):
            return qml.state()

        pipeline1 = CompilePipeline()  # no hybrid transforms
        assert pipeline1.cotransform_cache is None
        pipeline1.set_classical_component(f, (1,), {"a": 2})
        assert pipeline1.cotransform_cache is None

        new_t = qml.transform(
            qml.gradients.param_shift.transform, classical_cotransform=lambda *args: 0
        )

        hybrid_t = TransformContainer(new_t, (), {"hybrid": True})
        pipeline2 = CompilePipeline((hybrid_t,))
        pipeline2.set_classical_component(f, (1,), {"a": 2})
        assert pipeline2.cotransform_cache == CotransformCache(f, (1,), {"a": 2})

        pipeline3 = pipeline1 + pipeline2
        assert pipeline3.cotransform_cache == CotransformCache(f, (1,), {"a": 2})

        pipeline4 = pipeline2 + pipeline1
        assert pipeline4.cotransform_cache == CotransformCache(f, (1,), {"a": 2})

        with pytest.raises(ValueError, match=r"pipelines with cotransform caches"):
            _ = pipeline2 + pipeline2

    @pytest.mark.parametrize("arg", (0.5, rx.PyGraph()))
    def test_error_on_numpy_qnode(self, arg):
        """Test an error is raised if there are no trainable parameters for a hybrid pipeline."""

        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        circuit = qml.gradients.param_shift(circuit, hybrid=True)
        circuit.transform_program.set_classical_component(circuit, (arg,), {})

        tape = qml.tape.QuantumScript([], [])
        with pytest.raises(QuantumFunctionError, match="No trainable parameters"):
            circuit.transform_program((tape,))


class TestCompilePipelineCall:
    """Tests for calling a CompilePipeline on a batch of quantum tapes."""

    def test_call_on_empty_pipeline(self):
        """Tests that an empty pipeline returns input tapes with the null postprocessing function."""

        batch = qml.tape.QuantumScript([], [qml.state()])

        prog = CompilePipeline()
        new_batch, postprocessing = prog(batch)

        assert new_batch is batch
        assert postprocessing is null_postprocessing

        obj = [1, 2, 3, "b"]
        assert null_postprocessing(obj) is obj

    def test_single_compile_pipeline(self):
        """Basic test with a single transform that only modifies the tape but not the results."""

        def single_null_postprocessing(results):
            return results[0]

        def remove_operation_at_index(
            tape: QuantumScript, index: int
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            """A valid transform."""
            new_ops = list(tape.operations)
            new_ops.pop(index)  # pylint:disable=protected-access
            return (
                qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots),
            ), single_null_postprocessing

        container = TransformContainer(transform(remove_operation_at_index), kwargs={"index": 1})
        prog = CompilePipeline((container,))

        tape0 = qml.tape.QuantumScript(
            [qml.S(0), qml.T(1), qml.SX(2)], [qml.expval(qml.PauliZ(0))], shots=100
        )
        batch = (tape0,)
        new_batch, fn = prog(batch)

        assert len(new_batch) == 1
        expected = [qml.S(0), qml.SX(2), qml.expval(qml.PauliZ(0))]
        for op1, op2 in zip(expected, new_batch[0]):
            qml.assert_equal(op1, op2)
        assert new_batch[0].shots == qml.measurements.Shots(100)

        assert fn.func is _apply_postprocessing_stack
        assert fn.args == tuple()

        assert len(fn.keywords["postprocessing_stack"]) == 1
        postprocessing0 = fn.keywords["postprocessing_stack"][0]
        assert postprocessing0.func is _batch_postprocessing
        assert postprocessing0.args == tuple()
        assert postprocessing0.keywords["individual_fns"] == [single_null_postprocessing]
        assert postprocessing0.keywords["slices"] == [slice(0, 1)]

        results = (2.0,)
        assert fn(results) == (2.0,)

    def test_chain_two_postprocessings(self):
        """Test postprocessing functions applied in reverse order."""

        def add_one(results):
            return results[0] + 1.0

        def scale_two(results):
            return results[0] * 2.0

        def transform_add(tape: qml.tape.QuantumTape):
            """A valid transform."""
            return (tape,), add_one

        def transform_mul(tape: qml.tape.QuantumTape):
            return (tape,), scale_two

        container1 = TransformContainer(transform(transform_add))
        container2 = TransformContainer(transform(transform_mul))
        prog = CompilePipeline((container1, container2))

        tape0 = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))], shots=100)
        batch = (tape0,)
        new_batch, fn = prog(batch)

        assert len(new_batch) == 1
        assert new_batch[0] is tape0

        assert fn.func is _apply_postprocessing_stack
        assert fn.args == tuple()
        assert len(fn.keywords["postprocessing_stack"]) == 2

        postprocessing0 = fn.keywords["postprocessing_stack"][0]
        assert postprocessing0.func is _batch_postprocessing
        assert postprocessing0.args == tuple()
        assert postprocessing0.keywords["individual_fns"] == [
            add_one,
        ]
        assert postprocessing0.keywords["slices"] == [slice(0, 1)]

        postprocessing1 = fn.keywords["postprocessing_stack"][1]
        assert postprocessing1.func is _batch_postprocessing
        assert postprocessing1.args == tuple()
        assert postprocessing1.keywords["individual_fns"] == [
            scale_two,
        ]
        assert postprocessing1.keywords["slices"] == [slice(0, 1)]

        results = (1.0,)
        expected = (3.0,)  # 2.0 * 1.0 + 1.0
        assert fn(results) == expected

        # Test reverse direction

        prog_reverse = CompilePipeline((container2, container1))
        new_batch, fn = prog_reverse(batch)

        assert len(new_batch) == 1
        assert new_batch[0] is tape0

        assert fn.func is _apply_postprocessing_stack
        assert fn.args == tuple()
        assert len(fn.keywords["postprocessing_stack"]) == 2

        postprocessing0 = fn.keywords["postprocessing_stack"][0]
        assert postprocessing0.func is _batch_postprocessing
        assert postprocessing0.args == tuple()
        assert postprocessing0.keywords["individual_fns"] == [
            scale_two,
        ]
        assert postprocessing0.keywords["slices"] == [slice(0, 1)]

        postprocessing1 = fn.keywords["postprocessing_stack"][1]
        assert postprocessing1.func is _batch_postprocessing
        assert postprocessing1.args == tuple()
        assert postprocessing1.keywords["individual_fns"] == [
            add_one,
        ]
        assert postprocessing1.keywords["slices"] == [slice(0, 1)]

        results = (1.0,)
        expected = (4.0,)  # (1.0 + 1.0) * 2.0
        assert fn(results) == expected

    def test_postprocessing_batch_circuit_ragged(self):
        """Tests postprocessing when the input is a batch and the transform outputs different sizes of batches
        for each input tape.
        """

        # note this does not work for partitioned shots
        def sum_measurements(results: ResultBatch) -> Result:
            return sum(results)

        def split_sum_terms(tape):
            sum_obj = tape.measurements[0].obs
            new_tapes = tuple(
                QuantumScript(tape.operations, [qml.expval(o)], shots=tape.shots) for o in sum_obj
            )

            return new_tapes, sum_measurements

        container = TransformContainer(transform(split_sum_terms))
        prog = CompilePipeline((container,))

        op = qml.Rot(1.2, 2.3, 3.4, wires=0)

        orig1 = qml.tape.QuantumScript([op], [qml.expval(qml.sum(qml.PauliX(0), qml.PauliZ(0)))])
        orig2 = qml.tape.QuantumScript(
            [op], [qml.expval(qml.sum(qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)))]
        )
        orig3 = qml.tape.QuantumScript(
            [op], [qml.expval(qml.sum(*(qml.PauliX(i) for i in range(5))))]
        )  # contributes 5 terms

        batch, fn = prog((orig1, orig2, orig3))

        assert len(batch) == 10

        assert fn.func is _apply_postprocessing_stack
        assert not fn.args
        fn_stack = fn.keywords["postprocessing_stack"]
        assert len(fn_stack) == 1

        assert fn_stack[0].func is _batch_postprocessing
        assert fn_stack[0].keywords["individual_fns"] == [
            sum_measurements,
            sum_measurements,
            sum_measurements,
        ]
        assert fn_stack[0].keywords["slices"] == [slice(0, 2), slice(2, 5), slice(5, 10)]

        dummy_results = (1, 2, 3, 4, 5, 1, 1, 1, 1, 1)
        assert fn(dummy_results) == (3, 12, 5)

    @pytest.mark.capture
    def test_call_jaxpr_empty(self):
        """Test that calling an empty CompilePipeline with jaxpr returns untransformed ClosedJaxpr."""
        # pylint: disable=import-outside-toplevel
        import jax

        pipeline = CompilePipeline()
        const = jax.numpy.array(3.5)

        def f(x, n):
            qml.IsingXX(x, [0, 1])

            @qml.for_loop(n)
            def loop_fn(i):
                qml.Hadamard(i)
                qml.RX(const, i)

            loop_fn()
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.5, 5)
        transformed_jaxpr = pipeline(jaxpr.jaxpr, jaxpr.consts, 1.5, 5)
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
        assert transformed_jaxpr.consts == jaxpr.consts

        for eqn1, eqn2 in zip(jaxpr.eqns, transformed_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive
            # Jaxpr equality is based on identity and so since they were constructed
            # seperately, they will not be equal (hence the string check)
            assert str(eqn1.params) == str(eqn2.params)

    @pytest.mark.capture
    def test_call_jaxpr_single_transform(self):
        """Test that calling a CompilePipeline with a single transform with jaxpr works correctly."""
        # pylint: disable=import-outside-toplevel
        import jax

        pipeline = CompilePipeline()
        pipeline.add_transform(qml.transforms.cancel_inverses)

        def f():
            qml.H(0)
            qml.X(1)
            qml.H(0)
            qml.X(1)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        transformed_jaxpr = pipeline(jaxpr.jaxpr, jaxpr.consts)
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
        assert transformed_jaxpr.consts == jaxpr.consts

        assert len(transformed_jaxpr.eqns) == 2
        # pylint: disable=protected-access
        assert transformed_jaxpr.eqns[0].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.measurements.ExpectationMP._obs_primitive

    @pytest.mark.capture
    def test_call_jaxpr_multiple_transforms(self):
        """Test that calling a CompilePipeline with multiple transforms with jaxpr works correctly."""
        # pylint: disable=import-outside-toplevel
        import jax

        pipeline = CompilePipeline()
        pipeline.add_transform(qml.transforms.cancel_inverses)
        pipeline.add_transform(qml.transforms.defer_measurements, num_wires=3)
        pipeline.add_transform(
            qml.transforms.decompose,
            stopping_condition=lambda op: op.name != "IsingXX",
        )

        def f():
            qml.H(0)
            qml.H(0)
            qml.measure(1)
            qml.IsingXX(0.5, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        transformed_jaxpr = pipeline(jaxpr.jaxpr, jaxpr.consts)
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)

        # pylint: disable=protected-access
        isingxx_decomp = [qml.CNOT(wires=[0, 1]), qml.RX(0.5, wires=[0]), qml.CNOT(wires=[0, 1])]
        expected_primitives = [
            qml.CNOT._primitive,
            *[op._primitive for op in isingxx_decomp],
            qml.PauliZ._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        for eqn, expected_primitive in zip(
            transformed_jaxpr.eqns, expected_primitives, strict=True
        ):
            assert eqn.primitive == expected_primitive


class TestCompilePipelineIntegration:
    """Test the compile pipeline and its integration with QNodes"""

    def test_qnode_integration(self):
        """Test the integration with QNode wiht two similar transforms."""

        dispatched_transform = transform(first_valid_transform)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        new_qnode = dispatched_transform(dispatched_transform(qnode_circuit, 0), 0)

        pipeline = new_qnode.transform_program
        transformed_qnode_rep = repr(pipeline)
        assert (
            transformed_qnode_rep
            == "CompilePipeline("
            + str(first_valid_transform.__name__)
            + ", "
            + str(first_valid_transform.__name__)
            + ")"
        )

        assert not pipeline.is_empty()
        assert len(pipeline) == 2
        assert pipeline[0].transform is first_valid_transform
        assert pipeline[1].transform is first_valid_transform

    def test_qnode_integration_informative_transform(self):
        """Test the integration with QNode with two transforms, one of which is
        informative."""
        dispatched_transform_1 = transform(first_valid_transform)
        dispatched_transform_2 = transform(informative_transform, is_informative=True)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        new_qnode = dispatched_transform_2(dispatched_transform_1(qnode_circuit, 0))

        pipeline = new_qnode.transform_program
        transformed_qnode_rep = repr(pipeline)
        assert (
            transformed_qnode_rep
            == "CompilePipeline("
            + str(first_valid_transform.__name__)
            + ", "
            + str(informative_transform.__name__)
            + ")"
        )

        assert not pipeline.is_empty()
        assert len(pipeline) == 2
        assert pipeline[0].transform is first_valid_transform
        assert pipeline[1].transform is informative_transform

        result = new_qnode(0.1)
        assert result == (3,)

    def test_qnode_integration_different_transforms(self):
        """Test the integration with QNode with two different transforms."""

        dispatched_transform_1 = transform(first_valid_transform)
        dispatched_transform_2 = transform(second_valid_transform)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        new_qnode = dispatched_transform_2(dispatched_transform_1(qnode_circuit, 0), 0)

        pipeline = new_qnode.transform_program
        transformed_qnode_rep = repr(pipeline)
        assert (
            transformed_qnode_rep
            == "CompilePipeline("
            + str(first_valid_transform.__name__)
            + ", "
            + str(second_valid_transform.__name__)
            + ")"
        )

        assert not pipeline.is_empty()
        assert len(pipeline) == 2
        assert pipeline[0].transform is first_valid_transform
        assert pipeline[1].transform is second_valid_transform
