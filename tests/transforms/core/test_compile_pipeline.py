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
    BoundTransform,
    CompilePipeline,
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

        transform1 = BoundTransform(qml.transform(first_valid_transform))
        populated_prog = CompilePipeline((transform1,))
        assert populated_prog

    def test_iter_pipeline(self):
        """Test iteration over the compile pipeline."""
        compile_pipeline = CompilePipeline()
        transform1 = BoundTransform(qml.transform(first_valid_transform))

        for _ in range(10):
            compile_pipeline.append(transform1)

        assert len(compile_pipeline) == 10

        for elem in compile_pipeline:
            assert isinstance(elem, BoundTransform)
            assert elem.tape_transform is first_valid_transform

    def test_getitem(self):
        """Tests for the getitem dunder."""

        t0 = BoundTransform(qml.transform(first_valid_transform))
        t1 = BoundTransform(qml.transform(second_valid_transform))
        t2 = BoundTransform(qml.transform(informative_transform))
        pipeline = CompilePipeline([t0, t1, t2])

        assert pipeline[0] == t0
        assert pipeline[1] == t1
        assert pipeline[2] == t2

        assert pipeline[:2] == CompilePipeline([t0, t1])
        assert pipeline[::-1] == CompilePipeline([t2, t1, t0])

    def test_contains(self):
        """Test that we can check whether a transform or transform container exists in a transform."""

        t0 = BoundTransform(qml.transform(first_valid_transform))
        t1 = BoundTransform(qml.transform(second_valid_transform))
        t2 = BoundTransform(qml.transform(informative_transform))
        pipeline = CompilePipeline([t0, t1, t2])

        assert t0 in pipeline
        assert t1 in pipeline
        assert t2 in pipeline
        assert qml.compile not in pipeline

        assert t0 in pipeline
        assert t1 in pipeline
        assert t2 in pipeline

        t_not = BoundTransform(qml.compile)
        assert t_not not in pipeline

        assert "a" not in pipeline

    # ============ Parametrized addition tests ============
    @pytest.mark.parametrize(
        "left, right, expected_first, expected_second",
        [
            # container + container -> pipeline with 1 then 2
            pytest.param(
                BoundTransform(qml.transform(first_valid_transform)),
                BoundTransform(qml.transform(second_valid_transform)),
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
                BoundTransform(qml.transform(second_valid_transform)),
                first_valid_transform,
                second_valid_transform,
                id="dispatcher+container",
            ),
            # container + dispatcher -> pipeline with container then dispatcher
            pytest.param(
                BoundTransform(qml.transform(first_valid_transform)),
                qml.transform(second_valid_transform),
                first_valid_transform,
                second_valid_transform,
                id="container+dispatcher",
            ),
            # pipeline + container -> new pipeline with container at end
            pytest.param(
                CompilePipeline([BoundTransform(qml.transform(first_valid_transform))]),
                BoundTransform(qml.transform(second_valid_transform)),
                first_valid_transform,
                second_valid_transform,
                id="pipeline+container",
            ),
            # pipeline + dispatcher -> new pipeline with dispatcher at end
            pytest.param(
                CompilePipeline([BoundTransform(qml.transform(first_valid_transform))]),
                qml.transform(second_valid_transform),
                first_valid_transform,
                second_valid_transform,
                id="pipeline+dispatcher",
            ),
            # dispatcher + pipeline -> pipeline with dispatcher first, then pipeline contents
            pytest.param(
                qml.transform(first_valid_transform),
                CompilePipeline([BoundTransform(qml.transform(second_valid_transform))]),
                first_valid_transform,
                second_valid_transform,
                id="dispatcher+pipeline",
            ),
            # container + pipeline -> pipeline with container first, then pipeline contents
            pytest.param(
                BoundTransform(qml.transform(first_valid_transform)),
                CompilePipeline([BoundTransform(qml.transform(second_valid_transform))]),
                first_valid_transform,
                second_valid_transform,
                id="container+pipeline",
            ),
            # pipeline + pipeline -> new pipeline with one followed by two
            pytest.param(
                CompilePipeline([BoundTransform(qml.transform(first_valid_transform))]),
                CompilePipeline([BoundTransform(qml.transform(second_valid_transform))]),
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
        assert result[0].tape_transform is expected_first
        assert result[1].tape_transform is expected_second

    @pytest.mark.parametrize(
        "first, second, expected",
        [
            pytest.param(
                qml.transform(first_valid_transform, expand_transform=expand_transform),
                qml.transform(second_valid_transform),
                [expand_transform, first_valid_transform, second_valid_transform],
            ),
            pytest.param(
                qml.transform(first_valid_transform),
                qml.transform(second_valid_transform, expand_transform=expand_transform),
                [first_valid_transform, expand_transform, second_valid_transform],
            ),
            pytest.param(
                qml.transform(first_valid_transform, expand_transform=expand_transform),
                CompilePipeline([BoundTransform(qml.transform(second_valid_transform))]),
                [expand_transform, first_valid_transform, second_valid_transform],
            ),
            pytest.param(
                CompilePipeline([BoundTransform(qml.transform(second_valid_transform))]),
                qml.transform(first_valid_transform, expand_transform=expand_transform),
                [second_valid_transform, expand_transform, first_valid_transform],
            ),
        ],
    )
    def test_additions_with_expand_transforms(self, first, second, expected):
        """Tests that the expand_transform is included in the result of additions."""
        result = first + second
        assert isinstance(result, CompilePipeline)
        for actual, exp in zip(result, expected, strict=True):
            assert actual.tape_transform == exp

    # ============ Parametrized multiplication tests ============
    @pytest.mark.parametrize(
        "obj",
        [
            BoundTransform(qml.transform(first_valid_transform)),
            qml.transform(first_valid_transform),
            CompilePipeline([BoundTransform(qml.transform(first_valid_transform))]),
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
        assert all(t.tape_transform is first_valid_transform for t in result)

        # Test right multiplication (n * obj)
        result = n * obj
        assert isinstance(result, CompilePipeline)
        assert len(result) == n
        assert all(t.tape_transform is first_valid_transform for t in result)

    def test_multiplication_with_expand_transform(self):
        """Tests that the expand_transform is multiplied with the original transform."""

        result = qml.transform(first_valid_transform, expand_transform=expand_transform) * 3
        assert len(result) == 6
        for i in range(0, 6, 2):
            assert result[i].tape_transform == expand_transform
        for i in range(1, 6, 2):
            assert result[i].tape_transform == first_valid_transform

    # ============ Error tests for invalid types ============
    @pytest.mark.parametrize(
        "obj",
        [
            qml.transform(first_valid_transform),
            BoundTransform(qml.transform(first_valid_transform)),
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
            BoundTransform(qml.transform(first_valid_transform)),
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
            BoundTransform(qml.transform(first_valid_transform)),
            CompilePipeline([BoundTransform(qml.transform(first_valid_transform))]),
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
            BoundTransform(qml.transform(first_valid_transform)),
            CompilePipeline([BoundTransform(qml.transform(first_valid_transform))]),
        ],
        ids=["dispatcher", "container", "pipeline"],
    )
    def test_mul_negative_raises_error(self, obj):
        """Test that negative multiplication raises ValueError."""
        with pytest.raises(ValueError, match=r"Cannot multiply (transform)|(compile)"):
            _ = obj * -1

    def test_pipeline_rmul_final_transform_error(self):
        """Test that multiplying a pipeline with a final transform raises an error."""
        transform1 = BoundTransform(qml.transform(first_valid_transform))
        transform2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))
        pipeline = CompilePipeline([transform1, transform2])

        with pytest.raises(
            TransformError,
            match="Cannot multiply a compile pipeline that has a terminal transform",
        ):
            _ = 2 * pipeline

    def test_add_two_pipelines(self):
        """Test adding two compile pipelines"""
        transform1 = BoundTransform(qml.transform(first_valid_transform))
        transform2 = BoundTransform(qml.transform(second_valid_transform))

        compile_pipeline1 = CompilePipeline()
        compile_pipeline1.append(transform1)
        compile_pipeline1.append(transform1)
        compile_pipeline1.append(transform1)

        compile_pipeline2 = CompilePipeline()
        compile_pipeline1.append(transform2)
        compile_pipeline1.append(transform2)

        compile_pipeline = compile_pipeline1 + compile_pipeline2

        assert len(compile_pipeline) == 5

        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is first_valid_transform

        assert isinstance(compile_pipeline[1], BoundTransform)
        assert compile_pipeline[1].tape_transform is first_valid_transform

        assert isinstance(compile_pipeline[2], BoundTransform)
        assert compile_pipeline[2].tape_transform is first_valid_transform

        assert isinstance(compile_pipeline[3], BoundTransform)
        assert compile_pipeline[3].tape_transform is second_valid_transform

        assert isinstance(compile_pipeline[4], BoundTransform)
        assert compile_pipeline[4].tape_transform is second_valid_transform

    def test_add_both_final_compile_pipelines(self):
        """Test that an error is raised if two pipelines are added when both have
        terminal transforms"""
        transform1 = BoundTransform(qml.transform(first_valid_transform))
        transform2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))

        compile_pipeline1 = CompilePipeline()
        compile_pipeline1.append(transform1)
        compile_pipeline1.append(transform2)

        compile_pipeline2 = CompilePipeline()
        compile_pipeline2.append(transform1)
        compile_pipeline2.append(transform2)

        with pytest.raises(
            TransformError, match="The compile pipeline already has a terminal transform"
        ):
            _ = compile_pipeline1 + compile_pipeline2

    def test_add_pipelines_with_one_final_transform(self):
        """Test that compile pipelines are added correctly when one of them has a terminal
        transform."""
        transform1 = BoundTransform(qml.transform(first_valid_transform))
        transform2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))

        compile_pipeline1 = CompilePipeline()
        compile_pipeline1.append(transform1)

        compile_pipeline2 = CompilePipeline()
        compile_pipeline2.append(transform1)
        compile_pipeline2.append(transform2)

        merged_pipeline1 = compile_pipeline1 + compile_pipeline2
        assert len(merged_pipeline1) == 3

        assert isinstance(merged_pipeline1[0], BoundTransform)
        assert merged_pipeline1[0].tape_transform is first_valid_transform

        assert isinstance(merged_pipeline1[1], BoundTransform)
        assert merged_pipeline1[1].tape_transform is first_valid_transform

        assert isinstance(merged_pipeline1[2], BoundTransform)
        assert merged_pipeline1[2].tape_transform is second_valid_transform

        merged_pipeline2 = compile_pipeline2 + compile_pipeline1
        assert len(merged_pipeline2) == 3

        assert isinstance(merged_pipeline2[0], BoundTransform)
        assert merged_pipeline2[0].tape_transform is first_valid_transform

        assert isinstance(merged_pipeline2[1], BoundTransform)
        assert merged_pipeline2[1].tape_transform is first_valid_transform

        assert isinstance(merged_pipeline2[2], BoundTransform)
        assert merged_pipeline2[2].tape_transform is second_valid_transform

    @pytest.mark.parametrize(
        "right",
        [
            pytest.param(
                BoundTransform(qml.transform(second_valid_transform)),
                id="pipeline+container",
            ),
            pytest.param(qml.transform(second_valid_transform), id="pipeline+dispatcher"),
        ],
    )
    def test_pipeline_add_maintains_final_transform_at_end(self, right):
        """Test that adding to a pipeline with final_transform keeps final at end."""
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        pipeline = CompilePipeline([container1])

        result = pipeline + right
        assert isinstance(result, CompilePipeline)
        assert len(result) == 2
        # Final transform should be at the end
        assert result[0].tape_transform is second_valid_transform
        assert result[1].tape_transform is first_valid_transform
        assert result[1].is_final_transform

    @pytest.mark.parametrize(
        "right",
        [
            pytest.param(
                BoundTransform(qml.transform(second_valid_transform, final_transform=True)),
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
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
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
        container = BoundTransform(qml.transform(second_valid_transform, final_transform=True))
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
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        container2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))
        with pytest.raises(TransformError, match="are final transforms and cannot be combined"):
            _ = container1 + container2

    def test_container_add_dispatcher_both_final_error(self):
        """Test that adding a final dispatcher to a final container raises an error."""
        container = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        dispatcher = qml.transform(second_valid_transform, final_transform=True)
        with pytest.raises(TransformError, match="are final transforms and cannot be combined"):
            _ = container + dispatcher

    def test_container_mul_final_transform_error(self):
        """Test that multiplying a final container by n > 1 raises an error."""
        container = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        with pytest.raises(
            TransformError, match="is a final transform and cannot be applied more than once"
        ):
            _ = container * 2

    # ============ __radd__ tests ============
    @pytest.mark.parametrize(
        "left",
        [
            pytest.param(
                BoundTransform(qml.transform(first_valid_transform)),
                id="container+pipeline",
            ),
            pytest.param(qml.transform(first_valid_transform), id="dispatcher+pipeline"),
        ],
    )
    def test_pipeline_radd(self, left):
        """Test that __radd__ prepends a transform to a pipeline."""
        container2 = BoundTransform(qml.transform(second_valid_transform))
        pipeline = CompilePipeline([container2])

        result = left + pipeline
        assert isinstance(result, CompilePipeline)
        assert len(result) == 2
        assert result[0].tape_transform is first_valid_transform
        assert result[1].tape_transform is second_valid_transform

    def test_pipeline_radd_with_final_transform_error(self):
        """Test that __radd__ raises error when adding final to pipeline with final."""
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        container2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))
        pipeline = CompilePipeline([container2])

        with pytest.raises(TransformError, match="already has a terminal transform"):
            _ = container1 + pipeline

    # ============ __iadd__ tests ============
    def test_pipeline_iadd_container(self):
        """Test that __iadd__ appends a container in place."""
        container1 = BoundTransform(qml.transform(first_valid_transform))
        container2 = BoundTransform(qml.transform(second_valid_transform))
        pipeline = CompilePipeline([container1])

        original_id = id(pipeline)
        pipeline += container2

        assert id(pipeline) == original_id  # same object
        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is second_valid_transform

    def test_pipeline_iadd_dispatcher(self):
        """Test that __iadd__ appends a dispatcher in place."""
        container1 = BoundTransform(qml.transform(first_valid_transform))
        dispatcher = qml.transform(second_valid_transform)
        pipeline = CompilePipeline([container1])

        original_id = id(pipeline)
        pipeline += dispatcher

        assert id(pipeline) == original_id
        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is second_valid_transform

    def test_pipeline_iadd_pipeline(self):
        """Test that __iadd__ extends with another pipeline in place."""
        container1 = BoundTransform(qml.transform(first_valid_transform))
        container2 = BoundTransform(qml.transform(second_valid_transform))
        pipeline1 = CompilePipeline([container1])
        pipeline2 = CompilePipeline([container2])

        original_id = id(pipeline1)
        pipeline1 += pipeline2

        assert id(pipeline1) == original_id
        assert len(pipeline1) == 2
        assert pipeline1[0].tape_transform is first_valid_transform
        assert pipeline1[1].tape_transform is second_valid_transform

    def test_pipeline_iadd_maintains_final_transform_at_end(self):
        """Test that __iadd__ keeps final transform at the end."""
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        container2 = BoundTransform(qml.transform(second_valid_transform))
        pipeline = CompilePipeline([container1])

        pipeline += container2

        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is second_valid_transform
        assert pipeline[1].tape_transform is first_valid_transform
        assert pipeline[1].is_final_transform

    def test_pipeline_iadd_with_both_final_transform_error(self):
        """Test that __iadd__ raises error when adding final to pipeline with final."""
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        container2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))
        pipeline = CompilePipeline([container1])

        with pytest.raises(TransformError, match="already has a terminal transform"):
            pipeline += container2

    def test_pipeline_iadd_pipeline_with_both_final_transform_error(self):
        """Test that __iadd__ raises error when adding pipeline with final to pipeline with final."""
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        container2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))
        pipeline1 = CompilePipeline([container1])
        pipeline2 = CompilePipeline([container2])

        with pytest.raises(TransformError, match="already has a terminal transform"):
            pipeline1 += pipeline2

    def test_pipeline_iadd_pipeline_maintains_final_transform_at_end(self):
        """Test that __iadd__ with pipeline keeps final transform at the end."""
        container1 = BoundTransform(qml.transform(first_valid_transform, final_transform=True))
        container2 = BoundTransform(qml.transform(second_valid_transform))
        pipeline1 = CompilePipeline([container1])
        pipeline2 = CompilePipeline([container2])

        pipeline1 += pipeline2

        assert len(pipeline1) == 2
        assert pipeline1[0].tape_transform is second_valid_transform
        assert pipeline1[1].tape_transform is first_valid_transform
        assert pipeline1[1].is_final_transform

    def test_pipeline_iadd_pipeline_with_cotransform_cache(self):
        """Test that __iadd__ correctly handles cotransform_cache when adding pipelines."""

        @qml.qnode(qml.device("default.qubit"))
        def f(*_, **__):
            return qml.state()

        new_t = qml.transform(
            qml.gradients.param_shift.tape_transform, classical_cotransform=lambda *args: 0
        )
        hybrid_t = BoundTransform(new_t, (), {"hybrid": True})

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
            qml.gradients.param_shift.tape_transform, classical_cotransform=lambda *args: 0
        )
        hybrid_t = BoundTransform(new_t, (), {"hybrid": True})

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
        container = BoundTransform(qml.transform(first_valid_transform))
        pipeline = CompilePipeline([container])

        with pytest.raises(TypeError):
            pipeline += "invalid"

        with pytest.raises(TypeError):
            pipeline += 42

    def test_pipeline_add_invalid_type_raises_error(self):
        """Test that __add__ with invalid type raises TypeError."""
        container = BoundTransform(qml.transform(first_valid_transform))
        pipeline = CompilePipeline([container])

        with pytest.raises(TypeError):
            _ = pipeline + "invalid"

        with pytest.raises(TypeError):
            _ = pipeline + 42

    def test_pipeline_radd_invalid_type_raises_error(self):
        """Test that __radd__ with invalid type raises TypeError."""
        container = BoundTransform(qml.transform(first_valid_transform))
        pipeline = CompilePipeline([container])

        with pytest.raises(TypeError):
            _ = "invalid" + pipeline

        with pytest.raises(TypeError):
            _ = 42 + pipeline

    def test_repr_pipeline(self):
        """Test the string representation of a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = BoundTransform(qml.transform(first_valid_transform))
        transform2 = BoundTransform(qml.transform(second_valid_transform))

        compile_pipeline.append(transform1)
        compile_pipeline.append(transform2)

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
        t1 = BoundTransform(qml.transforms.compile, kwargs={"num_passes": 2})
        t2 = BoundTransform(qml.transforms.compile, kwargs={"num_passes": 2})
        t3 = BoundTransform(qml.transforms.transpile, kwargs={"coupling_map": [(0, 1), (1, 2)]})

        p1 = CompilePipeline([t1, t3])
        p2 = CompilePipeline([t2, t3])
        p3 = CompilePipeline([t3, t2])

        # test for equality of identical objects
        assert p1 == p2
        # test for inequality of different objects
        assert p1 != p3
        assert p1 != t1

        # Test inequality with different transforms
        t4 = BoundTransform(qml.transforms.transpile, kwargs={"coupling_map": [(0, 1), (2, 3)]})
        p4 = CompilePipeline([t1, t4])
        assert p1 != p4


class TestCompilePipelineConstruction:
    """Tests the different ways to initialize a CompilePipeline."""

    def test_empty_pipeline(self):
        """Test an empty pipeline."""

        pipeline = CompilePipeline()
        assert not pipeline
        assert pipeline.cotransform_cache is None
        assert len(pipeline) == 0

        with pytest.raises(IndexError):
            _ = pipeline[-1]

    def test_list_of_transforms(self):
        """Tests constructing a CompilePipeline with a list of transforms."""

        pipeline = CompilePipeline(
            [
                BoundTransform(qml.transforms.compile),
                BoundTransform(qml.transforms.decompose),
                BoundTransform(qml.transforms.cancel_inverses),
            ]
        )
        assert len(pipeline) == 3

    def test_list_of_transforms_arbitrary(self):
        """Tests constructing a CompilePipeline with a list of transforms."""

        pipeline = CompilePipeline(
            [
                qml.transforms.compile,
                BoundTransform(qml.transforms.decompose),
                CompilePipeline(qml.transforms.cancel_inverses),
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
            BoundTransform(qml.transforms.decompose, kwargs={"gate_set": {qml.Rot, qml.CNOT}}),
        )
        assert len(pipeline) == 4

    def test_invalid_object_in_transforms(self):
        """Tests that an error is raised when something is not a transform."""

        with pytest.raises(TypeError, match="CompilePipeline can only be constructed"):
            # matrix is not a transform
            CompilePipeline(qml.transforms.cancel_inverses, qml.matrix)


class TestCompilePipeline:
    """Test the compile pipeline class and its method."""

    def test_get_last(self):
        """Tests the get_last method"""
        pipeline = CompilePipeline()
        pipeline.add_transform(transform(first_valid_transform))
        pipeline.add_transform(transform(second_valid_transform))
        assert pipeline[-1] == BoundTransform(transform(second_valid_transform))

    def test_append(self):
        """Test to push back multiple transforms into a pipeline and also the different methods of a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = BoundTransform(transform(first_valid_transform))
        compile_pipeline.append(transform1)

        assert compile_pipeline
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is first_valid_transform

        transform2 = BoundTransform(transform(second_valid_transform))
        compile_pipeline.append(transform2)

        assert compile_pipeline
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[1], BoundTransform)
        assert compile_pipeline[1].tape_transform is second_valid_transform

        compile_pipeline.append(transform1)
        compile_pipeline.append(transform2)

        sub_pipeline_transforms = compile_pipeline[2:]
        assert len(sub_pipeline_transforms) == 2
        assert sub_pipeline_transforms[0] is transform1
        assert sub_pipeline_transforms[1] is transform2

        with pytest.raises(TransformError, match="does not appear to be a valid Python function"):
            compile_pipeline.append(10.0)

    def test_append_with_list_raises_helpful_error(self):
        """Test that append with a list raises an error pointing to extend."""
        pipeline = CompilePipeline()
        t1 = transform(first_valid_transform)
        t2 = transform(second_valid_transform)

        with pytest.raises(TypeError, match="Use extend\\(\\) to add multiple transforms"):
            pipeline.append([t1, t2])

        with pytest.raises(TypeError, match="Use extend\\(\\) to add multiple transforms"):
            pipeline.append((t1, t2))

    def test_extend_list(self):
        """Test extending a pipeline with a list of transforms."""
        pipeline = CompilePipeline()
        t1 = transform(first_valid_transform)
        t2 = transform(second_valid_transform)

        # Extend with a list of transforms
        pipeline.extend([t1, t2])

        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is second_valid_transform

    def test_extend_list_with_bound_transforms(self):
        """Test extending with a list containing both Transform and BoundTransform."""
        pipeline = CompilePipeline()
        t1 = transform(first_valid_transform)
        t2_bound = BoundTransform(transform(second_valid_transform))

        pipeline.extend([t1, t2_bound])

        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is second_valid_transform

    def test_extend_multiplied_transform(self):
        """Test extending with a multiplied transform (which returns a CompilePipeline)."""
        pipeline = CompilePipeline()
        t1 = BoundTransform(transform(first_valid_transform))

        # Multiplying a BoundTransform returns a CompilePipeline
        multiplied = 2 * t1
        assert isinstance(multiplied, CompilePipeline)

        pipeline.extend(multiplied)

        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is first_valid_transform

    def test_extend_compile_pipeline(self):
        """Test extending a CompilePipeline with another CompilePipeline."""
        pipeline1 = CompilePipeline()
        pipeline1.append(BoundTransform(transform(first_valid_transform)))

        pipeline2 = CompilePipeline()
        pipeline2.append(BoundTransform(transform(second_valid_transform)))

        pipeline1.extend(pipeline2)

        assert len(pipeline1) == 2
        assert pipeline1[0].tape_transform is first_valid_transform
        assert pipeline1[1].tape_transform is second_valid_transform

    def test_extend_tuple(self):
        """Test extending a pipeline with a tuple of transforms."""
        pipeline = CompilePipeline()
        t1 = transform(first_valid_transform)
        t2 = transform(second_valid_transform)

        # Extend with a tuple of transforms
        pipeline.extend((t1, t2))

        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is second_valid_transform

    def test_extend_list_with_multiplied_transform(self):
        """Test extending with a list containing a multiplied transform (CompilePipeline)."""
        pipeline = CompilePipeline()
        t1 = transform(first_valid_transform)
        t2 = transform(second_valid_transform)

        # t2 * 2 creates a CompilePipeline, which should be flattened when inside a list
        pipeline.extend([t1, t2 * 2])

        assert len(pipeline) == 3
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is second_valid_transform
        assert pipeline[2].tape_transform is second_valid_transform

    def test_add_transform(self):
        """Test to add multiple transforms into a pipeline and also the different methods of a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform)
        compile_pipeline.add_transform(transform1)

        assert compile_pipeline
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is first_valid_transform

        transform2 = transform(second_valid_transform)
        compile_pipeline.add_transform(transform2)

        assert compile_pipeline
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[1], BoundTransform)
        assert compile_pipeline[1].tape_transform is second_valid_transform

        compile_pipeline.add_transform(transform1)
        compile_pipeline.add_transform(transform2)

        sub_pipeline_transforms = compile_pipeline[2:]
        assert len(sub_pipeline_transforms) == 2
        assert sub_pipeline_transforms[0].tape_transform is first_valid_transform
        assert sub_pipeline_transforms[1].tape_transform is second_valid_transform

        with pytest.raises(TransformError, match="Only transforms can be added"):
            compile_pipeline.add_transform(10.0)

    def test_add_transform_with_expand(self):
        """Test to add a transform with expand into a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform, expand_transform=expand_transform)
        compile_pipeline.add_transform(transform1)

        assert compile_pipeline
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is expand_transform

        assert isinstance(compile_pipeline[1], BoundTransform)
        assert compile_pipeline[1].tape_transform is first_valid_transform

    def test_pop(self):
        """Test the pop method of the compile pipeline."""

        first = qml.transform(first_valid_transform)
        second = qml.transform(second_valid_transform, expand_transform=expand_transform)
        pipeline = first + second + second

        assert len(pipeline) == 5
        result = pipeline.pop(0)
        assert len(pipeline) == 4
        assert result.tape_transform == first_valid_transform

        result = pipeline.pop(1)
        assert result.tape_transform == second_valid_transform
        assert len(pipeline) == 2
        assert pipeline[0].tape_transform == expand_transform
        assert pipeline[1].tape_transform == second_valid_transform

    def test_insert_front(self):
        """Test to insert a transform (container) at the beginning of a compile pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = BoundTransform(transform(first_valid_transform))
        compile_pipeline.append(transform1)

        assert compile_pipeline
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is first_valid_transform

        transform2 = BoundTransform(transform(second_valid_transform))
        compile_pipeline.insert(0, transform2)

        assert compile_pipeline
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0] is transform2
        assert isinstance(compile_pipeline[1], BoundTransform)
        assert compile_pipeline[1] is transform1

        transform3 = BoundTransform(transform(second_valid_transform, is_informative=True))

        with pytest.raises(TransformError, match="can only be added to the end"):
            compile_pipeline.insert(0, transform3)

    def test_insert_transform(self):
        """Test to insert a transform (dispatcher) at the beginning of a compile pipeline."""

        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform)
        compile_pipeline.insert(0, transform1)

        assert compile_pipeline
        assert len(compile_pipeline) == 1
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is first_valid_transform

        transform2 = transform(second_valid_transform)
        compile_pipeline.insert(0, transform2)

        assert compile_pipeline
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is second_valid_transform
        assert isinstance(compile_pipeline[1], BoundTransform)
        assert compile_pipeline[1].tape_transform is first_valid_transform

        transform3 = transform(second_valid_transform, is_informative=True)

        with pytest.raises(TransformError, match="can only be added to the end"):
            compile_pipeline.insert(0, transform3)

    def test_insert_transform_with_expand(self):
        """Test to insert front a transform with expand into a pipeline."""
        compile_pipeline = CompilePipeline()

        transform1 = transform(first_valid_transform, expand_transform=expand_transform)
        compile_pipeline.insert(0, transform1)

        assert compile_pipeline
        assert len(compile_pipeline) == 2
        assert isinstance(compile_pipeline[0], BoundTransform)
        assert compile_pipeline[0].tape_transform is expand_transform

        assert isinstance(compile_pipeline[1], BoundTransform)
        assert compile_pipeline[1].tape_transform is first_valid_transform

    def test_valid_transforms(self):
        """Test adding transforms to a pipeline with a terminal transform."""
        compile_pipeline = CompilePipeline()
        transform1 = BoundTransform(qml.transform(first_valid_transform, is_informative=True))
        compile_pipeline.append(transform1)

        t_normal = BoundTransform(qml.transform(second_valid_transform))
        compile_pipeline.append(t_normal)
        assert len(compile_pipeline) == 2
        assert compile_pipeline[0] is t_normal
        assert compile_pipeline[1] is transform1

        t_normal2 = BoundTransform(qml.transform(first_valid_transform))
        compile_pipeline.append(t_normal2)
        assert compile_pipeline[0] is t_normal
        assert compile_pipeline[1] is t_normal2
        assert compile_pipeline[2] is transform1

        with pytest.raises(
            TransformError, match="The compile pipeline already has a terminal transform."
        ):
            compile_pipeline.append(transform1)

        transform2 = BoundTransform(qml.transform(second_valid_transform, final_transform=True))
        with pytest.raises(
            TransformError, match="The compile pipeline already has a terminal transform."
        ):
            compile_pipeline.append(transform2)

    def test_remove_by_container(self):
        """Test removing a specific TransformContainer from a program."""
        dispatched_transform = transform(first_valid_transform)
        container1 = BoundTransform(dispatched_transform)
        container2 = BoundTransform(dispatched_transform, args=(1,))

        program = CompilePipeline([container1, container2])
        assert len(program) == 2

        program.remove(container1)
        assert len(program) == 1
        assert program[0] == container2

    def test_remove_by_dispatcher(self):
        """Test removing all containers matching a Transform from a program."""

        dispatched_transform = transform(first_valid_transform)
        container1 = BoundTransform(dispatched_transform)
        container2 = BoundTransform(dispatched_transform, args=(1,))

        program = CompilePipeline([container1, container2])
        assert len(program) == 2

        program.remove(dispatched_transform)
        assert len(program) == 0

    def test_remove_with_expand_transform(self):
        """Tests that the expand_transform is removed with the original transform."""

        first = qml.transform(first_valid_transform)
        second = qml.transform(second_valid_transform, expand_transform=expand_transform)

        pipeline = second + first + second + second
        assert len(pipeline) == 7

        pipeline.remove(second)
        assert len(pipeline) == 1
        assert pipeline[0].tape_transform == first_valid_transform

    def test_remove_invalid_type(self):
        """Test that removing an invalid type raises TypeError."""
        dispatched_transform = transform(first_valid_transform)
        container = BoundTransform(dispatched_transform)
        program = CompilePipeline([container])

        with pytest.raises(TypeError, match="Only BoundTransform or Transform"):
            program.remove("not_a_container_or_dispatcher")

        with pytest.raises(TypeError, match="Only BoundTransform or Transform"):
            program.remove(42)


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
            qml.gradients.param_shift.tape_transform, classical_cotransform=lambda *args: 0
        )

        hybrid_t = BoundTransform(new_t, (), {"hybrid": True})
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

        container = BoundTransform(transform(remove_operation_at_index), kwargs={"index": 1})
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

        container1 = BoundTransform(transform(transform_add))
        container2 = BoundTransform(transform(transform_mul))
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

        container = BoundTransform(transform(split_sum_terms))
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

    def test_call_single_quantumscript_converts_to_batch(self):
        """Test that calling with a single QuantumScript (not a tuple) converts it to a batch."""

        def identity_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            """A transform that returns the tape unchanged."""
            return (tape,), lambda results: results[0]

        container = BoundTransform(transform(identity_transform))
        prog = CompilePipeline((container,))

        # Create a single QuantumScript (not wrapped in a tuple)
        single_tape = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1])], [qml.expval(qml.PauliZ(0))], shots=50
        )

        # Call with single QuantumScript - should trigger the isinstance check
        new_batch, fn = prog(single_tape)

        # Verify it was processed correctly
        assert len(new_batch) == 1
        assert isinstance(new_batch, tuple)
        assert new_batch[0] is single_tape

        # Verify postprocessing works
        dummy_results = (0.5,)
        assert fn(dummy_results) == (0.5,)

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

    def test_call_fallback_on_qnode(self):
        """Test that a CompilePipeline can be applied to a QNode using the fallback."""

        program = CompilePipeline()
        program += qml.transforms.cancel_inverses
        program += transform(first_valid_transform)(0)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def circuit(a):
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.PauliX(wires=0)  # Should be cancelled
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        # Apply the program to the QNode
        new_qnode = program(circuit)

        assert isinstance(new_qnode, qml.QNode)
        # The QNode should have the transforms from the program
        assert len(new_qnode.transform_program) == 2
        assert (
            new_qnode.transform_program[0].tape_transform
            is qml.transforms.cancel_inverses.tape_transform
        )
        assert new_qnode.transform_program[1].tape_transform is first_valid_transform

    def test_call_fallback_on_qnode_already_transformed(self):
        """Test that a CompilePipeline can be applied to a QNode that already has transforms."""

        program = CompilePipeline()
        program += transform(first_valid_transform)(0)

        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.cancel_inverses
        @qml.qnode(device=dev)
        def circuit(a):
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.PauliX(wires=0)  # Should be cancelled
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        # Apply the program to the QNode
        new_qnode = program(circuit)

        assert isinstance(new_qnode, qml.QNode)
        # The QNode should have the transforms from the program
        assert len(new_qnode.transform_program) == 2
        assert (
            new_qnode.transform_program[0].tape_transform
            is qml.transforms.cancel_inverses.tape_transform
        )
        assert new_qnode.transform_program[1].tape_transform is first_valid_transform

    def test_call_fallback_on_qnode_empty_program(self):
        """Test that an empty program returns the original QNode."""

        program = CompilePipeline()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        new_qnode = program(circuit)

        # For empty program, the fallback returns the original object unchanged
        assert new_qnode is circuit

    def test_call_fallback_on_callable(self):
        """Test that a CompilePipeline can be applied to a callable using the fallback."""

        program = CompilePipeline()
        program += transform(first_valid_transform)(0)

        def qfunc():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        # Apply the program to a callable
        transformed_qfunc = program(qfunc)

        assert callable(transformed_qfunc)
        assert transformed_qfunc is not qfunc

    def test_call_fallback_chain_applies_transforms(self):
        """Test that the fallback chain-applies each transform in order."""

        # Track how many times each transform is applied
        call_order = []

        def tracking_transform_1(tape):
            call_order.append(1)
            return [tape], lambda x: x[0]

        def tracking_transform_2(tape):
            call_order.append(2)
            return [tape], lambda x: x[0]

        program = CompilePipeline()
        program += transform(tracking_transform_1)
        program += transform(tracking_transform_2)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        # Apply the program - transforms should be in the QNode's transform_program
        new_qnode = program(circuit)

        assert len(new_qnode.transform_program) == 2
        # First transform in program should be first in QNode's transform_program
        assert new_qnode.transform_program[0].tape_transform is tracking_transform_1
        assert new_qnode.transform_program[1].tape_transform is tracking_transform_2

    def test_call_on_qnode_execution(self):
        """Test that a CompilePipeline applied to a QNode actually transforms execution."""

        program = CompilePipeline()
        program += transform(qml.transforms.cancel_inverses)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.PauliX(wires=0)  # These should be cancelled
            qml.PauliY(wires=1)
            qml.PauliY(wires=1)  # These should be cancelled
            return qml.expval(qml.PauliZ(0))

        # Apply the program to the QNode
        transformed_qnode = program(circuit)

        # Execute and verify the transform was applied
        with dev.tracker:
            result = transformed_qnode()

        # Check that the transform was applied: only Hadamard should remain
        # (X-X and Y-Y pairs should be cancelled)
        resources = dev.tracker.history["resources"][0]
        assert resources.gate_types == {"Hadamard": 1}
        assert resources.num_gates == 1
        assert resources.depth == 1

        # Check the numerical output: H|0> gives |+>, expectation of Z is 0
        assert qml.math.allclose(result, 0.0)

    def test_call_on_device(self):
        """Test that a CompilePipeline can be applied to a Device."""

        # Create a dummy device with a custom preprocess_transforms method
        class DummyDevice(qml.devices.Device):
            def preprocess_transforms(
                self, execution_config=None
            ):  # pylint: disable=unused-argument
                prog = CompilePipeline()
                prog.add_transform(qml.defer_measurements)
                return prog

            def execute(self, circuits, execution_config=None):  # pylint: disable=unused-argument
                return [0] * len(circuits)

        original_dev = DummyDevice()

        # Create a program with transforms
        program = CompilePipeline()
        program += transform(qml.transforms.cancel_inverses)
        program += transform(first_valid_transform)(0)

        # Apply the program to the device
        transformed_dev = program(original_dev)

        # Verify the device was transformed (it's wrapped twice, once for each transform)
        # The outer wrapper is for the second transform
        assert repr(transformed_dev).startswith("Transformed Device")

        # The original device is nested inside
        inner_dev = transformed_dev.original_device
        assert repr(inner_dev).startswith("Transformed Device")
        assert inner_dev.original_device is original_dev

        # Check that the device's preprocess_transforms includes the new transforms
        original_program = original_dev.preprocess_transforms()
        new_program = transformed_dev.preprocess_transforms()

        assert isinstance(original_program, CompilePipeline)
        assert isinstance(new_program, CompilePipeline)

        # Original program has 1 transform (defer_measurements)
        assert len(original_program) == 1
        # New program should have 3 transforms (original + 2 from our program)
        assert len(new_program) == 3

        # Verify the transforms are in the right order and are the right ones
        assert new_program[-2].tape_transform is qml.transforms.cancel_inverses.tape_transform
        assert new_program[-1].tape_transform is first_valid_transform


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

        assert pipeline
        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is first_valid_transform

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

        assert pipeline
        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is informative_transform

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

        assert pipeline
        assert len(pipeline) == 2
        assert pipeline[0].tape_transform is first_valid_transform
        assert pipeline[1].tape_transform is second_valid_transform
