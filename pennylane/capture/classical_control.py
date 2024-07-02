from pennylane.math import is_abstract

from .bind_nested_plxpr import QFunc, bind_nested_plxpr
from .switches import enabled


def for_loop_transform_def(qfunc: QFunc, start: int, stop: int, step: int) -> QFunc:
    def new_qfunc(x):
        for i in range(start, stop, step):
            x = qfunc(i, x)
        return x

    return new_qfunc


for_loop_transform = bind_nested_plxpr(
    for_loop_transform_def, name="for_loop", additional_args=(0,)
)


def for_loop(start, stop, step):
    def qfunc_transform(qfunc):
        return for_loop_transform(qfunc, start, stop, step)

    return qfunc_transform
