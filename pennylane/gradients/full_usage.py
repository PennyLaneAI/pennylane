def backward_jvp_gradient_transform(primals, tangents):
    at_max_diff = _n == max_diff
    new_tapes = set_parameters_on_copy_and_unwrap(tapes, primals[0], unwrap=at_max_diff)
    _args = (
        new_tapes,
        tangents[0],
        gradient_fn,
        jvp_shots,
    )
    _kwargs = {
        "reduction": "append",
        "gradient_kwargs": gradient_kwargs,
    }
    if at_max_diff:
        jvp_tapes, processing_fn = qml.gradients.batch_jvp(*_args, **_kwargs)
        jvps = processing_fn(execute_fn(jvp_tapes)[0])
    else:
        jvp_tapes, processing_fn = qml.gradients.batch_jvp(*_args, **_kwargs)

        jvps = processing_fn(
            execute(
                jvp_tapes,
                device,
                execute_fn,
                gradient_fn,
                gradient_kwargs,
                _n=_n + 1,
                max_diff=max_diff,
            )
        )
    res = execute_wrapper(primals[0])
