from pennylane.capture.collect_ops_and_meas import CollectOpsandMeas

from .qscript import QuantumScript


def plxpr_to_tape(jaxpr: "jax.core.Jaxpr", consts, *args, shots=None) -> QuantumScript:
    """Convert a jaxpr into a tape.

    Args:
        jaxpr (jax.core.Jaxpr): a pennylane variant jaxpr
        consts (list): the consts for the jaxpr
        *args : the arguments to execute the jaxpr with

    Keyword Args:
        shots (None, int, Sequence[int], Shots): the shots for the tape.

    .. code-block:: python

        @qml.for_loop(3)
        def loop(i):
            qml.X(i)

        def f(x):
            loop()
            qml.adjoint(qml.S)(0)
            m0 = qml.measure(0)
            qml.RX(2*x, 0)
            return qml.probs(wires=0), qml.expval(qml.Z(1))

        jaxpr = jax.make_jaxpr(f)(0.5)
        tape = qml.capture.convert_to_tape(jaxpr.jaxpr, jaxpr.consts, 1.2)
        print(qml.drawer.tape_text(tape, decimals=2))

    .. code-block::

        0: ──X──S†──┤↗├──RX(2.40)─┤  Probs
        1: ──X────────────────────┤  <Z>
        2: ──X────────────────────┤

    """

    collector = CollectOpsandMeas()
    collector.eval(jaxpr, consts, *args)
    assert collector.state
    return QuantumScript(collector.state["ops"], collector.state["measurements"], shots=shots)
