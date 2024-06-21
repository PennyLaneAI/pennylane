```python
import pennylane as qml
import jax

from pennylane.capture.interpreters import PlxprInterpreter, DefaultQubitInterpreter, LightningInterpreter, DecompositionInterpreter, ConvertToTape, CancelInverses, MergeRotations
qml.capture.enable()
```

### Demonstrating Existing Implementations


```python
def f(x):
    qml.X(0)
    qml.adjoint(qml.X(0))
    qml.Hadamard(0)
    qml.IsingXX(x, wires=(0,1))
    return qml.expval(qml.Z(0)), qml.probs(wires=(0,1))

plxpr = jax.make_jaxpr(f)(0.5)
```


```python
DefaultQubitInterpreter(num_wires=2)(plxpr.jaxpr, plxpr.consts, 1.2)
```




    [0.0, array([0.34058944, 0.15941056, 0.34058944, 0.15941056])]




```python
LightningInterpreter(num_wires=2)(plxpr.jaxpr, plxpr.consts, 1.2)
```




    [0.0, array([0.34058944, 0.15941056, 0.34058944, 0.15941056])]




```python
tape = ConvertToTape()(plxpr.jaxpr, plxpr.consts, 1.2)
print(tape.draw())
```

    0: ──X──X†──H─╭IsingXX─┤  <Z> ╭Probs
    1: ───────────╰IsingXX─┤      ╰Probs



```python
DecompositionInterpreter().call_jaxpr(plxpr.jaxpr, plxpr.consts)(2.5)
```




    { lambda ; a:f64[]. let
        _:AbstractOperator() = PhaseShift[n_wires=1] 1.5707963267948966 0
        _:AbstractOperator() = RX[n_wires=1] 3.141592653589793 0
        _:AbstractOperator() = PhaseShift[n_wires=1] 1.5707963267948966 0
        _:AbstractOperator() = PauliX[n_wires=1] 0
        _:AbstractOperator() = PhaseShift[n_wires=1] 1.5707963267948966 0
        _:AbstractOperator() = RX[n_wires=1] 1.5707963267948966 0
        _:AbstractOperator() = PhaseShift[n_wires=1] 1.5707963267948966 0
        _:AbstractOperator() = CNOT[n_wires=2] 0 1
        _:AbstractOperator() = RX[n_wires=1] a 0
        _:AbstractOperator() = CNOT[n_wires=2] 0 1
        b:AbstractOperator() = PauliZ[n_wires=1] 0
        c:AbstractMeasurement(n_wires=None) = expval_obs b
        d:AbstractMeasurement(n_wires=2) = probs_wires 0 1
      in (c, d) }




```python
CancelInverses().call_jaxpr(plxpr.jaxpr, plxpr.consts)(2.5)
```




    { lambda ; a:f64[]. let
        _:AbstractOperator() = IsingXX[n_wires=2] a 0 1
        b:AbstractOperator() = PauliZ[n_wires=1] 0
        c:AbstractMeasurement(n_wires=None) = expval_obs b
        d:AbstractMeasurement(n_wires=2) = probs_wires 0 1
      in (c, d) }




```python
def g(x):
    qml.RX(x, 0)
    qml.RX(2*x, 0)
    qml.RX(-4*x, 0)
    qml.X(0)
    qml.RX(0.5, 0)

plxpr = jax.make_jaxpr(g)(1.0)
MergeRotations().call_jaxpr(plxpr.jaxpr, plxpr.consts)(1.0)
```




    { lambda ; a:f64[]. let
        b:f64[] = mul 2.0 a
        c:f64[] = add b a
        d:f64[] = mul -4.0 a
        e:f64[] = add d c
        _:AbstractOperator() = RX[n_wires=1] e 0
        _:AbstractOperator() = PauliX[n_wires=1] 0
        _:AbstractOperator() = RX[n_wires=1] 0.5 0
      in () }



### Writing a new interpreter


```python
class AddSWAPNoise(PlxprInterpreter):

    def __init__(self, scale, prng_key=jax.random.key(12345)):
        self.scale = scale
        self.prng_key = prng_key
    
    def interpret_operation(self, op):
        if isinstance(op, qml.SWAP):
            self.prng_key, subkey = jax.random.split(self.prng_key)
            phi = self.scale*jax.random.uniform(subkey)
            qml.PhaseShift(phi, op.wires[0])
        val, structure = jax.tree_util.tree_flatten(op)
        jax.tree_util.tree_unflatten(structure, val)

    def interpret_measurement(self, m):
        vals, structure = jax.tree_util.tree_flatten(m)
        return jax.tree_util.tree_unflatten(structure, vals)
```


```python
def f():
    qml.SWAP((0,1))
    qml.SWAP((1,2))
    return qml.expval(qml.Z(0))

plxpr = jax.make_jaxpr(f)()
AddSWAPNoise(0.1).call_jaxpr(plxpr.jaxpr, plxpr.consts)()
```




    let _uniform = { lambda ; a:key<fry>[] b:f64[] c:f64[]. let
        d:f64[] = convert_element_type[new_dtype=float64 weak_type=False] b
        e:f64[] = convert_element_type[new_dtype=float64 weak_type=False] c
        f:u64[] = random_bits[bit_width=64 shape=()] a
        g:u64[] = shift_right_logical f 12
        h:u64[] = or g 4607182418800017408
        i:f64[] = bitcast_convert_type[new_dtype=float64] h
        j:f64[] = sub i 1.0
        k:f64[] = sub e d
        l:f64[] = mul j k
        m:f64[] = add l d
        n:f64[] = reshape[dimensions=None new_sizes=()] m
        o:f64[] = max d n
      in (o,) } in
    { lambda p:key<fry>[]; . let
        q:key<fry>[2] = random_split[shape=(2,)] p
        r:key<fry>[1] = slice[limit_indices=(1,) start_indices=(0,) strides=(1,)] q
        s:key<fry>[] = squeeze[dimensions=(0,)] r
        t:key<fry>[1] = slice[limit_indices=(2,) start_indices=(1,) strides=(1,)] q
        u:key<fry>[] = squeeze[dimensions=(0,)] t
        v:f64[] = pjit[name=_uniform jaxpr=_uniform] u 0.0 1.0
        w:f64[] = mul 0.1 v
        _:AbstractOperator() = PhaseShift[n_wires=1] w 0
        _:AbstractOperator() = SWAP[n_wires=2] 0 1
        x:key<fry>[2] = random_split[shape=(2,)] s
        y:key<fry>[1] = slice[limit_indices=(1,) start_indices=(0,) strides=(1,)] x
        _:key<fry>[] = squeeze[dimensions=(0,)] y
        z:key<fry>[1] = slice[limit_indices=(2,) start_indices=(1,) strides=(1,)] x
        ba:key<fry>[] = squeeze[dimensions=(0,)] z
        bb:f64[] = pjit[name=_uniform jaxpr=_uniform] ba 0.0 1.0
        bc:f64[] = mul 0.1 bb
        _:AbstractOperator() = PhaseShift[n_wires=1] bc 1
        _:AbstractOperator() = SWAP[n_wires=2] 1 2
        bd:AbstractOperator() = PauliZ[n_wires=1] 0
        be:AbstractMeasurement(n_wires=None) = expval_obs bd
      in (be,) }




```python

```
