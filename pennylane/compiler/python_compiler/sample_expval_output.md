Circuit returning qml.sample() (for all wires):

```python
@qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
@MeasurementsFromSamplesPass
@qml.qnode(dev)
def deleteme():
    qml.H(0)
    return qml.sample()
```

gives the MLIR:

```mlir
func.func public @deleteme() -> (tensor<10x2xi64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
  %0 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
  %1 = tensor.extract %0[] : tensor<i64>
  "quantum.device"(%1) {kwargs = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", lib = "/home/joseph.carter/work/pennylane/pyenv-catalyst-dev/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", name = "LightningSimulator"} : (i64) -> ()
  %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
  %3 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
  %4 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
  %5 = tensor.extract %4[] : tensor<i64>
  %6 = "quantum.extract"(%3, %5) : (!quantum.reg, i64) -> !quantum.bit
  %7 = "quantum.custom"(%6) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>, gate_name = "Hadamard"}> : (!quantum.bit) -> !quantum.bit
  %8 = tensor.extract %4[] : tensor<i64>
  %9 = "quantum.insert"(%3, %8, %7) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
  %10 = quantum.compbasis qreg %9 : !quantum.obs
  %11 = quantum.sample %10 : tensor<10x2xf64>
  %12 = "stablehlo.convert"(%11) : (tensor<10x2xf64>) -> tensor<10x2xi64>
  quantum.dealloc %9 : !quantum.reg
  quantum.device_release
  func.return %12 : tensor<10x2xi64>
}
```

Circuit that returns expval (per wire):

```python
@qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
@MeasurementsFromSamplesPass
@qml.qnode(dev)
def deleteme():
    qml.H(0)
    return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))
```

gives the MLIR:

```mlir
func.func public @deleteme() -> (tensor<f64>, tensor<f64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
  %0 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
  %1 = tensor.extract %0[] : tensor<i64>
  "quantum.device"(%1) {kwargs = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", lib = "/home/joseph.carter/work/pennylane/pyenv-catalyst-dev/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", name = "LightningSimulator"} : (i64) -> ()
  %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
  %3 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
  %4 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
  %5 = tensor.extract %4[] : tensor<i64>
  %6 = "quantum.extract"(%3, %5) : (!quantum.reg, i64) -> !quantum.bit
  %7 = "quantum.custom"(%6) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>, gate_name = "Hadamard"}> : (!quantum.bit) -> !quantum.bit
  %8 = quantum.namedobs %7[PauliZ] : !quantum.obs
  %9 = quantum.expval %8 : f64
  %10 = "tensor.from_elements"(%9) : (f64) -> tensor<f64>
  %11 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
  %12 = tensor.extract %11[] : tensor<i64>
  %13 = "quantum.extract"(%3, %12) : (!quantum.reg, i64) -> !quantum.bit
  %14 = quantum.namedobs %13[PauliZ] : !quantum.obs
  %15 = quantum.expval %14 : f64
  %16 = "tensor.from_elements"(%15) : (f64) -> tensor<f64>
  %17 = tensor.extract %4[] : tensor<i64>
  %18 = "quantum.insert"(%3, %17, %7) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
  quantum.dealloc %18 : !quantum.reg
  quantum.device_release
  func.return %10, %16 : tensor<f64>, tensor<f64>
}
```

Circuit that returns sample (per wire):

```python
@qml.qjit(pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
@MeasurementsFromSamplesPass
@qml.qnode(dev)
def deleteme():
    qml.H(0)
    return qml.sample(wires=0), qml.sample(wires=1)
```

gives the MLIR:

```mlir
func.func public @deleteme() -> (tensor<10x1xi64>, tensor<10x1xi64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
  %0 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
  %1 = tensor.extract %0[] : tensor<i64>
  "quantum.device"(%1) {kwargs = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", lib = "/home/joseph.carter/work/pennylane/pyenv-catalyst-dev/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", name = "LightningSimulator"} : (i64) -> ()
  %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
  %3 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
  %4 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
  %5 = tensor.extract %4[] : tensor<i64>
  %6 = "quantum.extract"(%3, %5) : (!quantum.reg, i64) -> !quantum.bit
  %7 = "quantum.custom"(%6) <{operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>, gate_name = "Hadamard"}> : (!quantum.bit) -> !quantum.bit
  %8 = quantum.compbasis qubits %7 : !quantum.obs
  %9 = quantum.sample %8 : tensor<10x1xf64>
  %10 = "stablehlo.convert"(%9) : (tensor<10x1xf64>) -> tensor<10x1xi64>
  %11 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
  %12 = tensor.extract %11[] : tensor<i64>
  %13 = "quantum.extract"(%3, %12) : (!quantum.reg, i64) -> !quantum.bit
  %14 = quantum.compbasis qubits %13 : !quantum.obs
  %15 = quantum.sample %14 : tensor<10x1xf64>
  %16 = "stablehlo.convert"(%15) : (tensor<10x1xf64>) -> tensor<10x1xi64>
  %17 = tensor.extract %4[] : tensor<i64>
  %18 = "quantum.insert"(%3, %17, %7) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
  quantum.dealloc %18 : !quantum.reg
  quantum.device_release
  func.return %10, %16 : tensor<10x1xi64>, tensor<10x1xi64>
}
```
