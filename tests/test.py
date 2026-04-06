PYTHONPATH="/home/godseye/Public/Contributions/pennylane python3"
import pennylane as qml

op = qml.TrotterProduct(qml.X(0) + qml.Y(0), 1.0, n=3, order=2)

print("Decomposition:")
for i, d in enumerate(op.decomposition(), start=1):
    print(i, d)

print("\nResources:")
print(op.resources())
