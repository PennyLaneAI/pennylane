from pennylane.wires import Wires
import timeit
import random

def small_wires():
    a = Wires([1, 2])
    b = Wires(['x', 2, 'a', 1])
    c = Wires(['x', 'y', 'z'])
    d = Wires([3, 4])
    return [a, b, c, d]


def large_wires():
    a = Wires(list(range(30)))
    b = Wires(list(range(5, 25)))
    c = Wires(list(map(chr, range(97, 117))))
    d = Wires([3, 4, 'x', 'y'])
    return [a, b, c, d]



def benchmark(wires, function):
    def run():
        function(wires)
    return timeit.timeit(run, number=10000)


def many_small(size, num):
    return [Wires(list(random.sample(range(0, size), 3))) for _ in range(num)]

"""
print(f"all_wires small {benchmark(small_wires, Wires.all_wires):1.3f}")
print(f"shared_wires small {benchmark(small_wires, Wires.shared_wires):1.3f}")
print(f"unique_wires small {benchmark(small_wires, Wires.unique_wires):1.3f}")


print(f"shared_wires large {benchmark(large_wires, Wires.shared_wires):1.3f}")
print(f"unique_wires large {benchmark(large_wires, Wires.unique_wires):1.3f}")

"""

for _ in range(3):
    print(f"all_wires small {benchmark(small_wires(), Wires.all_wires):1.3f}")
    print(f"all_wires large {benchmark(large_wires(), Wires.all_wires):1.3f}")
    print(f"all_wires many {benchmark(many_small(30, 100), Wires.all_wires):1.3f}")