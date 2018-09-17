import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def func(x, y):

    def RX(_x):
        return np.array([[np.cos(_x/2), - 1j*np.sin(_x/2)],
                         [- 1j*np.sin(_x/2), np.cos(_x/2)]])

    def RY(_y):
        return np.array([[np.cos(_y/2), np.sin(_y/2)],
                         [- np.sin(_y/2), np.cos(_y/2)]])

    a = np.array([1, 0])

    PauliZ = np.array([[1, 0],
                       [0, -1]])

    psi = np.dot(RY(y), np.dot(RX(x), a))
    exp = np.vdot(psi, np.dot(PauliZ, psi))

    return exp

print(func(3.14, 0.5))


fig = plt.figure(figsize =(5, 3))
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-3, 3, 0.1)
Y = np.arange(-3, 3, 0.1)
length = len(X)

xx, yy = np.meshgrid(X, Y)

Z = np.array([[func(x, y) for x in X] for y in Y]).reshape(length, length)
Z = Z.real

# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.0, 1.0)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()