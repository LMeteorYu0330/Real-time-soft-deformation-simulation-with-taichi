import matplotlib.pyplot as plt
import numpy as np

force = np.loadtxt("force.txt")
d = np.loadtxt("d.txt") * 10
ddde = np.loadtxt("de.txt")
force = np.linalg.norm(force, axis=1)
# result = np.vstack((ddde))
# result = result[:, 1500:-1]
# plt.plot(result[1], ".:", label='x')
# plt.plot(result[0], ":", label='f')
result = np.sum(ddde, 1)
print(result.shape, ddde.shape)
plt.plot(result, label='de')
plt.legend(fontsize=15)
plt.show()
