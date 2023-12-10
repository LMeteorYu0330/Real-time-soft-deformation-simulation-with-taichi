import matplotlib.pyplot as plt
import numpy as np

force = np.loadtxt("force.txt")
d = np.loadtxt("d.txt") * 10
force = np.linalg.norm(force, axis=1)
result = np.vstack((force, d))
# result = result[:, 1100:1500]
# plt.plot(result[1], ".:", label='x')
plt.plot(result[0], ":", label='f')
plt.legend(fontsize=15)
plt.show()
