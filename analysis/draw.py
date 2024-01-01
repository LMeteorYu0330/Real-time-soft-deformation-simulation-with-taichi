import matplotlib.pyplot as plt
import numpy as np

force = np.loadtxt("force.txt")
d = np.loadtxt("d.txt")
dde = np.loadtxt("de.txt")
dde = abs(dde[1:]-dde[:-1])*20
de = np.append(dde, dde[-1])
force = np.linalg.norm(force, axis=1)
result = np.vstack((force, d, de))
# result = result[:, 1500:-1]
plt.plot(result[1], ".:", label='x')
plt.plot(result[0], ":", label='f')
plt.plot(result[2], label='de')
plt.legend(fontsize=15)
plt.show()
