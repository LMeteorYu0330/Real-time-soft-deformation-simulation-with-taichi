import matplotlib.pyplot as plt
import numpy as np

force = np.loadtxt("force.txt")
d = np.loadtxt("d.txt") * 10
ddde = np.loadtxt("de.txt")+0.4
dde = abs(ddde[1:]-ddde[:-1])
de = np.append(dde, dde[-1])
force = np.linalg.norm(force, axis=1)
result = np.vstack((force, d, ddde))
# result = result[:, 1500:-1]
# plt.plot(result[1], ".:", label='x')
# plt.plot(result[0], ":", label='f')
plt.plot(result[2], label='de')
plt.legend(fontsize=15)
plt.show()
