import matplotlib.pyplot as plt
import numpy as np

force = np.loadtxt("force.txt")
# force = force[700:800]
force_scalar = np.linalg.norm(force, axis=1)
plt.plot(force_scalar)
plt.show()

