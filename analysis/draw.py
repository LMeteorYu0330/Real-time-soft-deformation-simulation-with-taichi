import matplotlib.pyplot as plt
import numpy as np

# force = np.loadtxt("force.txt")
# d = np.loadtxt("d.txt") * 1000
# ddde = np.loadtxt("de.txt")
# fi = np.loadtxt("fi.txt")
# force = np.linalg.norm(force, axis=1)
# fi = fi[1::2]
# ddde = ddde[1::2]
# d = d[1::2]
# # result = np.vstack((ddde))
# # result = result[:, 1500:-1]
# # plt.plot(result[1], ".:", label='x')
# # plt.plot(result[0], ":", label='f')
# # ddde = -np.sum(ddde, 1)
# result = np.vstack((ddde, d, fi))
# # plt.plot(result[0], label='de')
# # plt.plot(result[1], ".:", label='x')
# plt.plot(result[2], ":", label='f')
# plt.legend(fontsize=15)
# plt.show()

##############################################

ana = np.loadtxt("de.txt")
for i in range(ana.shape[1]):
    ma = max(ana[:, i])
    mi = min(ana[:, i])
    ana[:, i] = (ana[:, i] - mi) / (ma - mi)
vis_f = (ana[:, 0] + ana[:, 1] + ana[:, 2])/3# * -1 + 1
vis_c = (ana[:, 3] + ana[:, 4] + ana[:, 5])/3
# plt.plot(ana[::2, 0], label='fx')
# plt.plot(-ana[::2, 1], label='fy')
# plt.plot(ana[::2, 2], label='fz')
# plt.plot(ana[::2, 3], label='cx')
# plt.plot(ana[::2, 4], label='cy')
# plt.plot(ana[::2, 5], label='cz')
plt.plot(vis_c[::2], label='f')
plt.plot(vis_f[::2], label='c')
plt.legend(fontsize=15)
plt.show()
