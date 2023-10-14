import pyhaptics as ph
import taichi as ti
import numpy as np


@ti.data_oriented
class haptices:
    def __init__(self, verts, direct, init_rota):
        self.mat_row = np.zeros([4, 4])
        self.verts = verts
        self.mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=1)
        self.rota_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=1)

        self.rota(direct, init_rota)

    @ti.kernel
    def rota(self, direction: ti.i32, alpha: ti.f32):
        if direction == 0:
            self.rota_mat[0] = ([[1, 0, 0],
                                 [0, ti.cos(alpha), ti.sin(alpha)],
                                 [0, -ti.sin(alpha), ti.cos(alpha)]
                                 ])
        elif direction == 1:
            self.rota_mat[0] = ([[ti.cos(alpha), 0, -ti.sin(alpha)],
                                 [0, 1, 0],
                                 [ti.sin(alpha), 0, ti.cos(alpha)],
                                 ])
        else:
            self.rota_mat[0] = ([[ti.cos(alpha), ti.sin(alpha), 0],
                                 [-ti.sin(alpha), ti.cos(alpha), 0],
                                 [0, 0, 1]
                                 ])
        for vert in self.verts.ox:
            T = self.verts.ox[vert] @ self.rota_mat[0]
            self.verts.ox[vert] = T

    def get_mat(self):
        self.mat_row = np.array(ph.get_transform()).reshape(4, -1)

    @ti.kernel
    def model_transpose(self, mat: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i in range(4):
            for j in range(4):
                if i == 3 and j != 3:
                    self.mat[0][i, j] = mat[i, j] * 0.02
                else:
                    self.mat[0][i, j] = mat[i, j]

        for vert in self.verts.ox:
            X = ti.Vector([self.verts.ox[vert].x, self.verts.ox[vert].y, self.verts.ox[vert].z, 1])
            T = X @ self.mat[0]
            # print(T)
            self.verts.x[vert].x = T[0]
            self.verts.x[vert].y = T[1] + 0.5
            self.verts.x[vert].z = T[2]
            self.verts.rx[vert] = self.verts.x[vert]

    @staticmethod
    def set_force(x: ti.f32, y: ti.f32, z: ti.f32):
        ph.set_force(x, y, z)

    def run(self):
        self.get_mat()
        self.model_transpose(self.mat_row)
