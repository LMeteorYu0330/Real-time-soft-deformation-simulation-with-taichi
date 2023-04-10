import pyhaptics as ph
import taichi as ti
import fem_class as fem
import collide_detection as cd
import numpy as np
from math import pi


@ti.data_oriented
class haptices:
    def __init__(self, verts):
        self.mat_row = np.zeros([4, 4])
        self.verts = verts
        self.mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=1)
        self.rota_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=1)

        self.rota(1, -pi / 2)

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
                    self.mat[0][i, j] = mat[i, j] * 0.01
                else:
                    self.mat[0][i, j] = mat[i, j]

        for vert in self.verts.ox:
            X = ti.Vector([self.verts.ox[vert].x, self.verts.ox[vert].y, self.verts.ox[vert].z, 1])
            T = X @ self.mat[0]
            # print(T)
            self.verts.x[vert].x = T[0]
            self.verts.x[vert].y = T[1] + 0.5
            self.verts.x[vert].z = T[2]

    def run(self):
        self.get_mat()
        self.model_transpose(self.mat_row)


if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    ph.init()
    obj = "model/equipment/Scissors.stl"
    model = fem.LoadModel(obj)
    hap = haptices(model.mesh.verts)
    bvt = cd.aabb_obj(model.mesh.verts, layer_num=0)

    window = ti.ui.Window("FEM", (768, 768), vsync=False)
    canvas = window.get_canvas()
    canvas.set_background_color(color=(1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.up(0, 1, 0)
    camera.position(0, 0, -0.6)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(75)

    while window.running:
        bvt.run()

        hap.run()

        camera.track_user_inputs(window, 0.0008, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(camera.curr_position, (0.7, 0.7, 0.7))
        scene.mesh(model.mesh.verts.x, model.indices, color=(1.0, 0.3, 0.3))
        scene.particles(bvt.aabb_root, 0.002, (0.2, 0.2, 0.2))
        scene.particles(bvt.aabb_tree, 0.002, (0.9, 0.9, 0.9), index_offset=2, index_count=bvt.tree_size - 2)
        # scene.lines(bvt.aabb_tree, 0.1, color=(0.5, 0.5, 0.5), vertex_count=bvt.tree_size-3, vertex_offset=3)
        canvas.scene(scene)
        window.show()
