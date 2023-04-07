import pyhaptics as ph
import taichi as ti
import fem_class as fem
import collide_detection as cd
import numpy as np


@ti.data_oriented
class haptices:
    def __init__(self, verts):
        self.mat_row = np.zeros([4, 4])
        self.verts = verts
        self.mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=1)

    def get_mat(self):
        self.mat_row = np.array(ph.get_transform()).reshape(4, -1)
        for i in range(4):
            for j in range(4):
                self.mat[0][i, j] = self.mat_row[i, j]

    @ti.kernel
    def model_transpose(self):
        for vert in self.verts.ox:
            X = ti.Vector([self.verts.ox[vert].x, self.verts.ox[vert].y, self.verts.ox[vert].z, 1])
            T = X @ self.mat[0]
            # print(T)
            self.verts.x[vert].x = T[0]
            self.verts.x[vert].y = T[1]
            self.verts.x[vert].z = T[2]


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
    camera.position(3.5535414e-02, -6.5504539e+01, -8.8083954e+01)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(75)

    while window.running:
        bvt.run()
        hap.get_mat()
        hap.model_transpose()
        print(model.mesh.verts.x)

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
