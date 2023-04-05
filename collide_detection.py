import taichi as ti
import numpy as np
import fem_class as fem


@ti.data_oriented
class aabb_obj:
    def __init__(self, verts):
        self.aabb_root = ti.Vector.field(3, ti.f32, shape=8)
        self.min_x = ti.Vector.field(3, ti.f32, shape=1)
        self.max_x = ti.Vector.field(3, ti.f32, shape=1)
        self.verts = verts

    def get_maxmin(self):
        x_np = self.verts.x.to_numpy()
        self.min_x[0] = x_np.min(0)
        self.max_x[0] = x_np.max(0)

    @ti.kernel
    def get_aabb_root(self):
        self.aabb_root[0] = self.min_x[0]
        self.aabb_root[1] = [self.min_x[0].x, self.min_x[0].y, self.max_x[0].z]
        self.aabb_root[2] = [self.min_x[0].x, self.max_x[0].y, self.max_x[0].z]
        self.aabb_root[3] = [self.min_x[0].x, self.max_x[0].y, self.min_x[0].z]
        self.aabb_root[4] = [self.max_x[0].x, self.min_x[0].y, self.min_x[0].z]
        self.aabb_root[5] = [self.max_x[0].x, self.min_x[0].y, self.max_x[0].z]
        self.aabb_root[6] = [self.max_x[0].x, self.max_x[0].y, self.min_x[0].z]
        self.aabb_root[7] = self.max_x[0]

    @ti.kernel
    def aabb_tree(self):
        pass

if __name__ == '__main__':
    ti.init(arch=ti.cuda)
    obj = "model/liver/liver0.node"
    mesh = fem.Implicit(obj, v_norm=1)
    bvt = aabb_obj(mesh.mesh.verts)
