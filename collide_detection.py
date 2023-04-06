import taichi as ti
import numpy as np
import fem_class as fem


@ti.data_oriented
class aabb_obj:
    def __init__(self, verts, layer_num=3):
        self.verts = verts
        self.layer_num = layer_num

        self.tree_size = int((8 ** (self.layer_num - 1) + 13) / 7)
        self.aabb_root = ti.Vector.field(3, ti.f32, shape=8)
        self.aabb_tree = ti.Vector.field(3, ti.f32, shape=self.tree_size)

    def get_root(self):
        x_np = self.verts.x.to_numpy()
        self.aabb_tree[0] = x_np.min(0)
        self.aabb_tree[1] = x_np.max(0)

    @ti.kernel
    def get_aabb_root(self):
        self.aabb_root[0] = self.aabb_tree[0]
        self.aabb_root[1] = [self.aabb_tree[0].x, self.aabb_tree[0].y, self.aabb_tree[1].z]
        self.aabb_root[2] = [self.aabb_tree[0].x, self.aabb_tree[1].y, self.aabb_tree[0].z]
        self.aabb_root[3] = [self.aabb_tree[0].x, self.aabb_tree[1].y, self.aabb_tree[1].z]
        self.aabb_root[4] = [self.aabb_tree[1].x, self.aabb_tree[0].y, self.aabb_tree[0].z]
        self.aabb_root[5] = [self.aabb_tree[1].x, self.aabb_tree[0].y, self.aabb_tree[1].z]
        self.aabb_root[6] = [self.aabb_tree[1].x, self.aabb_tree[1].y, self.aabb_tree[0].z]
        self.aabb_root[7] = self.aabb_tree[1]

    @ti.kernel
    def get_aabb_tree(self):
        for layer in range(1, self.layer_num):
            pass

    def run(self):
        self.get_root()
        self.get_aabb_root()
        self.get_aabb_tree()


if __name__ == '__main__':
    ti.init(arch=ti.cuda)
    obj = "model/liver/liver0.node"
    mesh = fem.Implicit(obj, v_norm=1)
    bvt = aabb_obj(mesh.mesh.verts)
