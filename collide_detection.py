import taichi as ti


@ti.data_oriented
class aabb_obj:
    def __init__(self, verts, layer_num=3):
        self.verts = verts
        self.layer_num = layer_num

        self.tree_size = int((8 ** self.layer_num + 13) / 7)
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
        base_z = (self.aabb_root[1] - self.aabb_tree[0]).z
        base_x = (self.aabb_root[4] - self.aabb_tree[0]).x
        base_y = (self.aabb_root[2] - self.aabb_tree[0]).y
        layer_node_num = 1
        for layer in ti.static(range(1, self.layer_num + 1)):
            m = 2 ** layer
            for node in ti.static(range(1, 1 + (2 ** (3 * (layer - 1))))):
                n1 = 2 * ti.ceil(node / (4 ** (layer - 1))) - 1
                k2 = (node % (4 ** (layer - 1)))
                n2 = 2 * int(k2 / (m / 2) + 1) - 1
                n3 = 2 * (node % (m / 2) + 1) - 1
                self.aabb_tree[layer_node_num + node] = \
                    self.aabb_tree[0] + [n1 / m * base_x, n2 / m * base_y, n3 / m * base_z]
            layer_node_num += (2 ** (3 * (layer - 1)))

    def run(self):
        self.get_root()
        self.get_aabb_root()
        self.get_aabb_tree()


