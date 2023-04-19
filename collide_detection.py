import taichi as ti


@ti.data_oriented
class aabb_obj:
    def __init__(self, meshs, layer_num=4):
        self.meshs = meshs
        self.verts = self.meshs.verts
        self.faces = self.meshs.faces
        self.layer_num = layer_num
        self.face_num = len(self.faces)

        self.min_box = ti.Vector.field(3, ti.f32, shape=self.face_num * 8)
        self.min_box_for_draw = ti.Vector.field(3, ti.f32, shape=self.face_num * 24)
        self.tree_size = int((8 ** self.layer_num + 13) / 7)
        self.aabb_root = ti.Vector.field(3, ti.f32, shape=8)
        self.aabb_tree = ti.Vector.field(3, ti.f32, shape=self.tree_size)
        self.box_edge_len = ti.Vector.field(3, ti.f32, shape=1)

    def get_root(self):
        x_np = self.verts.x.to_numpy()
        self.aabb_tree[0] = x_np.min(0)
        self.aabb_tree[1] = x_np.max(0)

    @ti.kernel
    def get_aabb_tree(self):
        self.box_edge_len[0].x = self.aabb_tree[1].x - self.aabb_tree[0].x
        self.box_edge_len[0].y = self.aabb_tree[1].y - self.aabb_tree[0].y
        self.box_edge_len[0].z = self.aabb_tree[1].z - self.aabb_tree[0].z
        layer_node_num = 1
        for layer in ti.static(range(1, self.layer_num + 1)):
            m = 2 ** layer
            for node in ti.static(range(1, 1 + (2 ** (3 * (layer - 1))))):
                n1 = 2 * ti.ceil(node / (4 ** (layer - 1))) - 1
                k2 = (node % (4 ** (layer - 1)))
                n2 = 2 * int(k2 / (m / 2) + 1) - 1
                n3 = 2 * (node % (m / 2) + 1) - 1
                self.aabb_tree[layer_node_num + node] = \
                    self.aabb_tree[0] + \
                    [n1 / m * self.box_edge_len[0].x, n2 / m * self.box_edge_len[0].y, n3 / m * self.box_edge_len[0].z]
            layer_node_num += (2 ** (3 * (layer - 1)))

    @ti.kernel
    def get_min_box(self):
        for face in self.faces:
            MAX = ti.max(self.verts.x[face.verts[0].id],
                         self.verts.x[face.verts[1].id],
                         self.verts.x[face.verts[2].id])
            MIN = ti.min(self.verts.x[face.verts[0].id],
                         self.verts.x[face.verts[1].id],
                         self.verts.x[face.verts[2].id])
            self.min_box[face.id * 8 + 0] = [MIN.x, MIN.y, MIN.z]
            self.min_box[face.id * 8 + 1] = [MIN.x, MIN.y, MAX.z]
            self.min_box[face.id * 8 + 2] = [MIN.x, MAX.y, MIN.z]
            self.min_box[face.id * 8 + 3] = [MIN.x, MAX.y, MAX.z]
            self.min_box[face.id * 8 + 4] = [MAX.x, MIN.y, MIN.z]
            self.min_box[face.id * 8 + 5] = [MAX.x, MIN.y, MAX.z]
            self.min_box[face.id * 8 + 6] = [MAX.x, MAX.y, MIN.z]
            self.min_box[face.id * 8 + 7] = [MAX.x, MAX.y, MAX.z]

    @ti.kernel
    def box_for_draw(self):
        for i in range(0, int(self.face_num)):
            self.min_box_for_draw[i * 24 + 0] = self.min_box[i * 8 + 0]
            self.min_box_for_draw[i * 24 + 1] = self.min_box[i * 8 + 4]
            self.min_box_for_draw[i * 24 + 2] = self.min_box[i * 8 + 4]
            self.min_box_for_draw[i * 24 + 3] = self.min_box[i * 8 + 5]
            self.min_box_for_draw[i * 24 + 4] = self.min_box[i * 8 + 5]
            self.min_box_for_draw[i * 24 + 5] = self.min_box[i * 8 + 1]
            self.min_box_for_draw[i * 24 + 6] = self.min_box[i * 8 + 1]
            self.min_box_for_draw[i * 24 + 7] = self.min_box[i * 8 + 0]
            self.min_box_for_draw[i * 24 + 8] = self.min_box[i * 8 + 2]
            self.min_box_for_draw[i * 24 + 9] = self.min_box[i * 8 + 6]
            self.min_box_for_draw[i * 24 + 10] = self.min_box[i * 8 + 6]
            self.min_box_for_draw[i * 24 + 11] = self.min_box[i * 8 + 7]
            self.min_box_for_draw[i * 24 + 12] = self.min_box[i * 8 + 7]
            self.min_box_for_draw[i * 24 + 13] = self.min_box[i * 8 + 3]
            self.min_box_for_draw[i * 24 + 14] = self.min_box[i * 8 + 3]
            self.min_box_for_draw[i * 24 + 15] = self.min_box[i * 8 + 2]
            self.min_box_for_draw[i * 24 + 16] = self.min_box[i * 8 + 2]
            self.min_box_for_draw[i * 24 + 17] = self.min_box[i * 8 + 0]
            self.min_box_for_draw[i * 24 + 18] = self.min_box[i * 8 + 3]
            self.min_box_for_draw[i * 24 + 19] = self.min_box[i * 8 + 1]
            self.min_box_for_draw[i * 24 + 20] = self.min_box[i * 8 + 6]
            self.min_box_for_draw[i * 24 + 21] = self.min_box[i * 8 + 4]
            self.min_box_for_draw[i * 24 + 22] = self.min_box[i * 8 + 7]
            self.min_box_for_draw[i * 24 + 23] = self.min_box[i * 8 + 5]

    def run(self):
        self.get_min_box()
        self.box_for_draw()
        # self.get_root()
        # self.get_aabb_tree()
