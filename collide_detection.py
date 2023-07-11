import taichi as ti


@ti.data_oriented
class aabb_obj:
    def __init__(self, model, layer_num=3):
        self.model = model
        self.mesh = self.model.mesh
        self.verts = self.mesh.verts
        self.faces = self.mesh.faces
        self.layer_num = layer_num
        self.face_num = len(self.faces)
        self.vert_num = self.model.vert_num

        self.min_box = ti.Vector.field(3, ti.f32, shape=self.face_num * 8)
        self.min_box_for_draw = ti.Vector.field(3, ti.f32, shape=self.face_num * 24)
        self.aabb_tree_num = ti.Vector.field(layer_num, ti.i32, shape=self.face_num)
        self.face_barycenter = ti.Vector.field(3, ti.f32, shape=self.face_num)
        self.layer0_box = ti.Vector.field(3, ti.f32, shape=1 * 8)
        self.layer0_box_for_draw = ti.Vector.field(3, ti.f32, shape=1 * 24)
        self.layer1_box = ti.Vector.field(3, ti.f32, shape=8 * 8)
        self.layer1_box_for_draw = ti.Vector.field(3, ti.f32, shape=8 * 24)

        self.tree = ti.root.dense(ti.i, 8).dynamic(ti.j, self.face_num)
        self.box_struct = ti.types.struct(vert0=ti.i32, vert1=ti.i32, vert2=ti.i32, fid=ti.i32)
        self.box = self.box_struct.field()
        self.tree.place(self.box)

    def get_root(self):
        x_np = self.verts.x.to_numpy()
        self.layer0_box[0] = x_np.min(0)
        self.layer0_box[7] = x_np.max(0)

    @ti.kernel
    def get_box(self):
        for i in range(8):
            self.box[i].deactivate()

        for face in self.faces:
            # 这段循环仅用于可视化，但是不能删也不能挪位置，不然会掉帧（未查明原因）
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

        self.layer1_box.fill(self.model.center[0])
        for face in self.faces:
            self.face_barycenter[face.id] = (self.verts.x[face.verts[0].id]
                                             + self.verts.x[face.verts[1].id]
                                             + self.verts.x[face.verts[2].id]) / 3.0
            if self.face_barycenter[face.id].x < self.model.center[0].x \
                    and self.face_barycenter[face.id].y < self.model.center[0].y \
                    and self.face_barycenter[face.id].z < self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 0
                self.box[0].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))
            elif self.face_barycenter[face.id].x < self.model.center[0].x \
                    and self.face_barycenter[face.id].y < self.model.center[0].y \
                    and self.face_barycenter[face.id].z >= self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 1
                self.box[1].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))
            elif self.face_barycenter[face.id].x < self.model.center[0].x \
                    and self.face_barycenter[face.id].y >= self.model.center[0].y \
                    and self.face_barycenter[face.id].z < self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 2
                self.box[2].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))
            elif self.face_barycenter[face.id].x < self.model.center[0].x \
                    and self.face_barycenter[face.id].y >= self.model.center[0].y \
                    and self.face_barycenter[face.id].z >= self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 3
                self.box[3].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))
            elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                    and self.face_barycenter[face.id].y < self.model.center[0].y \
                    and self.face_barycenter[face.id].z < self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 4
                self.box[4].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))
            elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                    and self.face_barycenter[face.id].y < self.model.center[0].y \
                    and self.face_barycenter[face.id].z >= self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 5
                self.box[5].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))
            elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                    and self.face_barycenter[face.id].y >= self.model.center[0].y \
                    and self.face_barycenter[face.id].z < self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 6
                self.box[6].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))
            elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                    and self.face_barycenter[face.id].y >= self.model.center[0].y \
                    and self.face_barycenter[face.id].z >= self.model.center[0].z:
                self.aabb_tree_num[face.id][0] = 7
                self.box[7].append(self.box_struct(face.verts[0].id, face.verts[1].id, face.verts[2].id, face.id))

            if self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 0].x > self.face_barycenter[face.id].x:
                self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 0].x = self.face_barycenter[face.id].x
            if self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 7].x <= self.face_barycenter[face.id].x:
                self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 7].x = self.face_barycenter[face.id].x
            if self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 0].y > self.face_barycenter[face.id].y:
                self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 0].y = self.face_barycenter[face.id].y
            if self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 7].y <= self.face_barycenter[face.id].y:
                self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 7].y = self.face_barycenter[face.id].y
            if self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 0].z > self.face_barycenter[face.id].z:
                self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 0].z = self.face_barycenter[face.id].z
            if self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 7].z <= self.face_barycenter[face.id].z:
                self.layer1_box[self.aabb_tree_num[face.id][0] * 8 + 7].z = self.face_barycenter[face.id].z

    @ti.kernel
    def box_for_draw(self):

        self.layer0_box[1] = [self.layer0_box[0].x, self.layer0_box[0].y, self.layer0_box[7].z]
        self.layer0_box[2] = [self.layer0_box[0].x, self.layer0_box[7].y, self.layer0_box[0].z]
        self.layer0_box[3] = [self.layer0_box[0].x, self.layer0_box[7].y, self.layer0_box[7].z]
        self.layer0_box[4] = [self.layer0_box[7].x, self.layer0_box[0].y, self.layer0_box[0].z]
        self.layer0_box[5] = [self.layer0_box[7].x, self.layer0_box[0].y, self.layer0_box[7].z]
        self.layer0_box[6] = [self.layer0_box[7].x, self.layer0_box[7].y, self.layer0_box[0].z]

        for box in range(8):
            self.layer1_box[box * 8 + 1] = [self.layer1_box[box * 8 + 0].x, self.layer1_box[box * 8 + 0].y,
                                            self.layer1_box[box * 8 + 7].z]
            self.layer1_box[box * 8 + 2] = [self.layer1_box[box * 8 + 0].x, self.layer1_box[box * 8 + 7].y,
                                            self.layer1_box[box * 8 + 0].z]
            self.layer1_box[box * 8 + 3] = [self.layer1_box[box * 8 + 0].x, self.layer1_box[box * 8 + 7].y,
                                            self.layer1_box[box * 8 + 7].z]
            self.layer1_box[box * 8 + 4] = [self.layer1_box[box * 8 + 7].x, self.layer1_box[box * 8 + 0].y,
                                            self.layer1_box[box * 8 + 0].z]
            self.layer1_box[box * 8 + 5] = [self.layer1_box[box * 8 + 7].x, self.layer1_box[box * 8 + 0].y,
                                            self.layer1_box[box * 8 + 7].z]
            self.layer1_box[box * 8 + 6] = [self.layer1_box[box * 8 + 7].x, self.layer1_box[box * 8 + 7].y,
                                            self.layer1_box[box * 8 + 0].z]

        self.layer0_box_for_draw[0] = self.layer0_box[0]
        self.layer0_box_for_draw[1] = self.layer0_box[4]
        self.layer0_box_for_draw[2] = self.layer0_box[4]
        self.layer0_box_for_draw[3] = self.layer0_box[5]
        self.layer0_box_for_draw[4] = self.layer0_box[5]
        self.layer0_box_for_draw[5] = self.layer0_box[1]
        self.layer0_box_for_draw[6] = self.layer0_box[1]
        self.layer0_box_for_draw[7] = self.layer0_box[0]
        self.layer0_box_for_draw[8] = self.layer0_box[2]
        self.layer0_box_for_draw[9] = self.layer0_box[6]
        self.layer0_box_for_draw[10] = self.layer0_box[6]
        self.layer0_box_for_draw[11] = self.layer0_box[7]
        self.layer0_box_for_draw[12] = self.layer0_box[7]
        self.layer0_box_for_draw[13] = self.layer0_box[3]
        self.layer0_box_for_draw[14] = self.layer0_box[3]
        self.layer0_box_for_draw[15] = self.layer0_box[2]
        self.layer0_box_for_draw[16] = self.layer0_box[2]
        self.layer0_box_for_draw[17] = self.layer0_box[0]
        self.layer0_box_for_draw[18] = self.layer0_box[3]
        self.layer0_box_for_draw[19] = self.layer0_box[1]
        self.layer0_box_for_draw[20] = self.layer0_box[6]
        self.layer0_box_for_draw[21] = self.layer0_box[4]
        self.layer0_box_for_draw[22] = self.layer0_box[7]
        self.layer0_box_for_draw[23] = self.layer0_box[5]

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

        if self.layer_num >= 3:
            for i in range(8):
                self.layer1_box_for_draw[i * 24 + 0] = self.layer1_box[i * 8 + 0]
                self.layer1_box_for_draw[i * 24 + 1] = self.layer1_box[i * 8 + 4]
                self.layer1_box_for_draw[i * 24 + 2] = self.layer1_box[i * 8 + 4]
                self.layer1_box_for_draw[i * 24 + 3] = self.layer1_box[i * 8 + 5]
                self.layer1_box_for_draw[i * 24 + 4] = self.layer1_box[i * 8 + 5]
                self.layer1_box_for_draw[i * 24 + 5] = self.layer1_box[i * 8 + 1]
                self.layer1_box_for_draw[i * 24 + 6] = self.layer1_box[i * 8 + 1]
                self.layer1_box_for_draw[i * 24 + 7] = self.layer1_box[i * 8 + 0]
                self.layer1_box_for_draw[i * 24 + 8] = self.layer1_box[i * 8 + 2]
                self.layer1_box_for_draw[i * 24 + 9] = self.layer1_box[i * 8 + 6]
                self.layer1_box_for_draw[i * 24 + 10] = self.layer1_box[i * 8 + 6]
                self.layer1_box_for_draw[i * 24 + 11] = self.layer1_box[i * 8 + 7]
                self.layer1_box_for_draw[i * 24 + 12] = self.layer1_box[i * 8 + 7]
                self.layer1_box_for_draw[i * 24 + 13] = self.layer1_box[i * 8 + 3]
                self.layer1_box_for_draw[i * 24 + 14] = self.layer1_box[i * 8 + 3]
                self.layer1_box_for_draw[i * 24 + 15] = self.layer1_box[i * 8 + 2]
                self.layer1_box_for_draw[i * 24 + 16] = self.layer1_box[i * 8 + 2]
                self.layer1_box_for_draw[i * 24 + 17] = self.layer1_box[i * 8 + 0]
                self.layer1_box_for_draw[i * 24 + 18] = self.layer1_box[i * 8 + 3]
                self.layer1_box_for_draw[i * 24 + 19] = self.layer1_box[i * 8 + 1]
                self.layer1_box_for_draw[i * 24 + 20] = self.layer1_box[i * 8 + 6]
                self.layer1_box_for_draw[i * 24 + 21] = self.layer1_box[i * 8 + 4]
                self.layer1_box_for_draw[i * 24 + 22] = self.layer1_box[i * 8 + 7]
                self.layer1_box_for_draw[i * 24 + 23] = self.layer1_box[i * 8 + 5]

    def run(self):
        self.model.cal_barycenter()
        self.get_root()
        self.get_box()
        # self.box_for_draw()


@ti.data_oriented
class deceteor:
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        self.box1 = self.obj1.box
        self.box2 = self.obj2.box
        self.face_barycenter1 = self.obj1.face_barycenter
        self.face_barycenter2 = self.obj2.face_barycenter
        self.line = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.box_is_cross = ti.field(ti.i32, shape=3)
        self.min_box_decete = ti.field(ti.i32, shape=1)

        self.cross_tree = ti.root.dynamic(ti.i, 16)
        self.cross = ti.types.struct(tree1=ti.i32, tree2=ti.i32)
        self.cross_num = self.cross.field()
        self.cross_tree.place(self.cross_num)

        self.D_tree = ti.root.dynamic(ti.i, 2048)
        self.dist = ti.types.struct(D=ti.f32)
        self.DF = self.dist.field()
        self.D_tree.place(self.DF)

    @ti.kernel
    def aabb_cross_detect0(self):
        self.cross_num.deactivate()
        self.min_box_decete[0] = 0

        self.detect(self.obj1.layer0_box[0],
                    self.obj2.layer0_box[0],
                    self.obj1.layer0_box[7],
                    self.obj2.layer0_box[7], 0)
        if self.box_is_cross[0] == 1:
            for i in range(64):
                box1 = (i // 8) % 8
                box2 = i % 8
                self.detect(self.obj1.layer1_box[8 * box1 + 0],
                            self.obj2.layer1_box[8 * box2 + 0],
                            self.obj1.layer1_box[8 * box1 + 7],
                            self.obj2.layer1_box[8 * box2 + 7],
                            1)
                if self.box_is_cross[1] == 1:
                    self.cross_num.append(self.cross(box1, box2))
                    self.min_box_decete[0] = 1

    line_type = ti.types.ndarray(dtype=ti.i32, ndim=1)

    @ti.kernel
    def line_tri_detect(self, lmin: line_type, lmax: line_type):
        self.line[0] = self.obj2.verts.x[lmin[0]]
        self.line[1] = self.obj2.verts.x[lmax[0]]
        line = self.line[0] - self.line[1]


    @ti.kernel
    def aabb_cross_detect1_ballball(self):
        self.DF.deactivate()
        if self.min_box_decete[0] == 1:
            for i in range(self.cross_num.length()):
                # print(self.cross_num[i].tree1)
                for j in range(self.box1[self.cross_num[i].tree1].length()):
                    for k in range(self.box2[self.cross_num[i].tree2].length()):
                        P1 = self.face_barycenter1[self.box1[self.cross_num[i].tree1, j].fid]
                        P2 = self.face_barycenter2[self.box2[self.cross_num[i].tree2, k].fid]
                        P12 = (P1 - P2) ** 2
                        D = (P12.x + P12.y + P12.z) ** 0.5
                        if D <= 0.005:
                            self.DF.append(D)
        if self.DF.length() != 0:
            print(self.DF.length())
        # print(self.box1[self.cross_num[i].tree1, j].vert0,
        #       self.box1[self.cross_num[i].tree1, j].vert1,
        #       self.box1[self.cross_num[i].tree1, j].vert2)
        # print(self.box1[self.cross_num[i].tree1, j].vert0)
        # print(self.obj1.verts.x[self.box1[self.cross_num[i].tree1, j].vert0])

    # @ti.kernel
    # def aabb_cross_detect1_ballball(self):
    #     self.DF.deactivate()
    #     if self.min_box_decete[0] == 1:
    #         for i in range(self.cross_num.length()):
    #             len1 = self.box1[self.cross_num[i].tree1].length()
    #             len2 = self.box2[self.cross_num[i].tree2].length()
    #             lenth = len1 * len2
    #             for j in range(lenth):
    #                 P1 = self.face_barycenter1[self.box1[self.cross_num[i].tree1, j // len2 % len1].fid]
    #                 P2 = self.face_barycenter2[self.box2[self.cross_num[i].tree2, j % len2].fid]
    #                 P12 = (P1 - P2) ** 2
    #                 D = P12.x + P12.y + P12.z
    #                 if D <= 0.003:
    #                     self.DF.append(D)
    #         if self.DF.length() != 0:
    #             print(self.DF.length())

    @ti.func
    def detect(self, aabb1min, aabb2min, aabb1max, aabb2max, i):
        if aabb1max.x > aabb2min.x and aabb1min.x < aabb2max.x \
                and aabb1max.y > aabb2min.y and aabb1min.y < aabb2max.y \
                and aabb1max.z > aabb2min.z and aabb1min.z < aabb2max.z:
            self.box_is_cross[i] = 1
        else:
            self.box_is_cross[i] = 0

    def run(self):
        self.aabb_cross_detect0()
        self.line_tri_detect(self.obj2.model.line0, self.obj2.model.line1)
        # self.aabb_cross_detect1_ballball()
