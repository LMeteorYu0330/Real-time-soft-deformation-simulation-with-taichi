import taichi as ti
import numpy as np


@ti.data_oriented
class aabb_obj:
    def __init__(self, model, layer_num=3):
        self.model = model
        self.mesh = self.model.mesh
        self.verts = self.mesh.verts
        self.faces = self.mesh.faces
        self.layer_num = layer_num
        self.face_num = len(self.faces)

        self.min_box = ti.Vector.field(3, ti.f32, shape=self.face_num * 8)
        self.min_box_for_draw = ti.Vector.field(3, ti.f32, shape=self.face_num * 24)
        self.aabb_tree_num = ti.Vector.field(layer_num, ti.f32, shape=self.face_num)
        self.face_barycenter = ti.Vector.field(3, ti.f32, shape=self.face_num)
        self.layer0_box = ti.Vector.field(3, ti.f32, shape=1 * 8)
        self.layer0_box_for_draw = ti.Vector.field(3, ti.f32, shape=1 * 24)
        self.layer1_box = ti.Vector.field(3, ti.f32, shape=8 * 8)
        self.layer1_box_for_draw = ti.Vector.field(3, ti.f32, shape=8 * 24)

    def get_root(self):
        x_np = self.verts.x.to_numpy()
        self.layer0_box[0] = x_np.min(0)
        self.layer0_box[7] = x_np.max(0)

    @ti.kernel
    def get_box(self):
        """
        这部分代码有很大的优化空间，并行化高，盒子变得不稳定，盒子稳定又无法并行，很奇怪
        """
        # self.layer0_box[0] = self.model.center[0]
        # self.layer0_box[7] = self.model.center[0]
        # for vert in self.verts:
        #     self.layer0_box[0] = ti.atomic_min(self.layer0_box[0], vert.x)
        #     self.layer0_box[7] = ti.atomic_max(self.layer0_box[7], vert.x)
        self.layer0_box[1] = [self.layer0_box[0].x, self.layer0_box[0].y, self.layer0_box[7].z]
        self.layer0_box[2] = [self.layer0_box[0].x, self.layer0_box[7].y, self.layer0_box[0].z]
        self.layer0_box[3] = [self.layer0_box[0].x, self.layer0_box[7].y, self.layer0_box[7].z]
        self.layer0_box[4] = [self.layer0_box[7].x, self.layer0_box[0].y, self.layer0_box[0].z]
        self.layer0_box[5] = [self.layer0_box[7].x, self.layer0_box[0].y, self.layer0_box[7].z]
        self.layer0_box[6] = [self.layer0_box[7].x, self.layer0_box[7].y, self.layer0_box[0].z]

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

        if self.layer_num >= 3:
            self.layer1_box.fill(self.model.center[0])
            for face in self.faces:
                self.face_barycenter[face.id] = (self.verts.x[face.verts[0].id]
                                                 + self.verts.x[face.verts[1].id]
                                                 + self.verts.x[face.verts[2].id]) / 3.0
                if self.face_barycenter[face.id].x < self.model.center[0].x \
                        and self.face_barycenter[face.id].y < self.model.center[0].y \
                        and self.face_barycenter[face.id].z < self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 0
                elif self.face_barycenter[face.id].x < self.model.center[0].x \
                        and self.face_barycenter[face.id].y < self.model.center[0].y \
                        and self.face_barycenter[face.id].z >= self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 1
                elif self.face_barycenter[face.id].x < self.model.center[0].x \
                        and self.face_barycenter[face.id].y >= self.model.center[0].y \
                        and self.face_barycenter[face.id].z < self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 2
                elif self.face_barycenter[face.id].x < self.model.center[0].x \
                        and self.face_barycenter[face.id].y >= self.model.center[0].y \
                        and self.face_barycenter[face.id].z >= self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 3
                elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                        and self.face_barycenter[face.id].y < self.model.center[0].y \
                        and self.face_barycenter[face.id].z < self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 4
                elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                        and self.face_barycenter[face.id].y < self.model.center[0].y \
                        and self.face_barycenter[face.id].z >= self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 5
                elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                        and self.face_barycenter[face.id].y >= self.model.center[0].y \
                        and self.face_barycenter[face.id].z < self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 6
                elif self.face_barycenter[face.id].x >= self.model.center[0].x \
                        and self.face_barycenter[face.id].y >= self.model.center[0].y \
                        and self.face_barycenter[face.id].z >= self.model.center[0].z:
                    self.aabb_tree_num[face.id][0] = 7

                for box in range(8):
                    if self.aabb_tree_num[face.id][0] == box:
                        if self.layer1_box[box * 8 + 0].x > self.face_barycenter[face.id].x:
                            self.layer1_box[box * 8 + 0].x = self.face_barycenter[face.id].x
                        if self.layer1_box[box * 8 + 7].x <= self.face_barycenter[face.id].x:
                            self.layer1_box[box * 8 + 7].x = self.face_barycenter[face.id].x
                        if self.layer1_box[box * 8 + 0].y > self.face_barycenter[face.id].y:
                            self.layer1_box[box * 8 + 0].y = self.face_barycenter[face.id].y
                        if self.layer1_box[box * 8 + 7].y <= self.face_barycenter[face.id].y:
                            self.layer1_box[box * 8 + 7].y = self.face_barycenter[face.id].y
                        if self.layer1_box[box * 8 + 0].z > self.face_barycenter[face.id].z:
                            self.layer1_box[box * 8 + 0].z = self.face_barycenter[face.id].z
                        if self.layer1_box[box * 8 + 7].z <= self.face_barycenter[face.id].z:
                            self.layer1_box[box * 8 + 7].z = self.face_barycenter[face.id].z

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

    @ti.kernel
    def box_for_draw(self):
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
        self.box_for_draw()


@ti.data_oriented
class deceteor:
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2

        self.layer_is_cross = ti.field(ti.i32, shape=3)

    @ti.kernel
    def aabb_cross_detect(self):
        self.detect(self.obj1.layer0_box, self.obj2.layer0_box)
        if self.layer_is_cross[0] == 1:
            print(1)

    @ti.func
    def detect(self, aabb1, aabb2):
        if aabb1[7].x > aabb2[0].x and aabb1[0].x < aabb2[7].x \
                and aabb1[7].y > aabb2[0].y and aabb1[0].y < aabb2[7].y \
                and aabb1[7].z > aabb2[0].z and aabb1[0].z < aabb2[7].z:
            self.layer_is_cross[0] = 1
        else:
            self.layer_is_cross[0] = 0
