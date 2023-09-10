import taichi as ti
import fem_class as fem


@ti.data_oriented
class dcd:
    def __init__(self, *args, **kwargs):
        self.obj_num = len(args)
        names = self.__dict__
        for i in range(self.obj_num):
            names['obj' + str(i)] = args[i]
            names['mesh' + str(i)] = args[i].mesh
        self.faces0_num = len(args[0].mesh.faces)
        self.line = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.F = ti.Vector.field(1, dtype=ti.f32, shape=())
        self.K = 0.5
        self.D = 1.5
        self.pre_d0 = ti.Vector.field(1, dtype=ti.f32, shape=())

    line_type = ti.types.ndarray(dtype=ti.i32, ndim=1)

    @ti.kernel
    def line_tri_detect(self, lmin: line_type, lmax: line_type):
        self.line[0] = self.mesh1.verts.x[lmin[0]]
        self.line[1] = self.mesh1.verts.x[lmax[0]]
        self.force.fill(0)
        for face0 in self.mesh0.faces:
            # if face0.cells.size == 1:
            if True:
                v0 = self.mesh0.verts.x[face0.verts[0].id]
                v1 = self.mesh0.verts.x[face0.verts[1].id]
                v2 = self.mesh0.verts.x[face0.verts[2].id]
                e1 = self.plucker(v1, v0)
                e2 = self.plucker(v2, v1)
                e3 = self.plucker(v0, v2)
                L = self.plucker(self.line[0], self.line[1])

                s1 = self.sideOp(L, e1)
                s2 = self.sideOp(L, e2)
                s3 = self.sideOp(L, e3)

                if (s1 > 0 and s2 > 0 and s3 > 0) or (s1 < 0 and s2 < 0 and s3 < 0):
                    l3 = self.plucker(v0, self.line[0])
                    l2 = self.plucker(self.line[1], v0)
                    ss1 = self.sideOp(e2, l3)
                    ss2 = self.sideOp(e2, l2)
                    if ss1 > 0 and ss2 > 0:
                        self.intersect(0, face0)  # 线穿三角形
                #     elif ss1 == 0 or ss2 == 0:
                #         self.intersect(0, face0)  # 线的一端触碰到面
                # elif (s1 == 0 and s2 * s3 > 0) or (s2 == 0 and s1 * s3 > 0) or (s3 == 0 and s1 * s2 > 0):
                #     self.intersect(2, face0)  # 线擦面
                # elif (s1 == 0 and (s2 == 0)) or (s1 == 0 and (s3 == 0)) or (s2 == 0 and (s3 == 0)):
                #     self.intersect(0, face0)  # 线过点

    @ti.func
    def intersect(self, mode, face):
        if mode == 0:
            edge1 = self.mesh0.verts.x[face.edges[0].verts[1].id] - self.mesh0.verts.x[face.edges[0].verts[0].id]
            edge2 = self.mesh0.verts.x[face.edges[1].verts[1].id] - self.mesh0.verts.x[face.edges[1].verts[0].id]
            face_n = ti.math.normalize(ti.math.cross(edge1, edge2))
            face_cen = (self.mesh0.verts.x[face.verts[0].id]
                        + self.mesh0.verts.x[face.verts[0].id]
                        + self.mesh0.verts.x[face.verts[0].id]) / 3
            d0 = ti.math.distance(face_cen, self.line[0])  # 计算三角形质心到线端点的距离
            self.F[None] = self.K * d0 + self.D * (d0 - self.pre_d0[None])
            self.mesh0.verts.v[face.verts[0].id] += -1 * self.F[None][0] * face_n  # 用距离和方向给顶点速度
            self.mesh0.verts.v[face.verts[1].id] += -1 * self.F[None][0] * face_n
            self.mesh0.verts.v[face.verts[2].id] += -1 * self.F[None][0] * face_n

            self.force[None] += 3 * self.F[None][0] * face_n  # 用速度的相反量给力反馈作用力
            self.pre_d0[None][0] = d0

        if mode == 1:  # 不合适的力计算函数，但是能work
            for i in range(3):
                d0 = ti.math.distance(self.mesh0.verts.x[face.verts[i].id], self.line[0])
                n = self.line[0] - self.mesh0.verts.x[face.verts[i].id]
                n_nor = ti.math.normalize(n)
                self.mesh0.verts.v[face.verts[i].id].x += d0 * n_nor[0]
                self.mesh0.verts.v[face.verts[i].id].y += d0 * n_nor[1]
                self.mesh0.verts.v[face.verts[i].id].z += d0 * n_nor[2]
                self.force[None].x += -3 * d0 * n_nor[0]
                self.force[None].y += -3 * d0 * n_nor[1]
                self.force[None].z += -3 * d0 * n_nor[2]

        # # 计算面元法线方向
        # edge1 = self.mesh0.verts.x[face.edges[0].verts[1].id] - self.mesh0.verts.x[face.edges[0].verts[0].id]
        # edge2 = self.mesh0.verts.x[face.edges[1].verts[1].id] - self.mesh0.verts.x[face.edges[1].verts[0].id]
        # face_n = ti.math.normalize(ti.math.cross(edge1, edge2))

    @ti.func
    def plucker(self, a, b):
        l0 = a[0] * b[1] - b[0] * a[1]
        l1 = a[0] * b[2] - b[0] * a[2]
        l2 = a[0] - b[0]
        l3 = a[1] * b[2] - b[1] * a[2]
        l4 = a[2] - b[2]
        l5 = b[1] - a[1]
        return [l0, l1, l2, l3, l4, l5]

    @ti.func
    def sideOp(self, a, b):
        res = a[0] * b[4] + a[1] * b[5] + a[2] * b[3] + a[3] * b[2] + a[4] * b[0] + a[5] * b[1]
        return res

    def run(self):
        self.line_tri_detect(self.obj1.line0, self.obj1.line1)


if __name__ == '__main__':
    ti.init(arch=ti.gpu)

    obj = "model/liver/liver0.node"
    equipment = "model/equipment/Scissors.stl"

    model = fem.Implicit(obj, v_norm=1)
    equipment_model = fem.LoadModel(equipment)

    cd = dcd(model, equipment_model)

    cd.run()
