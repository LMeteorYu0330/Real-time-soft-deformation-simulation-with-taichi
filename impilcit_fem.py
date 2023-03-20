import taichi as ti
import meshtaichi_patcher as mp


@ti.data_oriented
class LoadModel:
    def __init__(self,
                 filename,
                 ):
        # load_mesh
        model_type = filename.split('.')[-1]
        if model_type == "node":
            self.mesh_rawdata = mp.load_mesh_rawdata(filename)
            self.mesh = mp.load_mesh(self.mesh_rawdata, relations=["CV", "VV", "CE", "EV"])
            self.mesh.verts.place({
                'x': ti.math.vec3,
                'v': ti.math.vec3,
                'f': ti.math.vec3
            })
            self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
            self.mesh.verts.v.fill(0.0)
            self.mesh.verts.f.fill(0.0)
            self.indices = ti.field(ti.u32, shape=len(self.mesh.cells) * 4 * 3)
            self.init_tet_indices()

        else:
            self.mesh_rawdata = mp.load_mesh_rawdata(filename)
            self.mesh = mp.load_mesh(self.mesh_rawdata, relations=["FV"])
            self.mesh.verts.place({'x': ti.math.vec3})
            self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
            self.indices = ti.field(ti.i32, shape=len(self.mesh.faces) * 3)
            self.init_surf_indices()

        self.vert_num = len(self.mesh.verts)
        self.center = ti.Vector.field(3, ti.f32, shape=())
        self.I = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ti.i32)
        self.cal_barycenter()

    @ti.kernel
    def init_surf_indices(self):
        for f in self.mesh.faces:
            for j in ti.static(range(3)):
                self.indices[f.id * 3 + j] = f.verts[j].id

    @ti.kernel
    def init_tet_indices(self):
        for c in self.mesh.cells:
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    self.indices[(c.id * 4 + i) * 3 + j] = c.verts[ind[i][j]].id

    @ti.kernel
    def cal_barycenter(self):
        for i in self.mesh.verts.x:
            self.center[None] += self.mesh.verts.x[i]
        self.center[None] /= self.vert_num


@ti.data_oriented
class Simulation(LoadModel):  # This class only for tetrahedron
    def __init__(self, filename, v_norm=1):
        super().__init__(filename)
        self.v_norm = v_norm

        self.dt = 1.0 / 30  # / 30000
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.e = 2e6  # 杨氏模量
        self.nu = 0.1  # 泊松系数
        self.mu = self.e / (2 * (1 + self.nu))
        self.la = self.e * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.density = 1e5

        self.cell_num = len(self.mesh.cells)
        self.V = ti.field(dtype=ti.f32, shape=())
        self.Dm = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)  # Dm
        self.W = ti.field(ti.f32, shape=self.cell_num)  # 四面体体积
        self.B = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)  # Dm逆
        self.m = ti.field(ti.f32, shape=self.vert_num)  # 点的质量
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)
        self.E = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)
        self.dD = ti.Matrix.field(3, 3, ti.i32, shape=(4, 3))
        self.dF = ti.Matrix.field(3, 3, ti.f32, shape=(4, 3))
        self.dE = ti.Matrix.field(3, 3, ti.f32, shape=(4, 3))
        self.dP = ti.Matrix.field(3, 3, ti.f32, shape=(4, 3))
        self.dH = ti.Matrix.field(3, 3, ti.f32, shape=(4, 3))

        self.K = ti.field(dtype=ti.f32, shape=(3 * self.vert_num, 3 * self.vert_num))
        # self.A = ti.field(dtype=ti.f32, shape=(3 * self.vert_num, 3 * self.vert_num))
        self.b = ti.field(dtype=ti.f32, shape=(3 * self.vert_num))
        self.x = ti.field(dtype=ti.f32, shape=(3 * self.vert_num))
        self.x_new = ti.field(dtype=ti.f32, shape=(3 * self.vert_num))
        self.norm_volume()
        self.fem_pre_cal()

    @ti.func
    def cal_dD(self):
        for i in ti.static(range(1)):
            for j in ti.static((range(3))):
                for m in ti.static(range(3)):
                    self.dD[i, j][j, m] = -1
        for i in ti.static(range(3)):
            for j in ti.static((range(3))):
                self.dD[i + 1, j][j, i] = 1

    @ti.kernel
    def norm_volume(self):
        for cell in self.mesh.cells:
            v = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    v[j, i] = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.V[None] += -(1.0 / 6.0) * v.determinant()
        if self.v_norm == 1:
            for vert in self.mesh.verts:
                vert.x *= 1000 / self.V[None]

    @ti.kernel
    def fem_pre_cal(self):  # fem参数预计算
        self.V[None] = 0
        for cell in self.mesh.cells:
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    self.Dm[cell.id][j, i] \
                        = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.B[cell.id] = self.Dm[cell.id].inverse()
            self.W[cell.id] = -(1.0 / 6.0) * self.Dm[cell.id].determinant()
            self.V[None] += self.W[cell.id]
            for i in ti.static(range(4)):
                self.m[cell.verts[i].id] += 0.25 * self.density * self.W[cell.id]  # 把体元质量均分到四个顶点
        self.cal_dD()

    @ti.kernel
    def fem_runtime_cal(self):  # 实时力计算
        for vert in self.mesh.verts:
            vert.f = self.gravity * self.m[vert.id]
        for cell in self.mesh.cells:
            Ds = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    Ds[j, i] \
                        = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.F[cell.id] = Ds @ self.B[cell.id]
            self.E[cell.id] = 0.5 * (self.F[cell.id].transpose() @ self.F[cell.id] - self.I)
            # U, sig, V = self.ssvd(self.F[cell.id])
            # P = 2 * self.mu * (self.F[cell.id] - U @ V.transpose())
            P = self.F[cell.id] @ (2 * self.mu * self.E[cell.id] + self.la * self.E[cell.id].trace() * self.I)
            H = -self.W[cell.id] * P @ self.B[cell.id].transpose()
            for i in ti.static(range(3)):
                fi = ti.Vector([H[0, i], H[1, i], H[2, i]])
                self.mesh.verts.f[cell.verts[i].id] += fi
                self.mesh.verts.f[cell.verts[3].id] += -fi

    @ti.func
    def ssvd(self, fai):
        U, sig, V = ti.svd(fai)
        if U.determinant() < 0:
            for i in ti.static(range(3)):
                U[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        if V.determinant() < 0:
            for i in ti.static(range(3)):
                V[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        return U, sig, V

    @ti.kernel
    def explicit_time_integral(self):
        for vert in self.mesh.verts:
            vert.v += self.dt * vert.f / self.m[vert.id]
            vert.x += vert.v * self.dt

    @ti.kernel
    def implicit_time_integral(self):
        for vert in self.mesh.verts:
            vert.v[0] = self.x[vert.id * 3 + 0]
            vert.v[1] = self.x[vert.id * 3 + 1]
            vert.v[2] = self.x[vert.id * 3 + 2]
            vert.x += vert.v * self.dt

    @ti.kernel
    def compute_K(self):
        for i, j in self.K:
            self.K[i, j] = 0
            # self.A[i, j] = 0

        for cell in self.mesh.cells:
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    self.dF[i, j] = self.dD[i, j] @ self.B[cell.id]
                    self.dE[i, j] = \
                        0.5 * (self.dF[i, j].transpose() @ self.F[cell.id]
                               + self.F[cell.id].transpose() @ self.dF[i, j])
                    self.dP[i, j] = \
                        self.dF[i, j] @ (2 * self.mu * self.E[cell.id] + self.la * self.E[cell.id].trace() * self.I) \
                        + self.F[cell.id] @ (2 * self.mu * self.dE[i, j] + self.la * self.dE[i, j].trace() * self.I)
                    self.dH[i, j] = -self.W[cell.id] * self.dP[i, j] @ self.B[cell.id].transpose()
            for i in ti.static(range(4)):
                vert = cell.verts[i]
                for dim in ti.static(range(3)):
                    ind = vert.id * 3 + dim
                    for j in ti.static(range(3)):
                        self.K[cell.verts[j].id * 3 + 0, ind] += self.dH[i, dim][0, j]
                        self.K[cell.verts[j].id * 3 + 1, ind] += self.dH[i, dim][1, j]
                        self.K[cell.verts[j].id * 3 + 2, ind] += self.dH[i, dim][2, j]
                    self.K[cell.verts[3].id * 3 + 0, ind] += \
                        -(self.dH[i, dim][0, 0] + self.dH[i, dim][0, 1] + self.dH[i, dim][0, 2])
                    self.K[cell.verts[3].id * 3 + 1, ind] += \
                        -(self.dH[i, dim][1, 0] + self.dH[i, dim][1, 1] + self.dH[i, dim][1, 2])
                    self.K[cell.verts[3].id * 3 + 2, ind] += \
                        -(self.dH[i, dim][2, 0] + self.dH[i, dim][2, 1] + self.dH[i, dim][2, 2])

        for i in (range(self.vert_num)):
            for j in (range(3)):
                for k in (range(self.vert_num * 3)):
                    self.K[i * 3 + j, k] *= self.dt ** 2 / self.m[i] * -1

        for i in (range(self.vert_num)):
            for j in (range(3)):
                self.K[i * 3 + j, i * 3 + j] += 1

        for vert in self.mesh.verts:
            for j in ti.static(range(3)):
                self.x[vert.id * 3 + j] = vert.v[j]
                self.b[vert.id * 3 + j] = vert.v[j] + self.dt / self.m[vert.id] * vert.f[j]

    @ti.kernel
    def jacobi(self, max_iter_num: ti.i32, tol: ti.f32):  # Jacobi iteration
        n = self.vert_num * 3
        iter_i = 0
        res = 0.0
        while iter_i < max_iter_num:

            for i in range(n):  # every row
                r = self.b[i] * 1.0
                for j in range(n):  # every column
                    if i != j:
                        r -= self.K[i, j] * self.x[j]
                self.x_new[i] = r / self.K[i, i]

            for i in range(n):
                self.x[i] = self.x_new[i]

            res = 0.0  # !!!
            for i in range(n):
                r = self.b[i] * 1.0
                for j in range(n):
                    r -= self.K[i, j] * self.x[j]
                res += r * r

            if res < tol:
                break

            iter_i += 1
        print("Jacobi iteration:", iter_i, res)

    @ti.kernel
    def boundary_condition(self):
        bounds = ti.Vector([1.0, 0.06, 1.0])
        for vert in self.mesh.verts:
            for i in ti.static(range(3)):
                if vert.x[i] < -bounds[i]:
                    vert.x[i] = -bounds[i]
                    if vert.v[i] < 0.0:
                        vert.v[i] = 0.0
                if vert.x[i] > bounds[i]:
                    vert.x[i] = bounds[i]
                    if vert.v[i] > 0.0:
                        vert.v[i] = 0.0
