import taichi as ti
import meshtaichi_patcher as mp
import numpy as np


@ti.data_oriented
class LoadModel:
    def __init__(self,
                 filename,
                 v_norm=1
                 ):
        # load_mesh
        model_type = filename.split('.')[-1]
        if model_type == "node":
            self.mesh_rawdata = mp.load_mesh_rawdata(filename)
            self.mesh = mp.load_mesh(self.mesh_rawdata, relations=["CV", "VV", "CE", "EV", "FV", "FC", "FE"])
            self.mesh.verts.place({
                'x': ti.math.vec3,
                'v': ti.math.vec3,
                'pf': ti.math.vec3,
                'f': ti.math.vec3,
                'fe': ti.math.vec3,
                'ox': ti.math.vec3,
                'gx': ti.math.vec3
            })
            self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
            self.mesh.verts.ox.from_numpy(self.mesh.get_position_as_numpy())
            self.mesh.verts.gx.from_numpy(self.mesh.get_position_as_numpy())
            self.mesh.verts.v.fill(0.0)
            self.mesh.verts.pf.fill(0.0)
            self.mesh.verts.f.fill(0.0)
            self.mesh.verts.fe.fill(0.0)
            self.indices = ti.field(ti.u32, shape=len(self.mesh.cells) * 4 * 3)
            self.init_tet_indices()

        else:
            self.mesh_rawdata = mp.load_mesh_rawdata(filename)
            self.mesh = mp.load_mesh(self.mesh_rawdata, relations=["FV"])
            self.mesh.verts.place({'x': ti.math.vec3,
                                   'ox': ti.math.vec3,
                                   'rx': ti.math.vec3})
            self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
            self.mesh.verts.ox.from_numpy(self.mesh.get_position_as_numpy())
            self.mesh.verts.rx.from_numpy(self.mesh.get_position_as_numpy())
            self.indices = ti.field(ti.i32, shape=len(self.mesh.faces) * 3)
            self.init_surf_indices()
            x_np = self.mesh.verts.rx.to_numpy()[:, 1]
            line_min = np.where(x_np == x_np.min(0))[0]
            line_max = np.where(x_np == x_np.max(0))[0]
            self.min_len = len(line_min)
            self.max_len = len(line_max)
            self.line0 = ti.ndarray(dtype=ti.i32, shape=self.min_len)
            self.line1 = ti.ndarray(dtype=ti.i32, shape=self.max_len)
            self.line0 = line_min
            self.line1 = line_max

        self.v_norm = v_norm
        self.vert_num = len(self.mesh.verts)
        self.center = ti.Vector.field(3, ti.f32, shape=1)
        self.I = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ti.i32)
        self.norm_volume_equipment()

    @ti.kernel
    def norm_volume_equipment(self):
        if self.v_norm != 0:
            for vert in self.mesh.verts:
                vert.x *= self.v_norm
                vert.ox *= self.v_norm

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
        self.center[0].fill(0)
        for i in self.mesh.verts.x:
            self.center[0] += self.mesh.verts.x[i]
        self.center[0] /= self.vert_num


@ti.data_oriented
class Implicit(LoadModel):
    def __init__(self, filename, v_norm=1, replace_direction=0, replace_alpha=0):
        super().__init__(filename)
        self.de_list = []
        self.fi_list = []
        self.ana = ti.field(ti.f32, shape=6)

        self.v_norm = v_norm
        self.replace_direction = replace_direction
        self.replace_alpha = replace_alpha
        self.rota_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=1)

        self.dt = 0.2
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.e = 7e6  # 杨氏模量
        self.nu = 0.1  # 泊松系数
        self.mu = self.e / (2 * (1 + self.nu))
        self.la = self.e * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.density = 5e5
        self.eta = 2  # 粘滞阻尼系数
        self.E1 = 2.5
        self.E2 = 1.5

        self.cell_num = len(self.mesh.cells)
        self.V = ti.field(dtype=ti.f32, shape=())
        self.Dm = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)  # Dm
        self.W = ti.field(ti.f32, shape=self.cell_num)  # 四面体体积
        self.B = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)  # Dm逆
        self.m = ti.field(ti.f32, shape=self.vert_num)  # 点的质量
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)
        self.F_old = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)
        self.E = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)
        self.give_shape = ti.field(ti.i32, shape=1)
        self.give_shape[0] = -1

        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.vert_num)
        self.r0 = ti.Vector.field(3, dtype=ti.f32, shape=self.vert_num)
        self.p0 = ti.Vector.field(3, dtype=ti.f32, shape=self.vert_num)
        self.dot_ans = ti.field(ti.f32, shape=())
        self.r_2_scalar = ti.field(ti.f32, shape=())

        self.mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=self.vert_num)
        self.norm_volume()
        self.fem_pre_cal()
        if self.replace_alpha:
            self.replace(self.replace_direction, self.replace_alpha, bias=[0, 0, 0])

    @ti.kernel
    def reset(self):
        for vert in self.mesh.verts:
            vert.x = vert.ox
        self.mesh.verts.v.fill(0.0)
        self.mesh.verts.f.fill(0.0)
        self.mesh.verts.fe.fill(0.0)

    @ti.kernel
    def norm_volume(self):
        for cell in self.mesh.cells:
            v = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    v[j, i] = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.V[None] += -(1.0 / 6.0) * v.determinant()
        if self.v_norm != 0:
            for vert in self.mesh.verts:
                vert.x *= 1000 / self.V[None] * self.v_norm
                vert.ox *= 1000 / self.V[None] * self.v_norm
                vert.gx *= 1000 / self.V[None] * self.v_norm

    @ti.kernel
    def replace(self, direction: ti.i32, alpha: ti.f32, bias: ti.math.vec3):
        for vert in self.mesh.verts.ox:
            self.mesh.verts.ox[vert] += bias
            self.mesh.verts.x[vert] += bias
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
        elif direction == 2:
            self.rota_mat[0] = ([[ti.cos(alpha), ti.sin(alpha), 0],
                                 [-ti.sin(alpha), ti.cos(alpha), 0],
                                 [0, 0, 1]
                                 ])
        else:
            self.rota_mat[0] = ([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]
                                 ])
        for vert in self.mesh.verts.ox:
            Tox = self.mesh.verts.ox[vert] @ self.rota_mat[0]
            Tx = self.mesh.verts.x[vert] @ self.rota_mat[0]
            self.mesh.verts.ox[vert] = Tox
            self.mesh.verts.x[vert] = Tx

    @ti.kernel
    def fem_pre_cal(self):
        for cell in self.mesh.cells:
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    self.Dm[cell.id][j, i] = \
                        self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.B[cell.id] = self.Dm[cell.id].inverse()  # Dm逆
            self.W[cell.id] = -(1.0 / 6.0) * self.Dm[cell.id].determinant()  # 四面体体积
            for i in ti.static(range(4)):
                self.m[cell.verts[i].id] += 0.25 * self.density * self.W[cell.id]  # 把体元质量均分到四个顶点

    @ti.kernel
    def fem_get_force_sim_Co_rotated(self):  # 实时力计算
        for vert in self.mesh.verts:
            vert.fe += self.gravity * self.m[vert.id]
            vert.f += vert.fe
            vert.pf = vert.fe  # pf是外力
            # if vert.fe[0] != 0 or vert.fe[1] != 0 or vert.fe[2] != 0:
            #     print(vert.fe)
        for cell in self.mesh.cells:
            Ds = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    Ds[j, i] \
                        = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.F[cell.id] = Ds @ self.B[cell.id]
            U, sig, V = self.ssvd(self.F[cell.id])
            P = 2 * self.mu * (self.F[cell.id] - U @ V.transpose())
            H = -self.W[cell.id] * P @ self.B[cell.id].transpose()
            for i in ti.static(range(3)):
                fi = ti.Vector([H[0, i], H[1, i], H[2, i]])
                self.mesh.verts.f[cell.verts[i].id] += fi
                self.mesh.verts.f[cell.verts[3].id] += -fi

    @ti.kernel
    def fem_get_force_Kelvin(self):  # 实时力计算
        for vert in self.mesh.verts:
            vert.f = self.gravity * self.m[vert.id] + vert.fe
            # if vert.fe[0] != 0 or vert.fe[1] != 0 or vert.fe[2] != 0:
            #     print(vert.fe)
        for cell in self.mesh.cells:
            Ds = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    Ds[j, i] \
                        = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.F[cell.id] = Ds @ self.B[cell.id]
            U, sig, V = self.ssvd(self.F[cell.id])
            sigma = 1 / 2 * (self.F[cell.id].transpose() @ self.F[cell.id] - self.I)
            sigma_old = 1 / 2 * (self.F_old[cell.id].transpose() @ self.F_old[cell.id] - self.I)
            # delta_epsilon = sigma - sigma_old
            # sigma_c = self.eta * delta_epsilon / self.dt
            sigma_c = self.E1 ** 2 * sigma / (self.E1 + self.E2) * (
                    1 - ti.exp((self.E1 - self.E2) * self.dt / self.eta))
            P = 2 * self.mu * (self.F[cell.id] - U @ V.transpose()) + \
                self.la * ((U @ V.transpose()).transpose() @ self.F[cell.id] - self.I).trace() * (U @ V.transpose())
            P += sigma_c
            H = -self.W[cell.id] * P @ self.B[cell.id].transpose()
            for i in ti.static(range(3)):
                fi = ti.Vector([H[0, i], H[1, i], H[2, i]])
                self.mesh.verts.f[cell.verts[i].id] += fi
                self.mesh.verts.f[cell.verts[3].id] += -fi
            self.F_old[cell.id] = self.F[cell.id]

    @ti.kernel
    def fem_get_force_Neo_Hookean(self):  # 实时力计算
        for vert in self.mesh.verts:
            vert.f = self.gravity * self.m[vert.id] + vert.fe
        for cell in self.mesh.cells:
            Ds = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    Ds[j, i] \
                        = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.F[cell.id] = Ds @ self.B[cell.id]
            J = self.F[cell.id].determinant()
            logJ = ti.log(J)
            F_inv_tran = self.F[cell.id].inverse().transpose()
            P = self.mu * (self.F[cell.id] - F_inv_tran) + self.la * logJ * F_inv_tran
            H = -self.W[cell.id] * P @ self.B[cell.id].transpose()
            for i in ti.static(range(3)):
                fi = ti.Vector([H[0, i], H[1, i], H[2, i]])
                self.mesh.verts.f[cell.verts[i].id] += fi
                self.mesh.verts.f[cell.verts[3].id] += -fi

    @ti.kernel
    def fem_get_force_STVK(self):  # 实时力计算
        for vert in self.mesh.verts:
            vert.f = self.gravity * self.m[vert.id] + vert.fe
        for cell in self.mesh.cells:
            Ds = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    Ds[j, i] \
                        = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
            self.F[cell.id] = Ds @ self.B[cell.id]
            self.E[cell.id] = 0.5 * (self.F[cell.id].transpose() @ self.F[cell.id] - self.I)
            P = self.F[cell.id] @ (2 * self.mu * self.E[cell.id] + self.la * self.E[cell.id].trace() * self.I)
            H = -self.W[cell.id] * P @ self.B[cell.id].transpose()
            for i in ti.static(range(3)):
                fi = ti.Vector([H[0, i], H[1, i], H[2, i]])
                self.mesh.verts.f[cell.verts[i].id] += fi
                self.mesh.verts.f[cell.verts[3].id] += -fi

    @ti.kernel
    def fem_get_b(self):
        for vert in self.mesh.verts:
            self.b[vert.id] = self.m[vert.id] * vert.v + self.dt * vert.f

    @ti.kernel
    def mat_mul_sim_Co_rotated(self, ret: ti.template(), vel: ti.template()):
        for vert in self.mesh.verts:
            ret[vert.id] = vel[vert.id] * self.m[vert.id]
        for cell in self.mesh.cells:
            verts = cell.verts
            W_c = self.W[cell.id]
            B_c = self.B[cell.id]
            for u in ti.static(range(4)):
                for d in (range(3)):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if u == 3:
                        for j in ti.static(range(3)):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    dP = 2.0 * self.mu * dF
                    dH = -W_c * dP @ B_c.transpose()
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            tmp = (vel[verts[i].id][j] - vel[verts[3].id][j])
                            ret[verts[u].id][d] += -self.dt ** 2 * dH[j, i] * tmp

    @ti.kernel
    def mat_mul_Kelvin(self, ret: ti.template(), vel: ti.template()):
        for vert in self.mesh.verts:
            ret[vert.id] = vel[vert.id] * self.m[vert.id]
        for cell in self.mesh.cells:
            verts = cell.verts
            W_c = self.W[cell.id]
            B_c = self.B[cell.id]
            for u in ti.static(range(4)):
                for d in (range(3)):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if u == 3:
                        for j in ti.static(range(3)):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    sigma = 1 / 2 * (dF.transpose() @ self.F[cell.id]) + 1 / 2 * (self.F[cell.id].transpose() @ dF)
                    U, sig, V = self.ssvd(dF)
                    dP = 2 * self.mu * (dF - U @ V.transpose()) + \
                         self.la * ((U @ V.transpose()).transpose() @ self.F[cell.id] - self.I).trace()
                    sigma_c = self.E1 ** 2 / (self.E1 + self.E2) * (
                            1 - ti.exp((self.E1 - self.E2) * self.dt / self.eta))
                    sigma *= sigma_c
                    dP += sigma
                    dH = -W_c * dP @ B_c.transpose()
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            tmp = (vel[verts[u].id][d] - vel[verts[3].id][d])
                            ret[verts[u].id][d] += -self.dt ** 2 * dH[j, i] * tmp

    @ti.kernel
    def mat_mul_STVK(self, ret: ti.template(), vel: ti.template()):
        for vert in self.mesh.verts:
            ret[vert.id] = vel[vert.id] * self.m[vert.id]
        for cell in self.mesh.cells:
            verts = cell.verts
            W_c = self.W[cell.id]
            B_c = self.B[cell.id]
            E_c = self.E[cell.id]
            F_c = self.F[cell.id]
            for u in ti.static(range(4)):
                for d in (range(3)):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if u == 3:
                        for j in ti.static(range(3)):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    dE = 0.5 * (dF.transpose() @ F_c + F_c.transpose() @ dF)
                    dP = dF @ (2 * self.mu * E_c + self.la * E_c.trace() * self.I) + F_c @ (
                            2 * self.mu * dE + self.la * dE.trace() * self.I)
                    dH = -W_c * dP @ B_c.transpose()
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            tmp = (vel[verts[i].id][j] - vel[verts[3].id][j])
                            ret[verts[u].id][d] += -self.dt ** 2 * dH[j, i] * tmp

    @ti.kernel
    def mat_mul_sim_Neo_Hookean(self, ret: ti.template(), vel: ti.template()):
        for vert in self.mesh.verts:
            ret[vert.id] = vel[vert.id] * self.m[vert.id]
        for cell in self.mesh.cells:
            verts = cell.verts
            W_c = self.W[cell.id]
            B_c = self.B[cell.id]
            F = self.F[cell.id]
            J = F.determinant()
            logJ = ti.log(J)
            F_inv_tran = F.inverse().transpose()
            for u in ti.static(range(4)):
                for d in (range(3)):
                    dD = ti.Matrix.zero(ti.f32, 3, 3)
                    if u == 3:
                        for j in ti.static(range(3)):
                            dD[d, j] = -1
                    else:
                        dD[d, u] = 1
                    dF = dD @ B_c
                    term = (F.inverse() @ dF).trace() * F_inv_tran
                    FDFF = F_inv_tran @ dF.transpose() @ F_inv_tran
                    dP = self.mu * dF + (self.mu - self.la * logJ) * FDFF + self.la * term
                    dH = -W_c * dP @ B_c.transpose()
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            tmp = (vel[verts[i].id][j] - vel[verts[3].id][j])
                            ret[verts[i].id][j] += -self.dt ** 2 * dH[d, u] * tmp

    def cg(self, n_iter, epsilon):
        # self.mat_mul_STVK(self.mul_ans, self.mesh.verts.v)
        self.mat_mul_sim_Co_rotated(self.mul_ans, self.mesh.verts.v)
        # self.mat_mul_sim_Neo_Hookean(self.mul_ans, self.mesh.verts.v)
        # self.mat_mul_Kelvin(self.mul_ans, self.mesh.verts.v)
        self.add(self.r0, self.b, -1, self.mul_ans)
        self.p0.copy_from(self.r0)
        r_2 = self.dot(self.r0, self.r0)
        r_2_init = r_2
        r_2_new = r_2
        for _ in ti.static(range(n_iter)):
            # self.mat_mul_STVK(self.mul_ans, self.p0)
            # self.mat_mul_sim_Neo_Hookean(self.mul_ans, self.p0)
            self.mat_mul_sim_Co_rotated(self.mul_ans, self.p0)
            # self.mat_mul_Kelvin(self.mul_ans, self.p0)
            dot_ans = self.dot(self.p0, self.mul_ans)
            alpha = r_2_new / (dot_ans + epsilon)
            self.add(self.mesh.verts.v, self.mesh.verts.v, alpha, self.p0)
            self.add(self.r0, self.r0, -alpha, self.mul_ans)
            r_2 = r_2_new
            r_2_new = self.dot(self.r0, self.r0)
            if r_2_new <= r_2_init * epsilon ** 2:
                break
            beta = r_2_new / r_2
            self.add(self.p0, self.r0, beta, self.p0)
        self.add(self.mesh.verts.x, self.mesh.verts.x, self.dt, self.mesh.verts.v)

    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f32, x3: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * x3[i]

    @ti.kernel
    def dot(self, x1: ti.template(), x2: ti.template()) -> ti.f32:
        ans = 0.0
        for i in x1:
            ans += x1[i].dot(x2[i])
        return ans

    @ti.kernel
    def boundary_condition(self):
        bias = [0.01, -0.19, -0.11]
        # for vert in self.mesh.verts:
        #     if self.give_shape[0] == 1 and vert.id == 2610:
        #         vert.x = vert.gx
        #
        #     elif self.give_shape[0] == -1:
        #         vert.gx = vert.x
        bounds = ti.Vector([0.15, 0.05, 0.2])
        if self.give_shape[0] == 1:
            bounds = ti.Vector([0.13, 0.05, 0.2])
        for vert in self.mesh.verts:
            for i in ti.static(range(3)):
                if vert.x[i] < -bounds[i] + bias[i]:
                    vert.x[i] = -bounds[i] + bias[i]
                    if vert.v[i] < 0.0:
                        vert.v[i] = 0.0
                if vert.x[i] > bounds[i] + bias[i]:
                    vert.x[i] = bounds[i] + bias[i]
                    if vert.v[i] > 0.0:
                        vert.v[i] = 0.0

            if vert.x[1] + bounds[1] < 0.002 and (vert.v[0] != 0 or vert.v[2] != 0):
                vert.v[0] *= 0.1
                vert.v[2] *= 0.1

    @ti.kernel
    def decay(self):
        # for vert in self.mesh.verts:
        #     for i in range(3):
        #         if vert.v[i] <= 1e-4 and ti.math.length(vert.f) <= 0.1:
        #             vert.v[i] = 0
        self.mesh.verts.fe.fill(0)

    @ti.kernel
    def Viscoelasticity(self):
        for vert in self.mesh.verts:
            decay = vert.f - vert.pf  # decay是外力+内力-外力，即为纯内力
            vert.f -= 0.9 * decay  # 外力+内力-0.8*内力，即点力=外力+0.2*内力
        # for cell in self.mesh.cells:
        #     F = self.F[cell.id].transpose() @ self.F[cell.id]
        #     for i in range(3):
        #         self.mesh.verts.f[cell.verts[0].id][i] -= 0.1 * F[i, 0]
        #         self.mesh.verts.f[cell.verts[1].id][i] -= 0.1 * F[i, 1]
        #         self.mesh.verts.f[cell.verts[2].id][i] -= 0.1 * F[i, 2]
        #         self.mesh.verts.f[cell.verts[3].id][i] += 0.1 * (F[i, 0] + F[i, 1] + F[i, 2])
        # E1 = 0.1
        # E2 = 0.1
        # N = 0.9
        # for vert in self.mesh.verts:
        #     f = vert.f - vert.pf
        #     decay = E1*f + E1**2*f/(E1+E2)*(1-ti.exp(-(E1+E2)*self.dt/N))
        #     print(decay)
        #     vert.f -= decay

    def call_F(self):
        # de = self.F.to_numpy()
        # fi = self.mesh.verts.f.to_numpy()[0]
        # de = np.sum(de) / len(de)
        # fi = np.sum(fi) / len(fi)
        # self.de_list.append(de)
        # self.fi_list.append(fi)

        # self.de_list.append((a[3], a[4], a[5]))
        # self.fi_list.append((a[0], a[1], a[2]))

        self.call_F_sub()
        ana = self.ana.to_numpy()
        self.de_list.append(ana)

    @ti.kernel
    def call_F_sub(self):
        for cell in self.mesh.cells:
            if cell.id == 2:
                F = self.F[cell.id].transpose() @ self.F[cell.id]
                self.ana[0] = F[0, 0] + F[1, 0] + F[2, 0]  # fx
                self.ana[1] = F[0, 1] + F[1, 1] + F[2, 1]  # fy
                self.ana[2] = F[0, 2] + F[1, 2] + F[2, 2]  # fz
                self.ana[3] = -(self.mesh.verts.f[cell.verts[0].id].x + self.mesh.verts.f[cell.verts[1].id].x + self.mesh.verts.f[
                    cell.verts[2].id].x) # cx
                self.ana[4] = -(self.mesh.verts.f[cell.verts[0].id].y + self.mesh.verts.f[cell.verts[1].id].y + self.mesh.verts.f[
                    cell.verts[2].id].y)  # cy
                self.ana[5] = -(self.mesh.verts.f[cell.verts[0].id].z + self.mesh.verts.f[cell.verts[1].id].z + self.mesh.verts.f[
                    cell.verts[2].id].z)  # cz

    def substep(self, step):
        for i in range(step):
            self.fem_get_force_sim_Co_rotated()
            self.Viscoelasticity()
            # self.fem_get_force_Kelvin()
            # self.fem_get_force_STVK()
            # self.fem_get_force_Neo_Hookean()
            self.fem_get_b()
            self.cg(5, 1e-5)
            self.boundary_condition()
            self.decay()
            self.call_F()

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

# @ti.data_oriented
# class Explicit(LoadModel):  # This class only for tetrahedron
#     def __init__(self, filename, v_norm=1):
#         super().__init__(filename)
#         self.v_norm = v_norm
#
#         self.dt = 7e-4
#         self.gravity = ti.Vector([0.0, -9.8, 0.0])
#         self.e = 2e6  # 杨氏模量
#         self.nu = 0.1  # 泊松系数
#         self.mu = self.e / (2 * (1 + self.nu))
#         self.la = self.e * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
#         self.density = 1e5
#
#         self.cell_num = len(self.mesh.cells)
#         self.V = ti.field(dtype=ti.f32, shape=())
#         self.Dm = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)  # Dm
#         self.W = ti.field(ti.f32, shape=self.cell_num)  # 四面体体积
#         self.B = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)  # Dm逆
#         self.m = ti.field(ti.f32, shape=self.vert_num)  # 点的质量
#         self.F = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)
#         self.E = ti.Matrix.field(3, 3, ti.f32, shape=self.cell_num)
#
#         self.norm_volume()
#         self.fem_pre_cal()
#
#     @ti.kernel
#     def reset(self):
#         for vert in self.mesh.verts:
#             vert.x = vert.ox
#         self.mesh.verts.v.fill(0.0)
#         self.mesh.verts.f.fill(0.0)
#
#     @ti.kernel
#     def norm_volume(self):
#         for cell in self.mesh.cells:
#             v = ti.Matrix.zero(ti.f32, 3, 3)
#             for i in ti.static(range(3)):
#                 for j in ti.static(range(3)):
#                     v[j, i] = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
#             self.V[None] += -(1.0 / 6.0) * v.determinant()
#         if self.v_norm == 1:
#             for vert in self.mesh.verts:
#                 vert.x *= 1000 / self.V[None]
#                 vert.ox *= 1000 / self.V[None]
#
#     @ti.kernel
#     def fem_pre_cal(self):  # fem参数预计算
#         self.V[None] = 0
#         for cell in self.mesh.cells:
#             for i in ti.static(range(3)):
#                 for j in ti.static(range(3)):
#                     self.Dm[cell.id][j, i] \
#                         = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
#             self.B[cell.id] = self.Dm[cell.id].inverse()
#             self.W[cell.id] = -(1.0 / 6.0) * self.Dm[cell.id].determinant()
#             self.V[None] += self.W[cell.id]
#             for i in ti.static(range(4)):
#                 self.m[cell.verts[i].id] += 0.25 * self.density * self.W[cell.id]  # 把体元质量均分到四个顶点
#
#     @ti.kernel
#     def fem_get_force(self):  # 实时力计算
#         for vert in self.mesh.verts:
#             vert.f = self.gravity * self.m[vert.id]
#         for cell in self.mesh.cells:
#             Ds = ti.Matrix.zero(ti.f32, 3, 3)
#             for i in ti.static(range(3)):
#                 for j in ti.static(range(3)):
#                     Ds[j, i] \
#                         = self.mesh.verts.x[cell.verts[i].id][j] - self.mesh.verts.x[cell.verts[3].id][j]
#             self.F[cell.id] = Ds @ self.B[cell.id]
#             self.E[cell.id] = 0.5 * (self.F[cell.id].transpose() @ self.F[cell.id] - self.I)
#             U, sig, V = self.ssvd(self.F[cell.id])
#             P = 2 * self.mu * (self.F[cell.id] - U @ V.transpose())
#             # P = self.F[cell.id] @ (2 * self.mu * self.E[cell.id] + self.la * self.E[cell.id].trace() * self.I)
#             H = -self.W[cell.id] * P @ self.B[cell.id].transpose()
#             for i in ti.static(range(3)):
#                 fi = ti.Vector([H[0, i], H[1, i], H[2, i]])
#                 self.mesh.verts.f[cell.verts[i].id] += fi
#                 self.mesh.verts.f[cell.verts[3].id] += -fi
#
#     @ti.func
#     def ssvd(self, fai):
#         U, sig, V = ti.svd(fai)
#         if U.determinant() < 0:
#             for i in ti.static(range(3)):
#                 U[i, 2] *= -1
#             sig[2, 2] = -sig[2, 2]
#         if V.determinant() < 0:
#             for i in ti.static(range(3)):
#                 V[i, 2] *= -1
#             sig[2, 2] = -sig[2, 2]
#         return U, sig, V
#
#     @ti.kernel
#     def explicit_time_integral(self):
#         for vert in self.mesh.verts:
#             vert.v += self.dt * vert.f / self.m[vert.id] * 0.0000125
#             vert.x += vert.v * self.dt
#
#     @ti.kernel
#     def boundary_condition(self):
#         bounds = ti.Vector([1.0, 0.1, 1.0])
#         for vert in self.mesh.verts:
#             for i in ti.static(range(3)):
#                 if vert.x[i] < -bounds[i]:
#                     vert.x[i] = -bounds[i]
#                     if vert.v[i] < 0.0:
#                         vert.v[i] = 0.0
#                 if vert.x[i] > bounds[i]:
#                     vert.x[i] = bounds[i]
#                     if vert.v[i] > 0.0:
#                         vert.v[i] = 0.0
#
#     def substep(self, step):
#         for i in range(step):
#             self.fem_get_force()
#             self.explicit_time_integral()
#             self.boundary_condition()
