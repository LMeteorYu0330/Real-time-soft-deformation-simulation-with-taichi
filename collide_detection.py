import taichi as ti
import numpy as np
import fem_class as fem


@ti.data_oriented
class aabb_obj:
    def __init__(self, model):
        self.min_x = ti.Vector.field(3, ti.f32, shape=())
        self.model = model

        self.get_min()

    @ti.kernel
    def get_min(self):
        self.min_x[None][0] = 0


if __name__ == '__main__':
    ti.init(arch=ti.cuda)
    aabb_obj(fem.Implicit("model/liver/liver0.node", v_norm=1))
