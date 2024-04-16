import numpy as np
import taichi as ti
import fem_class as fem
import collide_detection as cd
import haptics as ha
import pyhaptics3 as ph
import DCD
def ggui_init():
    window = ti.ui.Window("FEM", (768, 768), vsync=False)
    canvas = window.get_canvas()
    canvas.set_background_color(color=(1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(-0.06839289, 0.27995652, -0.11273427)
    camera.lookat(-0.06953721, -0.71992012, -0.12839985)
    camera.fov(60)

    return window, canvas, scene, camera

def ggui_run(window, canvas, scene, camera):
    camera.track_user_inputs(window, 0.001, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(camera.curr_position, (0.7, 0.7, 0.7))

    scene.mesh(model.mesh.verts.x, model.indices, color=(1.0, 0.3, 0.3), show_wireframe=False)
    scene.mesh(equipment_model.mesh.verts.x, equipment_model.indices, color=(0.7, 0.7, 0.7))
    scene.mesh(body_model.mesh.verts.x, body_model.indices, color=(0.9, 0.85, 0.8), show_wireframe=True)
    scene.particles(cd.line, 0.005, (0.9, 0.9, 0.9))
    # scene.mesh(equipment_model1.mesh.verts.x, equipment_model.indices, color=(0.7, 0.7, 0.7))
    # scene.lines(cd.line, width=1, color=(0, 0, 0))
    # scene.particles(bvt_equipment.face_barycenter, 0.0008, (0.9, 0.9, 0.9))
    # scene.lines(bvt_obj.min_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_equipment.min_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_obj.layer1_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_equipment.layer1_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_obj.layer0_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_equipment.layer0_box_for_draw, width=1, color=(0, 0, 0))
    # scene.particles(cd.corss_pot, 0.008, (0.9, 0.9, 0.9))
    # scene.particles(cd.line, 0.008, (0.9, 0.9, 0.9))
    # force_vis[0] = cd.line[0]
    # force_vis[1] = cd.line[0] + cd.force[None]
    # if cd.force[None].x + cd.force[None].y + cd.force[None].z != 0:
    #     scene.lines(force_vis, width=1.2, color=(0, 0.8, 0.2))

    canvas.scene(scene)
    window.show()


if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    ph.init(1)
    window, canvas, scene, camera = ggui_init()
    force_vis = ti.Vector.field(3, dtype=ti.f32, shape=2)

    obj = "model/liver_houdini/liver.node"
    # obj1 = "model/liver/liver0.node"
    equipment = "model/equipment/zhen.obj"
    body = "model/body.obj"
    model = fem.Implicit(obj, v_norm=5e-3, replace_direction=0, replace_alpha=ti.math.pi/2)
    model.replace(1, ti.math.pi/4, bias = [0.01, -0.19, -0.11])
    # model = fem.Implicit(obj2, v_norm=1)
    equipment_model = fem.LoadModel(equipment, v_norm=1e-3)
    body_model = fem.LoadModel(body, v_norm=1e-3)
    # equipment_model1 = fem.LoadModel(equipment, v_norm=1e-3)
    hap = ha.haptices(equipment_model.mesh.verts, 0, 1 / 2 * ti.math.pi)
    # hap1 = ha.haptices1(equipment_model1.mesh.verts, 0, 1 / 2 * ti.math.pi)
    cd = DCD.dcd(model, equipment_model)

    gui_run = True
    while window.running:
        if window.is_pressed('r'):
            model.reset()
        if window.is_pressed('v'):
            print(camera.curr_position, camera.curr_lookat)
        # if window.is_pressed('t'):
        #     print("x:", model.mesh.verts.x)
        #     print("v:", model.mesh.verts.v)
        #     print("f:", model.mesh.verts.f)
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'p':
                gui_run = not gui_run
            if window.event.key == ti.ui.ESCAPE:
                break
            if window.event.key == 'f':
                cd.give_force[0] *= -1
            if window.event.key == 'g':
                model.give_shape[0] *= -1

        if gui_run:
            cd.run()
            # hap.run(cd.force[0].x, cd.force[0].y, cd.force[0].z)
            hap.run(0, 0, 0)
            # hap1.run(cd.force[0].x, cd.force[0].y, cd.force[0].z)
            model.substep(1)
        ggui_run(window, canvas, scene, camera)

# np.savetxt("analysis/force.txt", cd.force_list)
# np.savetxt("analysis/d.txt", cd.d_list)
# np.savetxt("analysis/de.txt", model.de_list)
# np.savetxt("analysis/fi.txt", model.fi_list)