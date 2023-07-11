import time

import taichi as ti
import fem_class as fem
import collide_detection as cd
import haptics as ha
import pyhaptics as ph

def ggui_init():
    window = ti.ui.Window("FEM", (768, 768), vsync=False)
    canvas = window.get_canvas()
    canvas.set_background_color(color=(1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.up(0, 1, 0)
    camera.position(0.0, 0.0, 0.35)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(75)
    return window, canvas, scene, camera

def ggui_run(window, canvas, scene, camera):
    camera.track_user_inputs(window, 0.001, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(camera.curr_position, (0.7, 0.7, 0.7))

    scene.mesh(model.mesh.verts.x, model.indices, color=(1.0, 0.3, 0.3))
    scene.mesh(equipment_model.mesh.verts.x, equipment_model.indices, color=(0.7, 0.7, 0.7))

    scene.lines(detector.line, width=1, color=(0, 0, 0))
    # scene.particles(bvt_equipment.face_barycenter, 0.0008, (0.9, 0.9, 0.9))
    # scene.lines(bvt_obj.min_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_equipment.min_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_obj.layer1_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_equipment.layer1_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_obj.layer0_box_for_draw, width=1, color=(0, 0, 0))
    # scene.lines(bvt_equipment.layer0_box_for_draw, width=1, color=(0, 0, 0))
    canvas.scene(scene)
    window.show()


if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    ph.init()
    window, canvas, scene, camera = ggui_init()

    obj = "model/liver/liver0.node"
    equipment = "model/equipment/Scissors.stl"

    model = fem.Implicit(obj, v_norm=1)
    equipment_model = fem.LoadModel(equipment)

    bvt_obj = cd.aabb_obj(model)
    bvt_equipment = cd.aabb_obj(equipment_model)

    detector = cd.deceteor(bvt_obj, bvt_equipment)

    hap = ha.haptices(equipment_model.mesh.verts)

    gui_run = True
    while window.running:
        if window.is_pressed('r'):
            model.reset()
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'p':
                gui_run = not gui_run
            if window.event.key == ti.ui.ESCAPE:
                break

        if gui_run:
            model.substep(1)

            bvt_obj.run()
            bvt_equipment.run()
            detector.run()

            hap.run()

        ggui_run(window, canvas, scene, camera)
