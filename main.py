import taichi as ti
import fem_class as fem
import collide_detection as cd
import haptics as ha
import pyhaptics as ph

if __name__ == '__main__':
    ti.init(arch=ti.cuda)
    ph.init()

    obj = "model/liver/liver0.node"
    equipment = "model/equipment/Scissors.stl"

    model = fem.Implicit(obj, v_norm=1)
    equipment_model = fem.LoadModel(equipment)

    bvt_obj = cd.aabb_obj(model.mesh.verts)
    bvt_equipment = cd.aabb_obj(equipment_model.mesh.verts)

    hap = ha.haptices(equipment_model.mesh.verts)

    window = ti.ui.Window("FEM", (768, 768), vsync=False)
    canvas = window.get_canvas()
    canvas.set_background_color(color=(1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.up(0, 1, 0)
    camera.position(0.0, 0.0, -0.2)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(75)

    while window.running:
        if window.is_pressed('r'):
            model.reset()
        if window.is_pressed(ti.ui.ESCAPE):
            break

        model.substep(1)

        bvt_obj.run()
        bvt_equipment.run()

        hap.run()

        camera.track_user_inputs(window, 0.008, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(camera.curr_position, (0.7, 0.7, 0.7))
        scene.mesh(model.mesh.verts.x, model.indices, color=(1.0, 0.3, 0.3))
        scene.mesh(equipment_model.mesh.verts.x, equipment_model.indices, color=(0.7, 0.7, 0.7))
        scene.particles(bvt_obj.aabb_root, 0.002, (0.2, 0.2, 0.2))
        scene.particles(bvt_obj.aabb_tree, 0.002, (0.9, 0.9, 0.9), index_offset=2, index_count=bvt_obj.tree_size - 2)
        scene.particles(bvt_equipment.aabb_root, 0.002, (0.2, 0.2, 0.2))
        scene.particles(bvt_equipment.aabb_tree, 0.002, (0.9, 0.9, 0.9),
                        index_offset=2, index_count=bvt_equipment.tree_size - 2)
        canvas.scene(scene)
        window.show()

    # ti.profiler.print_kernel_profiler_info()
