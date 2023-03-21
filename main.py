import taichi as ti
import fem_class as fem

if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    obj = "model/liver/liver0.node"
    # obj = "model/equipment/Scissors.stl"
    mesh = fem.Implicit(obj, v_norm=1)

    window = ti.ui.Window("FEM", (768, 768), vsync=False)
    canvas = window.get_canvas()
    canvas.set_background_color(color=(1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.up(0, 1, 0)
    camera.position(-0.2, 0.0, 0.2)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(75)

    while window.running:
        if window.is_pressed('r'):
            mesh.reset()
        mesh.substep(1)

        camera.track_user_inputs(window, 0.0008, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(camera.curr_position, (0.7, 0.7, 0.7))
        scene.mesh(mesh.mesh.verts.x, mesh.indices, color=(1.0, 0.3, 0.3))
        canvas.scene(scene)
        window.show()
