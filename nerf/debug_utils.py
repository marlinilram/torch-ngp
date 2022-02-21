import numpy as np

def gen_init_trescope():
    status = {'initialized': False}
    def init_trescope(open, output_manager, id='main'):
        from trescope import Trescope
        if not status['initialized']:
            Trescope().initialize(open, output_manager, id)
            status['initialized'] = True
    return init_trescope

init_trescope = gen_init_trescope()


def vis_camera(pose, H, W, focal, img_name, output_id):
    # pose -> [right, up, backward] -> r, g, b
    from trescope import Trescope
    from trescope.config import Vector3DConfig, LineSegment3DConfig
    loc = pose[:3, -1][..., None]
    axis_len = 0.2
    right = axis_len*pose[:3, 0:1]+loc
    up = -axis_len*pose[:3, 1:2]+loc
    forward = axis_len*pose[:3, 2:3]+loc
    with Trescope().batchCall(f'{img_name}_img', showMergeOptHint=False):
        pts = np.hstack([right,loc])
        Trescope().selectOutput(output_id).plotLineSegment3D(pts[0], pts[1], pts[2]).withConfig(LineSegment3DConfig().name(f'{img_name}_img').width(4).indices([0], [1]).color(0xffff0000))
        pts = np.hstack([up,loc])
        Trescope().selectOutput(output_id).plotLineSegment3D(pts[0], pts[1], pts[2]).withConfig(LineSegment3DConfig().name(f'{img_name}_img').width(4).indices([0], [1]).color(0xff00ff00))
        pts = np.hstack([forward,loc])
        Trescope().selectOutput(output_id).plotLineSegment3D(pts[0], pts[1], pts[2]).withConfig(LineSegment3DConfig().name(f'{img_name}_img').width(4).indices([0], [1]).color(0xff0000ff))
        # Trescope().selectOutput(output_id).plotVector3D(*right).withConfig(Vector3DConfig().locations(*loc).color(0xffff0000))
        # Trescope().selectOutput(output_id).plotVector3D(*up).withConfig(Vector3DConfig().locations(*loc).color(0xff00ff00))
        # Trescope().selectOutput(output_id).plotVector3D(*forward).withConfig(Vector3DConfig().locations(*loc).color(0xff0000ff))

