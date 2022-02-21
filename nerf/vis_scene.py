import os.path as osp
import numpy as np
from trimesh.exchange.ply import load_ply
from trimesh.exchange.obj import load_obj
from trescope import Trescope
from trescope.config import Mesh3DConfig
from .opt import get_opt
from .provider import NeRFDataset


def main():
    opt = get_opt()
    dataset = NeRFDataset(opt.path, 'train', radius=opt.radius)
    mesh = load_ply(open(osp.join(opt.workspace, 'meshes/ngp_200.ply'), 'rb'), fix_texture=False)
    verts, faces = mesh['vertices'], mesh['faces']
    (Trescope().selectOutput(0)
     .plotMesh3D(verts[:, 0], verts[:, 1], verts[:, 2])
     .withConfig(Mesh3DConfig()
                 .indices(faces[:, 0], faces[:, 1], faces[:, 2])
                 .flatShading(False)
                 .name('ngp_200')))
    pass


def generate_camera_path_from_blender():
    camera_path = load_obj(open('data/arkit_huiyishi/ar_demo/camera_path.obj'))
    camera_target = load_obj(open('data/arkit_huiyishi/ar_demo/camera_target.obj'))
    camera_target_floor = load_obj(open('data/arkit_huiyishi/ar_demo/camera_target_floor.obj'))

    camera_path = camera_path['vertices']
    target = camera_target['vertices'].mean(axis=0)
    target_floor = camera_target_floor['vertices'].mean(axis=0)

    # forward
    forward = target - camera_path
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)

    # right
    floor_up = target-target_floor
    floor_up = floor_up / np.linalg.norm(floor_up)
    right = np.cross(forward, floor_up)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)

    # up
    up = np.cross(right, forward)
    

    pass


if __name__ == '__main__':
    # main()
    generate_camera_path_from_blender()
