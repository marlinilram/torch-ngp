from os import makedirs
import os.path as osp
import numpy as np
import json
from tqdm import tqdm
from trimesh import Trimesh
from trimesh.exchange.ply import load_ply, export_ply
from trimesh.exchange.obj import load_obj, export_obj
from trescope import Trescope
from trescope.config import Mesh3DConfig
from .opt import get_opt
from .provider import NeRFDataset, NeRFRayDataset
from .pose_refine import PoseRefine
from .debug_utils import depth_to_pts


def main():
    opt = get_opt()
    dataset = NeRFDataset(opt.path, 'all', downscale=2, radius=opt.radius)
    mesh = load_ply(open(osp.join(opt.workspace, 'meshes/ngp_200.ply'), 'rb'), fix_texture=False)
    verts, faces = mesh['vertices'], mesh['faces']
    (Trescope().selectOutput(0)
     .plotMesh3D(verts[:, 0], verts[:, 1], verts[:, 2])
     .withConfig(Mesh3DConfig()
                 .indices(faces[:, 0], faces[:, 1], faces[:, 2])
                 .flatShading(False)
                 .name('ngp_200')))
    pass


def est_global_scale():
    opt = get_opt()
    dataset = NeRFDataset(opt.path, 'all', downscale=3.75, radius=opt.radius)
    # dataset = NeRFRayDataset(opt.path, 'all', downscale=3.75, radius=opt.radius).img_dataset
    # np.savez(open(osp.join(dataset.root_path, 'sdf_data.npz'),'wb'),poses=dataset.poses,depths=[d for d, _ in dataset.ori_depths],confidences=[c for _, c in dataset.ori_depths], intrinsic=dataset.intrinsic)
    all_pts = []
    for data in tqdm(dataset):
        pts = depth_to_pts(data['image'], data['depth'], dataset.H, dataset.W, dataset.intrinsic, data['pose'], False)
        pts = pts[data['mask']]
        all_pts.append(pts)
        pass
    all_pts = np.vstack(all_pts)[:, :3]
    bmin = all_pts.min(axis=0)
    bmax = all_pts.max(axis=0)
    diag = np.sqrt(((bmax-bmin)*(bmax-bmin)).sum())
    center = 0.5*(bmin+bmax)
    print(f'diag: {diag}', f'center: {center}')
    # default bound 2.0, scene diag 4
    scale = 4 / diag
    offset = -scale*center
    print(f'scale: {scale}', f'offset: {offset}')


def main_save_pts():
    opt = get_opt()
    dataset = NeRFDataset(opt.path, 'all', downscale=3.75, radius=opt.radius)
    # dataset = NeRFRayDataset(opt.path, 'all', downscale=3.75, radius=opt.radius).img_dataset
    # np.savez(open(osp.join(dataset.root_path, 'sdf_data.npz'),'wb'),poses=dataset.poses,depths=[d for d, _ in dataset.ori_depths],confidences=[c for _, c in dataset.ori_depths], intrinsic=dataset.intrinsic)
    pt_path = 'pointcloud_aligned'
    if not osp.exists(osp.join(dataset.root_path, pt_path)):
        makedirs(osp.join(dataset.root_path, pt_path))
    all_pts = []
    for data in tqdm(dataset):
        name = data['name']
        pts = depth_to_pts(data['image'], data['depth'], dataset.H, dataset.W, dataset.intrinsic, data['pose'], False)
        pts = pts[data['mask']]
        all_pts.append(pts)
        mesh = Trimesh(vertices=pts[:, :3], vertex_colors=pts[:, 3:], process=False)
        open(osp.join(dataset.root_path, f'{pt_path}/{name}.ply'), 'wb').write(export_ply(mesh))
        pass
    all_pts = np.vstack(all_pts)[:, :3]
    bmin = all_pts.min(axis=0)
    bmax = all_pts.max(axis=0)
    diag = np.sqrt(((bmax-bmin)*(bmax-bmin)).sum())
    center = 0.5*(bmin+bmax)
    print(f'diag: {diag}', f'center: {center}')
    # default bound 2.0, scene diag 4
    scale = 4 / diag
    offset = -scale*center
    print(f'scale: {scale}', f'offset: {offset}')


def generate_camera_path_from_blender():
    camera_path = load_obj(open('data/arkit_huiyishi/ar/camera_path.obj'))
    camera_target = load_obj(open('data/arkit_huiyishi/ar/camera_target.obj'))
    camera_target_floor = load_obj(open('data/arkit_huiyishi/ar/camera_target_floor.obj'))

    camera_path = camera_path['vertices']
    target = camera_target['vertices'].mean(axis=0)
    target_floor = camera_target_floor['vertices'].mean(axis=0)

    mean_forward = target-camera_path.mean(axis=0)
    mean_forward = mean_forward / np.linalg.norm(mean_forward)
    camera_path += 0.1*mean_forward

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

    # assemble transforms
    poses = np.stack([right, -up, forward, camera_path])
    poses = poses.transpose((1, 2, 0))

    # dump to json
    json.dump(
        {'render_frames': [{
            "file_path": f"{pi}.jpg",
            'transform_matrix': pose.tolist()}
            for pi, pose in enumerate(poses)]},
        open('data/arkit_huiyishi/ar/render_frames.json', 'w'),
        indent=2)
    pass


def generate_camera_path():
    from ff3d_planner.common import Matrix4
    camera_target_floor = np.array([0.0, -0.64, -0.4])
    # depth scale 0.5
    # 0.6m -> y+=0.3
    camera_target = camera_target_floor + np.array([0, 0.64, 0])

    # camera move around circle center at camera_target with same height
    # from [0, 0, 1] axis [0, 1, 0], +100 degree
    # camera height 1.4m -> y+=0.7
    # radius 1m -> 0.5
    step = 0.5
    radius = 1.1
    cam_height = 0.7
    cam_base_pos = np.array([0, 0, 1]) * radius
    camera_path = []
    for degree in np.arange(-15, 30, step):# range(-15, 30, step):
        rot = Matrix4().make_rotation_y(degree).mat
        cam_pos = np.matmul(rot[:3, :3], cam_base_pos)
        cam_pos += camera_target_floor + np.array([0, cam_height, 0])
        camera_path.append(cam_pos)

    camera_path = np.stack(camera_path)
    forward = camera_target - camera_path
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)

    # right
    floor_up = camera_target - camera_target_floor
    floor_up = floor_up / np.linalg.norm(floor_up)
    right = np.cross(forward, floor_up)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)

    # up
    up = np.cross(right, forward)

    # assemble transforms
    poses = np.stack([right, -up, forward, camera_path])
    poses = poses.transpose((1, 2, 0))
    # dump to json
    workspace = 'data/arkit_huiyishi_4'
    if not osp.exists(osp.join(workspace, 'ar')):
        makedirs(osp.join(workspace, 'ar'))
    json.dump(
        {'render_frames': [{
            "file_path": f"{pi}.jpg",
            'transform_matrix': pose.tolist()}
            for pi, pose in enumerate(poses)]},
        open(osp.join(workspace, 'ar/render_frames.json'), 'w'),
        indent=2)


def rescale_fusion_mesh():
    opt = get_opt()
    root_path = osp.dirname(opt.path)
    with open(opt.path, 'r') as f:
        transform = json.load(f)
        scale = transform['scale']
        offset = transform['offset']
    obj = load_obj(open(osp.join(root_path, 'mesh_002000.obj')))
    verts = obj['vertices']
    verts = verts*scale + offset
    mesh = Trimesh(vertices=verts, faces=obj['faces'], process=False)
    open(osp.join(root_path, 'fusion.obj'), 'w').write(export_obj(mesh))


def main_apply_pose_refine_to_transforms():
    import torch
    opt = get_opt()
    transforms = json.load(open(opt.path))
    pose_arr = np.load('data/arkit_huiyishi_2/pose_array_002000_rescale.npy')
    pose_arr = torch.tensor(pose_arr)
    assert pose_arr.shape[0] == len(transforms['frames'])
    pose_refine = PoseRefine(pose_arr.shape[0], 'euler')
    pose_refine.vars.data = pose_arr
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    for fi, frame in enumerate(transforms['frames']):
        R = pose_refine.get_rotation_matrices([fi]).detach().squeeze().numpy()
        t = pose_refine.get_translations([fi]).detach().squeeze().numpy()
        m = np.concatenate([np.concatenate([R, t[..., None]], 1), bottom], 0)
        frame['pose_refine'] = m.tolist()
    json.dump(transforms, open(opt.path, 'w'), indent=2)
    pass


def align_traj():
    from evo.tools import file_interface
    from io import StringIO
    from functools import reduce

    def pose_to_traj(poses):
        traj = [pose[:3, :].flatten().tolist() for pose in poses]
        traj = [[str(digit) for digit in digits] for digits in traj]
        string_file = StringIO('\n'.join([' '.join(t) for t in traj]))
        traj = file_interface.read_kitti_poses_file(string_file)
        return traj
    est_file = 'data/arkit_huiyishi_4/transforms.json'
    ref_file = 'data/arkit_huiyishi_4/transforms_arkit.json'
    dataset_est = NeRFDataset(est_file, 'all', downscale=3.75)
    traj_est = pose_to_traj(dataset_est.poses)
    dataset_ref = NeRFDataset(ref_file, 'all', downscale=3.75)
    traj_ref = pose_to_traj(dataset_ref.poses)
    param = traj_est.align(traj_ref, correct_scale=True)
    transforms_est = json.load(open(est_file))
    transforms_ref = json.load(open(ref_file))
    # write aligned poses to file, so post transform is unnecessary
    transforms_est['depth_scale'] = transforms_ref['depth_scale']
    transforms_est['scale'] = 1.0
    transforms_est['offset'] = [0., 0., 0.]
    transforms_est['coordinate'] = 'ngp'
    for frame, aligned in zip(transforms_est['frames'], traj_est.poses_se3):
        frame['transform_matrix'] = aligned.tolist()
    json.dump(transforms_est, open('data/arkit_huiyishi_4/transforms_aligned.json', 'w'), indent=2)
    pass


if __name__ == '__main__':
    # main()
    # generate_camera_path_from_blender()
    # rescale_fusion_mesh()
    # main_save_pts()
    # main_apply_pose_refine_to_transforms()
    # align_traj()
    generate_camera_path()
    # est_global_scale()
