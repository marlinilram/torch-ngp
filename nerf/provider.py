import os
import time
import glob
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.spatial.transform import Slerp, Rotation

# NeRF dataset
import json

from tqdm import tqdm
from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0], coordinate='nerf'):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    if 'nerf' == coordinate:
        new_pose = np.array([
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
            [0, 0, 0, 1],
        ])
    elif 'ngp' == coordinate:
        new_pose = np.array([
            [pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3] * scale + offset[0]],
            [pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3] * scale + offset[1]],
            [pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ])
    else:
        raise NotImplementedError
    return new_pose  # [right, down, forward, pos]



def load_depth(datadir, fname):
    if os.path.exists(os.path.join(datadir, f'depth/{fname}.npy')):
        depth = np.load(os.path.join(datadir, f'depth/{fname}.npy'))
        # depth *= 10. # meter to decimeter
    else:
        with open(os.path.join(datadir, 'depth', f'{fname}.raw'), 'rb') as reader:
            depth = [np.array(reader.read(4)).view(dtype=np.float32)
                    for _ in range(256*192)]
            depth = np.array(depth).reshape((192, 256))
    return depth


def load_depth_confidence(datadir, fname):
    if os.path.exists(os.path.join(datadir, f'confidence/{fname}.npy')):
        mask = np.load(os.path.join(datadir, f'confidence/{fname}.npy'))
        confidence = np.zeros_like(mask, dtype=np.uint8)
        confidence[mask] = 2
    else:
        with open(os.path.join(datadir, 'confidence', f'{fname}.raw'), 'rb') as rd:
            confidence = [np.array(rd.read(1)).view(dtype=np.uint8)
                        for _ in range(256*192)]
            confidence = np.array(confidence).reshape((192, 256))
    return confidence


class NeRFDataset(Dataset):
    def __init__(self, path, type='train', downscale=1, radius=1, n_test=10):
        super().__init__()
        # path: the json file path.

        self.root_path = os.path.dirname(path)
        self.type = type
        self.downscale = downscale
        self.radius = radius # TODO: generate custom views for test?

        # load nerf-compatible format data.
        with open(path, 'r') as f:
            transform = json.load(f)

        coordinate = transform.get('coordinate', 'nerf')

        # load image size
        self.H = int(transform['h'] // downscale)
        self.W = int(transform['w'] // downscale)

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = transform['fl_x'] / downscale
        self.intrinsic[1, 1] = transform['fl_y'] / downscale
        self.intrinsic[0, 2] = transform['cx'] / downscale
        self.intrinsic[1, 2] = transform['cy'] / downscale

        self.scale = transform['scale']
        self.offset = transform['offset']
        self.depth_scale = transform['depth_scale']
        if transform.get('fusion') is not None:
            from trimesh import Trimesh
            from trimesh.exchange.obj import load_obj
            from trimesh.ray.ray_pyembree import RayMeshIntersector

            mesh_path = os.path.join(self.root_path, transform.get('fusion'))
            mesh = load_obj(open(mesh_path))
            mesh = Trimesh(**mesh, process=False)
            self.fusion_intersector = RayMeshIntersector(mesh, scale_to_box=False)
        else:
            self.fusion_intersector = None

        if type == 'ar':
            self.poses = []
            self.image_names = []
            frames = json.load(open(os.path.join(self.root_path, 'ar/render_frames.json')))['render_frames']
            for f in frames:
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                self.poses.append(pose)
                self.image_names.append(f['file_path'])
        else:
            with open(os.path.join(self.root_path, 'train.txt')) as rd:
                lines = rd.readlines()
                names = [l[:-5] for l in lines]
                frame_train_idx = {n: ni for ni, n in enumerate(names)}

            # same order with poses_bounds.npy
            frames = transform["frames"]
            for frame in frames:
                name = os.path.splitext(os.path.basename(frame['file_path']))
                frame['file_idx'] = frame_train_idx[name[0]]
                frame['file_name'] = name[0]
            frames = sorted(frames, key=lambda d: d['file_idx'])

            if type == 'test':
                # choose two random poses, and interpolate between.
                f0, f1 = np.random.choice(frames, 2, replace=False)
                pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), self.scale, self.offset, coordinate) # [4, 4]
                pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), self.scale, self.offset, coordinate) # [4, 4]
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)

                self.poses = []
                for i in range(n_test + 1):
                    ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)

            else:
                if type == 'train':
                    frames = frames[0:]
                elif type == 'valid':
                    frames = frames[:1]

                self.poses = []
                self.images = []
                self.depths = []
                self.masks = []
                self.names = []
                self.ori_depths = []
                for f in frames:
                    f_path = os.path.join(self.root_path, f['file_path'])

                    # there are non-exist paths in fox...
                    if not os.path.exists(f_path):
                        continue

                    self.names.append(f['file_name'])
                    pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
                    pose = nerf_matrix_to_ngp(pose, self.scale, self.offset, coordinate)
                    if 'pose_refine' in f:
                        pose = np.matmul(np.array(f['pose_refine']), pose)

                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    # add support for the alpha channel as a mask.
                    if image.shape[-1] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    image = image.astype(np.float32) / 255 # [H, W, 3/4]

                    self.poses.append(pose)
                    self.images.append(image)

                    img_id = os.path.splitext(os.path.basename(f['file_path']))[0]
                    depth = load_depth(self.root_path, img_id)
                    mask = load_depth_confidence(self.root_path, img_id)
                    self.ori_depths.append((depth*self.depth_scale, mask))
                    depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    depth *= self.depth_scale
                    mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    mask = mask >= 2
                    self.depths.append(depth)
                    self.masks.append(mask)

            self.poses = np.stack(self.poses, axis=0).astype(np.float32)

        if False:
            from .debug_utils import vis_camera, init_trescope
            from trescope import Trescope, Layout
            from trescope.toolbox import simpleDisplayOutputs
            from trescope.config import Scatter3DConfig, OrthographicCamera
            init_trescope(True, simpleDisplayOutputs(1, 1))
            Trescope().selectOutput(0).updateLayout(Layout().legendOrientation('vertical').camera(OrthographicCamera().eye(0, 0, 5).up(0, 1, 0)))
            for pi, pose in enumerate(self.poses):
                vis_camera(pose, 0, 0, 0, self.names[pi], 0)
            Trescope().breakPoint('')

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsic,
            'index': index,
        }

        if self.type == 'test' or self.type == 'ar':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            return results
        else:
            results['shape'] = (self.H, self.W)
            results['image'] = self.images[index]
            results['depth'] = self.depths[index]
            results['mask'] = self.masks[index]
            results['name'] = self.names[index]
            return results


class NeRFRayDataset(Dataset):
    def __init__(self, path, type='train', downscale=1, radius=1, n_test=10):
        super().__init__()
        self.img_dataset = NeRFDataset(path, type, downscale, radius, n_test)
        # generate all rays
        self.rays_per_img = self.img_dataset.H * self.img_dataset.W
        self.generate_ray_bundle()

    def generate_ray_bundle(self):
        intersector = self.img_dataset.fusion_intersector
        bundle_rays, bundle_depths, bundle_masks, bundle_inds = [], [], [], []
        print(f'<===== processing ray bundle {"wo" if intersector is None else "w"} fusion mesh =====>')
        for data in tqdm(self.img_dataset):
            H, W = data['shape'][0], data['shape'][1]
            intrinsic = torch.tensor(data['intrinsic'][None, ...])
            pose = torch.tensor(data['pose'][None, ...])
            rays_o, rays_d, _ = get_rays(pose, intrinsic, H, W, -1)
            if intersector is None:
                rays_o = rays_o.reshape(1, H, W, -1).numpy().squeeze()
                rays_d = rays_d.reshape(1, H, W, -1).numpy().squeeze()
                ray_depth = data['depth'][..., None]
                ray_depth = ray_depth / np.matmul(rays_d, data['pose'][:3, 2:3])
                ray_rgb = np.concatenate([rays_o, rays_d, data['image']], axis=2)
                ray_mask = data['mask'][..., None]
            else:
                rays_o = rays_o.numpy().squeeze()
                rays_d = rays_d.numpy().squeeze()
                ray_depth = np.zeros((rays_o.shape[0], 1), dtype=rays_o.dtype)
                f_inds, ray_inds, locs = intersector.intersects_id(
                    rays_o, rays_d, False, 1, True)
                loc_ray_cos = np.einsum('ij,ij->i',
                                        intersector.mesh.face_normals[f_inds],
                                        rays_d[ray_inds])
                ray_mask = loc_ray_cos < 0
                ray_inds = ray_inds[ray_mask]
                f_inds = f_inds[ray_mask]
                locs = locs[ray_mask]
                ray_depth[ray_inds] = np.linalg.norm(rays_o[ray_inds]-locs, axis=1, keepdims=True)
                ray_mask = ray_depth > 0
                rays_o = rays_o.reshape((H, W, -1))
                rays_d = rays_d.reshape((H, W, -1))
                ray_rgb = np.concatenate([rays_o, rays_d, data['image']], axis=2)
                ray_depth = ray_depth.reshape((H, W, -1))
                ray_mask = ray_mask.reshape((H, W, -1))

                # update to data['depth']
                img_depth = ray_depth * np.matmul(rays_d, data['pose'][:3, 2:3])
                self.img_dataset.depths[data['index']] = img_depth.squeeze()
                self.img_dataset.masks[data['index']] = ray_mask.squeeze()

            bundle_rays.append(ray_rgb)
            bundle_depths.append(ray_depth)
            bundle_masks.append(ray_mask)
            bundle_inds.append(np.full_like(ray_mask, data['index'], dtype=int))

            pass
        self.bundles = {'ray_rgb': np.stack(bundle_rays).reshape((-1, 9)),
                        'ray_depth': np.stack(bundle_depths).reshape((-1, 1)),
                        'ray_mask': np.stack(bundle_masks).reshape((-1, 1)),
                        'ray_idx': np.stack(bundle_inds).reshape((-1, 1))}
        print('<===== processing ray bundle done =====>')

    def __len__(self):
        return self.bundles['ray_rgb'].shape[0]

    def __getitem__(self, index):
        return {
            'index': index,
            'ray_rgb': self.bundles['ray_rgb'][index],
            'ray_depth': self.bundles['ray_depth'][index],
            'ray_mask': self.bundles['ray_mask'][index],
            'ray_idx': self.bundles['ray_idx'][index],
        }
