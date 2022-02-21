from operator import index
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
from ff3d_planner.common import Quaternion, Matrix4


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
        [0, 0, 0, 1],
    ])
    return new_pose



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
        
        # load image size
        self.H = int(transform['h']) // downscale
        self.W = int(transform['w']) // downscale

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = transform['fl_x'] / downscale
        self.intrinsic[1, 1] = transform['fl_y'] / downscale
        self.intrinsic[0, 2] = transform['cx'] / downscale
        self.intrinsic[1, 2] = transform['cy'] / downscale

        self.scale = transform['scale']
        self.offset = transform['offset']
        self.depth_scale = transform['depth_scale']

        with open(os.path.join(self.root_path, 'train.txt')) as rd:
            lines = rd.readlines()
            names = [l[:-5] for l in lines]
            frame_train_idx = {n: ni for ni, n in enumerate(names)}

        # same order with poses_bounds.npy
        frames = transform["frames"]
        for frame in frames:
            name = os.path.splitext(os.path.basename(frame['file_path']))
            frame['file_idx'] = frame_train_idx[name[0]]
        frames = sorted(frames, key=lambda d: d['file_idx'])

        if type == 'test':
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), self.scale, self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), self.scale, self.offset) # [4, 4]
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
                frames = frames[1:]
            elif type == 'valid':
                frames = frames[:1]

            self.poses = []
            self.images = []
            self.depths = []
            self.masks = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, self.scale, self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)

                img_id = os.path.splitext(os.path.basename(f['file_path']))[0]
                depth = load_depth(self.root_path, img_id)
                mask = load_depth_confidence(self.root_path, img_id)
                depth = cv2.resize(depth, (self.W, self.H))
                depth *= self.depth_scale
                mask = cv2.resize(mask, (self.W, self.H))
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
                vis_camera(pose, 0, 0, 0, str(pi), 0)
            Trescope().breakPoint('')

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsic,
            'index': index,
        }

        if self.type == 'test':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            return results
        else:
            results['shape'] = (self.H, self.W)
            results['image'] = self.images[index]
            results['depth'] = self.depths[index]
            results['mask'] = self.masks[index]
            return results


class NeRFRayDataset(NeRFDataset):
    def __init__(self, path, type='train', downscale=1, radius=1, n_test=10):
        super().__init__(path, type, downscale, radius, n_test)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)