import numpy as np
import json
from os import path as osp
from .provider import NeRFDataset


def load_arkit_poses(datadir):
    perm, timestamps, poses = [], [], []
    with open(osp.join(datadir, 'vio_pose.txt')) as reader:
        lines = reader.readlines()
        for line in lines:
            datas = [word.split('\n')[0] for word in line.split(' ')]
            idx, timestamp = int(datas[0]), datas[1]
            pose = np.array([float(v) for v in datas[2:14]]).reshape((4, 3))
            R, t = pose[1:4, :], pose[0:1, :]
            pose = np.concatenate([R, t.T], axis=1)
            perm.append(idx)
            timestamps.append(timestamp)
            poses.append(pose)
            pass
    return perm, poses


def main(root_path):
    dataset = NeRFDataset(osp.join(root_path, 'transforms.json'), 'all', downscale=2)
    # transforms = json.load(open(osp.join(root_path, 'transforms.json')))
    vio_data = load_arkit_poses(root_path)

    # align image index between transforms and vio_pose
    vio_pose = np.stack(vio_data[1])[np.array([int(n) for n in dataset.names])]
    cam_pos = vio_pose[:, :, 3]
    diff = cam_pos - np.roll(cam_pos, 1, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    dist_mat = np.linalg.norm(cam_pos[:, None, :] - cam_pos[None, :, :], axis=-1)

    ngp_pose = dataset.poses
    cam_pos_ngp = ngp_pose[:, :3, 3]
    diff_ngp = cam_pos_ngp - np.roll(cam_pos_ngp, 1, axis=0)
    dist_ngp = np.linalg.norm(diff_ngp, axis=1)
    dist_mat_ngp = np.linalg.norm(cam_pos_ngp[:, None, :] - cam_pos_ngp[None, :, :], axis=-1)

    mask = dist_mat > 0
    depth_scale = (dist_mat_ngp[mask]/dist_mat[mask]).mean()
    print(f'est depth scale: {depth_scale}')
    pass


if __name__ == '__main__':
    # cam_pos = np.load('cam_pos.npy')
    # diff = cam_pos - np.roll(cam_pos, 1, axis=0)
    # dist = np.linalg.norm(diff, axis=1)

    # cam_pos_ngp = np.load('cam_pos_ngp.npy')
    # diff_ngp = cam_pos_ngp - np.roll(cam_pos_ngp, 1, axis=0)
    # dist_ngp = np.linalg.norm(diff_ngp, axis=1)
    main('data/arkit_huiyishi_2')
    print('test')
