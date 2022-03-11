import argparse
import json
import numpy as np
from os import path as osp
from .colmap2nerf import sharpness


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
    # [right, down, forward, pos]
    return perm, poses


def load_intrinsic(datadir, fname, factor=2):
    perm, intrinsics = [], []
    with open(osp.join(datadir, 'intrinsic.txt')) as reader:
        lines = reader.readlines()
        for line in lines:
            datas = [word.split('\n')[0] for word in line.split(' ')]
            idx = int(datas[0])
            intrinsic = np.array([float(v) for v in datas[1:7]])
            perm.append(idx)
            intrinsics.append(intrinsic)
            pass

    # W, H, fx, fy, cx, cy
    return intrinsics[int(fname)] / 2
    # return perm, intrinsics


def get_camera_parameter(root_path):
    w, h, fl_x, fl_y, cx, cy = load_intrinsic(root_path, '0')

    angle_x = np.arctan(w/(fl_x*2))*2
    angle_y = np.arctan(h/(fl_y*2))*2
    fovx = angle_x*180/np.pi
    fovy = angle_y*180/np.pi
    k1 = 0.0
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    print(
        f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")
    return {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h
    }


def generate_transforms(root_path, img_path, cam_data):
    # 不要用train.txt, NerfingMVS跑完以后会过滤train.txt中不好的帧
    with open(osp.join(root_path, 'valid_train.txt'), 'r') as f_list:
        lines = f_list.readlines()
        lines = [l.split('\n')[0] for l in lines]
    if osp.exists(osp.join(root_path, 'train_ignore.txt')):
        ignore = json.load(open(osp.join(root_path, 'train_ignore.txt')))
        ignore = ignore['train_index']
        ignore = [lines[i] for i in ignore]
    else:
        ignore = []

    perm, poses = load_arkit_poses(root_path)

    out = {
        **cam_data,
        'aabb_scale': 1.0,
        'scale': 1.0,
        'offset': [0.0, 0.0, 0.0],  # rescale and offset coornidate system so that scene is bounded in [-size, size]
        'depth_scale': 1.0,
        'coordinate': 'ngp',
        'frames': []
    }
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    vio_data = {raw_id: pose for raw_id, pose in zip(perm, poses)}
    for name in lines:
        raw_id = int(osp.splitext(name)[0])
        if raw_id not in vio_data:
            continue
        if name in ignore:
            continue

        image_rel = osp.relpath(img_path)
        b = sharpness(f'./{image_rel}/{name}')
        print(f'images/{name}', "sharpness=", b)

        c2w = np.concatenate([vio_data[raw_id], bottom], 0)
        out['frames'].append({
            'file_path': f'images/{name}',
            'sharpness': b,
            'transform_matrix': c2w.tolist()
        })
        pass
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="convert a arkit export to nerf format transforms.json")

    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument("--text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--aabb_scale", default=16, choices=["1", "2", "4", "8", "16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--out", default="transforms.json", help="output path")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cam_data = get_camera_parameter(args.text)
    out = generate_transforms(args.text, args.images, cam_data)
    print(len(out['frames']), 'frames')
    print(f'writing {args.out}')
    with open(args.out, 'w') as outfile:
        json.dump(out, outfile, indent=2)
