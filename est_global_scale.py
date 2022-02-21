import numpy as np

if __name__ == '__main__':
    cam_pos = np.load('cam_pos.npy')
    diff = cam_pos - np.roll(cam_pos, 1, axis=0)
    dist = np.linalg.norm(diff, axis=1)

    cam_pos_ngp = np.load('cam_pos_ngp.npy')
    diff_ngp = cam_pos_ngp - np.roll(cam_pos_ngp, 1, axis=0)
    dist_ngp = np.linalg.norm(diff_ngp, axis=1)
    print('test')
