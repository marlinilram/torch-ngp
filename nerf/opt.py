import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")

    parser.add_argument('--radius', type=float, default=2, help="assume the camera is located on sphere(0, radius))")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-size, size)")

    parser.add_argument('--cuda_raymarching', action='store_true', help="use CUDA raymarching instead of pytorch (unstable now)")
    parser.add_argument('--pose_refine', action='store_true', help="refine pose")
    parser.add_argument('--depth_loss', action='store_true', help="use depth")
    return parser.parse_args()
