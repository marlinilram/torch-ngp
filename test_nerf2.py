import torch

from nerf.provider import NeRFDataset
from nerf.utils import seed_everything, Trainer
from nerf.opt import get_opt

import argparse
from os.path import join as pjoin

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    opt = get_opt()

    print(opt)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    seed_everything(opt.seed)

    valid_dataset = NeRFDataset(opt.path, 'ar', downscale=0.5, radius=opt.radius)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics",
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64,
        cuda_ray=opt.cuda_ray,
    )

    if opt.pose_refine:
        from nerf.pose_refine import PoseRefine
        train_dataset = NeRFDataset(opt.path, 'train', radius=opt.radius)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
        pose_refine = PoseRefine(len(train_dataset))
        pose_params = [{'name': 'pose_refine', 'params': list(pose_refine.parameters())}]
    else:
        pose_refine = None
        pose_params = []

    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)

    print(model)

    trainer = Trainer('ngp', vars(opt), model, pose_refine=pose_refine, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')

    # test dataset
    # trainer.save_mesh(bound=opt.bound, threshold=model.mean_density if model.cuda_ray else 10)
    trainer.test(valid_loader, save_path=pjoin(opt.workspace, 'test'))
