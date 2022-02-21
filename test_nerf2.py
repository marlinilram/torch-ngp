import torch

from nerf.network import NeRFNetwork
# from nerf.network_ff import NeRFNetwork as NeRFNetwork_FF
from nerf.provider import NeRFDataset
from nerf.utils import *
from nerf.opt import get_opt

import argparse
from os.path import join as pjoin

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    opt = get_opt()

    print(opt)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        # Network = NeRFNetwork_FF
    else:
        Network = NeRFNetwork

    seed_everything(opt.seed)

    train_dataset = NeRFDataset(opt.path, 'train', radius=opt.radius)
    valid_dataset = NeRFDataset(opt.path, 'valid', downscale=1, radius=opt.radius)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    model = Network(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        density_grid_size=128 if opt.cuda_raymarching else -1,
    )

    if opt.pose_refine:
        from nerf.pose_refine import PoseRefine
        pose_refine = PoseRefine(len(train_dataset))
        pose_params = [{'name': 'pose_refine', 'params': list(pose_refine.parameters())}]
    else:
        pose_refine = None
        pose_params = []

    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)

    print(model)

    trainer = Trainer('ngp', vars(opt), model, pose_refine=pose_refine, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')

    # test dataset
    trainer.save_mesh(bound=opt.bound)
    trainer.test(valid_loader, save_path=pjoin(opt.workspace, 'test'))
