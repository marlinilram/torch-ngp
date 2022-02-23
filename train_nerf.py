import torch

from nerf.provider import NeRFDataset
from nerf.utils import *
from nerf.opt import get_opt

import argparse

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

    train_dataset = NeRFDataset(opt.path, 'train', downscale=1, radius=opt.radius)
    valid_dataset = NeRFDataset(opt.path, 'valid', downscale=1, radius=opt.radius)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray,
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

    criterion = torch.nn.SmoothL1Loss()

    optimizer = lambda model: torch.optim.Adam(pose_params+[
        {'name': 'encoding', 'params': list(model.encoder.parameters())},
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)

    trainer = Trainer('ngp', vars(opt), model, pose_refine=pose_refine, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=10)
    
    trainer.train(train_loader, valid_loader, 200)

    # test dataset
    trainer.save_mesh()
    test_dataset = NeRFDataset(opt.path, 'test', radius=opt.radius, n_test=10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    trainer.test(test_loader)
