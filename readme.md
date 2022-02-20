# torch-ngp

A pytorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).

**News**: With the CUDA ray marching option for NeRF, we can now:
* converge to a reasonable result in about 1min (50 epochs).
* render a 1920x1080 image in under 1s.
(tested on the fox dataset with a TITAN RTX)

For the LEGO dataset, we can reach **~10FPS** at 800x800.

(Although the speed is still 2x~5x slower compared to the original implementation.)

SDF | NeRF
:---: | :---:
![](assets/armadillo.jpg) | ![](assets/fox.gif)

# Progress

As the official pytorch extension [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) has been released, the following implementations can be used as modular alternatives. 
The performance and speed of these modules are guaranteed to be on-par, and we support using tinycudann as the backbone by the `--tcnn` flag.
Later development will be focused on reproducing the NeRF inference speed.

* Fully-fused MLP
    - [x] basic pytorch binding of the [original implementation](https://github.com/NVlabs/tiny-cuda-nn)
* HashGrid Encoder
    - [x] basic pytorch CUDA extension
    - [x] fp16 support 
* Experiments
    - SDF
        - [x] baseline
        - [ ] better SDF calculation (especially for non-watertight meshes)
    - NeRF
        - [x] baseline (although much slower)
        - [x] ray marching in CUDA.


# Install
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Tested on Ubuntu with torch 1.10 & CUDA 11.3 on TITAN RTX.

# Usage

We use the same data format as instant-ngp, e.g., [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `./data`.

First time running will take some time to compile the CUDA extensions.

```bash
# SDF experiment
bash scripts/run_sdf.sh

# NeRF experiment (see the shell script for more options)
bash scripts/run_nerf.sh

# use different backbones
python train_nerf.py data/fox/transforms.json --workspace trial_nerf # fp32 mode
python train_nerf.py data/fox/transforms.json --workspace trial_nerf --fp16 # fp16 mode (pytorch amp)
python train_nerf.py data/fox/transforms.json --workspace trial_nerf --fp16 --ff # fp16 mode + FFMLP (this repo's implementation)
python train_nerf.py data/fox/transforms.json --workspace trial_nerf --fp16 --tcnn # fp16 mode + official tinycudann's encoder & MLP

# [NEW] use CUDA to accelerate ray marching 
python train_nerf.py data/fox/transforms.json --workspace trial_nerf --fp16 --ff --cuda_ray # fp16 mode + FFMLP + cuda raymarching
```

# Difference from the original implementation
* Instead of assuming the scene is bounded in the unit box `[0, 1]` and centered at `(0.5, 0.5, 0.5)`, this repo assumes **the scene is bounded in box `[-bound, bound]`, and centered at `(0, 0, 0)`**. Therefore, the functionality of `aabb_scale` is replaced by `bound` here.
* For the hashgrid encoder, this repo only implement the linear interpolation mode.
* For the voxel pruning in ray marching kernels, this repo doesn't implement the multi-scale density grid (check the `mip` keyword), and only use one `128x128x128` grid for simplicity. Instead of updating the grid every 16 steps, we update it every epoch, which may lead to slower first few epochs if using `--cuda_ray`.

# Update Logs
* 2.20: cuda raymarching is finally stable now!
* 2.15: add the official [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) as an alternative backend.    
* 2.10: add cuda_ray, can train/infer faster, but performance is worse currently.
* 2.6: add support for RGBA image.
* 1.30: fixed atomicAdd() to use __half2 in HashGrid Encoder's backward, now the training speed with fp16 is as expected!
* 1.29: 
    * finished an experimental binding of fully-fused MLP.
    * replace SHEncoder with a CUDA implementation.
* 1.26: add fp16 support for HashGrid Encoder (requires CUDA >= 10 and GPU ARCH >= 70 for now...).


# Acknowledgement

* Credits to [Thomas Müller](https://tom94.net/) for the amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp):
    ```
    @misc{tiny-cuda-nn,
        Author = {Thomas M\"uller},
        Year = {2021},
        Note = {https://github.com/nvlabs/tiny-cuda-nn},
        Title = {Tiny {CUDA} Neural Network Framework}
    }

    @article{mueller2022instant,
        title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
        author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
        journal = {arXiv:2201.05989},
        year = {2022},
        month = jan
    }
    ```

* The framework of NeRF is adapted from [nerf_pl](https://github.com/kwea123/nerf_pl):
    ```
    @misc{queianchen_nerf,
        author = {Quei-An, Chen},
        title = {Nerf_pl: a pytorch-lightning implementation of NeRF},
        url = {https://github.com/kwea123/nerf_pl/},
        year = {2020},
    }
    ```