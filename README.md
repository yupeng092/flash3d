[![arXiv](https://img.shields.io/badge/arXiv-2406.04343-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2406.04343)
[![ProjectPage](https://img.shields.io/badge/Project_Page-Flash3D-blue)](https://www.robots.ox.ac.uk/~vgg/research/flash3d/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/szymanowiczs/flash3d) 


# Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image


<p align="center">
  <img src="assets/teaser_video.gif" alt="animated" />
</p>

> [Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image](https://www.robots.ox.ac.uk/~vgg/research/flash3d/)  
> Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, João F. Henriques, Christian Rupprecht, Andrea Vedaldi  
> 3DV, 2025.
> *[arXiv 2406.04343](https://arxiv.org/pdf/2406.04343.pdf)*  

# News
- [x] `19.07.2024`: Training code and data release

# Setup

## Create a python environment

Flash3D has been trained and tested with the followings software versions:

- Python 3.10
- Pytorch 2.2.2
- CUDA 11.8
- GCC 11.2 (or more recent)

Begin by installing CUDA 11.8 and adding the path containing the `nvcc` compiler to the `PATH` environmental variable.
Then the python environment can be created either via conda:

```sh
conda create -y python=3.10 -n flash3d
conda activate flash3d
```

or using Python's venv module (assuming you already have access to Python 3.10 on your system):

```sh
python3.10 -m venv .venv
. .venv/bin/activate
```

Finally, install the required packages as follows:

```sh
pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For the local UniDepth V1 source/weight layout used by this repository (including an offline CPU fallback), see [docs/UNIDEPTH_SETUP.md](docs/UNIDEPTH_SETUP.md).

For Ascend 910B setup, frozen Depth Anything V2 Base pre-training, and NPU
single-image multi-view inference, see [scripts/README_npu.md](scripts/README_npu.md).

For CPU rendering from exported `gaussians.pt`, use `render_cpu_multiview.py`.  It renders named multi-camera rigs (the default is centre/left/right/up/down) and writes the exact rig to `camera_rig.json`.

### Native Flash3D CUDA renderer for multi-view output

The multi-view command also exposes Flash3D's original CUDA
`diff-gaussian-rasterization` path.  Unlike the portable CPU reference
renderer, it uses the same projection matrix convention, `projmatrix_raw`, SH
coefficients, Gaussian scales and rotations as
`models/decoder/gauss_util.py`.  Use it on a CUDA machine with the project's
standard environment installed:

```sh
python render_cpu_multiview.py \
  --backend native \
  --gaussians outputs/flash3d_repaired_source_check/gaussians.pt \
  --rig cross5 --baseline 0.15 --vertical-baseline 0.10 \
  --use-source-intrinsics --height 256 --width 384 \
  --scale-modifier 0.55 \
  --output outputs/flash3d_native_cross5
```

The default `--backend torch` is a pure-PyTorch, CPU-capable implementation of
the same tile-binning, depth sorting and front-to-back compositing pipeline;
it needs neither CUDA nor the extension. `--backend cpu` is retained as an
alias for it. `--backend legacy` selects the earlier lightweight approximation
only for A/B comparisons. The native backend is intentionally not an NPU/CPU
fallback; it fails early with a clear message if CUDA or
`diff_gaussian_rasterization` is unavailable. Options such as `--chunk-size`
are specific to the legacy renderer and do not change native CUDA
rasterization.

`render_cpu_multiview.py` accepts Flash3D `gaussians.pt`, standard binary 3DGS `.ply` files (including log-scale/logit-opacity exports), and UniSHARP's `unisharp_gaussians` `.pt` export.  For cross-model comparisons, use the same camera rig with `--position-scale` set to the ratio of the model's scene depth scale to the reference scene depth scale.

For a UniSHARP export, add `--use-source-intrinsics`: its calibrated camera matrix is embedded in the export and is automatically resized to the requested output resolution.  The CPU renderer also evaluates the `f_rest_*` coefficients of a standard 3DGS PLY up to degree 3, so specular/view-dependent colour is not mistakenly kept constant while the camera moves.  Flash3D exports fixed RGB rather than higher-order SH, so this latter correction cannot add view-dependent appearance to Flash3D.

UniSHARP Gaussian colours are linear RGB.  Add `--linear-to-srgb` when writing ordinary PNGs, otherwise a standard image viewer interprets the linear values as sRGB and the result appears too dark.

`benchmark_cpu.py` saves the camera matrix that UniDepth actually used in `gaussians.pt` metadata.  For a calibrated photograph (or a RE10K frame), provide the original-image pixel intrinsics with `--intrinsics FX FY CX CY`; the script resizes and pads the matrix exactly as the RE10K loader does.  The exported Gaussian file can then be rendered with `render_cpu_multiview.py --use-source-intrinsics`.

## Optional Real-ESRGAN super-resolution

To upscale rendered multi-view RGB images after CPU rendering, install the optional dependency and run:

```powershell
python -m pip install -r requirements-realesrgan.txt
python superresolve_realesrgan.py `
  --input outputs/cpu_multiview_physical_ten_test/rgb `
  --output outputs/cpu_multiview_physical_ten_test/rgb_x2 `
  --outscale 2 --tile 128
```

The script downloads the official `RealESRGAN_x4plus` checkpoint to `weights/realesrgan/` on first use. Use `--model-path` to supply an offline checkpoint, and lower `--tile` if CPU memory is limited.

## Download training data

### RealEstate10K dataset

For downloading the RealEstate10K dataset we base our instructions on the [Behind The Scenes](https://github.com/Brummi/BehindTheScenes/tree/main?tab=readme-ov-file#-datasets) scripts.
First you need to download the video sequence metadata including camera poses from https://google.github.io/realestate10k/download.html and unpack it into `data/` such that the folder layout is as follows:

```
data/RealEstate10K/train
data/RealEstate10K/test
```

Finally download the training and test sets of the dataset with the following commands:

```sh
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m train
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m test
```

This step will take several days to complete. Finally, download additional data for the RealEstate10K dataset.
In particular, we provide pre-processed COLMAP cache containing sparse point clouds which are used to estimate the scaling factor for depth predictions.
The last two commands filter the training and testing set from any missing video sequences.

```sh
sh datasets/dowload_realestate10k_colmap.sh
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s train
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s test
```

## Download and evaluate the pretrained model

We provide model weights that could be downloaded and evaluated on RealEstate10K test set:

```sh
python -m misc.download_pretrained_models -o exp/re10k_v2
sh evaluate.sh exp/re10k_v2
```

## Training

In order to train the model on RealEstate10K dataset execute this command:
```sh
python train.py \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=false 
```

For multiple GPU, we can run with this command:
```sh
sh train.sh
```
You can modify the cluster information in ```configs/hydra/cluster```.


## BibTeX
```
@article{szymanowicz2024flash3d,
      author = {Szymanowicz, Stanislaw and Insafutdinov, Eldar and Zheng, Chuanxia and Campbell, Dylan and Henriques, Joao and Rupprecht, Christian and Vedaldi, Andrea},
      title = {Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image},
      journal = {arxiv},
      year = {2024},
}
```
