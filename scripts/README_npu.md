# 单卡 Ascend 910B 预训练流程

所有命令均在 Flash3D 项目根目录执行。目标为 Linux + Ascend 910B + Python
3.10。先安装驱动、固件和 CANN；然后严格按 Ascend 的版本对应表安装与 CANN
匹配的 CPU PyTorch 2.7.1 与 `torch_npu 2.7.1.post4` wheel。不要安装 CUDA
版 PyTorch，也不要安装 `requirements.txt` 中的 xformers/Triton/CUDA 高斯
光栅器。

```bash
python3.10 -m venv .venv-npu
source .venv-npu/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 先按 CANN 对应关系安装 torch 和 torch_npu wheel，再安装其余 Python 包
pip install -r requirements_npu.txt
```

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/check_npu_env.py
python scripts/test_npu_renderer.py
```

Depth Anything V2 Base 需放在 `third_party/Depth-Anything-V2`，加载其
`metric_depth` 实现；权重默认路径为
`weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth`（VKITTI
室外 80m 版本）。NPU 配置
`layered_re10k_npu` 明确设置 `encoder: vitb`、`freeze: true`；训练时仅更新
Flash3D 的 ResNet 编码器、高斯解码器和层深度解码器。即使外层进入 train 模式，
代码也会让冻结的 Depth Anything 始终处于 eval 模式。

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
mkdir -p weights/depth-anything-v2
# 将官方 metric ViT-B 权重复制为：
# weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth
```

## 可选：预计算冻结深度

这一步减少训练时的 NPU 延迟和显存。输出目录可随后作为 `dataset.depth_path` 使用。

```bash
python scripts/precompute_depth_anything_re10k.py \
  --data-path /datasets/RealEstate10K \
  --output /datasets/RealEstate10K-depth-anything-vitb \
  --device npu
```

## 实测峰值显存

```bash
python scripts/profile_npu_pretrain.py \
  --data-path /datasets/RealEstate10K \
  --depth-path /datasets/RealEstate10K-depth-anything-vitb
```

若峰值超过目标，先将 `model.npu_renderer_max_gaussians` 从 65536 下调至 32768，
再降低每 tile 的上限；不要先增大 batch size。

## 单卡预训练

```bash
NPU_ID=0 bash scripts/pretrain_npu.sh \
  --data-path /datasets/RealEstate10K \
  --depth-path /datasets/RealEstate10K-depth-anything-vitb \
  --output outputs/layered_re10k_npu
```

`pretrain_npu.sh` 强制 `model.depth.freeze=true`。若未使用预计算深度，冻结的
ViT-B 仍会在 NPU 上前向执行，但不会保存反向图；先预计算能进一步降低单步延迟与
显存。

## 单图 NPU 推理与多机位输出

```bash
NPU_ID=0 bash scripts/infer_npu.sh \
  --image data/DS0304.jpg \
  --checkpoint outputs/layered_re10k_npu/checkpoints \
  --output outputs/npu_dsc0304 \
  --rig cross5 --baseline 0.15 --vertical-baseline 0.10
```

对于 RE10K 帧，请传入原图像素坐标下的内参，避免使用默认焦距：

```bash
NPU_ID=0 bash scripts/infer_npu.sh \
  --image /datasets/RealEstate10K/test/SEQ/FRAME.jpg \
  --intrinsics FX FY CX CY \
  --checkpoint outputs/layered_re10k_npu/checkpoints \
  --output outputs/npu_re10k
```

推理会保存 `gaussians.pt`、每个机位的 PNG、多视角拼图和
`inference_report.json`。默认最多使用 65536 个可见高斯、每 tile 128 个高斯，
用于控制 910B 峰值内存；若显存充足，可将两个限制设为 `0` 以保留全部候选。

## 评估

```bash

NPU_ID=0 bash scripts/evaluate_npu.sh \
  dataset.data_path=/datasets/RealEstate10K
```

`layered_re10k_npu` 使用 Depth Anything V2 ViT-B、冻结深度、单卡和
`npu_torch` 可微高斯渲染器。它不是多卡 HCCL 启动配置。
