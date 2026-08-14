# 单卡 Ascend 910B 预训练流程

所有命令均在 Flash3D 项目根目录执行，并假定已安装 CANN 和与之匹配的 `torch_npu`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/check_npu_env.py
python scripts/test_npu_renderer.py
```

Depth Anything V2 Base 需放在 `third_party/Depth-Anything-V2`，权重默认路径为
`weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth`。

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

## 单卡训练与评估

```bash
NPU_ID=0 bash scripts/train_npu.sh \
  dataset.data_path=/datasets/RealEstate10K \
  dataset.preload_depths=true \
  dataset.depth_path=/datasets/RealEstate10K-depth-anything-vitb

NPU_ID=0 bash scripts/evaluate_npu.sh \
  dataset.data_path=/datasets/RealEstate10K
```

`layered_re10k_npu` 使用 Depth Anything V2 ViT-B、冻结深度、单卡和
`npu_torch` 可微高斯渲染器。它不是多卡 HCCL 启动配置。
