# UniDepth V1 本地下载与配置

本项目的 `model.name=unidepth` 分支使用 **UniDepth V1 ViT-L/14** 作为冻结的单目深度先验。为保证 CPU 离线推理可复现，UniDepth 的 Python 源码与模型权重需要分别准备。

> 本文只说明如何从官方来源下载依赖。请不要把下载的权重直接提交到普通 Git 仓库。

## 1. 目录结构

完成后，请确认目录如下：

```text
flash3d-main/
├─ third_party/
│  └─ unidepth_offline/
│     ├─ hubconf.py
│     ├─ configs/config_v1_vitl14.json
│     └─ unidepth/
└─ weights/
   └─ unidepth-v1-cnvnxtl/
      └─ unidepth_v1_vitl14.bin
```

其中目录名 `unidepth-v1-cnvnxtl` 是本项目历史配置保留的名称；其中实际放置的是 ViT-L/14 权重，文件名必须保持为 `unidepth_v1_vitl14.bin`。

## 2. 下载 UniDepth 源码（不含权重）

在项目根目录运行：

```powershell
python third_party\fetch_unidepth_source.py
```

该脚本从 [官方 UniDepth 仓库](https://github.com/lpiccinelli-eth/UniDepth) 下载 CPU fallback 所需的 Python 文件和 `config_v1_vitl14.json`，保存到 `third_party/unidepth_offline`。脚本内置重试；网络不稳定时可直接重新执行。

也可以手动克隆官方仓库后，将其内容复制/链接为：

```text
third_party/unidepth_offline
```

此时必须确保 `hubconf.py` 位于该目录的第一层。

## 3. 下载 UniDepth V1 ViT-L/14 权重

官方权重页面：

- [Hugging Face：lpiccinelli/UniDepth](https://huggingface.co/lpiccinelli/UniDepth)
- [官方 UniDepth 模型说明](https://github.com/lpiccinelli-eth/UniDepth#model-zoo)

### 推荐：Hugging Face CLI

先完成 Hugging Face 登录（模型仓库若要求访问授权，请先在网页接受条款）：

```powershell
hf auth login
```

再执行：

```powershell
hf download lpiccinelli/UniDepth unidepth_v1_vitl14.bin `
  --local-dir weights\unidepth-v1-cnvnxtl
```

下载后检查：

```powershell
Get-Item weights\unidepth-v1-cnvnxtl\unidepth_v1_vitl14.bin
```

文件约为 1.3 GB。若 CLI 因网络 SSL 问题无法访问，请在 Hugging Face 网页的 **Files and versions** 页面下载同名文件，再手动复制到上述目录。权重文件不是 `.ckpt`；`.bin` 是 PyTorch state-dict 的常见发布格式，项目会通过 `load_pretrained()` 读取它。

### `config.json` 是否需要下载？

不需要。Flash3D 的 UniDepth 封装只会检查并加载：

```text
weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin
```

模型结构配置由第 2 步下载的 `third_party/unidepth_offline/configs/config_v1_vitl14.json` 提供。你已手动保存的 `config.json` 或 `config.txt` 可以保留作记录，但不参与运行。

## 4. CPU 运行说明

CPU 环境不安装 xFormers/Triton。若 xFormers 不可用，本项目在 `third_party/unidepth_offline/unidepth/layers/nystrom_attention.py` 中将 Nyström attention 回退到 PyTorch 的 `scaled_dot_product_attention`。它可以正确运行，但比 CUDA/xFormers 慢得多；建议先用 256×384 或更低分辨率验证。

安装 CPU 推理依赖：

```powershell
python -m pip install -r requirements-cpu.txt
```

## 5. 许可证与发布注意事项

官方 UniDepth 软件和预训练权重采用 [CC BY-NC 4.0](https://github.com/lpiccinelli-eth/UniDepth?tab=readme-ov-file#license)；使用和再分发前请确认符合其署名、非商业等条件。

此外，该权重约 1.32 GB，超过 GitHub 普通 Git 的 100 MiB 单文件限制。项目的 `.gitignore` 默认忽略 `weights/`。如需让其他人复现，请保留本文件的下载步骤，而非把权重提交到公开仓库；只有在许可证允许、且明确接受 Git LFS 存储与下载额度消耗时，才应使用 Git LFS 单独托管。
