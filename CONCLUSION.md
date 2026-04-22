# FSBI 项目实验总结与论文写作说明

最后更新：2026-04-25

本文档用于完整记录当前 FSBI 项目已经完成的工作，包括数据集准备、训练过程、测试命令、实验结果、论文中应如何表述，以及当前仍需注意的风险点。目标是让网页端 ChatGPT 或其他读者在阅读本文件后，能够快速理解本项目做了什么、怎么做的、结果如何，以及如何据此辅助毕业论文写作。

## 1. 项目目标

本项目围绕深度伪造视频检测展开，核心是在 SBI 框架基础上引入 DWT 频域增强，从而提升模型的泛化能力和压缩鲁棒性。

在当前代码仓库中：

- `--variant sbi` 表示 SBI 基线方法。
- `--variant esbi` 表示加入 DWT 频域增强后的方法。
- 在论文写作中，可以将 `esbi` 路径描述为“在 SBI 框架中引入离散小波变换进行频域增强的方法”。

本项目主要完成了以下实验目标：

- 在 FaceForensics++ c23 上验证基础检测性能。
- 在 Celeb-DF-v2 上验证跨库泛化能力。
- 在 FaceForensics++ c40 上验证压缩鲁棒性。
- 通过不同 DWT 配置对比完成消融实验。

## 2. 代码结构说明

重要文件如下：

| 路径 | 作用 |
|---|---|
| `src/train_sbi.py` | 主训练脚本，支持 SBI 和 DWT 增强方法。 |
| `src/utils/sbi.py` | SBI 训练数据集逻辑。 |
| `src/utils/esbi.py` | DWT 增强训练数据集逻辑。 |
| `src/inference/inference_dataset.py` | 主视频级测试脚本。 |
| `src/inference/datasets.py` | FF++、Celeb-DF-v2 等数据集测试列表构建逻辑。 |
| `src/ff.py` | FaceForensics++ 下载脚本。 |
| `src/configs/sbi/base.json` | 主训练配置文件。 |
| `output/` | 保存训练权重、训练日志和测试日志。 |
| `data/` | 数据集目录。 |

当前主训练配置文件 `src/configs/sbi/base.json` 内容如下：

```json
{
    "epoch": 100,
    "batch_size": 16,
    "image_size": 380
}
```

需要注意的是，`src/train_sbi.py` 中训练 dataloader 实际使用的是 `batch_size // 2`，因此当配置文件中的 `batch_size` 为 16 时，训练阶段实际 dataloader batch size 是 8。

## 3. 实验环境

实验运行在 GCP 虚拟机上，GPU 为 NVIDIA L4。

关键环境经验：

- 长任务均建议放在 `tmux` 中运行。
- Python 建议使用仓库内虚拟环境：`./.venv/bin/python`。
- 不建议直接使用系统 `python`，因为系统环境可能缺少依赖。
- OpenCV 曾依赖系统库 `libgl1`。
- `dlib` 不建议源码编译，优先使用 `dlib-bin` 或其他预编译包。

常用监控命令：

```bash
nvidia-smi
```

```bash
watch -n 1 nvidia-smi
```

```bash
top
```

```bash
htop
```

查看 tmux 会话：

```bash
tmux ls
```

进入 tmux 会话：

```bash
tmux attach -t <session_name>
```

退出但不终止 tmux 会话：

```text
Ctrl+b d
```

## 4. 当前已有数据集

当前主要数据占用：

```text
data/FaceForensics++    约 16G
data/Celeb-DF-v2        约 9.5G
output/                 约 4.9G
```

### 4.1 FaceForensics++

本地路径：

```text
data/FaceForensics++
```

当前已下载的视频数量：

```text
FF++ original c23:       1000 个视频
FF++ original c40:       1000 个视频
FF++ manipulated c23:    4000 个视频
FF++ manipulated c40:    4000 个视频
```

四种官方伪造方法：

```text
Deepfakes
Face2Face
FaceSwap
NeuralTextures
```

数据划分文件：

```text
data/FaceForensics++/train.json    400 对视频
data/FaceForensics++/val.json       50 对视频
data/FaceForensics++/test.json      50 对视频
```

FF++ 测试协议：

- 使用 `test.json` 对应的真实视频。
- 使用同一批 ID 对应的某一种伪造方法视频。
- 每次测试一个伪造方法，例如 Deepfakes、Face2Face、FaceSwap 或 NeuralTextures。
- 指标采用视频级 AUC。

FF++ 下载命令模板：

```bash
./.venv/bin/python src/ff.py data/FaceForensics++ -d original -c c23 -t videos
./.venv/bin/python src/ff.py data/FaceForensics++ -d Deepfakes -c c23 -t videos
./.venv/bin/python src/ff.py data/FaceForensics++ -d Face2Face -c c23 -t videos
./.venv/bin/python src/ff.py data/FaceForensics++ -d FaceSwap -c c23 -t videos
./.venv/bin/python src/ff.py data/FaceForensics++ -d NeuralTextures -c c23 -t videos
```

c40 鲁棒性测试数据下载时将 `-c c23` 改为 `-c c40`。

### 4.2 Celeb-DF-v2

本地路径：

```text
data/Celeb-DF-v2
```

目录结构：

```text
data/Celeb-DF-v2/Celeb-real
data/Celeb-DF-v2/Celeb-synthesis
data/Celeb-DF-v2/YouTube-real
data/Celeb-DF-v2/List_of_testing_videos.txt
```

本地视频数量：

```text
Celeb-real:       590
Celeb-synthesis: 5639
YouTube-real:    300
```

当前测试使用官方测试列表：

```text
data/Celeb-DF-v2/List_of_testing_videos.txt
```

官方测试列表构成：

```text
总数：518 个视频
伪造视频：340 个，来自 Celeb-synthesis
真实视频：178 个，其中 108 个来自 Celeb-real，70 个来自 YouTube-real
```

## 5. 未作为最终实验使用的数据集

### 5.1 DeeperForensics-1.0

未作为最终核心实验使用的原因：

- 完整数据集非常大，约数百 GB。
- 当前本地只下载过部分数据，源视频覆盖不完整。
- 若严格按官方测试协议完成鲁棒性实验，需要补充大量数据，成本过高。
- 因此最终改用 FF++ c40 作为压缩鲁棒性测试。

论文中不要声称完成了完整 DeeperForensics-1.0 测试。

### 5.2 FFIW10K

未使用原因：

- 已填写申请表，但在实验收口时尚未收到下载链接。

### 5.3 DFDC / DFDCP

未使用原因：

- DFDC 需要 AWS 账号和 AWS account number。
- 当前项目没有完成 DFDC / DFDCP 的数据下载和测试。

论文中如果原来写了 DFDC、DFDCP、FFIW10K、DeeperForensics-1.0，需要删除或改写为未来工作。

## 6. 已完成训练

所有完整训练均基于：

```bash
./.venv/bin/python src/train_sbi.py src/configs/sbi/base.json ...
```

### 6.1 Sym2 + Reflect 方法

输出目录：

```text
output/full_base_04_13_17_35_59
```

后续测试使用的权重：

```text
output/full_base_04_13_17_35_59/weights/87_0.9998_val.tar
```

训练命令模板：

```bash
tmux new-session -s full_base
cd /home/fsbi/FSBI
./.venv/bin/python src/train_sbi.py src/configs/sbi/base.json \
  -n full_base \
  --variant esbi \
  -w sym2 \
  -m reflect
```

训练日志末尾示例：

```text
Epoch 100/100 | train loss: 0.0034, train acc: 0.9985, val loss: 0.0324, val acc: 0.9906, val auc: 0.9995
```

### 6.2 SBI 基线模型

输出目录：

```text
output/full_base_sbi_04_17_12_07_58
```

后续测试使用的权重：

```text
output/full_base_sbi_04_17_12_07_58/weights/99_0.9999_val.tar
```

训练命令模板：

```bash
tmux new-session -s sbi_train
cd /home/fsbi/FSBI
./.venv/bin/python src/train_sbi.py src/configs/sbi/base.json \
  -n full_base_sbi \
  --variant sbi
```

训练日志末尾示例：

```text
Epoch 100/100 | train loss: 0.0029, train acc: 0.9989, val loss: 0.0197, val acc: 0.9931, val auc: 0.9997
```

### 6.3 Haar + Reflect 方法

输出目录：

```text
output/full_haar_reflect_04_19_18_27_50
```

后续泛化和鲁棒性测试使用的权重：

```text
output/full_haar_reflect_04_19_18_27_50/weights/50_0.9998_val.tar
```

训练命令模板：

```bash
tmux new-session -s haar_reflect_train
cd /home/fsbi/FSBI
./.venv/bin/python src/train_sbi.py src/configs/sbi/base.json \
  -n full_haar_reflect \
  --variant esbi \
  -w haar \
  -m reflect
```

训练日志末尾示例：

```text
Epoch 100/100 | train loss: 0.0023, train acc: 0.9990, val loss: 0.0329, val acc: 0.9938, val auc: 0.9990
```

### 6.4 Sym2 + Constant 方法

输出目录：

```text
output/full_sym2_constant_04_22_11_39_16
```

后续测试使用的权重：

```text
output/full_sym2_constant_04_22_11_39_16/weights/65_0.9998_val.tar
```

训练命令模板：

```bash
tmux new-session -s sym2_constant_train
cd /home/fsbi/FSBI
./.venv/bin/python src/train_sbi.py src/configs/sbi/base.json \
  -n full_sym2_constant \
  --variant esbi \
  -w sym2 \
  -m constant
```

训练日志末尾示例：

```text
Epoch 100/100 | train loss: 0.0037, train acc: 0.9985, val loss: 0.0260, val acc: 0.9912, val auc: 0.9994
```

## 7. 测试协议

主测试脚本：

```text
src/inference/inference_dataset.py
```

主要参数：

| 参数 | 含义 |
|---|---|
| `-w` | 权重路径。 |
| `-d` | 数据集，例如 `FF` 或 `CDF`。 |
| `-n` | 每个视频采样帧数，主实验使用 16。 |
| `-t` | FF++ 伪造类型，例如 `Deepfakes`、`Face2Face`、`FaceSwap`、`NeuralTextures`。 |
| `-c` | FF++ 压缩等级，例如 `c23` 或 `c40`。 |
| `--wavelet` | 小波基，例如 `sym2`、`haar`。 |
| `--mode` | DWT 边界模式，例如 `reflect`、`constant`。 |

当前测试是视频级 AUC，而不是帧级 AUC。

具体计算逻辑：

- 对每个视频抽取若干帧。
- 对每帧做人脸检测和裁剪。
- 对裁剪后的人脸输入模型，得到伪造概率。
- 同一帧如果检测到多张脸，使用最大伪造概率作为该帧得分。
- 将同一视频的帧得分取平均，得到一个视频级得分。
- 使用所有视频级得分计算 ROC AUC。

因此当前 AUC 是 video-level AUC，不是把所有帧当成独立样本计算的 frame-level AUC。

## 8. 主要测试命令

### 8.1 FF++ c23 基础性能测试

命令模板：

```bash
./.venv/bin/python src/inference/inference_dataset.py \
  -w <checkpoint> \
  -d FF \
  -n 16 \
  -t <Deepfakes|Face2Face|FaceSwap|NeuralTextures> \
  -c c23 \
  --wavelet <wavelet> \
  --mode <mode>
```

Sym2 + Reflect 示例：

```bash
./.venv/bin/python src/inference/inference_dataset.py \
  -w output/full_base_04_13_17_35_59/weights/87_0.9998_val.tar \
  -d FF \
  -n 16 \
  -t Deepfakes \
  -c c23 \
  --wavelet sym2 \
  --mode reflect
```

本地辅助脚本：

```text
scripts/run_sym2_c23_rerun.sh
```

### 8.2 Celeb-DF-v2 跨库泛化测试

命令模板：

```bash
./.venv/bin/python src/inference/inference_dataset.py \
  -w <checkpoint> \
  -d CDF \
  -n 16 \
  --wavelet <wavelet> \
  --mode <mode>
```

Haar + Reflect 示例：

```bash
./.venv/bin/python src/inference/inference_dataset.py \
  -w output/full_haar_reflect_04_19_18_27_50/weights/50_0.9998_val.tar \
  -d CDF \
  -n 16 \
  --wavelet haar \
  --mode reflect
```

### 8.3 FF++ c40 压缩鲁棒性测试

命令模板：

```bash
./.venv/bin/python src/inference/inference_dataset.py \
  -w <checkpoint> \
  -d FF \
  -n 16 \
  -t <Deepfakes|Face2Face|FaceSwap|NeuralTextures> \
  -c c40 \
  --wavelet <wavelet> \
  --mode <mode>
```

本地辅助脚本：

```text
scripts/run_c40_eval.sh
scripts/run_haar_generalization_robustness.sh
```

## 9. 当前本地日志可复核实验结果

以下结果均为 AUC，表中数值已经换算为百分数。

### 9.1 FF++ c23 基础检测性能

结果来源：

```text
output/full_base_04_13_17_35_59/eval_logs/
output/full_base_sbi_04_17_12_07_58/eval_logs/
output/full_sym2_constant_04_22_11_39_16/eval_logs/
output/full_haar_reflect_04_19_18_27_50/eval_logs/
```

| 方法 / 配置 | Deepfakes | Face2Face | FaceSwap | NeuralTextures | 平均 |
|---|---:|---:|---:|---:|---:|
| SBI baseline | 90.09 | 83.67 | 88.41 | 69.66 | 82.96 |
| Sym2 + Reflect | 98.24 | 93.22 | 84.56 | 89.89 | 91.48 |
| Sym2 + Constant | 98.21 | 94.15 | 88.31 | 87.87 | 92.14 |
| Haar + Reflect，当前本地 c23 日志 | 96.52 | 90.85 | 84.71 | 86.36 | 89.61 |

重要说明：

此前对话记录中曾记录 `Haar + Reflect` 在 FF++ c23 上平均 AUC 约为 `92.30`，但当前本地日志文件 `output/full_haar_reflect_04_19_18_27_50/eval_logs/` 中可复核的 c23 结果是上表的 `89.61`。如果论文最终选择 Haar + Reflect 作为主方法，则建议在最终写作前重新核对或重跑 Haar + Reflect 的 c23 测试。

### 9.2 16 帧与 32 帧测试对比

Sym2 + Reflect 曾分别使用 16 帧和 32 帧测试。

| 每视频帧数 | Deepfakes | Face2Face | FaceSwap | NeuralTextures | 平均 |
|---:|---:|---:|---:|---:|---:|
| 16 | 98.24 | 93.22 | 84.56 | 89.89 | 91.48 |
| 32 | 98.65 | 94.06 | 84.01 | 88.99 | 91.45 |

结论：

- 32 帧没有带来平均性能提升。
- 主实验使用 16 帧即可，计算成本更低，也更统一。

### 9.3 Celeb-DF-v2 跨库泛化结果

| 方法 / 配置 | Celeb-DF-v2 AUC |
|---|---:|
| SBI baseline | 69.44 |
| Sym2 + Reflect | 72.80 |
| Haar + Reflect | 76.45 |

解释：

- DWT 增强方法在 Celeb-DF-v2 上优于 SBI baseline。
- Haar + Reflect 在当前已有泛化实验中表现最好。
- 该实验可以支撑“跨库泛化能力提升”的结论，但不能扩展为对所有跨库数据集都有效，因为 DFDC、DFDCP、FFIW10K 和完整 DeeperForensics-1.0 没有完成测试。

### 9.4 FF++ c40 压缩鲁棒性结果

| 方法 / 配置 | Deepfakes | Face2Face | FaceSwap | NeuralTextures | 平均 |
|---|---:|---:|---:|---:|---:|
| SBI baseline | 65.69 | 64.94 | 70.24 | 62.33 | 65.80 |
| Sym2 + Reflect | 77.45 | 73.18 | 61.97 | 71.79 | 71.10 |
| Haar + Reflect | 77.12 | 72.87 | 66.10 | 69.28 | 71.34 |

解释：

- 训练使用 FF++ c23，测试使用 FF++ c40，因此这是压缩鲁棒性实验。
- DWT 增强方法在 c40 平均 AUC 上明显优于 SBI baseline。
- Haar + Reflect 的 c40 平均值略高于 Sym2 + Reflect。
- 论文中应称为“压缩鲁棒性”或“强压缩条件下的鲁棒性”，不要泛化成所有类型的鲁棒性。

## 10. 论文建议实验表

### 10.1 基础性能表

建议使用 FF++ c23 四种伪造方法的 AUC：

```text
SBI baseline
Sym2 + Reflect
Sym2 + Constant
Haar + Reflect
```

但如果最终主方法选择 Haar + Reflect，需要先解决 Haar + Reflect 当前 c23 日志与此前记录不一致的问题。

### 10.2 跨库泛化表

可使用：

```text
SBI baseline:       69.44
Sym2 + Reflect:     72.80
Haar + Reflect:     76.45
```

数据集为 Celeb-DF-v2。

### 10.3 压缩鲁棒性表

可使用：

```text
SBI baseline c40 平均:       65.80
Sym2 + Reflect c40 平均:     71.10
Haar + Reflect c40 平均:     71.34
```

数据集为 FF++ c40。

### 10.4 DWT 消融实验表

可以使用以下配置：

```text
SBI baseline
Sym2 + Reflect
Sym2 + Constant
Haar + Reflect
```

如果论文最终选择 Haar + Reflect 作为主方法，则方法章节、实验设置和结果分析都要统一改成：通过消融对比发现 Haar + Reflect 在当前实验协议下表现较好，因此采用该配置作为最终方法。

## 11. 论文写作建议

### 11.1 关于泛化能力

建议表述：

```text
为评估模型的跨库泛化能力，本文将 FaceForensics++ c23 上训练得到的模型直接迁移到 Celeb-DF-v2 官方测试集上进行测试，测试过程中不进行额外微调。实验结果表明，引入 DWT 频域增强后的方法在 Celeb-DF-v2 上取得了高于 SBI 基线的 AUC，说明该方法在跨数据集场景下具有更好的泛化能力。
```

避免表述：

```text
该方法可以泛化到所有真实场景深度伪造数据集。
```

### 11.2 关于鲁棒性

建议表述：

```text
为评估模型在视频压缩退化条件下的鲁棒性，本文使用 FaceForensics++ c23 训练模型，并在对应的 c40 强压缩测试视频上进行测试。实验结果显示，DWT 频域增强方法在 c40 条件下的平均 AUC 高于 SBI 基线，说明该方法在强压缩条件下具有更好的鲁棒性。
```

避免表述：

```text
该方法对所有扰动和真实场景退化都具有鲁棒性。
```

### 11.3 关于数据集限制

建议表述：

```text
受限于数据集申请周期、存储空间和实验成本，本文主要使用 Celeb-DF-v2 进行跨库泛化测试，并使用 FaceForensics++ c40 进行压缩鲁棒性测试。DFDC、DFDCP、FFIW10K 和 DeeperForensics-1.0 等更大规模数据集将在后续工作中进一步补充验证。
```

### 11.4 关于 AUC 与 ACC

建议表述：

```text
本文主要采用 AUC 作为评价指标。AUC 能够衡量模型在不同分类阈值下区分真实与伪造样本的能力，在深度伪造检测任务中应用较为广泛。相比之下，ACC 依赖固定阈值，容易受到样本分布和模型分数校准的影响，因此本文以 AUC 作为主要评价指标。
```

## 12. 当前重要风险与注意事项

### 12.1 SBI baseline 推理路径问题

当前 `src/inference/inference_dataset.py` 在推理时会执行 DWT 相关预处理逻辑，并没有单独区分 SBI baseline 的原始 RGB 推理路径。

这意味着：

- SBI baseline 的训练确实使用了 `--variant sbi`。
- 但当前测试脚本在推理阶段仍可能应用 DWT 风格预处理。
- 如果需要严格复现 SBI baseline，应增加无 DWT 的推理分支，并重跑 SBI 测试。

论文中如果不处理这一点，不建议写成“严格复现 SBI 原文”，而应更谨慎地称为“本文实验协议下的 SBI 基线”。

### 12.2 Haar + Reflect c23 结果不一致

当前存在一个需要复查的问题：

```text
此前对话记录：Haar + Reflect FF++ c23 平均 AUC 约为 92.30
当前本地日志：Haar + Reflect FF++ c23 平均 AUC 为 89.61
```

如果论文最终主方法选择 Haar + Reflect，则建议最终提交前做以下检查：

```text
1. 用指定权重重新跑 Haar + Reflect 的 FF++ c23 四个子集。
2. 确认测试权重到底是 50_0.9998_val.tar 还是其他 epoch 权重。
3. 确认推理时使用的参数确实是 --wavelet haar --mode reflect。
4. 用新日志替换论文表格中的 Haar + Reflect c23 结果。
```

### 12.3 c40 只能代表压缩鲁棒性

FF++ c40 可以作为压缩鲁棒性测试，但不能代表所有鲁棒性场景。

推荐使用以下表述：

```text
压缩鲁棒性
强压缩条件下的鲁棒性
视频压缩退化场景下的鲁棒性
```

不推荐使用：

```text
完整鲁棒性
所有扰动下的鲁棒性
真实世界所有场景鲁棒性
```

## 13. 当前还剩什么工作

如果论文第四章已经简化为以下内容：

```text
1. FF++ c23 基础性能实验
2. Celeb-DF-v2 跨库泛化实验
3. FF++ c40 压缩鲁棒性实验
4. DWT 配置消融实验
```

那么严格来说，新的大规模训练和测试已经不是必须的。

建议最终写作前完成：

```text
1. 决定最终主方法到底采用 Sym2 + Reflect、Sym2 + Constant 还是 Haar + Reflect。
2. 如果采用 Haar + Reflect，复查 Haar + Reflect 的 FF++ c23 结果。
3. 如果需要严格基线对比，增加 SBI 无 DWT 推理路径并重跑 SBI。
4. 修改论文第四章，删除 DFDC、DFDCP、FFIW10K、完整 DeeperForensics-1.0 等未完成实验。
5. 全文统一使用 AUC 作为主要指标。
```

## 14. Git 与文件打包说明

创建本文档时，Git 中仍有以下未跟踪文件：

```text
.codex
archived_sessions/
output/full_haar_reflect_04_19_18_27_50/eval_logs/cdf_eval.log
output/full_haar_reflect_04_19_18_27_50/eval_logs/ffpp_deepfakes_c40_eval.log
output/full_haar_reflect_04_19_18_27_50/eval_logs/ffpp_face2face_c40_eval.log
output/full_haar_reflect_04_19_18_27_50/eval_logs/ffpp_faceswap_c40_eval.log
output/full_haar_reflect_04_19_18_27_50/eval_logs/ffpp_neuraltextures_c40_eval.log
```

大文件情况：

```text
data/FaceForensics++    约 16G
data/Celeb-DF-v2        约 9.5G
output/                 约 4.9G
```

`scripts/` 中的脚本主要是本地自动化辅助脚本，此前用户表示不一定需要上传 Git。

## 15. 复现实验的最小命令集合

### 15.1 Sym2 + Reflect 的 FF++ c23 测试

```bash
for fake_type in Deepfakes Face2Face FaceSwap NeuralTextures; do
  ./.venv/bin/python src/inference/inference_dataset.py \
    -w output/full_base_04_13_17_35_59/weights/87_0.9998_val.tar \
    -d FF \
    -n 16 \
    -t "$fake_type" \
    -c c23 \
    --wavelet sym2 \
    --mode reflect
done
```

### 15.2 SBI 的 FF++ c23 测试

```bash
for fake_type in Deepfakes Face2Face FaceSwap NeuralTextures; do
  ./.venv/bin/python src/inference/inference_dataset.py \
    -w output/full_base_sbi_04_17_12_07_58/weights/99_0.9999_val.tar \
    -d FF \
    -n 16 \
    -t "$fake_type" \
    -c c23 \
    --wavelet sym2 \
    --mode reflect
done
```

注意：该命令仍受“当前推理脚本没有无 DWT SBI 分支”的影响。

### 15.3 Haar + Reflect 的 Celeb-DF-v2 和 c40 测试

```bash
./.venv/bin/python src/inference/inference_dataset.py \
  -w output/full_haar_reflect_04_19_18_27_50/weights/50_0.9998_val.tar \
  -d CDF \
  -n 16 \
  --wavelet haar \
  --mode reflect

for fake_type in Deepfakes Face2Face FaceSwap NeuralTextures; do
  ./.venv/bin/python src/inference/inference_dataset.py \
    -w output/full_haar_reflect_04_19_18_27_50/weights/50_0.9998_val.tar \
    -d FF \
    -n 16 \
    -t "$fake_type" \
    -c c40 \
    --wavelet haar \
    --mode reflect
done
```

## 16. 总体结论

当前项目已经完成了一个较完整的端到端实验流程：

- 下载并整理了 FaceForensics++ c23、c40 数据。
- 下载并整理了 Celeb-DF-v2 数据。
- 训练了 SBI baseline 和多个 DWT 增强配置。
- 完成了 FF++ c23 基础性能测试。
- 完成了 Celeb-DF-v2 跨库泛化测试。
- 完成了 FF++ c40 压缩鲁棒性测试。
- 初步完成了 DWT 配置消融实验。

论文中最稳妥的总体表述是：

```text
本文在 SBI 框架基础上引入 DWT 频域增强，通过 FaceForensics++ c23、Celeb-DF-v2 和 FaceForensics++ c40 上的实验结果表明，频域增强方法在当前实验协议下能够提升模型的跨库泛化能力和强压缩条件下的鲁棒性。
```

最终提交前最需要确认的两个问题：

```text
1. 是否需要严格实现 SBI baseline 的无 DWT 推理路径。
2. 是否需要重新确认 Haar + Reflect 在 FF++ c23 上的消融结果。
```
