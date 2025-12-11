# 火灾监测 MoE 项目说明

本项目实现了 UMONS 火灾监测任务的完整训练与推理流程，严格禁止使用任何预训练模型。我们构建了三个互补的“专家”：

- **专家 A（HistMLP）**：基于 HSV/YCbCr 直方图与边缘密度，强调火焰亮度和暖色先验。
- **专家 B（FFTMLP）**：基于频域能量分布，捕获烟雾模糊、纹理变化等特征。
- **专家 C（SmallFireCNN）**：从零开始训练的轻量 CNN，学习剩余模式。

三个专家的输出经过 MoE（Mixture of Experts）门控层自适应融合，以满足作业要求的“传统特征 + 深度模型 + 集成”方案。

## 目录结构概览

```
dataset/
  train/{fire,start_fire,no_fire}/
  val/{fire,start_fire,no_fire}/
  test/
code/
  datasets.py          # 数据集/常量定义
  features.py          # 直方图 + FFT 特征提取与缓存
  utils.py             # 工具函数（随机种子、wandb、检查点）
  models/
    hist_model.py      # 专家 A MLP
    fft_model.py       # 专家 B MLP
    cnn_model.py       # 专家 C CNN
    moe.py             # MoE 门控网络
  train_hist.py        # 训练专家 A，并记录 wandb
  train_fft.py         # 训练专家 B
  train_cnn.py         # 训练专家 C，记录增强样本
  train_moe.py         # 冻结专家后训练 MoE 门控
  inference.py         # 生成 Kaggle 提交用 result.csv
  ablation.py          # 在 val 集跑消融实验
  configs/             # YAML 配置，方便统一调参
organize_data.py       # 已运行，用于构建 dataset/ 结构
requirements.txt
README.md
```

所有模型权重与日志保存于 `artifacts/`，特征缓存位于 `.feature_cache/`。

## 环境配置

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

如需使用 Weights & Biases，提前执行 `wandb login` 或配置 `WANDB_API_KEY`。所有训练脚本均可使用 `--use-wandb`、`--wandb-project`、`--run-name` 等参数。

## 数据准备

1. 将官方 ZIP 放在仓库根目录。
2. 运行 `python organize_data.py`，自动解压到 `dataset/`，并删除文件名带 `copy/copìe` 的重复样本。

当前统计：train 集 fire/start_fire/no_fire = 1309/614/1030；val 集 = 18/82/126；test 集（无标签）= 1541。
## 配置化调参

所有脚本都支持 --config path/to.yaml，可以把常用超参写进 YAML，避免频繁敲命令行。示例文件位于 configs/：

`bash
python code/train_hist.py --config configs/train_hist.yaml
python code/train_cnn.py --config configs/train_cnn.yaml --use-wandb  # 命令行参数仍可覆盖配置
python code/inference.py --config configs/inference.yaml
`

YAML 顶层键名需与脚本参数一致，例如：

`yaml
# configs/train_moe.yaml
data_root: dataset
cache_dir: .feature_cache
hist_checkpoint: artifacts/hist_expert.pth
fft_checkpoint: artifacts/fft_expert.pth
cnn_checkpoint: artifacts/cnn_expert.pth
`

如果同一参数同时在 YAML 和命令行中出现，以命令行为准；--config 支持相对或绝对路径。



## 训练流程

在仓库根目录依次执行：

1. **专家 A（直方图）**
   ```bash
   python code/train_hist.py --use-wandb
   ```
   读取 `.feature_cache/hist/` 中的 HSV/YCbCr 直方图特征，训练 `HistMLP`。

2. **专家 B（FFT）**
   ```bash
   python code/train_fft.py --use-wandb
   ```
   利用 `.feature_cache/fft/` 中的频域统计，训练 `FFTMLP`，偏向捕捉烟雾/模糊模式。

3. **专家 C（CNN）**
   ```bash
   python code/train_cnn.py --use-wandb --model-type small --epochs 50 --batch-size 64
   # 若希望启用手写 ResNet18，可改为 --model-type resnet18
   ```
   - `small`：轻量 CNN，收敛快、便于调参与消融。
   - `resnet18`：我们手写的 ResNet18 风格骨干（`code/models/cnn_model.py`），完全随机初始化，更适合 3090 级别算力跑更深层实验。

4. **MoE 融合层**
   ```bash
   python code/train_moe.py --use-wandb
   ```
   获取前三位专家的权重（默认 rtifacts/*.pth），冻结专家参数只训练门控网络，为每个样本自适应加权；门控输入包含三位专家的 logits 与 softmax 最大概率（置信度），鼓励“更自信”的专家在融合时获得更高权重。

脚本均支持 `--epochs`、`--batch-size`、`--lr`、`--num-workers`、`--artifacts-dir` 等参数，方便调度与打包。

## 消融实验

```bash
python code/ablation.py
```

在 `dataset/val` 上分别评估：
- 单专家（hist / fft / cnn）
- 任意两个专家简单平均
- 三专家平均
- MoE（若存在 `artifacts/moe_gating.pth`）

结果会打印在终端，并写入 `ablation.json`，可直接用于实验报告。

## 生成 Kaggle 提交 CSV

```bash
python code/inference.py --data-root dataset --output result.csv
```

常用参数：
- `--test-dir`: 若要在隐藏 `TestData/` 上推理，指向对应目录。
- `--moe-checkpoint`: 可选，若缺失则默认平均三位专家的 logits。

脚本会生成 `result.csv`，格式为 `ImageID,Label`，并按照竞赛要求映射 `fire→0`、`no_fire→1`、`start_fire→2`。

## 打包提交

- 保留每次训练的控制台日志（或 wandb 链接）作为“训练日志”。
- 将 `artifacts/` 下的 `.pth` 权重、`ablation.json`、`result.csv` 与实验报告、README 一起打包，满足 e-learning 上传要求。
- 报告中可直接引用 wandb 的 Loss/Accuracy 曲线及消融统计。

## 后续建议

1. 跑完四个训练脚本并确认 wandb 日志与权重齐全。
2. 使用 `ablation.py` 获取实验数据写入报告。
3. 分别对 `dataset/test` 与官方 `TestData` 执行 `inference.py`，得到本地验证与提交文件。
4. 汇总代码、日志、权重、报告与 README，制作最终 ZIP 包。
