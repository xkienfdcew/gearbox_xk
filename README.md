# 齿轮箱故障检测（基于声学信号的深度学习故障诊断）

大学生创新创业训练计划项目。通过对电机/齿轮箱声学信号提取 **梅尔频谱 + MFCC** 特征，使用深度学习模型（CNN / LSTM / ResNet / 双分支结构）实现 8 种故障状态的自动识别，并系统研究了损失函数、特征融合方式与数据增强策略对诊断性能的影响。

## 目录

- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [数据与特征](#数据与特征)
- [脚本总览](#脚本总览)
- [训练流程](#训练流程)
- [实验设计（三组对比实验）](#实验设计三组对比实验)
- [模型与损失](#模型与损失)
- [输出文件说明](#输出文件说明)
- [常见问题](#常见问题)

## 项目结构

```
D:/大创/
├── data/                    # 原始 .mat 数据（8 类故障，按类别分文件夹）
├── features/                # 预构建特征集（full_v*.pt / label_map_v*.pt）
├── output/                  # 训练输出（模型、summary、图片、EXPERIMENTS.md）
├── logs/                    # 实验脚本日志
├── src/
│   ├── dataset/             # 数据处理
│   │   ├── feature.py       #   特征构建（切片/提取/归一化/保存）
│   │   ├── processor.py     #   特征提取器（mel/MFCC + 信号级增强）
│   │   └── mydataset.py     #   在线频谱增强数据集（SpecAugment）
│   ├── net/                 # 模型实现
│   │   ├── base.py          #   基础模块（SE/CBAM/残差块等）
│   │   ├── cnn.py           #   CNN 系列
│   │   ├── lstm.py          #   LSTM 系列
│   │   └── resnet.py        #   ResNet 系列
│   ├── net_creator.py       # ★ 模型工厂（注册表/dual 包装/分类头）
│   ├── train/
│   │   ├── trainer.py       #   训练循环（早停/mixup/warmup）
│   │   └── losses.py        #   损失函数（CE/Focal/CosFace/ArcFace/SubArc）
│   └── utils.py             # 工具
├── scripts/                 # ★ 脚本（见下表）
├── report/                  # 论文/报告相关
├── docs/                    # 参考文献 PDF
└── requirements.txt         # Python 依赖
```

## 环境安装

```bash
# 推荐 conda 创建环境
conda create -n dachuang python=3.11 -y
conda activate dachuang
pip install -r requirements.txt
```

本机 Windows 已备好虚拟环境 `.conda/`（内含 python.exe，脚本自动探测使用）。Linux 服务器部署请参考 [scripts/run_experiments_all.sh](#实验总脚本run_experiments_allsh) 顶部注释。

## 数据与特征

### 数据集
- 渥太华大学电机故障声学数据集，8 类故障状态（BR/FB/HH/KA/RM/RU/SW/VU）
- 每类在 16 种转速下采集，共 128 条原始信号
- `.mat` 格式，采样率 42000 Hz

### 特征构建

特征 = 128 阶 mel 频谱 + 42 阶 MFCC + 一阶差分 + 二阶差分 = **254 维 × 时间帧**

```bash
# 全量特征集（供交叉验证使用，推荐）
python scripts/build_features.py --all --no_aug --version 1.0 --piece_sec 1 --tag "1秒数据"
python scripts/build_features.py --all --no_aug --version 2.0 --piece_sec 2 --tag "2秒数据"

# 常用参数
--n_mels 128      # mel 滤波器个数
--n_mfcc 42       # MFCC 系数个数
--piece_sec 1     # 切片长度（秒）
--overlap 0.5     # 切片重叠率
--use_pcen        # 用 PCEN 替代 dB 归一化
--normalize       # 按通道 z-score（统计量来自训练集）
```

输出：`features/full_v*.pt`（含特征、标签、信号归属 id）+ `features/DATASETS.md`（版本记录）。

## 脚本总览

| 脚本 | 作用 | 状态 |
|------|------|------|
| `build_features.py` | 从 .mat 构建特征集（支持 --all 全量模式） | ✅ 推荐 |
| `cv_train.py` | ★ 交叉验证训练主脚本（5 折 CV，自动记录结果） | ✅ 推荐 |
| `run_experiments_all.sh` | ★ 实验总脚本（三组实验、并行、断点续跑） | ✅ 推荐 |
| `run_tasks.py` | 并行调度器（多进程训练、按显存估算并发） | ✅ 辅助 |
| `generate_exp_figures.py` | 按实验生成图表（热力图/条形图） | ✅ 辅助 |
| `train.py` | 旧版单次训练（train/test 划分，不含 CV） | ⚠️ 旧版 |
| `eval.py` | 旧版模型评估 | ⚠️ 旧版 |
| `generate_figures.py` | 旧版图表生成 | ⚠️ 旧版 |
| `pyside6.py` | PySide6 图形界面（开发中） | 🔧 开发中 |

## 训练流程

### 核心：`cv_train.py`

5 折交叉验证训练，按**原始信号分组**（同一条信号的相邻切片只进同一折，杜绝数据泄露）。

```bash
# 基本用法
python scripts/cv_train.py --cv 5 --model dual_lstm --features_version 1.0

# 常用组合
python scripts/cv_train.py --model dual_cnn_se --loss arcface \
    --arcface_s 16 --arcface_m 0.3 --online_aug --mixup 0.7

# 查看全部参数
python scripts/cv_train.py --help
```

**关键参数**：

| 类别 | 参数 | 说明 |
|------|------|------|
| 数据 | `--features_version` | 特征集版本（默认最新 full_vX.pt） |
| 模型 | `--model` | 模型名（见 [模型与损失](#模型与损失)） |
| 损失 | `--loss` | cross_entropy / focal / cosface / arcface / sub_arcface |
| 损失参数 | `--cosface_s/m`、`--arcface_s/m`、`--sub_arcface_s/m/k` | 大间隔损失超参 |
| 增强 | `--online_aug` | 在线频谱增强（SpecAugment + 时间翻转） |
| | `--mixup 0.7` | Mixup 数据增强 |
| 训练 | `--epochs 200 --patience 30 --lr 1e-3` | 常规训练超参 |
| 调度 | `--scheduler cosine/plateau` | 学习率调度器 |
| 其他 | `--output_dir` | 输出目录（默认 output/，实验分离时用） |

**输出**（每个实验一个目录）：
```
output/{model}_v{ver}_cv5/
├── best_model.pt          # 最优模型权重
├── config.json            # 实验配置
├── confusion_matrix.png   # 混淆矩阵
└── summary.json           # 5 折汇总（acc/f1 均值±std）
```
每次实验后自动追加一行记录到 `{output_dir}/EXPERIMENTS.md`。

### 多卡/并行：`run_tasks.py`

并发运行多个训练任务，充分利用服务器 GPU 显存。

```bash
# 任务文件每行一个: <tag> <cv_train 参数...>
python scripts/run_tasks.py --tasks tasks.txt \
    --prog_dir output/exp/.progress --log logs/exp.log \
    --jobs 8          # 0 = 按 GPU 显存自动估算
```

### 实验总脚本：`run_experiments_all.sh`

一键跑完论文三组实验，支持**并行**与**断点续跑**。

```bash
# 全部实验（自动按显存估算并发）
bash scripts/run_experiments_all.sh

# 指定并发 8，只跑实验二（续跑）
JOBS=8 bash scripts/run_experiments_all.sh 2

# Linux 服务器（指定 conda python）
PYTHON=/path/to/python JOBS=8 nohup bash run_experiments_all.sh > logs/all.log 2>&1 &
```

断点续跑：每个实验完成后在 `{实验目录}/.progress/` 生成标记，中断后重跑同命令自动跳过已完成部分。

## 实验设计（三组对比实验）

详见 Obsidian 笔记 `大创/实验设计.md`，概要：

| 实验 | 目标 | 实验数 |
|------|------|--------|
| 实验一 | 探索 dual_lstm/dual_cnn 上 cosface/arcface/sub_arcface 的最优参数（77 组网格） | 308 |
| 实验二 | 8 模型（含双分支）× 4 损失 × 2 数据集的性能对比 | 64 |
| 实验三 | top3 模型上评估 mixup 与在线增强的 4 种组合 | 24 |

**输出结构**（每个实验独立目录，含模型/图表/记录）：
```
output/
├── exp1_loss_param/{ds1,ds2}/   # + figures/ + best_params_ds{1,2}.json
├── exp2_model_loss/{ds1,ds2}/
├── exp3_aug_mixup/{ds1,ds2}/
└── (全局 EXPERIMENTS.md 可选)
```

### 图表生成：`generate_exp_figures.py`

```bash
python scripts/generate_exp_figures.py --exp 1 --out_dir output/exp1_loss_param
python scripts/generate_exp_figures.py --exp 2 --out_dir output/exp2_model_loss
python scripts/generate_exp_figures.py --exp 3 --out_dir output/exp3_aug_mixup
```
实验一额外输出 `best_params_ds{1,2}.json`（供实验二/三自动读取最优参数）。

## 模型与损失

### 模型（`src/net_creator.py` 注册表 + `--model` 参数）

| 系列 | 单分支 | 双分支（dual_ 前缀） |
|------|--------|---------------------|
| CNN | cnn / cnn2 / cnn_se / cnn2_se | dual_cnn / dual_cnn2_se |
| CNN+CBAM | cnn_cbam / cnn2_cbam | dual_cnn_cbam |
| LSTM | lstm / lstm_se | dual_lstm / dual_lstm_se |
| ResNet | resnet / resnet_se | dual_resnet_se |

双分支 = mel 频谱与 MFCC 分别由独立编码器提取后拼接融合。
`create_model('dual_cnn_se', input_size=(128,126))` 传元组即触发双分支。

### 损失（`src/train/losses.py`）

| 损失 | 最优参数（网格搜索结论） | 说明 |
|------|------------------------|------|
| cross_entropy | - | 基准 |
| focal | gamma=2.0 | 类别不平衡 |
| cosface | s=64, m=0.2 | 余弦间隔 |
| arcface | s=16, m=0.3 | 角度间隔 |
| sub_arcface | s=128, m=0.3, K=4 | 多子中心（应对工况差异） |

## 输出文件说明

| 文件 | 内容 |
|------|------|
| `output/EXPERIMENTS.md` | 所有实验记录汇总表（时间/模型/损失/参数/Acc/F1/tag） |
| `output/{run}/config.json` | 实验完整配置（可复现） |
| `output/{run}/summary.json` | 5 折 CV 汇总（含每折明细） |
| `features/DATASETS.md` | 特征集版本记录 |
| `report/figures/` | 论文用图 |

## 常见问题

1. **找不到 python**：脚本自动探测 `.conda/python.exe`（Win）/`.conda/bin/python`（Linux），也可用 `PYTHON=` 环境变量指定。
2. **实验一没跑完，能先跑实验二吗**：不能，实验二/三依赖实验一生成的 `best_params_ds*.json`，需先完成实验一（或手动准备该文件）。
3. **显存不足（OOM）**：降低并发 `JOBS=N`，或减小 `--batch_size`。
4. **脚本在 Linux 报 `\r` 错误**：确保脚本为 LF 换行（本仓库已处理，若从 Windows 重新编辑需转换）。
5. **CRLF 被截断**：若从 Windows 重新编辑需转换。
