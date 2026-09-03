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
└── requirements.txt         # Python 依赖
```

## 环境安装

```bash
# 推荐 conda 创建环境
conda create -n dachuang python=3.11 -y
conda activate dachuang
pip install -r requirements.txt
```

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
├──fold1
|   ├── best_model.pt          # 最优模型权重
|   ├── config.json            # 实验配置
|   ├── confusion_matrix.png  # 混淆矩阵
├──fold2
|   ├── ...
├── ...
└── summary.json           # 5 折汇总（acc/f1 均值±std）
```
每次实验后自动追加一行记录到 `{output_dir}/EXPERIMENTS.md`。

