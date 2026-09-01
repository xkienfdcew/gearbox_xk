"""
交叉验证训练脚本
================
在【信号层面】做 K 折交叉验证，每折用折内训练集早停验证，得到可靠的性能估计。

与 train.py 的区别：
    1. 使用 build_features.py --all 构建的全量特征集 full_vX.pt
       （含 signal_ids，记录每个样本来自哪条信号）
    2. 按信号分组做 StratifiedGroupKFold——同一信号的所有片段只会出现在
       同一折，避免相邻片段（50% 重叠切分）在训练/验证间信息泄露
    3. 每折用【训练折】的统计量做 z-score 归一化，验证折不参与统计
    4. 验证集不再等于测试集，早停指标可信
    5. 最终报告每折指标 + 均值±标准差

用法:
    # 第一步：构建全量特征集（不切分、不归一化）
    python scripts/build_features.py --all --no_aug --version 6.0 --tag "cv全量"

    # 第二步：5 折交叉验证
    python scripts/cv_train.py --cv 5 --model dual_cnn_se --features_version 6.0 --online_aug
    python scripts/cv_train.py --cv 5 --model cnn_se --features_version 6.0 --loss cosface
    python scripts/cv_train.py --help
"""

import os
import re
import sys
import json
from datetime import datetime

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, worker_init_fn
from src.net_creator import (create_model, FEAT_DIM_MAP, MODEL_REGISTRY,
                             ModelWithHead, get_feat_dim)
from src.train.trainer import train_model, evaluate_model, create_optimizer, create_scheduler
from src.train.losses import CenterLoss, FocalLoss, CosFaceHead, ArcFaceHead, SubCenterArcFaceHead
from src.dataset.mydataset import AugDataset


def _list_versions(features_dir):
    """列出特征目录中可用的 full 版本号。"""
    if not os.path.isdir(features_dir):
        return []
    pattern = re.compile(r"^full_v(\d+\.\d+)\.pt$")
    versions = set()
    for fname in os.listdir(features_dir):
        m = pattern.match(fname)
        if m:
            versions.add(m.group(1))
    return sorted(versions, key=lambda v: float(v))


def load_full_dataset(features_dir, version):
    """加载全量特征集: (features, labels, signal_ids), label_map"""
    full_path = os.path.join(features_dir, f"full_v{version}.pt")
    map_path = os.path.join(features_dir, f"label_map_v{version}.pt")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"全量特征集不存在: {full_path}\n"
            f"请先运行: python scripts/build_features.py --all --no_aug --version {version}"
        )
    feats, labels, signal_ids = torch.load(full_path, weights_only=False)
    label_map = torch.load(map_path, weights_only=False) if os.path.exists(map_path) else None
    return (feats, labels, signal_ids), label_map


def _infer_mel_mfcc(feat_channels, n_mfcc=42):
    """从拼接特征通道数推断 mel / MFCC 分支的通道划分。

    特征 = mel(n_mels) + MFCC(n_mfcc) + Δ + Δ²，MFCC 部分固定为 n_mfcc*3。
    190 通道 → (64, 126)；254 通道 → (128, 126)。
    """
    mfcc_ch = n_mfcc * 3
    mel_ch = feat_channels - mfcc_ch
    if mel_ch <= 0:
        raise click.UsageError(f"无法推断 mel 通道数: 特征 {feat_channels} 通道, MFCC 部分 {mfcc_ch} 通道")
    return mel_ch, mfcc_ch


def build_net_and_losses(model, is_dual, model_kwarg, loss_type, num_classes,
                         input_shape, class_weights, device, cosface_s, cosface_m,
                         arcface_s, arcface_m, sub_arcface_s, sub_arcface_m, sub_arcface_k):
    """构建 backbone + 分类头 + 损失函数。

    feat_dim 用 get_feat_dim 从模型实例动态推断（CNN 的 out_features 语义
    与 LSTM 池化后维度不同，不能用 model_kwarg 里的 out_features 直接当 feat_dim）。
    input_size 传 2 元组即自动触发 dual 模式（create_model 内部判断）。
    """
    backbone = create_model(model, num_classes=num_classes, **model_kwarg)
    feat_dim = get_feat_dim(backbone, input_shape=input_shape)

    if loss_type == "cosface":
        head = CosFaceHead(feat_dim, num_classes, s=cosface_s, m=cosface_m)
        head_type = "CosFaceHead"
    elif loss_type == "arcface":
        head = ArcFaceHead(feat_dim, num_classes, s=arcface_s, m=arcface_m)
        head_type = "ArcFaceHead"
    elif loss_type == "sub_arcface":
        head = SubCenterArcFaceHead(feat_dim, num_classes, K=sub_arcface_k,
                                    s=sub_arcface_s, m=sub_arcface_m)
        head_type = "SubCenterArcFaceHead"
    else:
        head = nn.Linear(feat_dim, num_classes)
        head_type = "Linear"

    net = ModelWithHead(backbone, head).to(device)

    # 主损失
    if loss_type == "focal":
        alpha = class_weights.to(device) if class_weights is not None else None
        criterion = FocalLoss(gamma=2.0, alpha=alpha)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            reduction='mean'
        )
    return net, criterion, head_type, feat_dim


def _append_experiment_log(summary_dir, summary):
    """把一次 CV 实验的结果追加记录到 output/EXPERIMENTS.md（自动建表头）。"""
    exp_path = os.path.join(PROJECT_ROOT, "output", "EXPERIMENTS.md")
    header = (
        "| 时间 | 运行名 | 模型 | 数据集 | 损失 | 参数(s,m,K) | 调度器 | online_aug | mixup | Acc(%)±std | F1(%) | 目录 | tag |\n"
        "|------|--------|------|--------|------|------------|--------|-----------|------|------------|-------|------|-----|\n"
    )
    lc = summary.get("loss_config", {})
    tc = summary.get("train_config", {})
    loss = summary.get("loss_type", "")
    if loss == "cosface":
        param = f"{lc.get('cosface_s')},{lc.get('cosface_m')},-"
    elif loss == "arcface":
        param = f"{lc.get('arcface_s')},{lc.get('arcface_m')},-"
    elif loss == "sub_arcface":
        param = f"{lc.get('sub_arcface_s')},{lc.get('sub_arcface_m')},{lc.get('sub_arcface_k')}"
    else:
        param = "-,-,-"
    ts = summary.get('timestamp', '')[:16].replace('T', ' ')
    row = (
        f"| {ts} | {summary.get('run','')} | {summary.get('model','')} "
        f"| v{summary.get('features_version','')} | {loss} | {param} "
        f"| {tc.get('scheduler','-')} | {tc.get('online_aug', False)} | {tc.get('mixup_alpha','-')} "
        f"| {summary.get('cv_acc_mean','')}±{summary.get('cv_acc_std','')} "
        f"| {summary.get('cv_f1_mean','')} | {os.path.basename(summary_dir)} "
        f"| {summary.get('tag','') or '-'} |\n"
    )
    os.makedirs(os.path.dirname(exp_path), exist_ok=True)
    if not os.path.exists(exp_path):
        with open(exp_path, "w", encoding="utf-8") as f:
            f.write("# 交叉验证实验结果记录\n\n"
                    "> 每次运行 scripts/cv_train.py 自动追加，所有实验的准确率汇总\n\n"
                    + header)
    with open(exp_path, "a", encoding="utf-8") as f:
        f.write(row)
    return exp_path


@click.command()
# ── 数据 ──
@click.option("--features_dir", default="features", help="特征集目录 (默认: features/)")
@click.option("--features_version", default=None, type=str,
              help="全量特征集版本号 (默认自动取最新 full_vX.pt)")
@click.option("--cv", "cv_folds", default=5, type=int,
              help="交叉验证折数 (默认: 5)")
# ── 模型 ──
@click.option("--model", default="cnn", type=str,
              help="模型名称（支持 dual_ 前缀，如 dual_cnn_se）")
# ── 训练超参 ──
@click.option("--epochs", default=200, type=int, help="最大训练轮数 (默认: 200)")
@click.option("--batch_size", default=64, type=int, help="批次大小 (默认: 64)")
@click.option("--lr", default=1e-3, type=float, help="学习率 (默认: 1e-3)")
@click.option("--weight_decay", default=1e-2, type=float, help="权重衰减 (默认: 1e-2)")
@click.option("--patience", default=30, type=int, help="早停耐心值 (默认: 30)")
@click.option("--grad_clip", default=None, type=float, help="梯度裁剪最大范数")
# ── 优化器 & 调度器 ──
@click.option("--optim", "optimizer_type", default="adamw",
              type=click.Choice(["adamw", "adam", "sgd"]), help="优化器")
@click.option("--scheduler", "scheduler_type", default="cosine",
              type=click.Choice(["cosine", "plateau", "none"]), help="学习率调度器")
@click.option("--t_0", default=15, type=int, help="CosineAnnealing T_0")
@click.option("--t_mult", default=2, type=int, help="CosineAnnealing T_mult")
# ── 损失 ──
@click.option("--loss", "loss_type", default="cross_entropy",
              type=click.Choice(["cross_entropy", "focal", "cosface", "arcface", "sub_arcface"]),
              help="损失函数类型")
@click.option("--center_loss", is_flag=True, help="启用 CenterLoss 辅助损失")
@click.option("--center_lambda", default=0.003, type=float, help="CenterLoss 权重")
@click.option("--class_weight", is_flag=True, help="按训练折分布自动计算类别权重")
# ── 正则化 & 增强 ──
@click.option("--online_aug", is_flag=True, help="启用在线频谱增强 (SpecAugment + 翻转)")
@click.option("--mixup", "mixup_alpha", default=None, type=float, help="Mixup alpha")
@click.option("--cosface_s", default=30.0, type=float)
@click.option("--cosface_m", default=0.20, type=float)
@click.option("--arcface_s", default=30.0, type=float)
@click.option("--arcface_m", default=0.30, type=float)
@click.option("--sub_arcface_s", default=30.0, type=float)
@click.option("--sub_arcface_m", default=0.30, type=float)
@click.option("--sub_arcface_k", default=3, type=int)
@click.option("--warmup_epochs", default=10, type=int, help="CosFace/ArcFace s/m 递增轮数")
# ── 设备 & 输出 ──
@click.option("--seed", default=42, type=int)
@click.option("--device", default="auto", type=str, help="auto / cuda / cpu")
@click.option("--output_dir", default="output", help="输出目录")
@click.option("--tag", default=None, type=str, help="备注标签")
@click.option("--verbose/--quiet", default=True)
def main(features_dir, features_version, cv_folds, model, epochs, batch_size,
         lr, weight_decay, patience, grad_clip,
         optimizer_type, scheduler_type, t_0, t_mult,
         loss_type, center_loss, center_lambda, class_weight,
         online_aug, mixup_alpha, cosface_s, cosface_m,
         arcface_s, arcface_m, sub_arcface_s, sub_arcface_m, sub_arcface_k,
         warmup_epochs, seed, device, output_dir, tag, verbose):
    """K 折交叉验证训练入口。"""
    set_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = datetime.now()

    # ── 0. 版本 & 模型校验 ──
    features_path = os.path.join(PROJECT_ROOT, features_dir)
    versions = _list_versions(features_path)
    if not versions:
        raise click.UsageError(
            f"未找到 full_vX.pt，请先运行:\n"
            f"  python scripts/build_features.py --all --no_aug --version 6.0 --tag 'cv全量'"
        )
    if features_version is None:
        features_version = versions[-1]
    elif features_version not in versions:
        raise click.UsageError(f"版本 {features_version} 不存在，可用: {versions}")

    is_dual = model.startswith("dual_")
    core_model = model[5:] if is_dual else model
    if core_model not in MODEL_REGISTRY:
        raise click.UsageError(f"未知模型 '{model}'，可用: {sorted(MODEL_REGISTRY.keys())}")

    # 默认模型超参（与 train.py 保持一致）；input_size 按实际特征通道动态推断
    model_kwarg = {
        'out_features': 256, 'hidden_size': 256, 'num_layers': 1,
        'base_filters': 32,
        'dropblock_p': 0.2, 'dropblock_block_size': 5, 'lstm_dropout': 0.1,
        'cbam_reduction': 16, 'cbam_kernel': 7,
        'se_reduction': 16
    }

    click.echo("=" * 68)
    click.echo(f"  {cv_folds}-Fold Cross-Validation")
    click.echo("=" * 68)
    click.echo(f"  全量特征集:   v{features_version} ({features_path})")
    click.echo(f"  模型:         {model}")
    click.echo(f"  设备:         {device}")

    # ── 1. 加载全量特征 ──
    (all_feats, all_labels, signal_ids), label_map = load_full_dataset(features_path, features_version)
    num_classes = len(torch.unique(all_labels))
    if label_map is not None:
        class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    else:
        class_names = [str(i) for i in range(num_classes)]
    n_signals = len(torch.unique(signal_ids))
    click.echo(f"  样本数:       {all_feats.shape[0]} (来自 {n_signals} 条信号, 特征 {list(all_feats.shape[1:])})")
    click.echo(f"  类别数:       {num_classes} {class_names}")

    # 按实际特征通道设置 input_size（单分支=通道数；双分支=mel/mfcc 划分）
    if is_dual:
        mel_ch, mfcc_ch = _infer_mel_mfcc(all_feats.shape[1])
        model_kwarg['input_size'] = (mel_ch, mfcc_ch)
        click.echo(f"  双分支划分:   mel={mel_ch} + mfcc={mfcc_ch}")
    else:
        model_kwarg['input_size'] = all_feats.shape[1]

    # ── 2. 按信号分组做 StratifiedGroupKFold ──
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_indices = list(sgkf.split(all_feats, all_labels, groups=signal_ids.numpy()))

    # 输出根目录: output/{model}_v{ver}_cv{cv}
    outputs_path = os.path.join(PROJECT_ROOT, output_dir)
    run_name = f"{model}_v{features_version}_cv{cv_folds}"
    cv_root = os.path.join(outputs_path, run_name)
    if os.path.exists(cv_root):
        idx = 2
        while os.path.exists(f"{cv_root}_{idx}"):
            idx += 1
        cv_root = f"{cv_root}_{idx}"
        run_name = os.path.basename(cv_root)
    os.makedirs(cv_root, exist_ok=False)

    # ── 3. 逐折训练 ──
    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(fold_indices, 1):
        click.echo()
        click.echo("#" * 68)
        click.echo(f"  Fold {fold}/{cv_folds}  训练 {len(tr_idx)} 样本 / 验证 {len(va_idx)} 样本")
        click.echo("#" * 68)
        set_seed(seed + fold)  # 每折独立随机性

        tr_feats = all_feats[tr_idx]
        tr_labels = all_labels[tr_idx]
        va_feats = all_feats[va_idx]
        va_labels = all_labels[va_idx]

        # ── 每折用训练折统计量归一化（验证折不参与，避免泄露） ──
        # eps = 1e-6
        # mean = tr_feats.mean(dim=(0, 2), keepdim=True)
        # std = tr_feats.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(eps)
        # tr_feats = (tr_feats - mean) / std
        # va_feats = (va_feats - mean) / std

        # ── 类别权重（只按训练折计算） ──
        class_weights = None
        if class_weight:
            counts = torch.bincount(tr_labels, minlength=num_classes).float()
            class_weights = 1.0 / counts
            class_weights = class_weights / class_weights.sum() * num_classes

        net, criterion, head_type, feat_dim = build_net_and_losses(
            core_model, is_dual, model_kwarg, loss_type, num_classes,
            input_shape=tuple(all_feats.shape[1:]),
            class_weights=class_weights, device=device,
            cosface_s=cosface_s, cosface_m=cosface_m,
            arcface_s=arcface_s, arcface_m=arcface_m,
            sub_arcface_s=sub_arcface_s, sub_arcface_m=sub_arcface_m, sub_arcface_k=sub_arcface_k
        )

        center_loss_fn = None
        if center_loss:
            center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=feat_dim).to(device)

        optimizer = create_optimizer(net, lr=lr, weight_decay=weight_decay,
                                     optimizer_type=optimizer_type)
        scheduler = create_scheduler(optimizer, scheduler_type=scheduler_type,
                                     T_0=t_0, T_mult=t_mult)

        # ── DataLoader ──
        if online_aug:
            train_dataset = AugDataset(tr_feats, tr_labels, min_aug_num=2,
                                       spec_aug_prob=0.5, flip_prob=0.5)
        else:
            train_dataset = TensorDataset(tr_feats, tr_labels)
        val_dataset = TensorDataset(va_feats, va_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, pin_memory=(device == "cuda"),
                                  worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=(device == "cuda"), worker_init_fn=worker_init_fn)

        # ── 训练（早停基于该折验证集，不再用测试集） ──
        fold_dir = os.path.join(cv_root, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model_path = os.path.join(fold_dir, "best_model.pt")

        net, train_info = train_model(
            net, train_loader, val_loader, optimizer, device, model_path,
            scheduler=scheduler,
            criterion=criterion,
            center_loss=center_loss_fn if center_loss else None,
            center_lambda=center_lambda,
            mixup_alpha=mixup_alpha,
            epochs=epochs,
            patience=patience,
            grad_clip_norm=grad_clip,
            warmup_epochs=warmup_epochs if loss_type in ('cosface', 'arcface', 'sub_arcface') else 0,
            verbose=verbose,
        )

        # ── 评估验证折 ──
        cm_path = os.path.join(fold_dir, "confusion_matrix.png")
        metrics = evaluate_model(net, val_loader, device, num_classes,
                                 class_labels=class_names,
                                 save_cm_path=cm_path, verbose=False)
        acc = metrics["accuracy"] * 100
        f1 = metrics.get("macro_f1", 0) * 100
        click.echo(f"  [Fold {fold}] val_acc={acc:.2f}%  macro-F1={f1:.2f}%  "
                   f"best_epoch={train_info['best_epoch']}")

        fold_results.append({
            "fold": fold,
            "val_acc": round(acc, 4),
            "macro_f1": round(f1, 4),
            "best_epoch": train_info["best_epoch"],
            "best_val_acc": round(train_info["best_val_acc"] * 100, 4),
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
        })

        # 保存折内配置
        fold_cfg = {
            "fold": fold, "model": model, "features_version": features_version,
            "cv_folds": cv_folds, "loss_type": loss_type, "feat_dim": feat_dim,
            "head_type": head_type, "results": fold_results[-1],
        }
        with open(os.path.join(fold_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(fold_cfg, f, indent=2, ensure_ascii=False)

    # ── 4. 汇总 ──
    accs = np.array([r["val_acc"] for r in fold_results])
    f1s = np.array([r["macro_f1"] for r in fold_results])
    elapsed = (datetime.now() - t_start).total_seconds()

    summary = {
        "run": run_name,
        "model": model,
        "features_version": features_version,
        "cv_folds": cv_folds,
        "loss_type": loss_type,
        "loss_config": {
            "cosface_s": cosface_s, "cosface_m": cosface_m,
            "arcface_s": arcface_s, "arcface_m": arcface_m,
            "sub_arcface_s": sub_arcface_s, "sub_arcface_m": sub_arcface_m,
            "sub_arcface_k": sub_arcface_k,
        },
        "train_config": {
            "scheduler": scheduler_type,
            "mixup_alpha": mixup_alpha,
            "online_aug": online_aug,
            "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay, "patience": patience,
            "warmup_epochs": warmup_epochs,
        },
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "tag": tag or "",
        "fold_results": fold_results,
        "cv_acc_mean": round(float(accs.mean()), 4),
        "cv_acc_std": round(float(accs.std()), 4),
        "cv_f1_mean": round(float(f1s.mean()), 4),
        "cv_f1_std": round(float(f1s.std()), 4),
        "class_names": class_names,
    }
    summary_path = os.path.join(cv_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 追加记录到 EXPERIMENTS.md ──
    try:
        exp_path = _append_experiment_log(cv_root, summary)
        click.echo(f"  实验记录:     {exp_path}")
    except Exception as e:
        click.echo(f"  [警告] 实验记录写入失败: {e}", err=True)

    click.echo()
    click.echo("=" * 68)
    click.echo(f"  {cv_folds}-Fold CV 完成!")
    click.echo(f"  运行名称:     {run_name}")
    click.echo(f"  每折 val_acc: " + ", ".join(f"{a:.2f}%" for a in accs))
    click.echo(f"  每折 F1:      " + ", ".join(f"{f:.2f}%" for f in f1s))
    click.echo(f"  平均 Acc:     {accs.mean():.2f}% ± {accs.std():.2f}%")
    click.echo(f"  平均 F1:      {f1s.mean():.2f}% ± {f1s.std():.2f}%")
    click.echo(f"  总耗时:       {elapsed / 60:.1f} 分钟")
    click.echo(f"  汇总保存至:   {summary_path}")
    click.echo("=" * 68)


if __name__ == "__main__":
    main()
