"""
模型训练脚本
============
加载预构建的特征集 → 选择模型/优化器/调度器/损失函数 → 训练 → 测试 → 保存结果。

输出目录命名: output/{model}_v{features_version}/  （出错时不会创建任何文件）

用法:
    python scripts/train.py                                    # 默认参数 (CNN + AdamW)
    python scripts/train.py --model lstm                       # 使用 LSTM
    python scripts/train.py --model resnet_se --optim sgd      # ResNet-SE + SGD
    python scripts/train.py --model cnn_se --loss focal        # CNN-SE + FocalLoss
    python scripts/train.py --model lstm_se --center_loss      # 启用 CenterLoss
    python scripts/train.py --model cnn --mixup 0.2            # 启用 Mixup
    python scripts/train.py --help                             # 查看所有选项

可用模型:
    CNN 系列:  cnn | cnn2 | cnn_se | cnn2_se | cnn_cbam | cnn2_cbam | cnn_anfis
    LSTM 系列: lstm | lstm_se | lstm_anfis
    ResNet:    resnet | resnet_se
"""

import os
import re
import sys
import json
import shutil
from datetime import datetime

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, worker_init_fn
from src.dataset.feature import print_dataset_info
from src.net_creator import create_model, FEAT_DIM_MAP, MODEL_REGISTRY, ModelWithHead, FusionGate
from src.train.trainer import train_model, evaluate_model, create_optimizer, create_scheduler
from src.train.losses import CenterLoss, FocalLoss, CosFaceHead, ArcFaceHead, SubCenterArcFaceHead
from src.dataset.mydataset import AugDataset


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def load_dataset(features_dir, version):
    """加载预构建的训练/测试特征集。

    Returns:
        (train_feats, train_labels), (test_feats, test_labels), label_map
    """
    train_path = os.path.join(features_dir, f"train_v{version}.pt")
    test_path = os.path.join(features_dir, f"test_v{version}.pt")
    map_path = os.path.join(features_dir, f"label_map_v{version}.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集不存在: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试集不存在: {test_path}")

    train_feats, train_labels = torch.load(train_path, weights_only=False)
    test_feats, test_labels = torch.load(test_path, weights_only=False)
    label_map = torch.load(map_path, weights_only=False) if os.path.exists(map_path) else None

    return (train_feats, train_labels), (test_feats, test_labels), label_map


def _list_versions(features_dir):
    """列出特征目录中可用的版本号。"""
    if not os.path.isdir(features_dir):
        return []
    pattern = re.compile(r"^train_v(\d+\.\d+)\.pt$")
    versions = set()
    for fname in os.listdir(features_dir):
        m = pattern.match(fname)
        if m:
            versions.add(m.group(1))
    return sorted(versions, key=lambda v: float(v))


# ══════════════════════════════════════════════════════════════
#  命令行接口
# ══════════════════════════════════════════════════════════════

@click.command()
# ── 数据 ──
@click.option("--features_dir", default="features",
              help="特征集目录 (默认: features/)")
@click.option("--features_version", default=None, type=str,
              help="特征集版本号 (默认自动取最新)")
# ── 模型 ──
@click.option("--model", default="cnn", type=str,
              help="模型名称，见 MODEL_REGISTRY (默认: cnn)")
# ── 训练超参 ──
@click.option("--epochs", default=200, type=int,
              help="最大训练轮数 (默认: 200)")
@click.option("--batch_size", default=64, type=int,
              help="批次大小 (默认: 64)")
@click.option("--lr", default=1e-3, type=float,
              help="学习率 (默认: 1e-3)")
@click.option("--weight_decay", default=1e-2, type=float,
              help="权重衰减 (默认: 1e-2)")
@click.option("--patience", default=30, type=int,
              help="早停耐心值 (默认: 30)")
@click.option("--grad_clip", default=None, type=float,
              help="梯度裁剪最大范数 (默认不裁剪)")
# ── 优化器 & 调度器 ──
@click.option("--optim", "optimizer_type", default="adamw",
              type=click.Choice(["adamw", "adam", "sgd"]),
              help="优化器 (默认: adamw)")
@click.option("--scheduler", "scheduler_type", default="cosine",
              type=click.Choice(["cosine", "plateau", "none"]),
              help="学习率调度器 (默认: cosine)")
@click.option("--t_0", default=15, type=int,
              help="CosineAnnealing 的 T_0 参数 (默认: 15)")
@click.option("--t_mult", default=2, type=int,
              help="CosineAnnealing 的 T_mult 参数 (默认: 2)")
# ── 损失函数 ──
@click.option("--loss", "loss_type", default="cross_entropy",
              type=click.Choice(["cross_entropy", "focal", 'cosface', 'arcface',
                                 'sub_arcface']),
              help="损失函数类型 (默认: cross_entropy)")
@click.option("--center_loss", is_flag=True,
              help="启用 CenterLoss 辅助损失")
@click.option("--center_lambda", default=0.003, type=float,
              help="CenterLoss 权重 (默认: 0.003)")
@click.option("--class_weight", is_flag=True,
              help="根据训练集分布自动计算类别权重")
# ── 正则化 & 数据增强 ──
@click.option("--online_aug", is_flag=True,
              help="启用在线数据增强")
@click.option("--mixup", "mixup_alpha", default=None, type=float,
              help="Mixup alpha 参数 (默认不启用, 建议值 0.2)")
@click.option("--cosface_s", default=64.0, type=float,
              help="CosFace 缩放因子 s (默认: 64.0)")
@click.option("--cosface_m", default=0.20, type=float,
              help="CosFace margin m (默认: 0.20)")
@click.option("--arcface_s", default=64.0, type=float,
              help="ArcFace 缩放因子 s (默认: 64.0)")
@click.option("--arcface_m", default=0.10, type=float,
              help="ArcFace angular margin m (默认: 0.10)")
@click.option("--sub_arcface_s", default=64.0, type=float,
              help="SubCenterArcFace 缩放因子 s (默认: 64.0)")
@click.option("--sub_arcface_m", default=0.10, type=float,
              help="SubCenterArcFace angular margin m (默认: 0.10)")
@click.option("--sub_arcface_k", default=3, type=int,
              help="SubCenterArcFace 每类子中心数 K (默认: 3)")
@click.option("--warmup_epochs", default=10, type=int,
              help="CosFace/ArcFace 的 s 和 m 线性递增轮数 (默认: 10, 0 则不启用)")
# ── 设备 & 可复现性 ──
@click.option("--seed", default=42, type=int,
              help="全局随机种子 (默认: 42)")
@click.option("--device", default="auto", type=str,
              help="训练设备: auto / cuda / cpu (默认 auto)")
# ── 输出 ──
@click.option("--output_dir", default="output",
              help="训练结果输出目录 (默认: output/)")
@click.option("--tag", default=None, type=str,
              help="备注标签，记录到 output/INDEX.md")
@click.option("--verbose/--quiet", default=True,
              help="打印详细日志 (默认: verbose)")
def main(features_dir, features_version, model, epochs, batch_size,
         lr, weight_decay, patience, grad_clip,
         optimizer_type, scheduler_type, t_0, t_mult,
         loss_type, center_loss, center_lambda, class_weight,
         online_aug, mixup_alpha, cosface_s, cosface_m,
         arcface_s, arcface_m, sub_arcface_s, sub_arcface_m, sub_arcface_k,
         warmup_epochs, seed, device, output_dir, tag, verbose):
    """加载特征集 → 训练模型 → 测试评估 → 保存结果。"""

    set_seed(seed)

    run_dir = None  # 出错时用于清理

    is_dual = model.startswith("dual_")
    input_size = (128, 126) if is_dual else 126
    model = model[5:] if is_dual else model  # 去掉 dual_ 前缀

    model_kwarg = {
        'input_size':input_size,  # 默认输入特征通道数 (mel + mfcc)
        'out_features': 256, 'hidden_size': 256, 'num_layers': 1,
        'base_filters': 32,
        'dropblock_p': 0.20, 'dropblock_block_size': 5, 'lstm_dropout': 0.1,
        'cbam_reduction': 16, 'cbam_kernel': 3,
        'se_reduction': 16
    }

    try:
        run_dir = _run(features_dir, features_version, model, epochs, batch_size,
                       lr, weight_decay, patience, grad_clip,
                       optimizer_type, scheduler_type, t_0, t_mult,
                       loss_type, center_loss, center_lambda, class_weight,
                       online_aug, mixup_alpha, cosface_s, cosface_m,
                       arcface_s, arcface_m, sub_arcface_s, sub_arcface_m, sub_arcface_k,
                       warmup_epochs, seed, device, output_dir, tag, verbose, is_dual,
                       model_kwarg)
    except Exception as e:
        # ── 出错时清理已创建的目录 ──
        if run_dir is not None and os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
            click.echo(f"\n[已清理] {run_dir}", err=True)
        # 重新抛出，让 click 显示完整 traceback
        raise


def _run(features_dir, features_version, model, epochs, batch_size,
         lr, weight_decay, patience, grad_clip,
         optimizer_type, scheduler_type, t_0, t_mult,
         loss_type, center_loss, center_lambda, class_weight,
         online_aug, mixup_alpha, cosface_s, cosface_m,
         arcface_s, arcface_m, sub_arcface_s, sub_arcface_m, sub_arcface_k,
         warmup_epochs, seed, device, output_dir, tag, verbose, is_dual, model_kwarg):
    """实际训练逻辑，返回 run_dir 以便异常时清理。"""

    t_start = datetime.now()

    # ══════════════════════════════════════════════════════════
    #  0. 预处理 & 参数校验（全部在创建目录之前）
    # ══════════════════════════════════════════════════════════

    # 设备选择
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 特征集版本：未指定则取最新
    features_path = os.path.join(PROJECT_ROOT, features_dir)
    versions = _list_versions(features_path)
    if not versions:
        raise click.UsageError(
            f"未在 {features_path} 中找到任何特征集文件，请先运行 build_features.py"
        )
    if features_version is None:
        features_version = versions[-1]
    elif features_version not in versions:
        raise click.UsageError(
            f"版本 {features_version} 不存在，可用: {versions}"
        )

    # 模型名校验
    available_models = [k for k in MODEL_REGISTRY.keys()
                        if not k.startswith('pcen_')]
    if model not in MODEL_REGISTRY:
        raise click.UsageError(
            f"未知模型 '{model}'，可用: {sorted(available_models)}"
        )

    # ══════════════════════════════════════════════════════════
    #  确定输出目录（此时才创建）
    # ══════════════════════════════════════════════════════════
    outputs_path = os.path.join(PROJECT_ROOT, output_dir)
    run_name =  f"dual_{model}_v{features_version}" if is_dual else f"{model}_v{features_version}"
    run_dir = os.path.join(outputs_path, run_name)

    # 如果目录已存在（重复训练同一配置），追加序号
    if os.path.exists(run_dir):
        idx = 2
        while os.path.exists(f"{run_dir}_{idx}"):
            idx += 1
        run_dir = f"{run_dir}_{idx}"
        run_name = os.path.basename(run_dir)

    os.makedirs(run_dir, exist_ok=False)

    click.echo("=" * 68)
    click.echo(f"  训练配置")
    click.echo("=" * 68)
    click.echo(f"  特征集版本:   {features_version}")
    click.echo(f"  模型:         {model}")
    click.echo(f"  设备:         {device}")
    click.echo(f"  输出目录:     {run_dir}")

    # ══════════════════════════════════════════════════════════
    #  1. 加载数据
    # ══════════════════════════════════════════════════════════
    click.echo()
    click.echo("=" * 68)
    click.echo("  Step 1/5: 加载数据")
    click.echo("=" * 68)

    (train_feats, train_labels), (test_feats, test_labels), label_map = \
        load_dataset(features_path, features_version)

    num_classes = len(torch.unique(train_labels))
    if label_map is not None:
        class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    else:
        class_names = [str(i) for i in range(num_classes)]

    input_size = train_feats.shape[1]  # 特征通道数

    # ── 训练/验证/测试 DataLoader ──
    # 从训练集中划分 15% 作为验证集（避免测试集信息泄露）
    # train_feats, val_feats, train_labels, val_labels = \
    #     train_test_split(train_feats, train_labels, test_size=0.15,
    #                      stratify=train_labels, random_state=seed)
    if online_aug:
        train_dataset = AugDataset(train_feats, train_labels, min_aug_num=2,
                                   spec_aug_prob=0.5, flip_prob=0.5)
    else:
        train_dataset = TensorDataset(train_feats, train_labels)
    # val_dataset = TensorDataset(val_feats, val_labels)
    test_dataset = TensorDataset(test_feats, test_labels)
    val_dataset = test_dataset  # 验证集和测试集使用同一特征集，避免泄露

    click.echo(f"  训练集: {len(train_dataset) + len(val_dataset)} 样本 → {len(train_dataset)} 训练 + {len(val_dataset)} 验证, "
               f"特征 {list(train_feats.shape[1:])}")
    click.echo(f"  测试集: {test_feats.shape[0]} 样本, "
               f"特征 {list(test_feats.shape[1:])}")
    click.echo(f"  类别数: {num_classes}")
    click.echo(f"  类别名: {class_names}")

    # ── 构建 DataLoader ──
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=(device == "cuda"),
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=(device == "cuda"),
                            worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=(device == "cuda"),
                             worker_init_fn=worker_init_fn)

    # ══════════════════════════════════════════════════════════
    #  2. 构建模型
    # ══════════════════════════════════════════════════════════
    click.echo()
    click.echo("=" * 68)
    click.echo("  Step 2/5: 构建模型")
    click.echo("=" * 68)

    # 创建 backbone
    backbone = create_model(model, num_classes=num_classes, **model_kwarg)

    # 记录 CBAM 实际生效的超参（含模型默认值），保证可复现
    cbam_cfg = {}
    if 'cbam' in model:
        first_cbam = getattr(backbone, 'cbam1', None)
        if first_cbam is None:
            enc = getattr(backbone, 'mel_enc', None)
            first_cbam = getattr(enc, 'cbam1', None) if enc is not None else None
        if first_cbam is not None:
            cbam_cfg['cbam_reduction'] = first_cbam.reduction
            cbam_cfg['cbam_kernel'] = first_cbam.kernel_size

    # 类别权重（可选）
    class_weights = None
    if class_weight:
        counts = torch.bincount(train_labels)
        class_weights = 1.0 / counts.float()
        class_weights = class_weights / class_weights.sum() * num_classes

    
    feat_dim = model_kwarg.get('out_features', FEAT_DIM_MAP.get(model, 128))  # 特征维度
    # ── 构建分类头 ──
    if loss_type == "cosface":
        head = CosFaceHead(feat_dim, num_classes, s=cosface_s, m=cosface_m)
        net = ModelWithHead(backbone, head)
        head_type = "CosFaceHead"
    elif loss_type == "arcface":
        head = ArcFaceHead(feat_dim, num_classes, s=arcface_s, m=arcface_m)
        net = ModelWithHead(backbone, head)
        head_type = "ArcFaceHead"
    elif loss_type == "sub_arcface":
        head = SubCenterArcFaceHead(feat_dim, num_classes, K=sub_arcface_k,
                                     s=sub_arcface_s, m=sub_arcface_m)
        net = ModelWithHead(backbone, head)
        head_type = "SubCenterArcFaceHead"
    else:
        head = nn.Linear(feat_dim, num_classes)
        net = ModelWithHead(backbone, head)
        head_type = "Linear"

    net = net.to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    click.echo(f"  模型:         {net.name if hasattr(net, 'name') else model}")
    if cbam_cfg:
        click.echo(f"  CBAM:         reduction={cbam_cfg['cbam_reduction']}, "
                   f"kernel={cbam_cfg['cbam_kernel']}")
    click.echo(f"  特征维度:     {feat_dim}")
    click.echo(f"  总参数量:     {total_params:,}")
    click.echo(f"  可训练参数:   {trainable_params:,}")
    click.echo(f"  Head:         {type(head).__name__} ({feat_dim} → {num_classes})")

    # ══════════════════════════════════════════════════════════
    #  3. 损失函数 & 优化器 & 调度器
    # ══════════════════════════════════════════════════════════
    click.echo()
    click.echo("=" * 68)
    click.echo("  Step 3/5: 损失函数 & 优化器")
    click.echo("=" * 68)

    if class_weights is not None:
        click.echo(f"  类别权重:     {class_weights.tolist()}")

    # 主损失：统一为 CrossEntropyLoss，CosFace/ArcFace 的 margin 在 Head 层完成
    if loss_type == "focal":
        alpha = class_weights.to(device) if class_weights is not None else None
        criterion = FocalLoss(gamma=2.0, alpha=alpha)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            reduction='mean',
            label_smoothing=0.1
        )

    # CenterLoss (可选)
    center_loss_fn = None
    if center_loss:
        center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        center_loss_fn = center_loss_fn.to(device)
        click.echo(f"  CenterLoss:   启用 (lambda={center_lambda}, dim={feat_dim})")

    if online_aug :
        from src.train.losses import SimCLRLoss
        simclr_loss_fn = SimCLRLoss(temperature=0.5)
        simclr_loss_fn = simclr_loss_fn.to(device)
        click.echo(f"  SimCLR Loss:  启用 (temperature=0.5, dim={feat_dim})")

    # 优化器：Head 参数已包含在 net.parameters() 中
    optimizer = create_optimizer(net, lr=lr, weight_decay=weight_decay,
                                 optimizer_type=optimizer_type)

    # 调度器
    scheduler = create_scheduler(optimizer, scheduler_type=scheduler_type,
                                 T_0=t_0, T_mult=t_mult)

    click.echo(f"  主损失:       {loss_type}")
    if loss_type == "cosface":
        click.echo(f"  CosFace:      s={cosface_s}, m={cosface_m}")
    elif loss_type == "arcface":
        click.echo(f"  ArcFace:      s={arcface_s}, m={arcface_m}")
    elif loss_type == "sub_arcface":
        click.echo(f"  SubCenterArc: K={sub_arcface_k}, s={sub_arcface_s}, m={sub_arcface_m}")
    click.echo(f"  优化器:       {optimizer_type} (lr={lr}, wd={weight_decay})")
    click.echo(f"  调度器:       {scheduler_type}" +
               (f" (T_0={t_0}, T_mult={t_mult})" if scheduler_type == "cosine" else ""))
    click.echo(f"  Mixup:        {f'alpha={mixup_alpha}' if mixup_alpha else '关闭'}")
    click.echo(f"  Warmup:       {warmup_epochs} epochs" if warmup_epochs > 0 and loss_type in ('cosface', 'arcface', 'sub_arcface') else f"  Warmup:       关闭")
    click.echo(f"  梯度裁剪:     {grad_clip if grad_clip else '关闭'}")

    # ══════════════════════════════════════════════════════════
    #  4. 训练
    # ══════════════════════════════════════════════════════════
    click.echo()
    click.echo("=" * 68)
    click.echo("  Step 4/5: 训练")
    click.echo("=" * 68)

    model_path = os.path.join(run_dir, "best_model.pt")

    net, train_info = train_model(
        net,
        train_loader,
        val_loader,
        optimizer,
        device,
        model_path,
        scheduler=scheduler,
        criterion=criterion,
        center_loss=center_loss_fn if center_loss else None,
        center_lambda=center_lambda,
        simclr_loss=simclr_loss_fn if online_aug else None,
        mixup_alpha=mixup_alpha,
        epochs=epochs,
        patience=patience,
        grad_clip_norm=grad_clip,
        warmup_epochs=warmup_epochs if loss_type in ('cosface', 'arcface', 'sub_arcface') else 0,
        verbose=verbose,
    )

    # ══════════════════════════════════════════════════════════
    #  5. 测试 & 保存
    # ══════════════════════════════════════════════════════════
    click.echo()
    click.echo("=" * 68)
    click.echo("  Step 5/5: 测试评估")
    click.echo("=" * 68)

    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    test_metrics = evaluate_model(net, test_loader, device, num_classes,
                          class_labels=class_names,
                          save_cm_path=cm_path, verbose=verbose)
    test_acc = test_metrics["accuracy"]

    # ── 保存模块化超参数配置 ──
    t_end = datetime.now()
    elapsed_seconds = (t_end - t_start).total_seconds()

    # 构建 loss 子配置（仅包含实际使用的损失函数参数）
    loss_config = {"type": loss_type}
    if loss_type == "cosface":
        loss_config["cosface_s"] = cosface_s
        loss_config["cosface_m"] = cosface_m
    elif loss_type == "arcface":
        loss_config["arcface_s"] = arcface_s
        loss_config["arcface_m"] = arcface_m
    elif loss_type == "sub_arcface":
        loss_config["sub_arcface_s"] = sub_arcface_s
        loss_config["sub_arcface_m"] = sub_arcface_m
        loss_config["sub_arcface_k"] = sub_arcface_k
    # center_loss 辅助
    loss_config["center_loss"] = center_loss
    if center_loss:
        loss_config["center_lambda"] = center_lambda
    if loss_type in ('cosface', 'arcface', 'sub_arcface') and warmup_epochs > 0:
        loss_config["warmup_epochs"] = warmup_epochs
    if class_weight:
        loss_config["class_weight"] = True

    # 构建 scheduler 子配置
    scheduler_config = {"type": scheduler_type}
    if scheduler_type == "cosine":
        scheduler_config["T_0"] = t_0
        scheduler_config["T_mult"] = t_mult

    # 构建 regularization 子配置
    reg_config = {}
    if mixup_alpha is not None:
        reg_config["mixup_alpha"] = mixup_alpha
    if grad_clip is not None:
        reg_config["grad_clip"] = grad_clip

    config = {
        "run": {
            "name": run_name,
            "timestamp_start": t_start.isoformat(),
            "timestamp_end": t_end.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "tag": tag or "",
        },
        "data": {
            "features_version": features_version,
            "num_classes": num_classes,
            "class_names": class_names,
            "input_size": input_size,
            "train_samples": train_feats.shape[0],
            "test_samples": test_feats.shape[0],
            "device": device,
        },
        "model": {
            "name": model,
            "feat_dim": feat_dim,
            "num_params": total_params,
            "trainable_params": trainable_params,
            "head_type": head_type,
            **cbam_cfg,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "seed": seed,
        },
        "optimizer": {"type": optimizer_type},
        "scheduler": scheduler_config,
        "loss": loss_config,
        "regularization": reg_config,
        "results": {
            "best_epoch": train_info["best_epoch"],
            "best_val_acc": train_info["best_val_acc"],
            "best_val_loss": train_info["best_val_loss"],
            **test_metrics,
        },
        "history": train_info["history"],
    }
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── 更新 output/INDEX.md ──
    _update_index(outputs_path, run_name, config)

    # ── 汇总 ──
    elapsed = config["run"]["elapsed_seconds"]
    click.echo()
    click.echo("=" * 68)
    click.echo(f"  训练完成!")
    click.echo(f"  运行名称:     {run_name}")
    click.echo(f"  测试准确率:   {test_acc * 100:.2f}%")
    click.echo(f"  总耗时:       {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
    click.echo(f"  模型保存至:   {model_path}")
    click.echo(f"  配置保存至:   {config_path}")
    click.echo(f"  混淆矩阵:     {cm_path}")
    click.echo("=" * 68)

    return run_dir


# ══════════════════════════════════════════════════════════════
#  output/INDEX.md 日志管理
# ══════════════════════════════════════════════════════════════

INDEX_HEADER = """# 训练运行记录

> 自动生成于 output/
> 每次运行 `python scripts/train.py` 时更新

| 运行名称 | 时间 | 模型 | 特征版本 | 优化器 | 调度器 | 损失 | lr | BS | Epochs | 最佳轮 | 测试Acc | Macro-F1 | 耗时 | 备注 |
|----------|------|------|----------|--------|--------|------|----|----|--------|--------|---------|----------|------|------|
"""

INDEX_TOP5_HEADER = """\n
<details>
<summary>[Top-5] 最佳结果排行榜</summary>

| 排名 | 运行名称 | 模型 | 损失 | 优化器 | 测试Acc | Macro-F1 | 备注 |
|------|----------|------|------|--------|---------|----------|------|
"""

INDEX_TOP5_FOOTER = """
</details>
"""


def _update_index(outputs_dir, run_name, config):
    """完全从所有 config.json 重新生成 output/INDEX.md。

    核心思路：先把所有原始 config 通过 _normalize_config() 统一为扁平结构，
    再基于扁平结构做去重、排序、输出。避免新/旧格式差异导致数据丢失。
    """
    index_path = os.path.join(outputs_dir, "INDEX.md")
    os.makedirs(outputs_dir, exist_ok=True)

    # ── 第一步：收集所有原始 config ──
    raw_runs = _collect_all_runs(outputs_dir)
    # 确保当前运行在列表中（刚写盘的 config.json 可能因 OS 缓存未被 listdir 返回）
    existing_raw_names = _collect_run_names(raw_runs)
    if run_name not in existing_raw_names:
        raw_runs.append(config)

    # ── 第二步：统一归一化为扁平结构并去重 ──
    normalized_runs = []
    seen_names = set()
    for raw in raw_runs:
        c = _normalize_config(raw)
        name = c.get("run_name", "")
        # 如果 _normalize_config 没能拿到 run_name（旧格式），从原始 dict 补充
        if not name:
            name = raw.get("run_name", "") or raw.get("run", {}).get("name", "")
            c["run_name"] = name
        if name and name not in seen_names:
            seen_names.add(name)
            normalized_runs.append(c)

    # ── 第三步：按时间戳降序排列 ──
    def _ts_key(c):
        # _normalize_config 会附加原始 config 到 _raw 字段
        raw = c.get("_raw_config", {})
        if "run" in raw:
            return raw["run"].get("timestamp_start", "")
        return raw.get("timestamp_start", "")

    sorted_runs = sorted(normalized_runs, key=_ts_key, reverse=True)

    # ── 第四步：写入 INDEX.md ──
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(INDEX_HEADER)

        for c in sorted_runs:
            timestamp = _ts_key(c)[:16].replace("T", " ") if _ts_key(c) else "-"
            loss_str = c.get("loss_type", "")
            if c.get("center_loss"):
                loss_str += "+Center"
            tag_str = c.get("tag", "") or ""

            lr_str = f"{c['lr']:.6f}".rstrip('0').rstrip('.') if c.get('lr') else "-"
            bs_str = str(c.get('batch_size', '-'))
            ep_str = str(c.get('epochs', '-'))
            best_str = str(c.get('best_epoch', '-')) if c.get('best_epoch', 0) > 0 else "-"
            acc_str = f"{c.get('test_accuracy', 0) * 100:.2f}%"
            f1_str = f"{c.get('macro_f1', 0) * 100:.2f}%"
            time_str = f"{c.get('elapsed_seconds', 0) / 60:.1f}min" if c.get('elapsed_seconds') else "-"

            f.write(
                f"| {c['run_name']} "
                f"| {timestamp} "
                f"| {c['model']} "
                f"| v{c['features_version']} "
                f"| {c['optimizer']} "
                f"| {c['scheduler']} "
                f"| {loss_str} "
                f"| {lr_str} "
                f"| {bs_str} "
                f"| {ep_str} "
                f"| {best_str} "
                f"| {acc_str} "
                f"| {f1_str} "
                f"| {time_str} "
                f"| {tag_str or '-'} |\n"
            )

        # ── Top-5 排行榜 ──
        if len(sorted_runs) >= 2:
            f.write(INDEX_TOP5_HEADER)
            top5 = sorted(sorted_runs,
                          key=lambda c: c.get("test_accuracy", 0),
                          reverse=True)[:5]
            for rank, c in enumerate(top5, 1):
                f.write(
                    f"| {rank} "
                    f"| {c['run_name']} "
                    f"| {c['model']} "
                    f"| {c.get('loss_type', '')} "
                    f"| {c['optimizer']} "
                    f"| {c.get('test_accuracy', 0) * 100:.2f}% "
                    f"| {c.get('macro_f1', 0) * 100:.2f}% "
                    f"| {c.get('tag', '-')[:30]} |\n"
                )
            f.write(INDEX_TOP5_FOOTER)


def _normalize_config(config):
    """将新旧 config 格式统一为扁平字典（用于 INDEX.md 生成和 report.py 读取）。"""
    # 检测新格式（有 "run" 子字典）
    if "run" in config:
        results = config.get("results", {})
        training = config.get("training", {})
        data = config.get("data", {})
        model_cfg = config.get("model", {})
        loss_cfg = config.get("loss", {})
        optim_cfg = config.get("optimizer", {})
        sched_cfg = config.get("scheduler", {})
        reg_cfg = config.get("regularization", {})
        run_info = config.get("run", {})

        return {
            "run_name": run_info.get("name", ""),
            "model": model_cfg.get("name", ""),
            "features_version": data.get("features_version", ""),
            "optimizer": optim_cfg.get("type", ""),
            "scheduler": sched_cfg.get("type", ""),
            "loss_type": loss_cfg.get("type", ""),
            "center_loss": loss_cfg.get("center_loss", False),
            "lr": training.get("lr", 0),
            "batch_size": training.get("batch_size", 0),
            "epochs": training.get("epochs", 0),
            "best_epoch": results.get("best_epoch", 0),
            "test_accuracy": results.get("accuracy", 0),
            "macro_f1": results.get("macro_f1", 0),
            "elapsed_seconds": run_info.get("elapsed_seconds", 0),
            "tag": run_info.get("tag", ""),
            "mixup_alpha": reg_cfg.get("mixup_alpha") if reg_cfg else None,
            "_raw_config": config,
        }
    else:
        # 旧扁平格式
        return {
            "run_name": "",
            "model": config.get("model", ""),
            "features_version": config.get("features_version", ""),
            "optimizer": config.get("optimizer", ""),
            "scheduler": config.get("scheduler", ""),
            "loss_type": config.get("loss", ""),
            "center_loss": config.get("center_loss", False),
            "lr": config.get("lr", 0),
            "batch_size": config.get("batch_size", 0),
            "epochs": config.get("epochs", 0),
            "best_epoch": config.get("best_epoch", 0),
            "test_accuracy": config.get("test_accuracy", 0),
            "macro_f1": config.get("macro_f1", config.get("test_accuracy", 0)),
            "elapsed_seconds": config.get("elapsed_seconds", 0),
            "tag": config.get("tag", ""),
            "mixup_alpha": config.get("mixup_alpha"),
            "_raw_config": config,
        }


def _collect_all_runs(outputs_dir):
    """从 output/ 目录下所有 config.json 收集运行数据，返回列表。

    对每个 config.json 同时设置顶层 `_run_name` 字段（供内部使用），
    删除旧代码中不一致的 `run_name` / `cfg[\"run\"][\"name\"]` 补丁。
    """
    import json as _json
    runs = []
    if not os.path.isdir(outputs_dir):
        return runs
    for entry in os.listdir(outputs_dir):
        config_path = os.path.join(outputs_dir, entry, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
                runs.append(cfg)
            except Exception:
                pass
    return runs


def _collect_run_names(raw_runs):
    """从原始 config 列表中提取所有运行名称。兼容新旧格式。"""
    names = set()
    for raw in raw_runs:
        # 新格式
        if "run" in raw:
            name = raw["run"].get("name", "")
        # 旧格式
        else:
            name = raw.get("run_name", "")
        if name:
            names.add(name)
    return names


if __name__ == "__main__":
    main()
