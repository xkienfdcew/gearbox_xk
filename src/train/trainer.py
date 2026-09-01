import os
import math as _math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def _apply_margin_warmup(model, epoch, warmup_epochs):
    """CosFace/ArcFace/SubCenterArcFace warmup: 线性递增 s 和 m。

    - s: 从 s_target/10 递增到 s_target（保证初期有基本的梯度信号）
    - m: 从 0 递增到 m_target（margin 从无到有，模型先学会基本分类再逐步拉开）
    - ArcFace/SubCenterArcFace 的 cos_m/sin_m/th/mm 在每次 m 变化时重新计算

    调用时机: 每个 epoch 训练开始前（model.train() 之后）
    """
    if warmup_epochs <= 0:
        return

    from src.train.losses import CosFaceHead, ArcFaceHead, SubCenterArcFaceHead

    head = getattr(model, 'head', None)
    if head is None or not isinstance(
        head, (CosFaceHead, ArcFaceHead, SubCenterArcFaceHead)
    ):
        return

    # 首次调用时保存目标值
    if not hasattr(head, '_s_target'):
        head._s_target = head.s
        head._m_target = head.m

    # 线性 warmup 因子: epoch 0 → 1/warmup_epochs, epoch warmup_epochs-1 → 1.0
    factor = min(1.0, (epoch + 1) / warmup_epochs)

    # s 从 10% 开始（不能从 0 开始，否则梯度消失）
    head.s = head._s_target * max(0.1, factor)

    # m 从 0 开始线性增长
    head.m = head._m_target * factor

    # ArcFace 系列依赖 m 的三角函数缓存需要重新计算
    if isinstance(head, (ArcFaceHead, SubCenterArcFaceHead)) and head.m > 0:
        head.cos_m = _math.cos(head.m)
        head.sin_m = _math.sin(head.m)
        head.th = _math.cos(_math.pi - head.m)
        head.mm = _math.sin(_math.pi - head.m) * head.m


def train_model(model, train_loader, val_loader, optimizer, device,
                model_save_path, *,
                scheduler=None,
                criterion=None,
                center_loss=None,
                center_lambda=0.003,
                simclr_loss=None,
                mixup_alpha=None,
                epochs=200,
                patience=30,
                grad_clip_norm=None,
                warmup_epochs=0,
                verbose=True,
                num_classes=8):
    """
    统一训练函数

    参数:
        model:           ModelWithHead 实例 或 ANFIS 模型
                         forward(x, labels=None) 返回 (feature, logits)
        train_loader:    训练 DataLoader
        val_loader:      验证 DataLoader
        optimizer:       优化器（外部创建，必需）
        device:          设备
        model_save_path: 最佳模型保存路径
        scheduler:       学习率调度器（可选，每个 epoch 后 step）
        criterion:       主损失函数（默认 CrossEntropyLoss，始终接收 logits + labels）
        center_loss:     CenterLoss 实例（可选）
        center_lambda:   center loss 权重
        simclr_loss:     SimCLRLoss 实例（可选）
        mixup_alpha:     mixup Beta 分布参数（None 则不启用）
        epochs:          最大训练轮数
        patience:        早停耐心值
        grad_clip_norm:  梯度裁剪最大范数（None 则不裁剪）
        warmup_epochs:   CosFace/ArcFace 的 s 和 m 线性增长轮数（0 则不启用，建议 5）
        verbose:         是否打印训练日志

    返回:
        model: 加载了最佳权重的模型
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0


    accmat = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_classes
    ).to(device=device)


    # ── 训练历史记录（每个 epoch 的指标）──
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ---- 训练阶段 ----
        model.train()
        accmat.reset()

        # CosFace/ArcFace warmup: 每个 epoch 开始前递增 s 和 m
        _apply_margin_warmup(model, epoch, warmup_epochs)

        train_loss_sum = 0.0

        for features, labels in train_loader:           
            labels = torch.cat(labels, dim=0) if isinstance(labels, (list, tuple)) else labels
            features = torch.cat(features, dim=0) if isinstance(features, (list, tuple)) else features

            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            loss = torch.tensor(0.0, device=device)

            # ── Mixup 分支（可选） ──
            if mixup_alpha is not None:
                from src.train.losses import mixup
                mix_features, labels_a, labels_b, lam = mixup(
                    features, labels, alpha=mixup_alpha
                )
                # 用 backbone 提取特征，head 输出原始 logits（不带 margin）
                if hasattr(model, 'backbone'):
                    mix_feat = model.backbone(mix_features)
                    mix_logi = model.head(mix_feat)
                else:
                    mix_feat, mix_logi = model(mix_features)
                # 标准 Mixup loss: λ·CE(pred, y_a) + (1-λ)·CE(pred, y_b)
                loss += lam * criterion(mix_logi, labels_a) + \
                        (1.0 - lam) * criterion(mix_logi, labels_b)

            # ── 正常前向：labels 传入以触发 CosFace/ArcFace 的 margin ──
            feat, logi = model(features, labels)
            loss += criterion(logi, labels)

            # ── Center loss（可选） ──
            if center_loss is not None:
                loss += center_lambda * center_loss(feat, labels)

            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            # 更新中心（center loss 需要）
            if center_loss is not None:
                center_loss.update_centers(feat.detach(), labels)

            train_loss_sum += loss.item()

            # 准确率用无 margin 的 logits 计算，反映真实分类能力
            with torch.no_grad():
                _, logi_raw = model(features)
            pred = logi_raw.argmax(dim=1)
            targets = labels.argmax(dim=1) if labels.dim() > 1 else labels
            accmat.update(pred, targets)

        avg_train_loss = train_loss_sum / len(train_loader)
        train_acc = accmat.compute() * 100

        # ── 记录训练指标 ──
        history["train_loss"].append(round(avg_train_loss, 6))
        history["train_acc"].append(round(train_acc.item(), 2))

        val_loss_sum = 0

        accmat.reset()
        # ---- 验证阶段 ----
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                # 验证时不传 labels → head 输出原始 logits（不带 margin）
                feat, logi = model(features)
                val_loss_sum += criterion(logi, labels).item()
                pred = logi.argmax(dim=1)
                targets = labels.argmax(dim=1) if labels.dim() > 1 else labels
                accmat.update(pred, targets)

        avg_val_loss = val_loss_sum / len(val_loader)
        val_acc = accmat.compute() * 100

        # ── 记录验证指标 ──
        history["val_loss"].append(round(avg_val_loss, 6))
        history["val_acc"].append(round(val_acc.item(), 2))

        if verbose:
            warmup_hint = ''
            if warmup_epochs > 0 and epoch < warmup_epochs:
                from src.train.losses import CosFaceHead, ArcFaceHead, SubCenterArcFaceHead
                head = getattr(model, 'head', None)
                if isinstance(head, (CosFaceHead, ArcFaceHead, SubCenterArcFaceHead)):
                    warmup_hint = f' [warmup s={head.s:.1f} m={head.m:.3f}]'
            print(f'Epoch {epoch + 1:3d} | '
                  f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | '
                  f'Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%'
                  f'{warmup_hint}')

        # 学习率调度（plateau 模式监控验证损失）
        if scheduler is not None:
            scheduler.step(avg_train_loss)

        # 保存最佳模型：按验证集准确率选择，验证集损失作为平局判断
        is_better = (val_acc > best_val_acc or
                     (val_acc == best_val_acc and avg_val_loss < best_val_loss))
        if is_better:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                if best_epoch == 0:
                    # 尚未保存过任何模型：重试再给 patience/2 的机会
                    no_improve = patience // 2
                else:
                    if verbose:
                        print(f'Early stopping at epoch {epoch + 1}, '
                              f'best val acc: {best_val_acc:.2f}% at epoch {best_epoch}')
                    break

    # 加载最佳模型；若从未保存则保留最终状态
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
    elif best_epoch == 0:
        # 训练过程中验证准确率从未改善（例如 patience 未耗尽但 model 从未保存）
        torch.save(model.state_dict(), model_save_path)

    # ── 构建返回信息 ──
    info = {
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc.item(), 2),
        "best_val_loss": round(best_val_loss, 6) if best_val_loss != float('inf') else None,
        "history": history,
    }
    return model, info


def evaluate_model(model, test_loader, device, num_classes, class_labels=None,
               save_cm_path=None, verbose=True):
    """
    在测试集上评估模型

    参数:
        model:         模型实例
        test_loader:   测试 DataLoader
        device:        设备
        num_classes:   类别数
        class_labels:  类别名称列表（用于混淆矩阵）
        save_cm_path:  混淆矩阵保存路径（None 则不保存）
        verbose:       是否打印结果

    返回:
        metrics: dict，包含 accuracy, per_class, macro_precision, macro_recall, macro_f1
    """
    model.eval()
    confmat = torchmetrics.classification.MulticlassConfusionMatrix(
        num_classes=num_classes
    ).to(device)
    accmat = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_classes
    ).to(device)
    precmat = torchmetrics.classification.MulticlassPrecision(
        num_classes=num_classes, average=None
    ).to(device)
    recallmat = torchmetrics.classification.MulticlassRecall(
        num_classes=num_classes, average=None
    ).to(device)
    f1mat = torchmetrics.classification.MulticlassF1Score(
        num_classes=num_classes, average=None
    ).to(device)

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            feat, logi = model(features)
            targets = labels.argmax(dim=1) if labels.dim() > 1 else labels
            pred = logi.argmax(dim=1)
            confmat.update(pred, targets)
            accmat.update(pred, targets)
            precmat.update(pred, targets)
            recallmat.update(pred, targets)
            f1mat.update(pred, targets)

    acc = accmat.compute()
    cm = confmat.compute()
    per_class_precision = precmat.compute()
    per_class_recall = recallmat.compute()
    per_class_f1 = f1mat.compute()

    labels_for_display = class_labels if class_labels is not None else \
        [str(i) for i in range(num_classes)]

    if verbose:
        print(f'Test Accuracy: {acc * 100:.2f}%')
        for i, lbl in enumerate(labels_for_display[:num_classes]):
            print(f'  {lbl}: P={per_class_precision[i]:.4f} '
                  f'R={per_class_recall[i]:.4f} F1={per_class_f1[i]:.4f}')

    if save_cm_path:
        disp = ConfusionMatrixDisplay(
            cm.cpu().numpy(),
            display_labels=labels_for_display[:num_classes]
        )
        disp.plot(cmap='Blues')
        model_label = getattr(model, 'name', 'model')
        plt.title(f'{model_label}\n{100 * acc:.2f}%')
        plt.savefig(save_cm_path)
        if hasattr(plt.get_current_fig_manager(), 'window'):
            plt.show()
        else:
            plt.close()
        if verbose:
            print(f'Confusion matrix saved to {save_cm_path}')

    # ── 构建完整指标字典 ──
    metrics = {
        "accuracy": round(float(acc), 6),
        "macro_precision": round(float(per_class_precision.mean()), 6),
        "macro_recall": round(float(per_class_recall.mean()), 6),
        "macro_f1": round(float(per_class_f1.mean()), 6),
        "per_class": {}
    }
    for i, lbl in enumerate(labels_for_display[:num_classes]):
        metrics["per_class"][lbl] = {
            "precision": round(float(per_class_precision[i]), 4),
            "recall": round(float(per_class_recall[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
        }

    return metrics


def create_optimizer(model, lr=1e-3, weight_decay=1e-2, param_groups=None,
                     optimizer_type='adamw'):
    """创建优化器，支持自定义参数组（用于差速学习率）"""
    if param_groups is not None:
        params = param_groups
    else:
        params = model.parameters()

    if optimizer_type == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"未知优化器: {optimizer_type}，可选 adamw/adam/sgd")


def create_scheduler(optimizer, scheduler_type='cosine', T_0=15, T_mult=2):
    """创建学习率调度器

    CosineAnnealingWarmRestarts 忽略 step(metric) 中传入的 metric 值；
    ReduceLROnPlateau 需要 step(metric) 传入监控指标。
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_mult=2, T_0=15
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"未知调度器: {scheduler_type}，可选 cosine/plateau/none")
