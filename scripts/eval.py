"""
已训练模型评估脚本
==================
加载 output/ 下任意已训练的模型，在测试集上评估并输出完整指标。

用法:
    # 评估单个模型
    python scripts/eval.py --run output/dual_cnn_se_v1.0_7

    # 指定设备
    python scripts/eval.py --run output/resnet_se_v1.0 --device cpu

    # 批量评估 output/ 下所有模型，生成排行榜
    python scripts/eval.py --all

    # 批量评估 + 按测试准确率排序
    python scripts/eval.py --all --sort

    # 只评估前 N 个模型（按准确率）
    python scripts/eval.py --all --top 5

    # 列出所有可用模型
    python scripts/eval.py --list
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, worker_init_fn
from src.net_creator import create_model, FEAT_DIM_MAP, MODEL_REGISTRY, ModelWithHead
from src.train.trainer import evaluate_model
from src.train.losses import CosFaceHead, ArcFaceHead, SubCenterArcFaceHead


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def load_dataset(features_dir, version):
    """加载预构建的训练/测试特征集。"""
    train_path = os.path.join(features_dir, f"train_v{version}.pt")
    test_path = os.path.join(features_dir, f"test_v{version}.pt")
    map_path = os.path.join(features_dir, f"label_map_v{version}.pt")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试集不存在: {test_path}")

    train_feats, train_labels = torch.load(train_path, weights_only=False)
    test_feats, test_labels = torch.load(test_path, weights_only=False)
    label_map = torch.load(map_path, weights_only=False) if os.path.exists(map_path) else None

    return (train_feats, train_labels), (test_feats, test_labels), label_map


def load_config(run_dir):
    """加载运行目录中的 config.json。"""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_model(config, device):
    """根据 config 完全复原训练时的模型结构（backbone + head）。

    支持新旧两种 config 格式。
    """
    # ── 解析 config 字段 ──
    if "run" in config:
        # 新格式
        model_name = config["model"]["name"]
        head_type = config["model"]["head_type"]
        feat_dim = config["model"]["feat_dim"]
        num_classes = config["data"]["num_classes"]
        loss_cfg = config.get("loss", {})
    else:
        # 旧格式（兼容）
        model_name = config.get("model", "")
        head_type = config.get("head_type", "")
        feat_dim = config.get("feat_dim", FEAT_DIM_MAP.get(model_name, 128))
        num_classes = config.get("num_classes", 8)
        loss_cfg = {
            "type": config.get("loss", "cross_entropy"),
            "cosface_s": config.get("cosface_s", 64.0),
            "cosface_m": config.get("cosface_m", 0.20),
            "arcface_s": config.get("arcface_s", 64.0),
            "arcface_m": config.get("arcface_m", 0.10),
            "sub_arcface_s": config.get("sub_arcface_s", 64.0),
            "sub_arcface_m": config.get("sub_arcface_m", 0.10),
            "sub_arcface_k": config.get("sub_arcface_k", 3),
        }

    # ── 创建 backbone ──
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 '{model_name}'，已知: {list(MODEL_REGISTRY.keys())}")

    is_lstm = MODEL_REGISTRY[model_name][1]
    is_anfis = model_name.endswith('anfis')

    if is_anfis:
        if is_lstm:
            input_size = config.get("data", {}).get("input_size", 190)
            backbone = create_model(model_name, input_size=input_size, num_classes=num_classes)
        else:
            backbone = create_model(model_name, num_classes=num_classes)
        net = backbone  # ANFIS 自带分类头
    else:
        if is_lstm:
            input_size = config.get("data", {}).get("input_size", 190)
            backbone = create_model(model_name, input_size=input_size, num_classes=num_classes)
        else:
            backbone = create_model(model_name, num_classes=num_classes)

        # ── 创建分类头 ──
        if head_type == "CosFaceHead":
            s = loss_cfg.get("cosface_s", 64.0)
            m = loss_cfg.get("cosface_m", 0.20)
            head = CosFaceHead(feat_dim, num_classes, s=s, m=m)
        elif head_type == "ArcFaceHead":
            s = loss_cfg.get("arcface_s", 64.0)
            m = loss_cfg.get("arcface_m", 0.10)
            head = ArcFaceHead(feat_dim, num_classes, s=s, m=m)
        elif head_type == "SubCenterArcFaceHead":
            s = loss_cfg.get("sub_arcface_s", 64.0)
            m = loss_cfg.get("sub_arcface_m", 0.10)
            k = loss_cfg.get("sub_arcface_k", 3)
            head = SubCenterArcFaceHead(feat_dim, num_classes, K=k, s=s, m=m)
        else:
            head = nn.Linear(feat_dim, num_classes)

        net = ModelWithHead(backbone, head)

    net = net.to(device)
    return net, model_name


def load_model_weights(net, run_dir):
    """加载 best_model.pt 权重。"""
    model_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return net


def build_test_loader(test_feats, test_labels, batch_size=64, device="cuda"):
    """构建测试 DataLoader。"""
    test_dataset = TensorDataset(test_feats, test_labels)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      pin_memory=(device == "cuda"),
                      worker_init_fn=worker_init_fn)


def find_all_runs(output_dir):
    """扫描 output/ 下所有包含 config.json + best_model.pt 的运行目录。"""
    runs = []
    if not os.path.isdir(output_dir):
        return runs
    for entry in sorted(os.listdir(output_dir)):
        run_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        config_path = os.path.join(run_dir, "config.json")
        model_path = os.path.join(run_dir, "best_model.pt")
        if os.path.exists(config_path) and os.path.exists(model_path):
            runs.append(run_dir)
    return runs


def print_separator(title="", char="=", width=68):
    """打印分隔线。"""
    print()
    print(char * width)
    if title:
        print(f"  {title}")
        print(char * width)


def evaluate_single(run_dir, device="cuda", verbose=True):
    """评估单个已训练模型，返回指标字典。"""
    config = load_config(run_dir)

    # ── 提取基本信息 ──
    if "run" in config:
        run_name = config["run"]["name"]
        features_version = config["data"]["features_version"]
        num_classes = config["data"]["num_classes"]
        class_names = config["data"]["class_names"]
    else:
        run_name = os.path.basename(run_dir)
        features_version = config.get("features_version", "1.0")
        num_classes = config.get("num_classes", 8)
        class_names = config.get("class_names", [str(i) for i in range(num_classes)])

    if verbose:
        print_separator(f"评估: {run_name}")

    # ── 加载数据 ──
    features_dir = os.path.join(PROJECT_ROOT, "features")
    (train_feats, train_labels), (test_feats, test_labels), _ = \
        load_dataset(features_dir, features_version)

    if verbose:
        print(f"  特征版本:     v{features_version}")
        print(f"  测试样本数:   {test_feats.shape[0]}")
        print(f"  特征维度:     {list(test_feats.shape[1:])}")

    # ── 复原模型 ──
    net, model_name = reconstruct_model(config, device)
    net = load_model_weights(net, run_dir)
    net.eval()

    total_params = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f"  模型:         {model_name}")
        print(f"  参数量:       {total_params:,}")

    # ── 测试 ──
    test_loader = build_test_loader(test_feats, test_labels, batch_size=64, device=device)
    cm_path = os.path.join(run_dir, "confusion_matrix_reeval.png")

    metrics = evaluate_model(net, test_loader, device, num_classes,
                             class_labels=class_names,
                             save_cm_path=cm_path, verbose=verbose)

    # ── 打印汇总 ──
    if verbose:
        print(f"\n  ┌{'─' * 56}┐")
        print(f"  │  测试准确率:  {metrics['accuracy'] * 100:6.2f}%"
              f"{' ' * 37}│")
        print(f"  │  Macro-P:      {metrics['macro_precision'] * 100:6.2f}%"
              f"{' ' * 37}│")
        print(f"  │  Macro-R:      {metrics['macro_recall'] * 100:6.2f}%"
              f"{' ' * 37}│")
        print(f"  │  Macro-F1:     {metrics['macro_f1'] * 100:6.2f}%"
              f"{' ' * 37}│")
        print(f"  └{'─' * 56}┘")

        # ── 打印验证集历史最佳（如果 config 中有）──
        if "run" in config and "results" in config:
            results = config["results"]
            print(f"\n  训练时记录:")
            print(f"    最佳轮次:      {results.get('best_epoch', '?')}")
            print(f"    最佳验证Acc:   {results.get('best_val_acc', '?'):.2f}%" if isinstance(results.get('best_val_acc'), (int, float)) else f"    最佳验证Acc:   {results.get('best_val_acc', '?')}")
            print(f"    验证→测试差距: {results.get('best_val_acc', 0) - metrics['accuracy'] * 100:+.2f}%" if isinstance(results.get('best_val_acc'), (int, float)) else "")

    return metrics


def evaluate_all(output_dir, device="cuda", sort_by_acc=False, top_n=None):
    """批量评估 output/ 下所有模型。"""
    runs = find_all_runs(output_dir)

    if not runs:
        print("未找到任何可评估的模型（需要 config.json + best_model.pt）")
        return

    print_separator(f"批量评估: 共 {len(runs)} 个模型")
    print(f"  设备: {device}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    for i, run_dir in enumerate(runs):
        run_name = os.path.basename(run_dir)
        try:
            print(f"\n[{i + 1}/{len(runs)}] {run_name} ...", end=" ", flush=True)
            metrics = evaluate_single(run_dir, device=device, verbose=False)
            results.append({
                "run_name": run_name,
                "run_dir": run_dir,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                **{k: v for k, v in metrics.items() if k not in ("accuracy", "macro_f1")},
            })
            print(f"Acc={metrics['accuracy'] * 100:.2f}%  F1={metrics['macro_f1'] * 100:.2f}%")
        except Exception as e:
            print(f"失败: {e}")

    # ── 排序 ──
    if sort_by_acc:
        results.sort(key=lambda r: r["accuracy"], reverse=True)

    # ── 打印排行榜 ──
    print_separator("评估排行榜", "=")
    print(f"{'排名':<5} {'运行名称':<35} {'准确率':<10} {'Macro-F1':<10} {'每类最低F1':<15}")
    print("-" * 75)

    for rank, r in enumerate(results, 1):
        per_class = r.get("per_class", {})
        worst_cls = ""
        if per_class:
            worst = min(per_class.items(), key=lambda x: x[1].get("f1", 0))
            worst_cls = f"{worst[0]}:{worst[1]['f1']:.2f}"

        print(f"{rank:<5} {r['run_name']:<35} "
              f"{r['accuracy'] * 100:>6.2f}%   {r['macro_f1'] * 100:>6.2f}%   "
              f"{worst_cls:<15}")

    if top_n and len(results) > top_n:
        print(f"\n  (仅显示前 {top_n} 个，共 {len(results)} 个模型)")

    return results


def list_runs(output_dir):
    """列出所有可评估的运行目录及其关键信息。"""
    runs = find_all_runs(output_dir)

    if not runs:
        print("未找到任何可评估的模型。")
        return

    print_separator(f"可用模型: 共 {len(runs)} 个")
    print(f"{'运行名称':<35} {'模型':<18} {'准确率':<10} {'F1':<10} {'时间':<20}")
    print("-" * 95)

    for run_dir in runs:
        run_name = os.path.basename(run_dir)
        try:
            config = load_config(run_dir)
            if "run" in config:
                model = config["model"]["name"]
                acc = config["results"].get("accuracy", 0) * 100
                f1 = config["results"].get("macro_f1", 0) * 100
                ts = config["run"].get("timestamp_start", "-")[:19]
            else:
                model = config.get("model", "?")
                acc = config.get("test_accuracy", 0) * 100
                f1 = config.get("macro_f1", 0) * 100
                ts = config.get("timestamp_start", "-")[:19]
            print(f"{run_name:<35} {model:<18} {acc:>6.2f}%   {f1:>6.2f}%   {ts:<20}")
        except Exception as e:
            print(f"{run_name:<35} {'(读取失败)':<18} {'-':<10} {'-':<10} {str(e)[:20]:<20}")


# ══════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="评估已训练的电机故障诊断模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/eval.py --run output/dual_cnn_se_v1.0_7     # 评估单个模型
  python scripts/eval.py --all                                # 批量评估全部
  python scripts/eval.py --all --sort                         # 批量评估并按准确率排序
  python scripts/eval.py --all --top 5                        # 仅显示 Top-5
  python scripts/eval.py --list                               # 列出所有模型
        """,
    )

    parser.add_argument("--run", type=str, default=None,
                        help="要评估的运行目录路径，如 output/dual_cnn_se_v1.0_7")
    parser.add_argument("--all", action="store_true",
                        help="批量评估 output/ 下所有模型")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可评估的模型")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="模型输出目录 (默认: output/)")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备: auto / cuda / cpu (默认: auto)")
    parser.add_argument("--sort", action="store_true",
                        help="按测试准确率降序排列结果")
    parser.add_argument("--top", type=int, default=None,
                        help="只显示前 N 个结果")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")

    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    set_seed(args.seed)

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)

    if args.list:
        list_runs(output_dir)
    elif args.all:
        evaluate_all(output_dir, device=device, sort_by_acc=args.sort, top_n=args.top)
    elif args.run:
        run_dir = args.run
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(PROJECT_ROOT, run_dir)
        if not os.path.isdir(run_dir):
            print(f"错误: 目录不存在 — {run_dir}")
            sys.exit(1)
        evaluate_single(run_dir, device=device)
    else:
        parser.print_help()
        print("\n提示: 请指定 --run 或 --all 或 --list")


if __name__ == "__main__":
    main()
