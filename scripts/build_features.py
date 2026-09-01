"""
完整特征集构建脚本
==================
从 data/ 下的原始 .mat 文件读取信号 → 切分 → 提取特征 → 保存为 .pt 文件。

版本号自动管理：扫描输出目录中已有的版本，自动递增（如 1.0 → 1.1）。
每次构建同时更新 features/DATASETS.md，记录所有版本的参数与统计信息。

用法:
    python scripts/build_features.py                          # 使用默认参数，版本号自动递增
    python scripts/build_features.py --piece_sec 3 --sr 42000  # 自定义参数
    python scripts/build_features.py --no_aug                  # 不做数据增强
    python scripts/build_features.py --version 2.0             # 手动指定版本号
    python scripts/build_features.py --tag "尝试 mel=128"      # 为本次构建添加备注
    python scripts/build_features.py --no_normalize            # 关闭按通道 z-score 归一化
    python scripts/build_features.py --help                    # 查看所有选项
"""

import os
import re
import sys
import json
from datetime import datetime

import click
import torch

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed
from src.dataset.feature import (
    read_rawdata,
    slice_signal,
    split_dataset,
    build_features,
    build_all_features,
    save_dataset,
    print_dataset_info,
)


# ══════════════════════════════════════════════════════════════
#  版本号管理
# ══════════════════════════════════════════════════════════════

def _scan_existing_versions(output_dir):
    """扫描输出目录，返回已存在的版本号列表（按数值排序）。"""
    if not os.path.isdir(output_dir):
        return []
    pattern = re.compile(r"^train_v(\d+\.\d+)\.pt$")
    versions = []
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            try:
                versions.append(float(m.group(1)))
            except ValueError:
                pass
    return sorted(set(versions))


def _resolve_version(output_dir, explicit_version):
    """
    决定本次构建的版本号。

    优先级:
        1. 用户显式传入 --version → 使用该版本号
        2. 输出目录已有版本 → 取最大版本号 + 0.1
        3. 输出目录为空 → 起始版本 1.0
    """
    if explicit_version is not None:
        return explicit_version

    existing = _scan_existing_versions(output_dir)
    if existing:
        latest = existing[-1]
        next_ver = latest + 0.1
        # 浮点精度修正: 2.3 → 2.4, 1.9 → 2.0
        next_ver = round(next_ver, 1)
        return next_ver
    else:
        return 1.0


# ══════════════════════════════════════════════════════════════
#  变更日志
# ══════════════════════════════════════════════════════════════

CHANGELOG_HEADER = """# 特征集版本记录

> 自动生成于 {output_dir}/
> 每次运行 `python scripts/build_features.py` 时更新

| 版本 | 时间 | sr | piece_sec | overlap | n_mels | n_mfcc | PCEN | test_size | 增强 | 训练样本 | 测试样本 | 特征形状 | 类别数 | 备注 |
|------|------|----|-----------|---------|--------|--------|------|-----------|------|----------|----------|----------|--------|------|
"""


def _update_changelog(output_dir, version, params, stats):
    """
    更新 DATASETS.md：读取已有记录 → 更新 / 插入当前版本行 → 重写。

    Args:
        output_dir: 输出目录路径
        version:    版本号 (float)
        params:     dict，本次构建参数
        stats:      dict，本次构建统计 (train_samples, test_samples, feat_shape, num_classes)
    """
    changelog_path = os.path.join(output_dir, "DATASETS.md")
    os.makedirs(output_dir, exist_ok=True)

    ver_str = f"{version:.1f}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    aug_on = "是" if params.get("aug_enabled") else "否"
    pcen_on = "是" if params.get("use_pcen") else "否"

    new_row = (
        f"| {ver_str} "
        f"| {timestamp} "
        f"| {params['sr']} "
        f"| {params['piece_sec']} "
        f"| {params['overlap']} "
        f"| {params['n_mels']} "
        f"| {params['n_mfcc']} "
        f"| {pcen_on} "
        f"| {params['test_size']} "
        f"| {aug_on} "
        f"| {stats['train_samples']} "
        f"| {stats['test_samples']} "
        f"| {stats['feat_shape']} "
        f"| {stats['num_classes']} "
        f"| {params.get('tag', '-')} |\n"
    )

    if not os.path.exists(changelog_path):
        # 新建文件
        with open(changelog_path, "w", encoding="utf-8") as f:
            f.write(CHANGELOG_HEADER.format(output_dir=os.path.basename(output_dir)))
            f.write(new_row)
    else:
        # 更新已有文件：如果该版本已存在则替换对应行，否则追加
        with open(changelog_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 查找该版本是否已有记录行
        updated = False
        new_lines = []
        for line in lines:
            if line.startswith(f"| {ver_str} "):
                new_lines.append(new_row)
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(new_row)

        with open(changelog_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return changelog_path


# ══════════════════════════════════════════════════════════════
#  命令行接口
# ══════════════════════════════════════════════════════════════

@click.command()
@click.option("--data_dir", default="data",
              help="原始 .mat 数据根目录 (默认: data/)")
@click.option("--output_dir", default="features",
              help="输出 .pt 文件的目录 (默认: features/)")
@click.option("--sr", default=42000, type=int,
              help="采样率 (默认: 42000)")
@click.option("--piece_sec", default=2.0, type=float,
              help="每段信号长度/秒 (默认: 2.0)")
@click.option("--overlap", default=0.5, type=float,
              help="片段重叠率 (默认: 0.5)")
@click.option("--n_mels", default=64, type=int,
              help="Mel 频带数 (默认: 64)")
@click.option("--n_mfcc", default=42, type=int,
              help="MFCC 系数个数 (默认: 42)")
@click.option("--use_pcen", is_flag=True,
              help="使用 PCEN 归一化替代 amplitude_to_db (默认: 关闭)")
@click.option("--normalize/--no_normalize", default=False,
              help="按通道 z-score 归一化特征，统计量来自训练集 (默认: 关闭)")
@click.option("--test_size", default=0.2, type=float,
              help="测试集占比 (默认: 0.2)")
@click.option("--seed", default=42, type=int,
              help="全局随机种子，保证可复现 (默认: 42)")
@click.option("--random_state", default=42, type=int,
              help="train_test_split 的随机种子 (默认: 42)")
@click.option("--version", default=None, type=float,
              help="手动指定版本号（默认自动递增，如 1.0 → 1.1）")
@click.option("--tag", default=None, type=str,
              help="本次构建的备注标签，会记录到 DATASETS.md")
@click.option("--no_aug", is_flag=True,
              help="关闭数据增强（训练集也不做增强）")
@click.option("--all", "all_signals", is_flag=True,
              help="构建全量特征集（不划分训练/测试，保存 full_vX.pt，供交叉验证使用）")
@click.option("--gpu", is_flag=True,
              help="强制使用 GPU（默认优先 CPU 存盘）")
@click.option("--verbose/--quiet", default=True,
              help="打印详细信息 (默认: verbose)")
def main(data_dir, output_dir, sr, piece_sec, overlap, n_mels, n_mfcc,
         use_pcen, normalize, test_size, seed, random_state, version, tag, no_aug,
         all_signals, gpu, verbose):
    """从原始 .mat 文件构建训练/测试特征集。"""

    set_seed(seed)
    t_start = datetime.now()

    # ── 0. 解析版本号 ──
    output_path = os.path.join(PROJECT_ROOT, output_dir)
    dataset_version = _resolve_version(output_path, version)
    ver_str = f"{dataset_version:.1f}"

    click.echo(f"  数据集版本:   {ver_str}"
               f"{' (自动递增)' if version is None else ' (手动指定)'}")

    # ── 1. 读取原始数据 ──
    click.echo()
    click.echo("=" * 60)
    click.echo("  Step 1/5: 读取原始数据")
    click.echo("=" * 60)

    data_path = os.path.join(PROJECT_ROOT, data_dir)
    signals, labels, label_map = read_rawdata(data_path)

    # 逆映射 {idx: name}，供 build_features 确定增强倍数
    idx_to_name = {v: k for k, v in label_map.items()}

    click.echo(f"  数据目录:     {data_path}")
    click.echo(f"  故障类型数:   {len(label_map)}")
    for name, idx in label_map.items():
        count = labels.count(idx)
        click.echo(f"    [{idx}] {name}: {count} 条信号")
    click.echo(f"  总信号数:     {len(signals)}")

    # ── 2. 划分训练/测试集（--all 模式跳过） ──
    click.echo()
    click.echo("=" * 60)
    if all_signals:
        click.echo("  Step 2/5: 全量模式（不划分训练/测试，供交叉验证）")
        click.echo("=" * 60)
        datasets = None
    else:
        click.echo("  Step 2/5: 划分训练/测试集")
        click.echo("=" * 60)
        datasets = split_dataset(signals, labels,
                                 test_size=test_size,
                                 random_state=random_state)
        for phase, (sigs, labs) in datasets.items():
            click.echo(f"  {phase}: {len(sigs)} 条信号")

    # ── 3. 估算片段数量 ──
    click.echo()
    click.echo("=" * 60)
    click.echo("  Step 3/5: 估算与配置")
    click.echo("=" * 60)

    # 取一条信号估算每段可切多少片段
    sample_segments = slice_signal(signals[0], piece_sec, sr, overlap)
    click.echo(f"  采样率:       {sr} Hz")
    click.echo(f"  片段长度:     {piece_sec} 秒 = {int(piece_sec * sr)} 采样点")
    click.echo(f"  重叠率:       {overlap}")
    click.echo(f"  每条信号约:   {len(sample_segments)} 个片段")
    click.echo(f"  Mel 频带:     {n_mels}")
    click.echo(f"  MFCC 系数:    {n_mfcc}")
    click.echo(f"  PCEN 归一化:  {'开启' if use_pcen else '关闭 (dB)'}")
    click.echo(f"  按通道 z-score: {'开启 (统计量来自训练集)' if normalize else '关闭'}")
    click.echo(f"  数据增强:     {'关闭' if no_aug else '开启'}")

    # 增强概率配置（no_aug 时传 None）
    aug_pro_dict = None if no_aug else {
        "noise": 0.5,
        "translate": 0.0,
        "stretch": 0.0,
        "scale": 0.5,
        "pitch": 0.0,
    }

    # ── 4. 构建特征 ──
    click.echo()
    click.echo("=" * 60)
    click.echo("  Step 4/5: 提取特征" + ("（全量，无增强）" if all_signals else " & 数据增强"))
    click.echo("=" * 60)

    if all_signals:
        # 全量模式：不切分、不归一化（归一化由 CV 训练脚本按每折训练集动态计算，避免信息泄露）
        if normalize:
            click.echo("  [提示] --all 模式下忽略 --normalize（交叉验证需在每折训练集上单独归一化）")
        train_feats, train_labels, signal_ids = build_all_features(
            signals, labels,
            sr=sr,
            piece_sec=piece_sec,
            overlap=overlap,
            aug_pro_dict=aug_pro_dict,
            label_names=idx_to_name if not no_aug else None,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            use_pcen=use_pcen,
            normalize=False,
        )
        test_feats, test_labels = None, None
        click.echo(f"  全量特征:     {train_feats.shape}")
        click.echo(f"  全量标签:     {train_labels.shape}")
        click.echo(f"  信号归属:     {signal_ids.shape} (唯一信号 {len(torch.unique(signal_ids))} 条)")
    else:
        train_feats, train_labels, test_feats, test_labels = build_features(
            datasets,
            sr=sr,
            piece_sec=piece_sec,
            overlap=overlap,
            aug_pro_dict=aug_pro_dict,
            label_names=idx_to_name if not no_aug else None,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            use_pcen=use_pcen,
            normalize=normalize,
        )
        click.echo(f"  训练集特征:   {train_feats.shape}")
        click.echo(f"  训练集标签:   {train_labels.shape}")
        click.echo(f"  测试集特征:   {test_feats.shape}")
        click.echo(f"  测试集标签:   {test_labels.shape}")

    # ── 5. 保存 ──
    click.echo()
    click.echo("=" * 60)
    click.echo("  Step 5/5: 保存 & 写文档")
    click.echo("=" * 60)

    if all_signals:
        full_path = os.path.join(output_path, f"full_v{ver_str}.pt")
        torch.save((train_feats, train_labels, signal_ids), full_path)
        map_path = os.path.join(output_path, f"label_map_v{ver_str}.pt")
        torch.save(label_map, map_path)
        click.echo(f"  full:       {full_path}")
        click.echo(f"  label_map:  {map_path}")
    else:
        train_path = save_dataset(train_feats, train_labels,
                                  output_path, ver_str, "train")
        test_path = save_dataset(test_feats, test_labels,
                                 output_path, ver_str, "test")

        # 额外保存 label_map 供后续使用
        map_path = os.path.join(output_path, f"label_map_v{ver_str}.pt")
        torch.save(label_map, map_path)

        click.echo(f"  train:      {train_path}")
        click.echo(f"  test:       {test_path}")
        click.echo(f"  label_map:  {map_path}")

    # ── 更新 DATASETS.md ──
    tag_str = tag or "-"
    if normalize:
        tag_str = (tag + "，按通道z-score归一化") if tag else "按通道z-score归一化"

    params = {
        "sr": sr,
        "piece_sec": piece_sec,
        "overlap": overlap,
        "n_mels": n_mels,
        "n_mfcc": n_mfcc,
        "use_pcen": use_pcen,
        "normalize": normalize,
        "test_size": test_size,
        "aug_enabled": not no_aug,
        "tag": tag_str,
    }
    stats = {
        "train_samples": train_feats.shape[0],
        "test_samples": test_feats.shape[0] if test_feats is not None else 0,
        "feat_shape": str(list(train_feats.shape[1:])),
        "num_classes": len(label_map),
    }
    if all_signals:
        stats["signal_count"] = len(torch.unique(signal_ids))
    changelog_path = _update_changelog(output_path, dataset_version, params, stats)
    click.echo(f"  changelog:  {changelog_path}")

    # ── 汇总 ──
    elapsed = (datetime.now() - t_start).total_seconds()
    click.echo()
    click.echo("=" * 60)
    click.echo(f"  版本 {ver_str} 完成! 总耗时 {elapsed:.1f} 秒")
    click.echo("=" * 60)

    print_dataset_info(output_path, ver_str)


if __name__ == "__main__":
    main()
