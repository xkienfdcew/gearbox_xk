# -*- coding: utf-8 -*-
"""生成实验图表（细分版）：
fig1a/fig1b: 8 模型 × 4 损失 柱状图（1s / 2s 分开，柱顶标数值）
fig2a/2b/2c: CosFace / ArcFace / SubArcFace 参数扫描热力图（格内标数值）
fig3:       dual_lstm 增强方式对比（无增强 / online_aug / online_aug+mixup）
"""
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 中文字体 ──
for fp in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

OUT = 'report/figures'
os.makedirs(OUT, exist_ok=True)

# ── 收集数据 ──
all_res = []
for f in glob.glob('output/*/summary.json'):
    try:
        d = json.load(open(f, encoding='utf-8'))
        all_res.append(d)
    except:
        pass

def find(tag_prefix=None, model=None, ver=None, loss=None, no_tag=False):
    """查找匹配的实验，返回 (acc_mean, acc_std, f1) 列表"""
    out = []
    for d in all_res:
        t = d.get('tag', '')
        if tag_prefix and not t.startswith(tag_prefix): continue
        if no_tag and t: continue
        if model and d.get('model') != model: continue
        if ver and d.get('features_version') != ver: continue
        if loss and d.get('loss_type') != loss: continue
        out.append((d.get('cv_acc_mean', np.nan), d.get('cv_acc_std', np.nan), d.get('cv_f1_mean', np.nan)))
    return out

def best(tag_prefix=None, model=None, ver=None, loss=None, no_tag=False):
    """返回最优实验的 (acc, std, f1) 或 None"""
    r = find(tag_prefix, model, ver, loss, no_tag)
    if not r: return None
    return max(r, key=lambda x: x[0])

models = ['dual_lstm', 'dual_cnn', 'cnn', 'dual_lstm_se',
          'dual_cnn2_se', 'dual_cnn_se', 'cnn_se', 'lstm']
losses = ['cross_entropy', 'cosface', 'arcface', 'sub_arcface']
loss_labels = {'cross_entropy': 'CE', 'cosface': 'CosFace',
               'arcface': 'ArcFace', 'sub_arcface': 'SubArcFace'}
colors = {'cross_entropy': '#7f7f7f', 'cosface': '#1f77b4',
          'arcface': '#ff7f0e', 'sub_arcface': '#2ca02c'}

# ══════════════════════════════════════════════════════════
#  图 1a / 1b: 模型 × 损失（1s / 2s 独立图，柱顶标数值）
# ══════════════════════════════════════════════════════════
for ver, suffix, title in [('1.0', '1s', '1 秒数据集 (DS-1)'),
                           ('2.0', '2s', '2 秒数据集 (DS-2)')]:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(models))
    width = 0.2
    for i, loss in enumerate(losses):
        accs, stds = [], []
        for m in models:
            r = best('ce_all' if loss == 'cross_entropy' else 's4_', m, ver, loss)
            accs.append(r[0] if r else np.nan)
            stds.append(r[1] if r else np.nan)
        accs = np.array(accs); stds = np.array(stds)
        mask = ~np.isnan(accs)
        bars = ax.bar(x[mask] + (i - 1.5) * width, accs[mask], width,
                      yerr=stds[mask], label=loss_labels[loss],
                      color=colors[loss], capsize=2.5, alpha=0.9)
        # 柱顶标注数值（2 位小数）
        for b, v in zip(bars, accs[mask]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(50, 100)
    ax.set_title(f'模型 × 损失函数（最优参数）准确率对比 — {title}', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1{("a" if suffix == "1s" else "b")}_models_loss_{suffix}.png', dpi=150)
    plt.close(fig)
    print(f"[图1{suffix}] 已保存: fig1{('a' if suffix=='1s' else 'b')}_models_loss_{suffix}.png")

# ══════════════════════════════════════════════════════════
#  图 2a/2b/2c: 参数扫描热力图（独立、格内标数值）
# ══════════════════════════════════════════════════════════
def param_matrix(loss, s_list, m_list, get_param, k_filter=None):
    mat = np.full((len(s_list), len(m_list)), np.nan)
    for i, s in enumerate(s_list):
        for j, m in enumerate(m_list):
            for d in all_res:
                t = d.get('tag', '')
                if not t.startswith('s3_'): continue
                if d.get('model') != 'dual_lstm' or d.get('features_version') != '1.0': continue
                if d.get('loss_type') != loss: continue
                lc = d.get('loss_config', {})
                if k_filter and lc.get('sub_arcface_k') != k_filter: continue
                if get_param(lc) == (s, m):
                    mat[i, j] = d.get('cv_acc_mean', np.nan)
    return mat

def draw_heatmap(loss, s_list, m_list, get_param, title, k_filter=None):
    mat = param_matrix(loss, s_list, m_list, get_param, k_filter)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=75, vmax=92)
    ax.set_xticks(range(len(m_list)))
    ax.set_xticklabels([f'm={x}' for x in m_list], fontsize=11)
    ax.set_yticks(range(len(s_list)))
    ax.set_yticklabels([f's={x}' for x in s_list], fontsize=11)
    ax.set_xlabel('margin m', fontsize=12)
    ax.set_ylabel('scale s', fontsize=12)
    ax.set_title(title, fontsize=14)
    # 格内标数值（1 位小数）
    for i in range(len(s_list)):
        for j in range(len(m_list)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=10,
                        color='black', fontweight='bold' if v == np.nanmax(mat) else 'normal')
    # 标最优
    idx = np.unravel_index(np.nanargmax(mat), mat.shape)
    ax.plot(idx[1], idx[0], 'k*', markersize=18)
    fig.colorbar(im, ax=ax, shrink=0.85, label='Accuracy (%)')
    fig.tight_layout()
    _fname = {'CosFace': 'fig2a_cosface_param', 'ArcFace': 'fig2b_arcface_param',
              'SubCenterArcFace': 'fig2c_subarcface_param'}[title.split()[0]]
    fig.savefig(f'{OUT}/{_fname}.png', dpi=150)
    plt.close(fig)
    print(f"  {title}: 最优 s={s_list[idx[0]]}, m={m_list[idx[1]]} → {mat[idx]:.2f}%")

draw_heatmap('cosface', [16,32,64,128,256], [0.05,0.1,0.2,0.3,0.5],
             lambda lc: (lc.get('cosface_s'), lc.get('cosface_m')),
             'CosFace 参数扫描（dual_lstm, 1s, 5折CV）')
draw_heatmap('arcface', [16,32,64,128,256], [0.1,0.2,0.3,0.5,0.7],
             lambda lc: (lc.get('arcface_s'), lc.get('arcface_m')),
             'ArcFace 参数扫描（dual_lstm, 1s, 5折CV）')
draw_heatmap('sub_arcface', [32,64,128], [0.1,0.3,0.5],
             lambda lc: (lc.get('sub_arcface_s'), lc.get('sub_arcface_m')),
             'SubCenterArcFace 参数扫描（dual_lstm, 1s, K=4）', k_filter=4)

# ══════════════════════════════════════════════════════════
#  图 3: dual_lstm 增强方式对比（无增强 vs online_aug vs online_aug+mixup）
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, ver, suffix in [(axes[0], '1.0', '1s'), (axes[1], '2.0', '2s')]:
    groups = {
        '无增强 (Stage4)': [],
        '+ online_aug': [],
        '+ mixup 0.7': [],
    }
    stds = {'无增强 (Stage4)': [], '+ online_aug': [], '+ mixup 0.7': []}
    for loss in losses:
        r4 = best('s4_', 'dual_lstm', ver, loss)
        r5 = best('s5_', 'dual_lstm', ver, loss)
        rm = best(None, 'dual_lstm', ver, loss, no_tag=True)  # 用户手动批次(带mixup)
        groups['无增强 (Stage4)'].append(r4[0] if r4 else np.nan)
        groups['+ online_aug'].append(r5[0] if r5 else np.nan)
        groups['+ mixup 0.7'].append(rm[0] if rm else np.nan)
        stds['无增强 (Stage4)'].append(r4[1] if r4 else np.nan)
        stds['+ online_aug'].append(r5[1] if r5 else np.nan)
        stds['+ mixup 0.7'].append(rm[1] if rm else np.nan)
    x = np.arange(len(losses))
    width = 0.25
    clrs = ['#7f7f7f', '#ff7f0e', '#2ca02c']
    for i, (gname, vals) in enumerate(groups.items()):
        vals = np.array(vals); sd = np.array(stds[gname])
        bars = ax.bar(x + (i - 1) * width, vals, width, yerr=sd,
                      label=gname, color=clrs[i], capsize=3, alpha=0.9)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([loss_labels[l] for l in losses])
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(70, 100)
    ax.set_title(f'dual_lstm 增强方式对比 — {suffix}', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle('增强策略对 dual_lstm 的影响（4 种损失，5 折 CV）', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT}/fig3_aug_compare.png', dpi=150)
plt.close(fig)
print("\n[图3] 已保存: fig3_aug_compare.png")
print("\n全部图表生成完成！")
