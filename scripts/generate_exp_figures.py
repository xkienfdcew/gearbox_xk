# -*- coding: utf-8 -*-
"""按实验生成图片（每个实验输出到自己的目录）。

用法:
  python scripts/generate_exp_figures.py --exp 1 --out_dir output/exp1_loss_param
  python scripts/generate_exp_figures.py --exp 2 --out_dir output/exp2_model_loss
  python scripts/generate_exp_figures.py --exp 3 --out_dir output/exp3_aug_mixup

实验一还会额外输出 best_params_ds1.json / best_params_ds2.json（供实验二/三引用）。
"""
import sys, os, json, glob, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fp in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

LOSS_NAMES = ['cross_entropy', 'cosface', 'arcface', 'sub_arcface']
LOSS_LABELS = {'cross_entropy': 'CE', 'cosface': 'CosFace',
               'arcface': 'ArcFace', 'sub_arcface': 'SubArcFace'}
COLORS = {'cross_entropy': '#7f7f7f', 'cosface': '#1f77b4',
          'arcface': '#ff7f0e', 'sub_arcface': '#2ca02c'}


def load_summaries(root):
    """递归收集 root 下所有 summary.json"""
    res = []
    for f in glob.glob(os.path.join(root, '**', 'summary.json'), recursive=True):
        try:
            d = json.load(open(f, encoding='utf-8'))
            if d.get('cv_acc_mean') is not None:
                d['_dir'] = os.path.basename(os.path.dirname(f))
                res.append(d)
        except Exception:
            pass
    return res


def find(results, model=None, ver=None, loss=None, tag_contains=None):
    out = []
    for d in results:
        if model and d.get('model') != model: continue
        if ver and d.get('features_version') != ver: continue
        if loss and d.get('loss_type') != loss: continue
        if tag_contains and tag_contains not in d.get('tag', ''): continue
        out.append(d)
    return out


def best(results, **kw):
    r = find(results, **kw)
    if not r: return None
    return max(r, key=lambda x: x.get('cv_acc_mean', 0))


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [图] {path}")


# ══════════════════════════════════════════════════════════
#  实验一: 热力图 + 最优参数 json
# ══════════════════════════════════════════════════════════
def exp1(out_dir):
    best_all = {}
    for ds in ['ds1', 'ds2']:
        root = os.path.join(out_dir, ds)
        results = load_summaries(root)
        if not results:
            print(f"  [警告] {root} 无结果，跳过")
            continue
        ver = '1.0' if ds == 'ds1' else '2.0'
        best_all[ds] = {}

        for model in ['dual_lstm', 'dual_cnn']:
            fig_dir = os.path.join(root, 'figures')
            # ── CosFace 热力图 ──
            s_list = [16, 32, 64, 128, 256]; m_list = [0.05, 0.1, 0.2, 0.3, 0.5]
            mat = np.full((len(s_list), len(m_list)), np.nan)
            for d in find(results, model=model, ver=ver, loss='cosface'):
                lc = d.get('loss_config', {})
                s, m = lc.get('cosface_s'), lc.get('cosface_m')
                if s in s_list and m in m_list:
                    mat[s_list.index(s), m_list.index(m)] = d.get('cv_acc_mean')
            _draw_heatmap(mat, s_list, m_list, f'{model} CosFace (DS-{ds[-1]})',
                          os.path.join(fig_dir, f'{model}_cosface_heatmap.png'))
            idx = np.unravel_index(np.nanargmax(mat), mat.shape)
            best_all[ds].setdefault('cosface', {})[model] = {
                's': s_list[idx[0]], 'm': m_list[idx[1]], 'acc': round(float(mat[idx]), 2)}

            # ── ArcFace 热力图 ──
            s_list = [16, 32, 64, 128, 256]; m_list = [0.1, 0.2, 0.3, 0.5, 0.7]
            mat = np.full((len(s_list), len(m_list)), np.nan)
            for d in find(results, model=model, ver=ver, loss='arcface'):
                lc = d.get('loss_config', {})
                s, m = lc.get('arcface_s'), lc.get('arcface_m')
                if s in s_list and m in m_list:
                    mat[s_list.index(s), m_list.index(m)] = d.get('cv_acc_mean')
            _draw_heatmap(mat, s_list, m_list, f'{model} ArcFace (DS-{ds[-1]})',
                          os.path.join(fig_dir, f'{model}_arcface_heatmap.png'))
            idx = np.unravel_index(np.nanargmax(mat), mat.shape)
            best_all[ds].setdefault('arcface', {})[model] = {
                's': s_list[idx[0]], 'm': m_list[idx[1]], 'acc': round(float(mat[idx]), 2)}

            # ── SubCenterArcFace 热力图（每 (s,m) 取最佳 K） ──
            s_list = [32, 64, 128]; m_list = [0.1, 0.3, 0.5]
            mat = np.full((len(s_list), len(m_list)), np.nan)
            k_map = {}
            for d in find(results, model=model, ver=ver, loss='sub_arcface'):
                lc = d.get('loss_config', {})
                s, m = lc.get('sub_arcface_s'), lc.get('sub_arcface_m')
                if s in s_list and m in m_list:
                    v = d.get('cv_acc_mean')
                    if np.isnan(mat[s_list.index(s), m_list.index(m)]) or v > mat[s_list.index(s), m_list.index(m)]:
                        mat[s_list.index(s), m_list.index(m)] = v
                        k_map[(s_list.index(s), m_list.index(m))] = lc.get('sub_arcface_k')
            _draw_heatmap(mat, s_list, m_list, f'{model} SubCenterArcFace (DS-{ds[-1]})',
                          os.path.join(fig_dir, f'{model}_subarcface_heatmap.png'))
            idx = np.unravel_index(np.nanargmax(mat), mat.shape)
            best_all[ds].setdefault('sub_arcface', {})[model] = {
                's': s_list[idx[0]], 'm': m_list[idx[1]],
                'K': k_map.get(idx, 3), 'acc': round(float(mat[idx]), 2)}

    # 最优参数汇总（跨模型取 acc 最高者）
    for ds in best_all:
        for loss in ['cosface', 'arcface', 'sub_arcface']:
            if loss in best_all[ds]:
                vals = best_all[ds][loss]
                if vals:
                    best_model = max(vals, key=lambda m: vals[m]['acc'])
                    best_all[ds][loss]['_best'] = vals[best_model]
                    best_all[ds][loss]['_best_model'] = best_model
        with open(os.path.join(out_dir, f'best_params_{ds}.json'), 'w', encoding='utf-8') as f:
            json.dump(best_all[ds], f, indent=2, ensure_ascii=False)
    print(f"  [参数] 最优参数已写入 {out_dir}/best_params_ds1.json / ds2.json")


def _draw_heatmap(mat, s_list, m_list, title, path):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=75, vmax=92)
    ax.set_xticks(range(len(m_list))); ax.set_xticklabels([f'm={x}' for x in m_list], fontsize=11)
    ax.set_yticks(range(len(s_list))); ax.set_yticklabels([f's={x}' for x in s_list], fontsize=11)
    ax.set_xlabel('margin m', fontsize=12); ax.set_ylabel('scale s', fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(len(s_list)):
        for j in range(len(m_list)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=10,
                        fontweight='bold' if v == np.nanmax(mat) else 'normal')
    idx = np.unravel_index(np.nanargmax(mat), mat.shape)
    ax.plot(idx[1], idx[0], 'k*', markersize=18)
    fig.colorbar(im, ax=ax, shrink=0.85, label='Accuracy (%)')
    fig.tight_layout()
    save_fig(fig, path)


# ══════════════════════════════════════════════════════════
#  实验二: 模型 × 损失条形图（单分支/双分支，每数据集各 2 张）
# ══════════════════════════════════════════════════════════
def exp2(out_dir):
    single = ['cnn', 'cnn_se', 'cnn2_se', 'lstm']
    double = ['dual_cnn', 'dual_cnn_se', 'dual_cnn2_se', 'dual_lstm']
    for ds in ['ds1', 'ds2']:
        root = os.path.join(out_dir, ds)
        results = load_summaries(root)
        if not results:
            print(f"  [警告] {root} 无结果，跳过")
            continue
        ver = '1.0' if ds == 'ds1' else '2.0'
        fig_dir = os.path.join(root, 'figures')
        for name, models in [('single_branch', single), ('double_branch', double)]:
            _draw_bar(results, models, ver, f'{name} 模型 × 损失 (DS-{ds[-1]})',
                      os.path.join(fig_dir, f'{name}.png'))


def _draw_bar(results, models, ver, title, path):
    x = np.arange(len(models)); width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, loss in enumerate(LOSS_NAMES):
        accs, stds = [], []
        for m in models:
            r = best(results, model=m, ver=ver, loss=loss)
            accs.append(r.get('cv_acc_mean') if r else np.nan)
            stds.append(r.get('cv_acc_std') if r else np.nan)
        accs = np.array(accs); stds = np.array(stds)
        mask = ~np.isnan(accs)
        bars = ax.bar(x[mask] + (i - 1.5) * width, accs[mask], width, yerr=stds[mask],
                      label=LOSS_LABELS[loss], color=COLORS[loss], capsize=2.5, alpha=0.9)
        for b, v in zip(bars, accs[mask]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.6,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, rotation=15)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(50, 100)
    ax.set_title(title, fontsize=13); ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, path)


# ══════════════════════════════════════════════════════════
#  实验三: 增强配置对比（每数据集 1 张）
# ══════════════════════════════════════════════════════════
def exp3(out_dir):
    for ds in ['ds1', 'ds2']:
        root = os.path.join(out_dir, ds)
        results = load_summaries(root)
        if not results:
            print(f"  [警告] {root} 无结果，跳过")
            continue
        ver = '1.0' if ds == 'ds1' else '2.0'
        fig_dir = os.path.join(root, 'figures')
        models = sorted(set(d.get('model') for d in results))
        x = np.arange(len(models)); width = 0.2
        fig, ax = plt.subplots(figsize=(11, 6))
        cfg_names = ['base', 'mixup', 'aug', 'aug+mixup']
        cfg_colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c']
        for i, cfg in enumerate(cfg_names):
            accs, stds = [], []
            for m in models:
                r = None
                for d in results:
                    t = d.get('tag', '')
                    if d.get('model') == m and d.get('features_version') == ver and \
                       t.endswith('_' + cfg):
                        r = d; break
                accs.append(r.get('cv_acc_mean') if r else np.nan)
                stds.append(r.get('cv_acc_std') if r else np.nan)
            accs = np.array(accs); stds = np.array(stds)
            mask = ~np.isnan(accs)
            bars = ax.bar(x[mask] + (i - 1.5) * width, accs[mask], width, yerr=stds[mask],
                          label=cfg, color=cfg_colors[i], capsize=2.5, alpha=0.9)
            for b, v in zip(bars, accs[mask]):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, rotation=15)
        ax.set_ylabel('Accuracy (%)'); ax.set_ylim(50, 100)
        ax.set_title(f'增强配置对比 (DS-{ds[-1]})', fontsize=13); ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        save_fig(fig, os.path.join(fig_dir, 'aug_compare.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    print(f"=== 生成实验{a.exp}图片: {a.out_dir} ===")
    {1: exp1, 2: exp2, 3: exp3}[a.exp](a.out_dir)
    print("完成!")


if __name__ == '__main__':
    main()
