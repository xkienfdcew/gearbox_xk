#!/bin/bash
# 大创实验总脚本（支持断点续跑）
# 用法: bash run_experiments_all.sh [1|2|3|all]
# 断点续跑: 完成后在 {实验目录}/.progress/ 生成 <tag>.done，重跑自动跳过
cd "$(dirname "$0")/.."
PY="python"
LOGDIR="logs"
mkdir -p "$LOGDIR"
EXP="$1"; EXP="${EXP:-all}"

run_once() {
    local TAG="$1"; shift
    if [ -f "$PROG_DIR/$TAG.done" ]; then
        echo "[跳过] $TAG"
        return 0
    fi
    echo "[运行] $TAG"
    $PY scripts/cv_train.py --cv 5 --epochs 200 --patience 30 \
        "$@" --tag "$TAG" >> "$LOG" 2>&1
    [ $? -eq 0 ] && touch "$PROG_DIR/$TAG.done" \
        || echo "  [失败] $TAG (重跑可续)"
}

exp_begin() {
    mkdir -p "$PROG_DIR"
    local dn=$(ls "$PROG_DIR"/*.done 2>/dev/null | wc -l)
    echo "════════════════════════════════════════"
    echo "  $1 → $OUT  [已完成 $dn 个]"
    echo "════════════════════════════════════════"
}
# ═══════════════ 实验一: 损失参数大网格 ═══════════════
exp1() {
    OUT="output/exp1_loss_param"
    LOG="$LOGDIR/exp1.log"
    PROG_DIR="$OUT/.progress"
    exp_begin "实验一: 损失参数搜索 (2模型×2数据集×77组)"
    for ver in 1.0 2.0; do
        for model in dual_lstm dual_cnn; do
            for s in 16 32 64 128 256; do for m in 0.05 0.1 0.2 0.3 0.5; do
                run_once "exp1_${model}_ds${ver}_cos_s${s}_m${m}" \
                    --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                    --loss cosface --cosface_s $s --cosface_m $m
            done; done
            for s in 16 32 64 128 256; do for m in 0.1 0.2 0.3 0.5 0.7; do
                run_once "exp1_${model}_ds${ver}_arc_s${s}_m${m}" \
                    --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                    --loss arcface --arcface_s $s --arcface_m $m
            done; done
            for s in 32 64 128; do for m in 0.1 0.3 0.5; do for k in 2 3 4; do
                run_once "exp1_${model}_ds${ver}_sub_s${s}_m${m}_k${k}" \
                    --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                    --loss sub_arcface --sub_arcface_s $s --sub_arcface_m $m --sub_arcface_k $k
            done; done; done
        done
    done
    $PY scripts/generate_exp_figures.py --exp 1 --out_dir "$OUT"
}
# ═══════════════ 实验二: 模型×损失 ═══════════════
read_best() {
    # read_best <json> <loss> <field>
    "$PY" -c "import json;print(json.load(open('$1'))['$2']['_best']['$3'])"
}
exp2() {
    OUT="output/exp2_model_loss"
    LOG="$LOGDIR/exp2.log"
    PROG_DIR="$OUT/.progress"
    exp_begin "实验二: 8模型×4损失×2数据集"
    MODELS="cnn cnn_se lstm dual_cnn dual_cnn_se dual_lstm"
    for ver in 1.0 2.0; do
        BP="output/exp1_loss_param/best_params_ds$ver.json"
        [ -f "$BP" ] || { echo "[错误] 缺 $BP，请先完成实验一"; exit 1; }
        CS=$(read_best $BP cosface s); CM=$(read_best $BP cosface m)
        AS=$(read_best $BP arcface s); AM=$(read_best $BP arcface m)
        SS=$(read_best $BP sub_arcface s); SM=$(read_best $BP sub_arcface m)
        SK=$(read_best $BP sub_arcface K)
        echo "  [DS-$ver 最优] cos($CS,$CM) arc($AS,$AM) sub($SS,$SM,K$SK)"
        for model in $MODELS; do
            run_once "exp2_${model}_ds${ver}_ce" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss cross_entropy
            run_once "exp2_${model}_ds${ver}_cos" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss cosface --cosface_s $CS --cosface_m $CM
            run_once "exp2_${model}_ds${ver}_arc" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM
            run_once "exp2_${model}_ds${ver}_sub" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss sub_arcface --sub_arcface_s $SS --sub_arcface_m $SM --sub_arcface_k $SK
        done
    done
    $PY scripts/generate_exp_figures.py --exp 2 --out_dir "$OUT"
}

# ═══════════════ 实验三: 增强对照 ═══════════════
exp3() {
    OUT="output/exp3_aug_mixup"
    LOG="$LOGDIR/exp3.log"
    PROG_DIR="$OUT/.progress"
    exp_begin "实验三: top3×4增强配置×2数据集"
    TOP="dual_lstm dual_cnn cnn"
    for ver in 1.0 2.0; do
        BP="output/exp1_loss_param/best_params_ds$ver.json"
        [ -f "$BP" ] || { echo "[错误] 缺 $BP，请先完成实验一"; exit 1; }
        AS=$(read_best $BP arcface s); AM=$(read_best $BP arcface m)
        for model in $TOP; do
            run_once "exp3_${model}_ds${ver}_base" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM
            run_once "exp3_${model}_ds${ver}_mixup" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM --mixup 0.7
            run_once "exp3_${model}_ds${ver}_aug" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM --online_aug
            run_once "exp3_${model}_ds${ver}_augmix" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM --online_aug --mixup 0.7
        done
    done
    $PY scripts/generate_exp_figures.py --exp 3 --out_dir "$OUT"
}

# ═══════════════ 执行 ═══════════════
echo "大创实验总脚本启动 $(date)"
case "$EXP" in
    1) exp1 ;;
    2) exp2 ;;
    3) exp3 ;;
    all) exp1; exp2; exp3; echo "全部完成 $(date)" ;;
    *) echo "用法: bash run_experiments_all.sh [1|2|3|all]"; exit 1 ;;
esac
