#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  大创实验总脚本 v2（并行版，支持断点续跑）
#  用法: JOBS=4 bash run_experiments_all.sh [1|2|3|all]
#    JOBS: 并发进程数（默认 0 = 按 GPU 显存自动估算）
#  断点续跑: {实验目录}/.progress/<tag>.done，重跑自动跳过
#  服务器: JOBS=8 nohup bash run_experiments_all.sh > logs/all.log 2>&1 &
# ══════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.."

# ── Python 自动探测（Windows / Linux 通用） ──
# 优先级: 环境变量 PYTHON > .conda/python.exe(Win) > .conda/bin/python(Linux) > python3
if [ -n "$PYTHON" ]; then
    PY="$PYTHON"
elif [ -f ".conda/python.exe" ]; then
    PY=".conda/python.exe"
elif [ -f ".conda/bin/python" ]; then
    PY=".conda/bin/python"
else
    PY="python3"
fi
LOGDIR="logs"
mkdir -p "$LOGDIR"
JOBS="${JOBS:-0}"
EXP="$1"; EXP="${EXP:-all}"

# 并发执行任务文件（.done 由调度器管理）
dispatch() {
    "$PY" scripts/run_tasks.py --tasks "$TASKFILE" \
        --prog_dir "$PROG_DIR" --log "$LOG" --jobs "$JOBS"
}

# 清空并追加任务行（跳过已完成）
task() {
    local TAG="$1"; shift
    [ -f "$PROG_DIR/$TAG.done" ] && return 0
    echo "$TAG $*" >> "$TASKFILE"
}
# ═══════════ 实验一: 损失参数大网格（并行） ═══════════
exp1() {
    OUT="output/exp1_loss_param"
    LOG="$LOGDIR/exp1.log"
    PROG_DIR="$OUT/.progress"
    TASKFILE="$OUT/tasks.txt"
    mkdir -p "$PROG_DIR" "$OUT/ds1" "$OUT/ds2"
    : > "$TASKFILE"
    echo "════════════════════════════════════════"
    echo "  实验一: 2模型×2数据集×77组 (并发 $JOBS)"
    echo "════════════════════════════════════════"
    for ver in 1.0 2.0; do
        for model in dual_lstm dual_cnn; do
            for s in 16 32 64 128 256; do for m in 0.05 0.1 0.2 0.3 0.5; do
                task "exp1_${model}_ds${ver}_cos_s${s}_m${m}" \
                    --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                    --loss cosface --cosface_s $s --cosface_m $m
            done; done
            for s in 16 32 64 128 256; do for m in 0.1 0.2 0.3 0.5 0.7; do
                task "exp1_${model}_ds${ver}_arc_s${s}_m${m}" \
                    --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                    --loss arcface --arcface_s $s --arcface_m $m
            done; done
            for s in 32 64 128; do for m in 0.1 0.3 0.5; do for k in 2 3 4; do
                task "exp1_${model}_ds${ver}_sub_s${s}_m${m}_k${k}" \
                    --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                    --loss sub_arcface --sub_arcface_s $s --sub_arcface_m $m --sub_arcface_k $k
            done; done; done
        done
    done
    dispatch
    "$PY" scripts/generate_exp_figures.py --exp 1 --out_dir "$OUT"
}
# ═══════════ 实验二: 模型×损失（并行） ═══════════
read_best() {
    "$PY" -c "import json;print(json.load(open('$1'))['$2']['_best']['$3'])"
}
exp2() {
    OUT="output/exp2_model_loss"
    LOG="$LOGDIR/exp2.log"
    PROG_DIR="$OUT/.progress"
    TASKFILE="$OUT/tasks.txt"
    mkdir -p "$PROG_DIR" "$OUT/ds1" "$OUT/ds2"
    : > "$TASKFILE"
    echo "════════════════════════════════════════"
    echo "  实验二: 8模型×4损失×2数据集 (并发 $JOBS)"
    echo "════════════════════════════════════════"
    MODELS="cnn cnn_se cnn2_se lstm dual_cnn dual_cnn_se dual_cnn2_se dual_lstm"
    for ver in 1.0 2.0; do
        BP="output/exp1_loss_param/best_params_ds$ver.json"
        [ -f "$BP" ] || { echo "[错误] 缺 $BP，请先完成实验一"; return 1; }
        CS=$(read_best $BP cosface s); CM=$(read_best $BP cosface m)
        AS=$(read_best $BP arcface s); AM=$(read_best $BP arcface m)
        SS=$(read_best $BP sub_arcface s); SM=$(read_best $BP sub_arcface m)
        SK=$(read_best $BP sub_arcface K)
        echo "  [DS-$ver 最优] cos($CS,$CM) arc($AS,$AM) sub($SS,$SM,K$SK)"
        for model in $MODELS; do
            task "exp2_${model}_ds${ver}_ce" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss cross_entropy
            task "exp2_${model}_ds${ver}_cos" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss cosface --cosface_s $CS --cosface_m $CM
            task "exp2_${model}_ds${ver}_arc" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM
            task "exp2_${model}_ds${ver}_sub" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss sub_arcface --sub_arcface_s $SS --sub_arcface_m $SM --sub_arcface_k $SK
        done
    done
    dispatch
    "$PY" scripts/generate_exp_figures.py --exp 2 --out_dir "$OUT"
}

# ═══════════ 实验三: 增强对照（并行） ═══════════
exp3() {
    OUT="output/exp3_aug_mixup"
    LOG="$LOGDIR/exp3.log"
    PROG_DIR="$OUT/.progress"
    TASKFILE="$OUT/tasks.txt"
    mkdir -p "$PROG_DIR" "$OUT/ds1" "$OUT/ds2"
    : > "$TASKFILE"
    echo "════════════════════════════════════════"
    echo "  实验三: top3×4增强配置×2数据集 (并发 $JOBS)"
    echo "════════════════════════════════════════"
    TOP="dual_lstm dual_cnn cnn"
    for ver in 1.0 2.0; do
        BP="output/exp1_loss_param/best_params_ds$ver.json"
        [ -f "$BP" ] || { echo "[错误] 缺 $BP，请先完成实验一"; return 1; }
        AS=$(read_best $BP arcface s); AM=$(read_best $BP arcface m)
        for model in $TOP; do
            task "exp3_${model}_ds${ver}_base" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM
            task "exp3_${model}_ds${ver}_mixup" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM --mixup 0.7
            task "exp3_${model}_ds${ver}_aug" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM --online_aug
            task "exp3_${model}_ds${ver}_augmix" \
                --features_version $ver --model $model --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $AS --arcface_m $AM --online_aug --mixup 0.7
        done
    done
    dispatch
    "$PY" scripts/generate_exp_figures.py --exp 3 --out_dir "$OUT"
}

# ═══════════ 执行 ═══════════
echo "大创实验总脚本 v2 (并行) 启动 $(date) | 并发: $JOBS"
case "$EXP" in
    1) exp1 ;;
    2) exp2 ;;
    3) exp3 ;;
    all) exp1 && exp2 && exp3 && echo "全部完成 $(date)" ;;
    *) echo "用法: JOBS=N bash run_experiments_all.sh [1|2|3|all]"; exit 1 ;;
esac
