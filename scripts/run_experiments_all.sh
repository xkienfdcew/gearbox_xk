#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  大创实验总脚本（全新实验，服务器一次跑完）
#  对应: Obsidian 实验设计.md
#
#  用法: bash run_experiments_all.sh          # 一次跑完三个实验
#        bash run_experiments_all.sh 1|2|3    # 只跑指定实验
#
#  输出结构:
#    output/exp1_loss_param/ds1,ds2/  + figures/ + best_params_ds{1,2}.json
#    output/exp2_model_loss/ds1,ds2/  + figures/
#    output/exp3_aug_mixup/ds1,ds2/   + figures/
#
#  推荐服务器运行: nohup bash run_experiments_all.sh > logs/exp_all.log 2>&1 &
# ══════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.."   # 项目根目录
PY=".conda/python.exe"
LOGDIR="logs"
mkdir -p "$LOGDIR"
EXP="$1"; EXP="${EXP:-all}"

# 实验一参数网格
COSFACE_SS="16 32 64 128 256"; COSFACE_MS="0.05 0.1 0.2 0.3 0.5"
ARCFACE_SS="16 32 64 128 256"; ARCFACE_MS="0.1 0.2 0.3 0.5 0.7"
SUB_SS="32 64 128"; SUB_MS="0.1 0.3 0.5"; SUB_KS="2 3 4"

# ══════════════════════════════════════════════════════════════
#  实验一: 损失参数大网格（2 模型 × 2 数据集 × 77 组 = 308 个）
# ══════════════════════════════════════════════════════════════
exp1() {
    OUT="output/exp1_loss_param"
    LOG="$LOGDIR/exp1.log"
    echo "=== 实验一 开始 $(date) ===" >> "$LOG"
    echo "[实验一] 损失参数大网格搜索 -> $OUT"

    for ver in 1.0 2.0; do
        for model in dual_lstm dual_cnn; do
            # CosFace 25
            for s in $COSFACE_SS; do for m in $COSFACE_MS; do
                $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                    --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                    --loss cosface --cosface_s $s --cosface_m $m \
                    --tag "exp1_${model}_ds${ver}_cosface_s${s}_m${m}" >> "$LOG" 2>&1
            done; done
            # ArcFace 25
            for s in $ARCFACE_SS; do for m in $ARCFACE_MS; do
                $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                    --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                    --loss arcface --arcface_s $s --arcface_m $m \
                    --tag "exp1_${model}_ds${ver}_arcface_s${s}_m${m}" >> "$LOG" 2>&1
            done; done
            # SubCenterArcFace 27
            for s in $SUB_SS; do for m in $SUB_MS; do for k in $SUB_KS; do
                $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                    --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                    --loss sub_arcface --sub_arcface_s $s --sub_arcface_m $m --sub_arcface_k $k \
                    --tag "exp1_${model}_ds${ver}_subarcface_s${s}_m${m}_k${k}" >> "$LOG" 2>&1
            done; done; done
        done
    done
    echo "=== 实验一 结束 $(date) ===" >> "$LOG"
    # 生成热力图 + 最优参数 json
    $PY scripts/generate_exp_figures.py --exp 1 --out_dir "$OUT"
}

# ══════════════════════════════════════════════════════════════
#  实验二: 6 模型 × 4 损失 × 2 数据集（ 48 个）
#  损失参数: 从实验一 best_params_ds{1,2}.json 读取
# ══════════════════════════════════════════════════════════════
exp2() {
    OUT="output/exp2_model_loss"
    LOG="$LOGDIR/exp2.log"
    echo "=== 实验二 开始 $(date) ===" >> "$LOG"
    echo "[实验二] 模型×损失 -> $OUT"

    MODELS="cnn cnn_se lstm dual_cnn dual_cnn_se dual_lstm"
    for ver in 1.0 2.0; do
        BP="$OUT/../exp1_loss_param/best_params_ds$ver"
        [ -f "$BP.json" ] || BP="output/exp1_loss_param/best_params_ds$ver"
        # 读取最优参数
        COS_S=$("$PY" -c "import json;print(json.load(open('$BP.json'))['cosface']['_best']['s'])")
        COS_M=$("$PY" -c "import json;print(json.load(open('$BP.json'))['cosface']['_best']['m'])")
        ARC_S=$("$PY" -c "import json;print(json.load(open('$BP.json'))['arcface']['_best']['s'])")
        ARC_M=$("$PY" -c "import json;print(json.load(open('$BP.json'))['arcface']['_best']['m'])")
        SUB_S=$("$PY" -c "import json;print(json.load(open('$BP.json'))['sub_arcface']['_best']['s'])")
        SUB_M=$("$PY" -c "import json;print(json.load(open('$BP.json'))['sub_arcface']['_best']['m'])")
        SUB_K=$("$PY" -c "import json;print(json.load(open('$BP.json'))['sub_arcface']['_best']['K'])")
        echo "  [DS-$ver] 最优参数: cosface($COS_S,$COS_M) arcface($ARC_S,$ARC_M) sub($SUB_S,$SUB_M,K$SUB_K)" | tee -a "$LOG"

        for model in $MODELS; do
            $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                --loss cross_entropy --tag "exp2_${model}_ds${ver}_ce" >> "$LOG" 2>&1
            $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                --loss cosface --cosface_s $COS_S --cosface_m $COS_M \
                --tag "exp2_${model}_ds${ver}_cosface" >> "$LOG" 2>&1
            $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                --loss arcface --arcface_s $ARC_S --arcface_m $ARC_M \
                --tag "exp2_${model}_ds${ver}_arcface" >> "$LOG" 2>&1
            $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                --loss sub_arcface --sub_arcface_s $SUB_S --sub_arcface_m $SUB_M --sub_arcface_k $SUB_K \
                --tag "exp2_${model}_ds${ver}_subarcface" >> "$LOG" 2>&1
        done
    done
    echo "=== 实验二 结束 $(date) ===" >> "$LOG"
    $PY scripts/generate_exp_figures.py --exp 2 --out_dir "$OUT"
}

# ══════════════════════════════════════════════════════════════
#  实验三: top3 模型 × 4 增强配置 × 2 数据集（24 个）
#  top3 = 实验二 DS-1 上 acc 最高的 3 个模型（此处按已知先验预设，
#  若需严格按实验二结果动态选取，可在 exp2 完成后手动调整下面列表）
# ══════════════════════════════════════════════════════════════
exp3() {
    OUT="output/exp3_aug_mixup"
    LOG="$LOGDIR/exp3.log"
    echo "=== 实验三 开始 $(date) ===" >> "$LOG"
    echo "[实验三] 增强对照 -> $OUT"

    # top3 模型（可在此手动调整）
    TOP_MODELS="dual_lstm dual_cnn cnn"
    MIXUP=0.7

    for ver in 1.0 2.0; do
        BP="$OUT/../exp1_loss_param/best_params_ds$ver"
        [ -f "$BP.json" ] || BP="output/exp1_loss_param/best_params_ds$ver"
        ARC_S=$("$PY" -c "import json;print(json.load(open('$BP.json'))['arcface']['_best']['s'])")
        ARC_M=$("$PY" -c "import json;print(json.load(open('$BP.json'))['arcface']['_best']['m'])")
        for model in $TOP_MODELS; do
            for cfg in base mixup aug aug+mixup; do
                EXTRA="--loss arcface --arcface_s $ARC_S --arcface_m $ARC_M"
                case $cfg in
                    mixup)     EXTRA="$EXTRA --mixup $MIXUP" ;;
                    aug)       EXTRA="$EXTRA --online_aug" ;;
                    aug+mixup) EXTRA="$EXTRA --online_aug --mixup $MIXUP" ;;
                esac
                $PY scripts/cv_train.py --cv 5 --features_version $ver --model $model \
                    --epochs 200 --patience 30 --output_dir "$OUT/ds$ver" \
                    $EXTRA --tag "exp3_${model}_ds${ver}_${cfg}" >> "$LOG" 2>&1
            done
        done
    done
    echo "=== 实验三 结束 $(date) ===" >> "$LOG"
    $PY scripts/generate_exp_figures.py --exp 3 --out_dir "$OUT"
}

# ══════════════════════════════════════════════════════════════
#  执行
# ══════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════"
echo "  大创实验总脚本启动 $(date)"
echo "  输出根目录: output/"
echo "══════════════════════════════════════════"

case "$EXP" in
    1) exp1 ;;
    2) exp2 ;;
    3) exp3 ;;
    all)
        exp1
        exp2
        exp3
        echo "══════════════════════════════════════════"
        echo "  全部实验完成 $(date)"
        echo "  结果: output/exp1_loss_param/ exp2_model_loss/ exp3_aug_mixup/"
        echo "══════════════════════════════════════════"
        ;;
    *) echo "用法: bash run_experiments_all.sh [1|2|3|all]"; exit 1 ;;
esac
