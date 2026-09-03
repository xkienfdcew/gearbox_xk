# -*- coding: utf-8 -*-
"""并行任务调度器：并发运行多个 cv_train 实验，充分利用 GPU 显存。

用法:
    python scripts/run_tasks.py --tasks <任务文件> --prog_dir <状态目录>         --log <日志文件> [--jobs N]

任务文件格式（每行）:
    <tag> <cv_train 完整参数...>

特性:
    - 最多 N 个进程同时训练（--jobs 0 时按 GPU 显存自动估算）
    - 断点续跑: 已完成的 tag（.done 存在）自动跳过
    - 单个任务失败不阻塞，重跑本脚本即可续
"""
import sys, os, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _find_python():
    """自动探测 python: Windows→.conda/python.exe, Linux→.conda/bin/python 或 python3"""
    if os.name == "nt":
        p = os.path.join(PROJECT_ROOT, ".conda", "python.exe")
        return p if os.path.exists(p) else sys.executable
    for cand in (os.path.join(PROJECT_ROOT, ".conda", "bin", "python"),
                 os.path.join(PROJECT_ROOT, "python")):
        if os.path.exists(cand):
            return cand
    return "python"

PY = _find_python()
CV = os.path.join(PROJECT_ROOT, "scripts", "cv_train.py")


def load_tasks(task_file):
    tasks = []
    with open(task_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tag, _, rest = line.partition(" ")
            tasks.append((tag, rest.split()))
    return tasks


def run_one(tag, args, prog_dir, log_path, lock):
    done_file = os.path.join(prog_dir, tag + ".done")
    if os.path.exists(done_file):
        with lock:
            print(f"[跳过] {tag}")
        return True
    # 默认参数；若任务行内已指定同参数则跳过（允许行内覆盖，便于测试/微调）
    present = {a for a in args if a.startswith("--")}
    base = []
    for k, v in (("--cv", "5"), ("--epochs", "200"), ("--patience", "30")):
        if k not in present:
            base += [k, v]
    cmd = [PY, CV] + base + args + ["--tag", tag]
    t0 = time.time()
    with lock:
        print(f"[{time.strftime('%H:%M:%S')}] [运行] {tag}")
    with open(log_path, "a", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=lf)
    if r.returncode == 0:
        open(done_file, "w").close()
        with lock:
            print(f"[{time.strftime('%H:%M:%S')}] [完成] {tag} ({time.time()-t0:.0f}s)")
        return True
    with lock:
        print(f"[失败] {tag} (exit={r.returncode}, 重跑可续)")
    return False


def auto_jobs():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vrams = [int(x.strip()) / 1024 for x in out.stdout.strip().splitlines() if x.strip()]
        if vrams:
            n = max(1, int(sum(vrams) // 4))   # 每个训练任务预留约 4 GiB
            print(f"[自动] GPU 显存合计 {sum(vrams):.0f} GiB -> 并发 {n}")
            return n
    except Exception:
        pass
    print("[自动] 无法查询 GPU 显存 -> 默认并发 2")
    return 2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--prog_dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--jobs", type=int, default=0, help="并发数（0=按显存自动）")
    a = ap.parse_args()

    os.makedirs(a.prog_dir, exist_ok=True)
    jobs = auto_jobs() if a.jobs <= 0 else a.jobs
    tasks = load_tasks(a.tasks)
    total = len(tasks)
    print(f"任务总数: {total} | 并发: {jobs} | 日志: {a.log}")

    if total == 0:
        print("无待运行任务（可能全部已完成）")
        return

    done_cnt = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, tag, args, a.prog_dir, a.log, lock): tag
                for tag, args in tasks}
        for fut in as_completed(futs):
            if fut.result():
                done_cnt += 1
            print(f"  进度: {done_cnt}/{total}")

    print(f"批次完成: {done_cnt}/{total} 成功")


if __name__ == "__main__":
    main()
