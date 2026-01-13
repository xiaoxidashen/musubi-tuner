#!/usr/bin/env python
"""
WAN 2.2 T2V LoRA 训练脚本
直接运行即可，支持自动断点续训
用法: python train_wan22.py
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ==================== 训练命令配置 ====================
# 直接修改下面的参数列表，参数和值写在一起

OUTPUT_DIR = r'output'
OUTPUT_NAME = 'lh_lora_v1'
LOGGING_DIR = './logs'

CMD = [
    "accelerate launch --num_cpu_threads_per_process 1",
    "src/musubi_tuner/wan_train_network.py",

    # 模型配置
    "--task t2v-A14B",
    r"--dit models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
    r"--vae models/split_files/vae/wan_2.1_vae.safetensors",
    r"--t5 models/umt5-xxl-enc-bf16.safetensors",

    # 数据集
    r"--dataset_config datasets/test5/dataset.toml",

    # 精度与加速
    "--mixed_precision fp16",
    "--fp8_base",
    "--blocks_to_swap 1",  # 将20个块交换到CPU，节省显存（模型共40层）
    "--img_in_txt_in_offloading",  # 将 img_in 和 txt_in 卸载到 CPU
    "--xformers",
    "--split_attn",  # 切分注意力计算以节省显存
    "--gradient_checkpointing",
    "--compile",
    "--compile_mode default",
    "--cuda_cudnn_benchmark",
    "--persistent_data_loader_workers",

    # 优化器
    "--optimizer_type adamw8bit",
    "--learning_rate 5e-4",

    # 学习率调度
    "--lr_scheduler cosine",
    "--lr_warmup_steps 20",

    # 数据加载
    "--max_data_loader_n_workers 2",
    "--gradient_accumulation_steps 2",

    # LoRA
    "--network_module networks.lora_wan",
    "--network_dim 32",
    "--network_alpha 32",  # 推荐设为 dim 的一半，即 16

    # 时间步（低噪声模型）
    "--timestep_sampling sigmoid",
    "--discrete_flow_shift 1.0",
    "--min_timestep 0",
    "--max_timestep 875",
    "--preserve_distribution_shape",

    # 训练与保存
    "--max_train_epochs 100",
    "--save_every_n_epochs 10",
    "--save_state",
    "--save_last_n_epochs_state 3",
    "--save_last_n_epochs 3",
    "--seed 42",

    # 输出
    f"--output_dir {OUTPUT_DIR}",
    f"--output_name {OUTPUT_NAME}",

    # 日志
    f"--logging_dir {LOGGING_DIR}",
    "--log_with tensorboard",
    "--log_config",

    # 采样（可选，注释掉则禁用）
    # "--sample_prompts ./sample_prompts.txt",
    # "--sample_every_n_epochs 20",
    # "--sample_at_first",
]


# ==================== 配置结束 ====================


def build_cmd():
    """展开命令列表"""
    result = []
    for item in CMD:
        result.extend(item.split())
    return result


def kill_tensorboard():
    """杀掉已存在的 TensorBoard 进程"""
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/F', '/IM', 'tensorboard.exe'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(['pkill', '-f', 'tensorboard'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_tensorboard():
    """启动 TensorBoard（先杀掉已存在的进程）"""
    port = 6006
    kill_tensorboard()
    time.sleep(1)

    try:
        proc = subprocess.Popen(
            ['tensorboard', '--logdir', LOGGING_DIR, '--port', str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        )
        time.sleep(2)
        if proc.poll() is None:
            url = f"http://localhost:{port}"
            print(f"TensorBoard 已启动: {url}")
            webbrowser.open(url)
            return proc
    except FileNotFoundError:
        print("未找到 tensorboard，请安装: pip install tensorboard")
    return None


def find_latest_state():
    """查找最新的训练状态"""
    output_dir = Path(OUTPUT_DIR)
    state_dirs = list(output_dir.glob(f"{OUTPUT_NAME}-*-state"))

    if not state_dirs:
        print("未找到已保存的状态，将从头开始训练")
        return None

    state_dirs.sort(key=lambda x: int(x.name.split('-')[-2]))
    latest = state_dirs[-1]
    epoch = latest.name.split('-')[-2]
    print(f"找到状态: {latest.name}，从 epoch {epoch} 恢复")
    return str(latest)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGGING_DIR).mkdir(parents=True, exist_ok=True)

    start_tensorboard()

    cmd = build_cmd()
    resume_state = find_latest_state()
    if resume_state:
        cmd.extend(["--resume", resume_state])

    print(f"\n{'=' * 60}")
    print(f"WAN 2.2 LoRA 训练: {OUTPUT_NAME}")
    print(f"{'=' * 60}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n训练完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n训练中断（退出码: {e.returncode}）")
        print("再次运行可从检查点恢复")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n手动中断，再次运行可恢复")
        sys.exit(0)


if __name__ == "__main__":
    main()
