#!/usr/bin/env python
"""
WAN 2.2 T2V LoRA 训练脚本
直接运行即可，支持自动断点续训
用法: python train_wan22.py -low  或  python train_wan22.py -high
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ==================== 训练命令配置 ====================
# 直接修改下面的参数列表，参数和值写在一起

OUTPUT_DIR = r'output'
OUTPUT_NAME_BASE = 'lh_lora_v1'
LOGGING_DIR = '/workspace' if sys.platform == 'linux' else './logs'


def build_cmd_list(noise_type: str):
    """根据噪声类型构建命令列表"""
    if noise_type == 'low':
        dit_model = "wan2.2_t2v_low_noise_14B_fp16.safetensors"
        min_timestep = 0
        max_timestep = 875
        output_name = f"{OUTPUT_NAME_BASE}_low"
    else:
        dit_model = "wan2.2_t2v_high_noise_14B_fp16.safetensors"
        min_timestep = 875
        max_timestep = 1000
        output_name = f"{OUTPUT_NAME_BASE}_high"

    return [
        "accelerate launch --num_cpu_threads_per_process 1",
        "src/musubi_tuner/wan_train_network.py",

        # 模型配置
        "--task t2v-A14B",
        f"--dit models/split_files/diffusion_models/{dit_model}",
        r"--vae models/split_files/vae/wan_2.1_vae.safetensors",
        r"--t5 models/umt5-xxl-enc-bf16.safetensors",

        # 数据集
        r"--dataset_config datasets/test6/dataset.toml",

        # 精度与加速
        "--mixed_precision fp16",
        "--fp8_base",
        "--blocks_to_swap 1",  # 将20个块交换到CPU，节省显存（模型共40层）
        "--img_in_txt_in_offloading",  # 将 img_in 和 txt_in 卸载到 CPU
        '--flash_attn' if sys.platform == 'linux' else '--xformers',
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
        "--lr_scheduler constant_with_warmup",
        "--lr_warmup_steps 20",

        # 数据加载
        "--max_data_loader_n_workers 2",
        "--gradient_accumulation_steps 2",

        # LoRA
        "--network_module networks.lora_wan",
        "--network_dim 32",
        "--network_alpha 32",  # 推荐设为 dim 的一半，即 16

        # 时间步
        "--timestep_sampling sigmoid",
        "--discrete_flow_shift 1.0",
        f"--min_timestep {min_timestep}",
        f"--max_timestep {max_timestep}",
        "--preserve_distribution_shape",

        # 训练与保存
        "--max_train_epochs 200",
        "--save_every_n_epochs 1",
        "--save_state",
        "--save_last_n_epochs_state 5",
        "--save_last_n_epochs 5",
        "--seed 42",

        # 输出
        f"--output_dir {OUTPUT_DIR}",
        f"--output_name {output_name}",

        # 日志
        f"--logging_dir {LOGGING_DIR}",
        "--log_with tensorboard",
        "--log_config",

        # 采样（可选，注释掉则禁用）
        # "--sample_prompts ./sample_prompts.txt",
        # "--sample_every_n_epochs 20",
        # "--sample_at_first",
    ], output_name


# ==================== 配置结束 ====================


def build_cmd(cmd_list):
    """展开命令列表"""
    result = []
    for item in cmd_list:
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
    """启动 TensorBoard（先杀掉已存在的进程），仅 Windows"""
    if sys.platform != 'win32':
        return None
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


def find_latest_state(output_name):
    """查找最新的训练状态（通过目录修改时间判断）"""
    output_dir = Path(OUTPUT_DIR)

    # 查找所有状态目录：
    # - 中间状态: lh_lora_v1_low-000190-state（带 epoch 编号）
    # - 完成状态: lh_lora_v1_low-state（无 epoch 编号）
    state_dirs = list(output_dir.glob(f"{output_name}-*-state"))
    final_state = output_dir / f"{output_name}-state"
    if final_state.exists():
        state_dirs.append(final_state)

    if not state_dirs:
        print("未找到已保存的状态，将从头开始训练")
        return None

    # 按目录修改时间排序，取最新的
    state_dirs.sort(key=lambda x: x.stat().st_mtime)
    latest = state_dirs[-1]

    # 判断是完成状态还是中间状态
    if latest.name == f"{output_name}-state":
        print(f"找到完成状态: {latest.name}，将恢复训练")
    else:
        epoch = latest.name.split('-')[-2]
        print(f"找到中间状态: {latest.name}，从 epoch {epoch} 恢复")

    return str(latest)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="WAN 2.2 T2V LoRA 训练脚本")
    noise_group = parser.add_mutually_exclusive_group(required=True)
    noise_group.add_argument("-low", action="store_true", help="使用低噪声模型训练")
    noise_group.add_argument("-high", action="store_true", help="使用高噪声模型训练")
    args = parser.parse_args()

    noise_type = 'low' if args.low else 'high'
    cmd_list, output_name = build_cmd_list(noise_type)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGGING_DIR).mkdir(parents=True, exist_ok=True)

    start_tensorboard()

    cmd = build_cmd(cmd_list)
    resume_state = find_latest_state(output_name)
    if resume_state:
        cmd.extend(["--resume", resume_state])

    print(f"\n{'=' * 60}")
    print(f"WAN 2.2 LoRA 训练: {output_name} ({noise_type} noise)")
    print(f"{'=' * 60}\n")

    process = None

    def handle_sigint(sig, frame):
        print("\n手动中断，杀掉进程...")
        if process:
            if sys.platform != 'win32':
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                # 释放 GPU 资源
                subprocess.run('fuser -k /dev/nvidia*', shell=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                process.kill()
        sys.exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, handle_sigint)

    # 启动训练进程（Linux 上创建新进程组）
    if sys.platform != 'win32':
        process = subprocess.Popen(cmd, preexec_fn=os.setsid)
    else:
        process = subprocess.Popen(cmd)

    returncode = process.wait()
    if returncode == 0:
        print("\n训练完成！")
    else:
        print(f"\n训练中断（退出码: {returncode}）")
        print("再次运行可从检查点恢复")
        sys.exit(1)


if __name__ == "__main__":
    main()
