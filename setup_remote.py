#!/usr/bin/env python
"""
远程服务器初始化脚本
下载模型文件到 datasets 目录，并缓存 latents 和 text encoder outputs
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """执行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"执行: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n错误: {description} 失败")
        sys.exit(1)
    print(f"\n✓ {description} 完成")


def find_dataset_configs():
    """扫描 datasets 目录下的所有 dataset.toml 文件"""
    datasets_dir = Path("datasets")
    configs = list(datasets_dir.glob("*/dataset.toml"))
    return configs


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="远程服务器初始化脚本")
    parser.add_argument("--skip_download", action="store_true", help="跳过模型下载")
    parser.add_argument("--skip_cache", action="store_true", help="跳过缓存步骤")
    args = parser.parse_args()

    print("=" * 60)
    print("远程服务器初始化脚本")
    print("=" * 60)

    # 保存原始目录
    original_dir = os.getcwd()

    # 安装 huggingface_hub
    print("\n检查并安装 huggingface_hub...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"警告: 安装 huggingface_hub 时出现问题")
        print(result.stderr)
    else:
        print("✓ huggingface_hub 已安装")

    # 切换到 datasets 目录下载模型
    print("\n切换到 datasets 目录...")
    os.chdir("datasets")
    print(f"当前目录: {os.getcwd()}")

    if not args.skip_download:
        # 下载模型文件
        run_command(
            "hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir .",
            "下载 DiT 模型"
        )

        run_command(
            "hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir .",
            "下载 VAE 模型"
        )

        run_command(
            "hf download Kijai/WanVideo_comfy umt5-xxl-enc-bf16.safetensors --local-dir .",
            "下载 T5 编码器"
        )
        print("\n✓ 模型下载完成！")
    else:
        print("\n跳过模型下载")

    # 切换回原始目录
    os.chdir(original_dir)

    if args.skip_cache:
        print("\n跳过缓存步骤")
        print("\n现在可以开始训练了:")
        print("  python train_wan22.py")
        return

    # 扫描数据集配置文件
    print("\n" + "=" * 60)
    print("扫描数据集配置文件")
    print("=" * 60)

    configs = find_dataset_configs()
    if not configs:
        print("未找到任何 dataset.toml 文件，跳过缓存步骤")
        print("\n现在可以开始训练了:")
        print("  python train_wan22.py")
        return

    print(f"\n找到 {len(configs)} 个数据集配置文件:")
    for config in configs:
        print(f"  - {config}")

    # 模型路径
    vae_path = "datasets/split_files/vae/wan_2.1_vae.safetensors"
    t5_path = "datasets/umt5-xxl-enc-bf16.safetensors"

    # 对每个数据集进行缓存
    for config_path in configs:
        print("\n" + "=" * 60)
        print(f"处理数据集: {config_path}")
        print("=" * 60)

        # 缓存 latents
        run_command(
            f"python src/musubi_tuner/wan_cache_latents.py "
            f"--dataset_config {config_path} "
            f"--vae {vae_path}",
            "缓存 VAE latents"
        )

        # 缓存 text encoder outputs
        run_command(
            f"python src/musubi_tuner/wan_cache_text_encoder_outputs.py "
            f"--dataset_config {config_path} "
            f"--t5 {t5_path}",
            "缓存 Text Encoder outputs"
        )

    print("\n" + "=" * 60)
    print("✓ 所有初始化步骤完成！")
    print("=" * 60)
    print("\n现在可以开始训练了:")
    print("  python train_wan22.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        sys.exit(1)
