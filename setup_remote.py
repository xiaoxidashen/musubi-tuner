#!/usr/bin/env python
"""
远程服务器初始化脚本
下载模型文件到 datasets 目录
"""
import subprocess
import sys


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="远程服务器初始化脚本")
    parser.add_argument("--skip_download", action="store_true", help="跳过模型下载")
    args = parser.parse_args()

    print("=" * 60)
    print("远程服务器初始化脚本")
    print("=" * 60)

    # 安装 huggingface_hub
    print("\n检查并安装 huggingface_hub...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"警告: 安装 huggingface_hub 时出现问题")
        print(result.stderr)
    else:
        print("✓ huggingface_hub 已安装")

    # 切换到 datasets 目录
    print("\n切换到 datasets 目录...")
    import os
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
    else:
        print("\n跳过模型下载")

    print("\n" + "=" * 60)
    print("✓ 模型下载完成！")
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
