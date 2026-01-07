#!/usr/bin/env python
"""
WAN 2.2 T2V LoRA 智能训练脚本 - 低噪声模型
直接运行即可，支持自动断点续训
用法: python train_wan22.py
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path


# ==================== 配置区域 ====================
# 修改这里的参数来自定义训练

CONFIG = {
    # 任务类型
    'task': 't2v-A14B',

    # 模型路径
    'dit_path': r'F:\ComfyUI\models\diffusion_models\wan2.2_t2v_low_noise_14B_fp16.safetensors',
    'vae_path': r'F:\ComfyUI\models\vae\Wan2.1_VAE.safetensors',
    't5_path': r'F:\ComfyUI\models\text_encoders\umt5-xxl-enc-bf16.safetensors',

    # 数据集配置
    'dataset_config': r'D:\Code\Github\Projects\ai-toolkit\datasets\test1\dataset.toml',

    # 训练精度
    'mixed_precision': 'fp16',
    'fp8_base': True,

    # Attention 优化（选择一个，已安装 xformers 推荐使用）
    'xformers': True,       # xformers 加速（更快，已安装）
    'sdpa': False,          # PyTorch 原生 SDPA（备用）

    # 训练加速
    'compile': True,                   # torch.compile 编译加速（可能提速 20-50%，首次编译慢）
    'compile_mode': 'default',          # 编译模式: default/reduce-overhead/max-autotune
    'cuda_allow_tf32': False,           # 启用 TF32（仅 RTX 30/40 系列，2080 Ti 不支持）
    'cuda_cudnn_benchmark': True,       # cuDNN 自动调优
    'persistent_data_loader_workers': True,  # 持久化数据加载器（减少 epoch 间等待）

    # 可视化训练曲线（会自动启动 TensorBoard 并打开浏览器）
    'logging_dir': './logs',            # TensorBoard 日志目录（设为 None 禁用自动启动）
    'log_with': 'tensorboard',          # 日志工具: tensorboard/wandb/all
    'log_config': True,                 # 记录训练配置

    # 优化器配置
    'optimizer_type': 'adamw',
    'learning_rate': 3e-4,
    'weight_decay': 0.1,
    'max_grad_norm': 0,

    # 学习率调度器
    'lr_scheduler': 'polynomial',
    'lr_scheduler_power': 8,
    'lr_scheduler_min_lr_ratio': 5e-5,

    # 训练参数
    'gradient_accumulation_steps': 1,
    'max_data_loader_n_workers': 2,

    # LoRA 参数
    'network_dim': 16,
    'network_alpha': 16,

    # 时间步配置（低噪声模型）
    'timestep_sampling': 'shift',
    'discrete_flow_shift': 1.0,
    'min_timestep': 0,      # 低噪声: 0
    'max_timestep': 875,    # 低噪声: 875

    # 保存配置
    'max_train_epochs': 100,
    'save_every_n_epochs': 10,      # 每 10 个 epoch 保存
    'save_last_n_epochs_state': 3,  # 只保留最近 3 个状态
    'seed': 5,

    # 输出配置
    'output_dir': r'F:\ComfyUI\models\loras',
    'output_name': 'WAN2.2-LowNoise_test1_v1',
}

# ==================== 配置结束 ====================


class WAN22Trainer:
    """WAN 2.2 训练管理器"""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_name = config['output_name']
        self.tensorboard_process = None

    def find_latest_state(self):
        """查找最新的训练状态"""
        # 查找所有 state 目录
        state_dirs = list(self.output_dir.glob(f"{self.output_name}-*-state"))

        if not state_dirs:
            print("📝 未找到已保存的状态，将从头开始训练")
            return None

        # 按照 epoch 数字排序，找最新的
        state_dirs.sort(key=lambda x: int(x.name.split('-')[-2]))
        latest_state = state_dirs[-1]

        epoch_num = latest_state.name.split('-')[-2]
        print(f"✅ 找到最新的训练状态: {latest_state.name}")
        print(f"📊 将从 epoch {epoch_num} 恢复训练")

        return str(latest_state)

    def start_tensorboard(self):
        """启动 TensorBoard 服务"""
        if not self.config.get('logging_dir'):
            return None

        logging_dir = self.config['logging_dir']
        port = 6006

        # 检查端口是否已被占用
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()

        if result == 0:
            print(f"⚠️  TensorBoard 可能已在运行 (端口 {port} 已占用)")
            print(f"   访问: http://localhost:{port}")
            return None

        print(f"🚀 正在启动 TensorBoard...")

        # 启动 TensorBoard
        try:
            self.tensorboard_process = subprocess.Popen(
                ['tensorboard', '--logdir', logging_dir, '--port', str(port), '--bind_all'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )

            # 等待 TensorBoard 启动
            time.sleep(3)

            # 检查是否成功启动
            if self.tensorboard_process.poll() is None:
                tensorboard_url = f"http://localhost:{port}"
                print(f"✅ TensorBoard 已启动: {tensorboard_url}")

                # 自动打开浏览器
                try:
                    webbrowser.open(tensorboard_url)
                    print("🌐 已自动打开浏览器")
                except:
                    print("💡 请手动在浏览器中打开上述链接")

                return self.tensorboard_process
            else:
                print("❌ TensorBoard 启动失败")
                return None

        except FileNotFoundError:
            print("❌ 未找到 tensorboard 命令，请安装: pip install tensorboard")
            return None
        except Exception as e:
            print(f"❌ 启动 TensorBoard 出错: {e}")
            return None

    def stop_tensorboard(self):
        """停止 TensorBoard 服务"""
        if self.tensorboard_process:
            try:
                print("\n🛑 正在关闭 TensorBoard...")
                if sys.platform == 'win32':
                    # Windows 使用 taskkill
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.tensorboard_process.pid)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    self.tensorboard_process.terminate()
                    self.tensorboard_process.wait(timeout=5)
                print("✅ TensorBoard 已关闭")
            except Exception as e:
                print(f"⚠️  关闭 TensorBoard 时出错: {e}")
                print("   你可能需要手动关闭 TensorBoard 进程")

    def build_command(self, resume_state=None):
        """构建训练命令"""
        config = self.config

        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            "src/musubi_tuner/wan_train_network.py",
            "--task", config['task'],
            "--dit", config['dit_path'],
            "--vae", config['vae_path'],
            "--t5", config['t5_path'],
            "--dataset_config", config['dataset_config'],
            "--mixed_precision", config['mixed_precision'],
        ]

        # 可选参数
        if config.get('fp8_base'):
            cmd.append("--fp8_base")

        # Attention 优化
        if config.get('xformers'):
            cmd.append("--xformers")
        elif config.get('sdpa'):
            cmd.append("--sdpa")

        # 训练加速
        if config.get('compile'):
            cmd.extend([
                "--compile",
                "--compile_mode", config.get('compile_mode', 'default'),
            ])

        if config.get('cuda_allow_tf32'):
            cmd.append("--cuda_allow_tf32")

        if config.get('cuda_cudnn_benchmark'):
            cmd.append("--cuda_cudnn_benchmark")

        if config.get('persistent_data_loader_workers'):
            cmd.append("--persistent_data_loader_workers")

        # 可视化日志
        if config.get('logging_dir'):
            cmd.extend([
                "--logging_dir", config['logging_dir'],
                "--log_with", config.get('log_with', 'tensorboard'),
                "--log_tracker_name", config['output_name'],  # 使用固定名称，让日志接续
            ])
            if config.get('log_config'):
                cmd.append("--log_config")

        # 优化器参数
        cmd.extend([
            "--optimizer_type", config['optimizer_type'],
            "--learning_rate", str(config['learning_rate']),
            "--optimizer_args", f"weight_decay={config['weight_decay']}",
            "--max_grad_norm", str(config['max_grad_norm']),
        ])

        # 学习率调度器
        cmd.extend([
            "--lr_scheduler", config['lr_scheduler'],
            "--lr_scheduler_power", str(config['lr_scheduler_power']),
            "--lr_scheduler_min_lr_ratio", str(config['lr_scheduler_min_lr_ratio']),
        ])

        # 训练参数
        cmd.extend([
            "--gradient_checkpointing",
            "--gradient_accumulation_steps", str(config['gradient_accumulation_steps']),
            "--max_data_loader_n_workers", str(config['max_data_loader_n_workers']),
        ])

        # LoRA 参数
        cmd.extend([
            "--network_module", "networks.lora_wan",
            "--network_dim", str(config['network_dim']),
            "--network_alpha", str(config['network_alpha']),
        ])

        # 时间步参数
        cmd.extend([
            "--timestep_sampling", config['timestep_sampling'],
            "--discrete_flow_shift", str(config['discrete_flow_shift']),
            "--preserve_distribution_shape",
            "--min_timestep", str(config['min_timestep']),
            "--max_timestep", str(config['max_timestep']),
        ])

        # 保存参数
        cmd.extend([
            "--max_train_epochs", str(config['max_train_epochs']),
            "--save_every_n_epochs", str(config['save_every_n_epochs']),
            "--save_state",  # 关键：保存完整状态
            "--save_last_n_epochs_state", str(config['save_last_n_epochs_state']),
            "--seed", str(config['seed']),
        ])

        # 输出参数
        cmd.extend([
            "--output_dir", config['output_dir'],
            "--output_name", config['output_name'],
        ])

        # 如果有恢复状态，添加 resume 参数
        if resume_state:
            cmd.extend(["--resume", resume_state])

        return cmd

    def train(self):
        """开始或恢复训练"""
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 启动 TensorBoard
        self.start_tensorboard()

        # 查找最新状态
        resume_state = self.find_latest_state()

        # 构建命令
        cmd = self.build_command(resume_state)

        # 打印配置信息
        print("\n" + "="*80)
        print("🚀 WAN 2.2 低噪声模型 LoRA 训练")
        print("="*80)
        print(f"📁 数据集: {self.config['dataset_config']}")
        print(f"🎯 模型: {self.config['output_name']}")
        print(f"📊 总 Epochs: {self.config['max_train_epochs']}")
        print(f"💾 保存间隔: 每 {self.config['save_every_n_epochs']} epochs")
        print(f"⏱️  时间步范围: {self.config['min_timestep']} - {self.config['max_timestep']}")

        # 优化设置
        attn_mode = "SDPA" if self.config.get('sdpa') else ("xformers" if self.config.get('xformers') else "默认")
        print(f"⚡ Attention: {attn_mode}")
        if self.config.get('compile'):
            print(f"⚡ Torch Compile: 是 ({self.config.get('compile_mode', 'default')})")
        if self.config.get('cuda_allow_tf32'):
            print(f"⚡ TF32: 已启用")

        # 日志设置
        if self.config.get('logging_dir'):
            print(f"📈 TensorBoard: {self.config['logging_dir']}")

        if resume_state:
            print(f"🔄 恢复训练: 是")
        else:
            print(f"🔄 恢复训练: 否（从头开始）")
        print("="*80 + "\n")

        # 执行训练
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ 训练完成！")
            print(f"📦 输出目录: {self.config['output_dir']}")

            # 如果启用了日志，提示如何查看
            if self.config.get('logging_dir'):
                print(f"\n📈 TensorBoard 仍在运行，可继续查看训练曲线")
                print(f"   访问: http://localhost:6006")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ 训练中断（退出码: {e.returncode}）")
            print("💡 提示: 再次运行此脚本可以从最新检查点恢复训练")
            self.stop_tensorboard()
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n⏸️  训练已手动中断（Ctrl+C）")
            print("💡 提示: 再次运行此脚本可以从最新检查点恢复训练")
            self.stop_tensorboard()
            sys.exit(0)
        finally:
            # 询问是否关闭 TensorBoard
            if self.tensorboard_process and self.tensorboard_process.poll() is None:
                print("\n" + "="*80)
                try:
                    response = input("📊 是否关闭 TensorBoard？(y/n，默认保持运行): ").strip().lower()
                    if response == 'y':
                        self.stop_tensorboard()
                    else:
                        print("💡 TensorBoard 继续运行，访问: http://localhost:6006")
                        print("   如需关闭，运行: taskkill /F /IM tensorboard.exe")
                except:
                    print("💡 TensorBoard 继续运行，访问: http://localhost:6006")
                    print("   如需关闭，运行: taskkill /F /IM tensorboard.exe")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("WAN 2.2 T2V LoRA 智能训练脚本")
    print("="*80)

    # 创建训练器
    trainer = WAN22Trainer(CONFIG)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
