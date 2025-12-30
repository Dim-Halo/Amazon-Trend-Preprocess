"""
Pipeline V3 - 使用基类的简化版本

支持自动检测 GPU 并选择合适的流水线
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.base_pipeline import CPUPipeline, GPUPipeline


def create_pipeline(device='auto'):
    """
    创建流水线实例

    Args:
        device: 'auto', 'cpu', 或 'cuda'

    Returns:
        流水线实例
    """
    if device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ 检测到 CUDA，使用 GPU 流水线")
                return GPUPipeline()
            else:
                print("ℹ️ 未检测到 CUDA，使用 CPU 流水线")
                return CPUPipeline()
        except ImportError:
            print("ℹ️ PyTorch 未安装，使用 CPU 流水线")
            return CPUPipeline()
    elif device == 'cpu':
        return CPUPipeline()
    elif device.startswith('cuda'):
        gpu_id = 0 if device == 'cuda' else int(device.split(':')[1])
        return GPUPipeline(gpu_id=gpu_id)
    else:
        raise ValueError(f"不支持的设备类型: {device}")


def run_pipeline(device='auto', force_rerun=False, modules=None):
    """
    运行流水线

    Args:
        device: 设备类型 ('auto', 'cpu', 'cuda')
        force_rerun: 是否强制重新运行
        modules: 要运行的模块列表
    """
    pipeline = create_pipeline(device)
    pipeline.run(force_rerun=force_rerun, modules=modules)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline V3 - 统一的 CPU/GPU 流水线')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='运行设备')
    parser.add_argument('--force', action='store_true', help='强制重新运行')
    parser.add_argument('--module', type=int, nargs='+', choices=[1, 2, 3, 4],
                       help='要运行的模块')

    args = parser.parse_args()

    run_pipeline(
        device=args.device,
        force_rerun=args.force,
        modules=args.module
    )
