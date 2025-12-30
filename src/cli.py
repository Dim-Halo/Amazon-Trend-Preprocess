#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Amazon Trend Pipeline CLI

使用示例:
    python -m src.cli                    # 运行完整流水线
    python -m src.cli --module 1         # 只运行模块 1
    python -m src.cli --module 1 2 3     # 运行模块 1, 2, 3
    python -m src.cli --force            # 强制重新运行所有模块
    python -m src.cli --reset            # 重置流水线状态
    python -m src.cli --status           # 查看流水线状态
"""

import argparse
import sys
import os
from pathlib import Path

# 设置 Windows 控制台编码
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline_v2 import run_pipeline
from src.utils.state import PipelineState
from src.utils.logger import setup_logger
from src.config import Config

def show_status():
    """显示流水线状态"""
    state = PipelineState()
    print("\n" + "=" * 50)
    print("📊 流水线状态")
    print("=" * 50)

    if not state.state['completed_modules']:
        print("❌ 没有已完成的模块")
        return

    print(f"\n✅ 已完成的模块: {', '.join(state.state['completed_modules'])}")
    print(f"\n⏰ 最后运行时间: {state.state.get('last_run', 'N/A')}")

    print("\n📝 模块详情:")
    for module, data in state.state['module_metadata'].items():
        print(f"\n  {module}:")
        print(f"    完成时间: {data.get('completed_at', 'N/A')}")
        if data.get('metadata'):
            for key, value in data['metadata'].items():
                print(f"    {key}: {value}")

    print("\n" + "=" * 50)

def reset_state(module=None):
    """重置流水线状态"""
    state = PipelineState()
    if module:
        state.reset(module)
        print(f"✅ 已重置模块 {module} 的状态")
    else:
        state.reset()
        print("✅ 已重置所有模块的状态")

def main():
    parser = argparse.ArgumentParser(
        description='Amazon Trend Pipeline - 数据处理流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                    运行完整流水线
  %(prog)s --module 1         只运行模块 1
  %(prog)s --module 1 2 3     运行模块 1, 2, 3
  %(prog)s --force            强制重新运行所有模块
  %(prog)s --reset            重置流水线状态
  %(prog)s --status           查看流水线状态

模块说明:
  1 - 全局词汇收集
  2 - 向量聚类与映射
  3 - 输出验证文件
  4 - 生成 TimesNet 矩阵
        """
    )

    parser.add_argument(
        '--module', '-m',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4],
        help='要运行的模块编号 (可指定多个)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新运行，忽略缓存'
    )

    parser.add_argument(
        '--reset', '-r',
        action='store_true',
        help='重置流水线状态'
    )

    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='显示流水线状态'
    )

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        help='指定运行设备 (覆盖配置文件)'
    )

    args = parser.parse_args()

    # 设置设备
    if args.device:
        Config.DEVICE = args.device

    # 处理命令
    if args.status:
        show_status()
        return

    if args.reset:
        reset_state()
        return

    # 运行流水线
    try:
        Config.ensure_folders()
        logger = setup_logger('cli', Config.LOG_FOLDER / 'cli.log')
        logger.info("启动 CLI...")

        run_pipeline(
            force_rerun=args.force,
            modules=args.module
        )

        print("\n✅ 流水线执行成功!")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
