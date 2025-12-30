"""
处理 Excel 文件大小限制的工具

Excel 限制：
- 最大行数：1,048,576
- 最大列数：16,384

解决方案：
1. 分割成多个 Excel 文件
2. 使用 CSV 格式
3. 使用 Parquet 格式（推荐）
"""

import pandas as pd
from pathlib import Path
import math


class LargeDataExporter:
    """处理超大数据集的导出"""

    EXCEL_MAX_ROWS = 1048576  # Excel 最大行数

    @staticmethod
    def export_to_multiple_excel(df, output_path, max_rows=None):
        """
        将大型 DataFrame 分割成多个 Excel 文件

        Args:
            df: DataFrame
            output_path: 输出路径（不含扩展名）
            max_rows: 每个文件的最大行数（默认为 Excel 限制 - 1）

        Returns:
            生成的文件列表
        """
        if max_rows is None:
            max_rows = LargeDataExporter.EXCEL_MAX_ROWS - 1  # 留一行给表头

        total_rows = len(df)
        num_files = math.ceil(total_rows / max_rows)

        output_path = Path(output_path)
        output_files = []

        print(f"数据共 {total_rows:,} 行，将分割成 {num_files} 个文件")

        for i in range(num_files):
            start_idx = i * max_rows
            end_idx = min((i + 1) * max_rows, total_rows)

            chunk = df.iloc[start_idx:end_idx]

            # 生成文件名
            if num_files == 1:
                file_path = output_path.with_suffix('.xlsx')
            else:
                file_path = output_path.parent / f"{output_path.stem}_part{i+1}.xlsx"

            chunk.to_excel(file_path, index=False)
            output_files.append(file_path)

            print(f"✅ 已保存: {file_path.name} ({len(chunk):,} 行)")

        return output_files

    @staticmethod
    def export_to_csv(df, output_path):
        """
        导出为 CSV（无行数限制）

        Args:
            df: DataFrame
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        output_path = Path(output_path).with_suffix('.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 已保存 CSV: {output_path.name} ({len(df):,} 行)")
        return output_path

    @staticmethod
    def export_to_parquet(df, output_path):
        """
        导出为 Parquet（推荐，压缩率高，读取快）

        Args:
            df: DataFrame
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        output_path = Path(output_path).with_suffix('.parquet')
        df.to_parquet(output_path, index=False, compression='snappy')
        print(f"✅ 已保存 Parquet: {output_path.name} ({len(df):,} 行)")
        return output_path

    @staticmethod
    def smart_export(df, output_path, prefer_format='auto'):
        """
        智能选择导出格式

        Args:
            df: DataFrame
            output_path: 输出路径
            prefer_format: 'auto', 'excel', 'csv', 'parquet'

        Returns:
            输出文件路径列表
        """
        total_rows = len(df)

        # 自动选择格式
        if prefer_format == 'auto':
            if total_rows < LargeDataExporter.EXCEL_MAX_ROWS - 1:
                prefer_format = 'excel'
            elif total_rows < 10_000_000:
                prefer_format = 'csv'
            else:
                prefer_format = 'parquet'

        print(f"数据行数: {total_rows:,}, 选择格式: {prefer_format}")

        if prefer_format == 'excel':
            if total_rows >= LargeDataExporter.EXCEL_MAX_ROWS:
                print("⚠️ 数据超过 Excel 限制，分割成多个文件")
                return LargeDataExporter.export_to_multiple_excel(df, output_path)
            else:
                output_path = Path(output_path).with_suffix('.xlsx')
                df.to_excel(output_path, index=False)
                print(f"✅ 已保存 Excel: {output_path.name}")
                return [output_path]

        elif prefer_format == 'csv':
            return [LargeDataExporter.export_to_csv(df, output_path)]

        elif prefer_format == 'parquet':
            return [LargeDataExporter.export_to_parquet(df, output_path)]

        else:
            raise ValueError(f"不支持的格式: {prefer_format}")


def fix_pipeline_export():
    """
    修复流水线中的 Excel 导出问题

    在 pipeline_v2.py 的 module_3_export_verification 中使用
    """
    from src.config import Config

    def module_3_export_verification_fixed(mapping_dict, change_log, state=None):
        """修复版本的模块 3"""
        from src.utils.logger import setup_logger

        logger = setup_logger('pipeline', Config.LOG_FOLDER / 'pipeline.log')
        logger.info("=" * 50)
        logger.info("📝 [模块 3] 启动: 生成验证文件与处理数据...")

        # 导出映射关系表（修复版本）
        if change_log:
            df_log = pd.DataFrame(change_log)
            log_path = Config.CHECK_FOLDER / '词义映射对照表_只看变化'

            # 使用智能导出
            output_files = LargeDataExporter.smart_export(
                df_log,
                log_path,
                prefer_format='auto'  # 自动选择最佳格式
            )

            logger.info(f"👁️ 映射对照表已保存至: {', '.join([f.name for f in output_files])}")

            # 如果生成了多个文件，创建一个说明文件
            if len(output_files) > 1:
                readme_path = Config.CHECK_FOLDER / 'README.txt'
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write("映射对照表因数据量过大，已分割成多个文件：\n\n")
                    for i, file in enumerate(output_files, 1):
                        f.write(f"{i}. {file.name}\n")
                    f.write(f"\n总行数: {len(df_log):,}\n")
                logger.info(f"📄 已创建说明文件: {readme_path.name}")

        else:
            logger.warning("没有发现任何相似词合并，请检查阈值是否太高")

        # 批量处理文件（保持不变）
        logger.info("批量处理原始文件 (增加映射列)...")
        files = [f for f in Config.INPUT_FOLDER.iterdir() if f.suffix == '.csv']

        processed_file_paths = []

        for file in files:
            in_path = file
            out_path = Config.PROCESSED_FOLDER / f"mapped_{file.name}"

            try:
                df = pd.read_csv(in_path, encoding='utf-8-sig', parse_dates=[Config.DATE_COL])
                df['mapped_term'] = df[Config.TERM_COL].map(mapping_dict).fillna(df[Config.TERM_COL])
                df.to_csv(out_path, index=False, encoding='utf-8-sig')
                processed_file_paths.append(out_path)
            except Exception as e:
                logger.error(f"处理 {file.name} 失败: {e}")

        logger.info(f"✅ 所有文件已处理并保存至: {Config.PROCESSED_FOLDER}")

        if state:
            state.mark_completed('module_3', {'processed_files': len(processed_file_paths)})

        return processed_file_paths

    return module_3_export_verification_fixed


# 使用示例
if __name__ == "__main__":
    # 示例 1: 处理超大 DataFrame
    # df = pd.DataFrame({'col1': range(2000000), 'col2': range(2000000)})

    # 方式 1: 自动选择格式
    # LargeDataExporter.smart_export(df, './output/data', prefer_format='auto')

    # 方式 2: 强制分割成多个 Excel
    # LargeDataExporter.export_to_multiple_excel(df, './output/data')

    # 方式 3: 导出为 CSV
    # LargeDataExporter.export_to_csv(df, './output/data.csv')

    # 方式 4: 导出为 Parquet（推荐）
    # LargeDataExporter.export_to_parquet(df, './output/data.parquet')

    print("使用示例请参考代码注释")
