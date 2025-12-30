"""数据验证工具"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

class DataValidator:
    """CSV 数据验证器"""

    @staticmethod
    def validate_csv_structure(file_path: Path, required_cols: List[str]) -> Dict[str, any]:
        """
        验证 CSV 文件结构

        Args:
            file_path: CSV 文件路径
            required_cols: 必需的列名列表

        Returns:
            验证结果字典，包含 is_valid, missing_cols, error
        """
        result = {
            'is_valid': False,
            'missing_cols': [],
            'extra_info': {},
            'error': None
        }

        try:
            # 只读取第一行来检查列名
            df = pd.read_csv(file_path, nrows=1, encoding='utf-8-sig')

            # 清理列名
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')

            # 检查缺失的列
            missing = set(required_cols) - set(df.columns)

            if missing:
                result['missing_cols'] = list(missing)
                result['error'] = f"缺失必需列: {', '.join(missing)}"
            else:
                result['is_valid'] = True
                result['extra_info']['total_cols'] = len(df.columns)

        except Exception as e:
            result['error'] = str(e)

        return result

    @staticmethod
    def validate_data_quality(file_path: Path, term_col: str, date_col: str, rank_col: str) -> Dict[str, any]:
        """
        验证数据质量

        Args:
            file_path: CSV 文件路径
            term_col: 搜索词列名
            date_col: 日期列名
            rank_col: 排名列名

        Returns:
            数据质量报告
        """
        result = {
            'is_valid': True,
            'warnings': [],
            'stats': {},
            'error': None
        }

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')

            # 统计信息
            result['stats']['total_rows'] = len(df)
            result['stats']['null_terms'] = df[term_col].isna().sum()
            result['stats']['null_dates'] = df[date_col].isna().sum()
            result['stats']['null_ranks'] = df[rank_col].isna().sum()
            result['stats']['unique_terms'] = df[term_col].nunique()

            # 检查空值
            if result['stats']['null_terms'] > 0:
                result['warnings'].append(f"{term_col} 列有 {result['stats']['null_terms']} 个空值")

            if result['stats']['null_dates'] > 0:
                result['warnings'].append(f"{date_col} 列有 {result['stats']['null_dates']} 个空值")

            if result['stats']['null_ranks'] > 0:
                result['warnings'].append(f"{rank_col} 列有 {result['stats']['null_ranks']} 个空值")

            # 检查排名列是否可转换为数字
            try:
                if df[rank_col].dtype == object:
                    test_rank = df[rank_col].astype(str).str.replace(',', '')
                    pd.to_numeric(test_rank, errors='raise')
            except:
                result['warnings'].append(f"{rank_col} 列包含无法转换为数字的值")

        except Exception as e:
            result['is_valid'] = False
            result['error'] = str(e)

        return result

    @staticmethod
    def validate_batch(folder: Path, required_cols: List[str], logger=None) -> Dict[str, Dict]:
        """
        批量验证文件夹中的所有 CSV 文件

        Args:
            folder: 文件夹路径
            required_cols: 必需的列名列表
            logger: 日志记录器（可选）

        Returns:
            验证结果字典，key 为文件名，value 为验证结果
        """
        results = {}
        files = [f for f in folder.iterdir() if f.suffix == '.csv']

        for file in files:
            result = DataValidator.validate_csv_structure(file, required_cols)
            results[file.name] = result

            if logger:
                if result['is_valid']:
                    logger.info(f"✅ {file.name} - 验证通过")
                else:
                    logger.warning(f"❌ {file.name} - {result['error']}")

        return results
