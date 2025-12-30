"""
使用 DuckDB 优化的数据处理模块

DuckDB 优势：
- 更快的 CSV 读取和处理
- SQL 查询能力
- 更好的内存管理
- 并行处理支持
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger('duckdb_pipeline', Config.LOG_FOLDER / 'duckdb_pipeline.log')


class DuckDBPipeline:
    """使用 DuckDB 优化的流水线"""

    def __init__(self, db_path=':memory:'):
        """
        初始化 DuckDB 流水线

        Args:
            db_path: 数据库路径，':memory:' 表示内存数据库
        """
        self.conn = duckdb.connect(db_path)
        self.logger = logger
        Config.setup_environment()
        Config.ensure_folders()

    def module_1_collect_vocab_duckdb(self):
        """
        模块 1: 使用 DuckDB 收集词汇

        性能优势：
        - 并行读取多个 CSV 文件
        - SQL DISTINCT 比 pandas unique() 更快
        - 内存效率更高
        """
        self.logger.info("=" * 50)
        self.logger.info("📦 [模块 1 - DuckDB] 启动: 全局词汇收集...")
        start_time = time.time()

        # 获取所有 CSV 文件
        csv_files = list(Config.INPUT_FOLDER.glob('*.csv'))
        self.logger.info(f"发现 {len(csv_files)} 个 CSV 文件")

        if not csv_files:
            self.logger.error("未找到 CSV 文件")
            return []

        # 使用 DuckDB 的 read_csv_auto 批量读取
        # 这比 pandas 逐个读取快得多
        file_pattern = str(Config.INPUT_FOLDER / '*.csv')

        try:
            # 创建临时表，合并所有文件
            query = f"""
            CREATE TEMP TABLE all_terms AS
            SELECT DISTINCT {Config.TERM_COL} as term
            FROM read_csv_auto('{file_pattern}',
                               union_by_name=true,
                               ignore_errors=true)
            WHERE {Config.TERM_COL} IS NOT NULL
            ORDER BY term
            """

            self.conn.execute(query)

            # 提取词汇列表
            vocab_list = self.conn.execute("SELECT term FROM all_terms").fetchdf()['term'].tolist()

            elapsed = time.time() - start_time
            self.logger.info(f"✅ 收集完成! 全网唯一词数: {len(vocab_list)}")
            self.logger.info(f"⏱️ 耗时: {elapsed:.2f}s")

            return vocab_list

        except Exception as e:
            self.logger.error(f"DuckDB 处理失败: {e}")
            self.logger.info("回退到 pandas 方法...")
            return self._fallback_collect_vocab()

    def _fallback_collect_vocab(self):
        """回退到 pandas 方法"""
        global_vocab = set()
        files = list(Config.INPUT_FOLDER.glob('*.csv'))

        for file in files:
            try:
                df = pd.read_csv(file, usecols=[Config.TERM_COL], encoding='utf-8-sig')
                terms = df[Config.TERM_COL].dropna().unique().tolist()
                global_vocab.update(terms)
            except Exception as e:
                self.logger.warning(f"跳过 {file.name}: {e}")

        return sorted(list(global_vocab))

    def module_3_export_with_duckdb(self, mapping_dict):
        """
        模块 3: 使用 DuckDB 批量处理文件

        性能优势：
        - 批量 JOIN 操作比 pandas map() 更快
        - 并行写入
        - 更少的内存占用
        """
        self.logger.info("=" * 50)
        self.logger.info("📝 [模块 3 - DuckDB] 启动: 批量处理文件...")
        start_time = time.time()

        # 将映射字典转换为 DuckDB 表
        mapping_df = pd.DataFrame([
            {'original_term': k, 'mapped_term': v}
            for k, v in mapping_dict.items()
        ])

        self.conn.register('mapping_table', mapping_df)

        files = list(Config.INPUT_FOLDER.glob('*.csv'))
        processed_count = 0

        for file in files:
            try:
                # 使用 DuckDB 读取和处理
                query = f"""
                SELECT
                    t.*,
                    COALESCE(m.mapped_term, t.{Config.TERM_COL}) as mapped_term
                FROM read_csv_auto('{file}', header=true) t
                LEFT JOIN mapping_table m ON t.{Config.TERM_COL} = m.original_term
                """

                result_df = self.conn.execute(query).fetchdf()

                # 保存结果
                output_path = Config.PROCESSED_FOLDER / f"mapped_{file.name}"
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

                processed_count += 1

            except Exception as e:
                self.logger.error(f"处理 {file.name} 失败: {e}")

        elapsed = time.time() - start_time
        self.logger.info(f"✅ 处理完成! 成功处理 {processed_count}/{len(files)} 个文件")
        self.logger.info(f"⏱️ 耗时: {elapsed:.2f}s")

        return processed_count

    def module_4_generate_matrix_duckdb(self):
        """
        模块 4: 使用 DuckDB 生成矩阵

        性能优势：
        - SQL PIVOT 比 pandas pivot 更快
        - 聚合操作更高效
        - 自动处理大数据集
        """
        self.logger.info("=" * 50)
        self.logger.info("📊 [模块 4 - DuckDB] 启动: 生成矩阵...")
        start_time = time.time()

        file_pattern = str(Config.PROCESSED_FOLDER / '*.csv')

        try:
            # 使用 DuckDB 读取所有文件并聚合
            query = f"""
            SELECT
                mapped_term,
                {Config.DATE_COL} as date,
                MIN(CAST(REPLACE({Config.RANK_COL}, ',', '') AS INTEGER)) as min_rank
            FROM read_csv_auto('{file_pattern}',
                               union_by_name=true,
                               ignore_errors=true)
            WHERE mapped_term IS NOT NULL
              AND {Config.DATE_COL} IS NOT NULL
            GROUP BY mapped_term, {Config.DATE_COL}
            """

            # 执行查询并转换为 pandas DataFrame
            df_agg = self.conn.execute(query).fetchdf()

            self.logger.info(f"聚合完成，共 {len(df_agg)} 条记录")

            # 使用 pandas 进行 pivot（DuckDB 的 PIVOT 语法较复杂）
            df_pivot = df_agg.pivot(index='mapped_term', columns='date', values='min_rank')

            # 处理和保存
            df_pivot = df_pivot.sort_index(axis=1)
            df_pivot.fillna(Config.FILL_VALUE, inplace=True)

            valid_mask = df_pivot.min(axis=1) < Config.RANK_THRESHOLD

            final_matrix = np.log10(df_pivot.loc[valid_mask].values + 1)
            kept_terms = df_pivot.index[valid_mask]
            kept_dates = df_pivot.columns

            # 保存文件
            np.save(Config.NPY_FOLDER / 'timesnet_input.npy', final_matrix)
            pd.Series(kept_terms, name='term').to_csv(
                Config.NPY_FOLDER / 'terms.csv', index=False, encoding='utf-8-sig'
            )
            pd.Series(kept_dates, name='date').to_csv(
                Config.NPY_FOLDER / 'dates.csv', index=False, encoding='utf-8-sig'
            )

            elapsed = time.time() - start_time
            self.logger.info(f"🎉 最终矩阵形状: {final_matrix.shape}")
            self.logger.info(f"⏱️ 总耗时: {elapsed:.2f}s")

            return final_matrix.shape

        except Exception as e:
            self.logger.error(f"DuckDB 处理失败: {e}")
            raise

    def analyze_data_quality(self):
        """
        使用 DuckDB 分析数据质量

        这是 DuckDB 的独特优势 - 快速的分析查询
        """
        self.logger.info("=" * 50)
        self.logger.info("📊 数据质量分析...")

        file_pattern = str(Config.INPUT_FOLDER / '*.csv')

        try:
            # 统计查询
            query = f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT {Config.TERM_COL}) as unique_terms,
                COUNT(DISTINCT {Config.DATE_COL}) as unique_dates,
                MIN({Config.RANK_COL}) as min_rank,
                MAX({Config.RANK_COL}) as max_rank,
                AVG(CAST(REPLACE({Config.RANK_COL}, ',', '') AS INTEGER)) as avg_rank
            FROM read_csv_auto('{file_pattern}',
                               union_by_name=true,
                               ignore_errors=true)
            WHERE {Config.TERM_COL} IS NOT NULL
            """

            stats = self.conn.execute(query).fetchdf()
            self.logger.info(f"\n数据统计:\n{stats.to_string()}")

            return stats

        except Exception as e:
            self.logger.error(f"分析失败: {e}")
            return None

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def benchmark_comparison():
    """
    性能对比：DuckDB vs Pandas

    测试模块 1 的性能差异
    """
    logger.info("=" * 50)
    logger.info("🏁 性能对比测试: DuckDB vs Pandas")

    # DuckDB 版本
    with DuckDBPipeline() as db_pipeline:
        start = time.time()
        vocab_duckdb = db_pipeline.module_1_collect_vocab_duckdb()
        duckdb_time = time.time() - start

    # Pandas 版本
    start = time.time()
    global_vocab = set()
    files = list(Config.INPUT_FOLDER.glob('*.csv'))
    for file in files:
        try:
            df = pd.read_csv(file, usecols=[Config.TERM_COL], encoding='utf-8-sig')
            terms = df[Config.TERM_COL].dropna().unique().tolist()
            global_vocab.update(terms)
        except:
            pass
    vocab_pandas = sorted(list(global_vocab))
    pandas_time = time.time() - start

    # 结果
    logger.info(f"\n性能对比结果:")
    logger.info(f"DuckDB: {duckdb_time:.2f}s, 词数: {len(vocab_duckdb)}")
    logger.info(f"Pandas: {pandas_time:.2f}s, 词数: {len(vocab_pandas)}")
    logger.info(f"速度提升: {(pandas_time / duckdb_time - 1) * 100:.1f}%")


if __name__ == "__main__":
    # 示例用法
    with DuckDBPipeline() as pipeline:
        # 数据质量分析
        pipeline.analyze_data_quality()

        # 收集词汇
        vocab = pipeline.module_1_collect_vocab_duckdb()

        # 性能对比
        # benchmark_comparison()
