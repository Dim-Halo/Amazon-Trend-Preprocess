"""流水线基类，提供 CPU 和 GPU 流水线的共享逻辑"""

import pandas as pd
import numpy as np
import os
import gc
import json
import time
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import faiss

from src.config import Config
from src.utils.logger import setup_logger
from src.utils.state import PipelineState
from src.utils.validator import DataValidator


class BasePipeline(ABC):
    """流水线基类"""

    def __init__(self, device='cpu', logger_name='pipeline'):
        """
        初始化流水线

        Args:
            device: 运行设备 ('cpu' 或 'cuda')
            logger_name: 日志记录器名称
        """
        self.device = device
        Config.DEVICE = device
        Config.setup_environment()
        Config.ensure_folders()
        self.logger = setup_logger(logger_name, Config.LOG_FOLDER / f'{logger_name}.log')
        self.state = PipelineState()

    @abstractmethod
    def _build_faiss_index(self, embeddings):
        """
        构建 FAISS 索引（子类实现）

        Args:
            embeddings: 向量数组

        Returns:
            FAISS 索引对象
        """
        pass

    def module_1_collect_vocab(self):
        """模块 1: 全局词汇收集"""
        self.logger.info("=" * 50)
        self.logger.info("📦 [模块 1] 启动: 全局词汇收集...")
        start_time = time.time()

        # 检查缓存
        if self.state.is_completed('module_1'):
            cache_file = self.state.get_cache_path('vocab_list.json')
            if cache_file.exists():
                self.logger.info("从缓存加载词汇表...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    vocab_list = json.load(f)
                self.logger.info(f"✅ 从缓存加载 {len(vocab_list)} 个词")
                return vocab_list

        # 验证文件结构
        self.logger.info("验证输入文件...")
        required_cols = [Config.TERM_COL, Config.DATE_COL, Config.RANK_COL]
        validation_results = DataValidator.validate_batch(Config.INPUT_FOLDER, required_cols, self.logger)

        invalid_files = [name for name, result in validation_results.items() if not result['is_valid']]
        if invalid_files:
            self.logger.warning(f"发现 {len(invalid_files)} 个无效文件，将跳过")

        global_vocab = set()
        files = [f for f in os.listdir(Config.INPUT_FOLDER) if f.endswith('.csv')]
        self.logger.info(f"发现 {len(files)} 个 CSV 文件")

        valid_count = 0
        for i, file in enumerate(files):
            if file in invalid_files:
                continue

            path = Config.INPUT_FOLDER / file
            try:
                df = pd.read_csv(path, usecols=[Config.TERM_COL], encoding='utf-8-sig')
                terms = df[Config.TERM_COL].dropna().unique().tolist()
                global_vocab.update(terms)
                valid_count += 1
            except Exception as e:
                self.logger.warning(f"跳过 {file}: {e}")

        vocab_list = sorted(list(global_vocab))
        elapsed = time.time() - start_time

        self.logger.info(f"✅ 收集完成! 处理了 {valid_count}/{len(files)} 个文件")
        self.logger.info(f"✅ 全网唯一词数: {len(vocab_list)}")
        self.logger.info(f"⏱️ 耗时: {elapsed:.2f}s")

        # 保存缓存
        cache_file = self.state.get_cache_path('vocab_list.json')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_list, f, ensure_ascii=False)
        self.state.mark_completed('module_1', {
            'vocab_count': len(vocab_list),
            'valid_files': valid_count,
            'total_files': len(files)
        })

        return vocab_list

    def module_2_build_mapping(self, vocab_list):
        """模块 2: 向量聚类与映射"""
        self.logger.info("=" * 50)
        self.logger.info(f"🧠 [模块 2] 启动: 向量化 + Top-K 相似搜索 ({self.device})")
        start_time = time.time()

        # 检查缓存
        if self.state.is_completed('module_2'):
            mapping_file = self.state.get_cache_path('mapping_dict.json')
            changes_file = self.state.get_cache_path('change_log.json')
            if mapping_file.exists() and changes_file.exists():
                self.logger.info("从缓存加载映射关系...")
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_dict = json.load(f)
                with open(changes_file, 'r', encoding='utf-8') as f:
                    change_log = json.load(f)
                self.logger.info(f"✅ 从缓存加载 {len(mapping_dict)} 个映射")
                return mapping_dict, change_log

        # 检查 FAISS 索引缓存
        embeddings_cache = self.state.get_cache_path('embeddings.npy')
        faiss_index_cache = self.state.get_cache_path('faiss_index.bin')

        if embeddings_cache.exists() and faiss_index_cache.exists():
            self.logger.info("从缓存加载向量和 FAISS 索引...")
            embeddings = np.load(embeddings_cache)
            index = faiss.read_index(str(faiss_index_cache))
            self.logger.info(f"✅ 从缓存加载完成，跳过向量计算")
        else:
            # 加载模型
            self.logger.info(f"加载模型 {Config.MODEL_NAME}...")
            model = SentenceTransformer(Config.MODEL_NAME, device=self.device)

            # 向量化
            self.logger.info(f"计算 {len(vocab_list)} 个词的向量...")
            embeddings = model.encode(
                vocab_list,
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # 归一化
            faiss.normalize_L2(embeddings)

            # 构建 FAISS Index（子类实现）
            index = self._build_faiss_index(embeddings)

            # 保存缓存
            self.logger.info("保存向量和 FAISS 索引到缓存...")
            np.save(embeddings_cache, embeddings)
            faiss.write_index(index, str(faiss_index_cache))
            self.logger.info("✅ 缓存保存完成")

        # Top-K 搜索
        self.logger.info(f"执行 Top-{Config.TOP_K} 相似搜索 (threshold={Config.SIMILARITY_THRESHOLD})...")
        D, I = index.search(embeddings, Config.TOP_K)

        # 构建映射
        mapping_dict = {}
        change_log = []

        for i, word in enumerate(vocab_list):
            sim_scores = D[i]
            neighbor_indices = I[i]

            neighbors = [
                vocab_list[j]
                for j, score in zip(neighbor_indices, sim_scores)
                if score >= Config.SIMILARITY_THRESHOLD
            ]

            if not neighbors:
                mapping_dict[word] = word
                continue

            canonical = min(neighbors, key=len)
            mapping_dict[word] = canonical

            if word != canonical:
                change_log.append({
                    '原始词 (Original)': word,
                    '映射后 (Mapped)': canonical,
                    '同组词数': len(neighbors)
                })

            if (i + 1) % 100000 == 0:
                self.logger.info(f"已处理 {i + 1:,}/{len(vocab_list):,} 个词")

        elapsed = time.time() - start_time
        self.logger.info(f"✅ 映射完成!")
        self.logger.info(f"🔁 发生映射的词数: {len(change_log):,}")
        self.logger.info(f"⏱️ 总耗时: {elapsed:.2f}s")

        # 保存缓存
        mapping_file = self.state.get_cache_path('mapping_dict.json')
        changes_file = self.state.get_cache_path('change_log.json')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_dict, f, ensure_ascii=False)
        with open(changes_file, 'w', encoding='utf-8') as f:
            json.dump(change_log, f, ensure_ascii=False)
        self.state.mark_completed('module_2', {'mapping_count': len(change_log)})

        return mapping_dict, change_log

    def module_3_export_verification(self, mapping_dict, change_log):
        """模块 3: 输出验证文件"""
        self.logger.info("=" * 50)
        self.logger.info("📝 [模块 3] 启动: 生成验证文件与处理数据...")

        # 导出映射关系表
        if change_log:
            df_log = pd.DataFrame(change_log)
            log_path = Config.CHECK_FOLDER / '词义映射对照表_只看变化.xlsx'
            df_log.to_excel(log_path, index=False)
            self.logger.info(f"👁️ 映射对照表已保存至: {log_path}")
        else:
            self.logger.warning("没有发现任何相似词合并，请检查阈值是否太高")

        # 批量处理文件
        self.logger.info("批量处理原始文件 (增加映射列)...")
        files = [f for f in os.listdir(Config.INPUT_FOLDER) if f.endswith('.csv')]

        processed_file_paths = []

        for file in files:
            in_path = Config.INPUT_FOLDER / file
            out_path = Config.PROCESSED_FOLDER / f"mapped_{file}"

            try:
                df = pd.read_csv(in_path, encoding='utf-8-sig', parse_dates=[Config.DATE_COL])
                df['mapped_term'] = df[Config.TERM_COL].map(mapping_dict).fillna(df[Config.TERM_COL])
                df.to_csv(out_path, index=False, encoding='utf-8-sig')
                processed_file_paths.append(out_path)
            except Exception as e:
                self.logger.error(f"处理 {file} 失败: {e}")

        self.logger.info(f"✅ 所有文件已处理并保存至: {Config.PROCESSED_FOLDER}")
        self.state.mark_completed('module_3', {'processed_files': len(processed_file_paths)})

        return processed_file_paths

    def module_4_generate_matrix(self, memory_limit_mb=2048):
        """模块 4: 生成 TimesNet 矩阵"""
        self.logger.info("=" * 50)
        self.logger.info("📊 [模块 4] 启动: 扫描文件夹并生成矩阵...")

        input_folder = Config.PROCESSED_FOLDER

        if not input_folder.exists():
            self.logger.error(f"输入文件夹不存在: {input_folder}")
            self.logger.error("请先运行模块 3 生成数据")
            return

        files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        self.logger.info(f"📂 发现 {len(files)} 个 CSV 文件")

        if not files:
            self.logger.error("文件夹是空的，没有 CSV 文件")
            return

        all_dfs = []
        partial_results = []

        try:
            import psutil
            memory_monitoring = True
            process = psutil.Process()
            self.logger.info(f"内存监控已启用，限制: {memory_limit_mb} MB")
        except ImportError:
            memory_monitoring = False
            self.logger.warning("psutil 未安装，内存监控已禁用")

        self.logger.info("开始读取数据...")
        for i, file in enumerate(files):
            file_path = input_folder / file

            # 检查内存使用
            if memory_monitoring:
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > memory_limit_mb and all_dfs:
                    self.logger.warning(f"内存使用 {memory_mb:.0f}MB 超过限制，保存部分结果...")
                    partial_df = pd.concat(all_dfs, axis=1)
                    partial_results.append(partial_df)
                    all_dfs = []
                    gc.collect()
                    self.logger.info(f"已保存部分结果 #{len(partial_results)}，内存已释放")

            try:
                df = pd.read_csv(file_path, parse_dates=[Config.DATE_COL], encoding='utf-8-sig')
                df.columns = df.columns.str.strip().str.replace('\ufeff', '')

                if df[Config.RANK_COL].dtype == object:
                    df[Config.RANK_COL] = df[Config.RANK_COL].astype(str).str.replace(',', '')
                df[Config.RANK_COL] = pd.to_numeric(df[Config.RANK_COL], errors='coerce')

                df_agg = df.groupby(['mapped_term', Config.DATE_COL])[Config.RANK_COL].min().reset_index()
                df_pivot = df_agg.pivot(index='mapped_term', columns=Config.DATE_COL, values=Config.RANK_COL)

                if not df_pivot.empty:
                    all_dfs.append(df_pivot)

                if (i + 1) % 10 == 0:
                    mem_info = f", 内存: {memory_mb:.0f}MB" if memory_monitoring else ""
                    self.logger.info(f"已处理 {i + 1}/{len(files)} 个文件{mem_info}")

            except Exception as e:
                self.logger.warning(f"跳过文件 {file}: {e}")

        self.logger.info(f"✅ 成功读取 {len(all_dfs)} 个有效文件的数据块")

        if not all_dfs and not partial_results:
            self.logger.error("所有文件处理后均为空！无法生成矩阵")
            return

        # 合并所有数据
        if partial_results:
            self.logger.info(f"合并 {len(partial_results)} 个部分结果和当前数据...")
            if all_dfs:
                partial_results.append(pd.concat(all_dfs, axis=1))
            final_df = pd.concat(partial_results, axis=1)
            gc.collect()
        else:
            self.logger.info("拼接全量矩阵...")
            final_df = pd.concat(all_dfs, axis=1)

        # 处理重复列
        self.logger.info("处理重复日期列...")
        final_df = final_df.T.groupby(level=0).min().T
        final_df = final_df.sort_index(axis=1)

        # 最终清洗与保存
        final_df.fillna(Config.FILL_VALUE, inplace=True)

        valid_mask = final_df.min(axis=1) < Config.RANK_THRESHOLD

        final_matrix = np.log10(final_df.loc[valid_mask].values + 1)
        kept_terms = final_df.index[valid_mask]
        kept_dates = final_df.columns

        # 保存文件
        np.save(Config.NPY_FOLDER / 'timesnet_input.npy', final_matrix)
        pd.Series(kept_terms, name='term').to_csv(Config.NPY_FOLDER / 'terms.csv', index=False, encoding='utf-8-sig')
        pd.Series(kept_dates, name='date').to_csv(Config.NPY_FOLDER / 'dates.csv', index=False, encoding='utf-8-sig')

        self.logger.info(f"🎉 最终矩阵形状: {final_matrix.shape}")
        self.logger.info(f"💾 结果已保存至: {Config.NPY_FOLDER}")

        self.state.mark_completed('module_4', {
            'matrix_shape': final_matrix.shape,
            'terms_count': len(kept_terms),
            'dates_count': len(kept_dates)
        })

    def run(self, force_rerun=False, modules=None):
        """
        运行流水线

        Args:
            force_rerun: 是否强制重新运行
            modules: 要运行的模块列表
        """
        if force_rerun:
            self.logger.info("强制重新运行所有模块")
            self.state.reset()

        if modules is None:
            modules = [1, 2, 3, 4]

        vocab = None
        mapping = None
        changes = None

        # 模块 1
        if 1 in modules:
            if not self.state.is_completed('module_1') or force_rerun:
                vocab = self.module_1_collect_vocab()
            else:
                self.logger.info("模块 1 已完成，跳过")
                cache_file = self.state.get_cache_path('vocab_list.json')
                with open(cache_file, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)

        # 模块 2
        if 2 in modules:
            if vocab is None:
                cache_file = self.state.get_cache_path('vocab_list.json')
                with open(cache_file, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)

            if not self.state.is_completed('module_2') or force_rerun:
                mapping, changes = self.module_2_build_mapping(vocab)
            else:
                self.logger.info("模块 2 已完成，跳过")
                mapping_file = self.state.get_cache_path('mapping_dict.json')
                changes_file = self.state.get_cache_path('change_log.json')
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                with open(changes_file, 'r', encoding='utf-8') as f:
                    changes = json.load(f)

        # 模块 3
        if 3 in modules:
            if mapping is None:
                mapping_file = self.state.get_cache_path('mapping_dict.json')
                changes_file = self.state.get_cache_path('change_log.json')
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                with open(changes_file, 'r', encoding='utf-8') as f:
                    changes = json.load(f)

            if not self.state.is_completed('module_3') or force_rerun:
                self.module_3_export_verification(mapping, changes)
            else:
                self.logger.info("模块 3 已完成，跳过")

        # 模块 4
        if 4 in modules:
            if not self.state.is_completed('module_4') or force_rerun:
                self.module_4_generate_matrix()
            else:
                self.logger.info("模块 4 已完成，跳过")

        self.logger.info("=" * 50)
        self.logger.info("🎉 流水线执行完成!")


class CPUPipeline(BasePipeline):
    """CPU 流水线"""

    def __init__(self):
        super().__init__(device='cpu', logger_name='pipeline_cpu')

    def _build_faiss_index(self, embeddings):
        """构建 CPU FAISS 索引"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index


class GPUPipeline(BasePipeline):
    """GPU 流水线"""

    def __init__(self, gpu_id=0):
        super().__init__(device=f'cuda:{gpu_id}', logger_name='pipeline_gpu')
        self.gpu_id = gpu_id

    def _build_faiss_index(self, embeddings):
        """构建 GPU FAISS 索引"""
        try:
            import torch
            if not torch.cuda.is_available():
                self.logger.warning("CUDA 不可用，回退到 CPU 索引")
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                return index

            dim = embeddings.shape[1]
            cpu_index = faiss.IndexFlatIP(dim)

            # 尝试使用 GPU
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
                gpu_index.add(embeddings)
                self.logger.info(f"✅ 使用 GPU {self.gpu_id} 构建 FAISS 索引")
                return gpu_index
            except Exception as e:
                self.logger.warning(f"GPU 索引构建失败，回退到 CPU: {e}")
                cpu_index.add(embeddings)
                return cpu_index

        except ImportError:
            self.logger.warning("PyTorch 未安装，使用 CPU 索引")
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            return index
