import pandas as pd
import numpy as np
import os
import gc
import json
import time
from sentence_transformers import SentenceTransformer
import faiss
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psutil
import sys

# ================= ⚙️ 全局配置 (Config) =================
CONFIG = {
    'input_folder': './clean_data',          # 输入：清洗后的数据文件夹
    'processed_folder': './processed_data',  # 输出：加了映射列的数据 (for model)
    'check_folder': './mapping_check',       # 输出：人工检查用 Excel
    'npy_folder': './final_npy',             # 输出：最终矩阵
    
    'term_col': 'normalized_term',           # 搜索词列名
    'date_col': '报告日期',                   # 日期列名
    'rank_col': '搜索频率排名',               # 排名列名
    
    'similarity_threshold': 0.725,           # 相似度阈值
    
    # GPU配置
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # 自动检测GPU
    'faiss_gpu': True,                        # 启用FAISS GPU加速
    'batch_size': 512,                        # GPU批次大小
    'vectorization_batch': 512,               # 向量化批次大小
    'parallel_process': True,                 # 是否并行处理文件
    'max_workers': min(4, cpu_count())        # 并行工作数
}

# 确保文件夹存在
for folder in [CONFIG['processed_folder'], CONFIG['check_folder'], CONFIG['npy_folder']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ================= 🔍 GPU状态检查 =================
def check_gpu_status():
    """检查GPU状态"""
    print("\n" + "="*50)
    print("🔍 GPU状态检查")
    print("="*50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   设备数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   显存总量: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
                print(f"   当前分配: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                print(f"   缓存保留: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
                
            # 设置默认GPU
            torch.cuda.set_device(0)
            print(f"\n📌 使用设备: GPU 0 - {torch.cuda.get_device_name(0)}")
        else:
            print("❌ CUDA不可用，将使用CPU")
            CONFIG['device'] = 'cpu'
            CONFIG['faiss_gpu'] = False
            print(f"📌 使用设备: CPU")
            
        # 检查FAISS GPU支持
        try:
            import faiss
            if hasattr(faiss, 'StandardGpuResources'):
                print("\n✅ FAISS GPU支持可用")
                if CONFIG['faiss_gpu'] and torch.cuda.is_available():
                    print("   🚀 FAISS GPU加速已启用")
            else:
                print("\n❌ FAISS GPU支持不可用")
                CONFIG['faiss_gpu'] = False
        except Exception as e:
            print(f"\n⚠️ FAISS导入失败: {e}")
            CONFIG['faiss_gpu'] = False
            
    except Exception as e:
        print(f"⚠️ GPU检查出错: {e}")
        CONFIG['device'] = 'cpu'
        CONFIG['faiss_gpu'] = False
    
    print("="*50)
    
    # 显示内存状态
    process = psutil.Process(os.getpid())
    print(f"💾 当前进程内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU显存使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB / "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("="*50)
    return CONFIG['device']

# ================= 📦 模块 1: 全局词汇收集 =================
def module_1_collect_vocab():
    """
    遍历所有文件，提取所有唯一的搜索词。
    """
    print("\n" + "="*50)
    print("📦 [模块 1] 启动: 全局词汇收集...")
    print("="*50)
    
    start_time = time.time()
    
    global_vocab = set()
    files = [f for f in os.listdir(CONFIG['input_folder']) if f.endswith('.csv')]
    
    print(f"📂 发现 {len(files)} 个CSV文件")
    print("🔍 正在收集词汇...")
    
    for i, file in enumerate(tqdm(files, desc="收集词汇")):
        path = os.path.join(CONFIG['input_folder'], file)
        try:
            # 只读一列，速度极快
            df = pd.read_csv(path, usecols=[CONFIG['term_col']], encoding='utf-8-sig')
            terms = df[CONFIG['term_col']].dropna().unique().tolist()
            global_vocab.update(terms)
        except Exception as e:
            print(f"   ⚠️ 跳过 {file}: {e}")
            
    vocab_list = sorted(list(global_vocab))
    
    # 内存清理
    del global_vocab
    gc.collect()
    
    print(f"\n✅ 收集完成!")
    print(f"   📊 全网唯一词数: {len(vocab_list):,}")
    print(f"   📈 词数统计:")
    print(f"      总词数: {len(vocab_list)}")
    
    # 显示词长统计
    if vocab_list:
        avg_len = np.mean([len(str(word)) for word in vocab_list[:1000]])  # 抽样统计
        print(f"      平均词长: {avg_len:.1f} 字符")
    
    print(f"   ⏱️ 耗时: {time.time() - start_time:.2f}秒")
    
    return vocab_list

# ================= 🧠 模块 2: GPU向量聚类与映射 =================
def module_2_build_mapping(vocab_list):
    """
    使用 KNN + 阈值过滤的 GPU 向量聚类与映射（稳定版）
    """
    print("\n" + "="*50)
    print("🧠 [模块 2] 启动: GPU向量化与KNN聚类（稳定版）")
    print("="*50)

    start_time = time.time()
    device = CONFIG['device']

    print(f"📊 处理词汇量: {len(vocab_list):,}")
    print("🚀 加载 SentenceTransformer 模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # ========= 1. 向量化 =========
    embeddings = model.encode(
        vocab_list,
        batch_size=CONFIG['vectorization_batch'],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    print(f"✅ 向量化完成: {embeddings.shape}")

    # ========= 2. 构建 FAISS Index =========
    if CONFIG['faiss_gpu'] and 'cuda' in device:
        print("🚀 使用 FAISS GPU Index (IndexFlatIP)")
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024)
        cpu_index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        print("💻 使用 FAISS CPU Index")
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    print(f"✅ 索引构建完成，向量数: {index.ntotal}")

    # ========= 3. KNN 搜索 =========
    K = 20  # ⭐ 可调：10~30 都安全
    print(f"🔎 执行 KNN 搜索 (K={K}) + 相似度阈值过滤...")
    distances, indices = index.search(embeddings, K)

    # ========= 4. 构建映射 =========
    vocab_array = np.array(vocab_list, dtype=object)
    mapping_dict = {}
    change_log = []

    for i in tqdm(range(len(vocab_list)), desc="构建映射"):
        sims = distances[i]
        nbrs = indices[i]

        # 阈值过滤（保留自己）
        valid_mask = sims >= CONFIG['similarity_threshold']
        valid_indices = nbrs[valid_mask]

        if len(valid_indices) > 1:
            neighbors = vocab_array[valid_indices]

            # ⭐ 新 canonical 规则：词数优先，其次长度
            canonical = min(
                neighbors,
                key=lambda x: (len(x.split()), len(x))
            )

            mapping_dict[vocab_list[i]] = canonical

            if vocab_list[i] != canonical:
                change_log.append({
                    '原始词 (Original)': vocab_list[i],
                    '映射后 (Mapped)': canonical,
                    '同组词数': len(neighbors),
                    '最高相似度': float(sims[valid_mask].max())
                })
        else:
            mapping_dict[vocab_list[i]] = vocab_list[i]

    print("\n✅ 映射构建完成!")
    print(f"   📊 总词数: {len(vocab_list):,}")
    print(f"   🔀 发生映射: {len(change_log):,}")
    print(f"   📉 映射比例: {len(change_log)/len(vocab_list)*100:.2f}%")

    # ========= 5. 清理 =========
    del embeddings
    del index
    if 'res' in locals():
        del res
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"⏱️ 模块耗时: {time.time() - start_time:.2f} 秒")
    return mapping_dict, change_log

# ================= 📝 模块 3: 输出验证文件 =================
def process_single_file(args):
    """处理单个文件的函数，用于并行处理"""
    file, mapping_dict = args
    in_path = os.path.join(CONFIG['input_folder'], file)
    out_path = os.path.join(CONFIG['processed_folder'], f"mapped_{file}")
    
    try:
        # 尝试不同编码读取
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
        df = None
        
        for enc in encodings:
            try:
                df = pd.read_csv(in_path, encoding=enc, parse_dates=[CONFIG['date_col']])
                break
            except:
                continue
        
        if df is None:
            print(f"   ❌ 无法读取 {file}，跳过")
            return None
        
        # 核心操作：映射并新增一列
        df['mapped_term'] = df[CONFIG['term_col']].map(mapping_dict).fillna(df[CONFIG['term_col']])
        
        # 保存
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        return out_path
        
    except Exception as e:
        print(f"   ❌ 处理 {file} 失败: {e}")
        return None

def module_3_export_verification(mapping_dict, change_log):
    """
    1. 生成一个汇总的 Change Log Excel 方便人工审查。
    2. 重新处理 52 个 CSV，增加 'mapped_term' 列并保存。
    """
    print("\n" + "="*50)
    print("📝 [模块 3] 启动: 生成验证文件与处理数据...")
    print("="*50)
    
    start_time = time.time()
    
    # --- 任务 A: 导出"映射关系表" ---
    if change_log:
        print("📊 生成映射关系表...")
        df_log = pd.DataFrame(change_log)
        
        # 按同组词数排序，方便查看
        df_log = df_log.sort_values(['同组词数', '原始词 (Original)'], ascending=[False, True])
        
        log_path = os.path.join(CONFIG['check_folder'], '词义映射对照表_只看变化.xlsx')
        
        # 使用Excel写入器，支持大文件
        with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
            df_log.to_excel(writer, index=False, sheet_name='映射关系')
            
            # 添加统计信息
            stats_df = pd.DataFrame({
                '统计项': ['总词数', '发生映射词数', '映射比例', '平均同组词数', '阈值'],
                '数值': [
                    len(mapping_dict),
                    len(change_log),
                    f"{len(change_log)/len(mapping_dict)*100:.1f}%",
                    f"{df_log['同组词数'].mean():.1f}",
                    CONFIG['similarity_threshold']
                ]
            })
            stats_df.to_excel(writer, index=False, sheet_name='统计信息')
        
        print(f"✅ [人工检查] 映射对照表已保存至: {log_path}")
        print(f"   📄 包含 {len(df_log)} 条映射记录")
    else:
        print("⚠️ 没有发现任何相似词合并，请检查阈值是否太高。")
    
    # --- 任务 B: 批量处理文件，增加新列 ---
    print("\n🚀 批量处理原始文件 (增加映射列)...")
    files = [f for f in os.listdir(CONFIG['input_folder']) if f.endswith('.csv')]
    print(f"📂 发现 {len(files)} 个文件需要处理")
    
    processed_file_paths = []
    
    if CONFIG['parallel_process'] and len(files) > 5:
        print(f"⚡ 启用并行处理，使用 {CONFIG['max_workers']} 个进程...")
        
        # 准备参数
        args_list = [(file, mapping_dict) for file in files]
        
        # 使用进程池并行处理
        with Pool(processes=CONFIG['max_workers']) as pool:
            results = list(tqdm(pool.imap(process_single_file, args_list), 
                              total=len(files), 
                              desc="并行处理文件"))
        
        processed_file_paths = [r for r in results if r is not None]
        
    else:
        print("💻 使用串行处理...")
        for file in tqdm(files, desc="处理文件"):
            args = (file, mapping_dict)
            result = process_single_file(args)
            if result:
                processed_file_paths.append(result)
    
    print(f"\n✅ 文件处理完成!")
    print(f"   📁 成功处理: {len(processed_file_paths)}/{len(files)} 个文件")
    print(f"   📂 保存至: {CONFIG['processed_folder']}")
    
    # 保存映射字典为JSON，方便后续使用
    print("💾 保存映射字典...")
    dict_path = os.path.join(CONFIG['check_folder'], 'mapping_dictionary.json')
    
    # 只保存部分映射（前1000条）作为示例
    sample_dict = dict(list(mapping_dict.items())[:1000])
    with open(dict_path, 'w', encoding='utf-8') as f:
        json.dump(sample_dict, f, ensure_ascii=False, indent=2)
    
    print(f"   📄 映射字典示例已保存: {dict_path}")
    print(f"⏱️ 耗时: {time.time() - start_time:.2f}秒")
    
    return processed_file_paths

# ================= 📊 模块 4: 生成 TimesNet 矩阵 =================
def module_4_generate_matrix():
    """
    直接扫描 CONFIG['processed_folder'] 文件夹中的所有 CSV 文件。
    执行 Pivot 和 Concat，生成最终矩阵。
    """
    print("\n" + "="*50)
    print("📊 [模块 4] 启动: 扫描文件夹并生成矩阵...")
    print("="*50)
    
    start_time = time.time()
    input_folder = CONFIG['processed_folder']
    
    # 1. 检查文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 错误: 输入文件夹不存在: {input_folder}")
        print("   请先运行模块 3 生成数据，或检查路径配置。")
        return
    
    # 2. 扫描所有 CSV 文件
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"📂 发现 {len(files)} 个 CSV 文件")
    
    if not files:
        print("❌ 错误: 文件夹是空的，没有 CSV 文件。")
        return
    
    all_dfs = []
    skipped_files = []
    
    # 3. 循环处理
    print("🚀 开始读取和处理数据...")
    
    for i, file in enumerate(tqdm(files, desc="处理文件")):
        file_path = os.path.join(input_folder, file)
        
        try:
            # 读取文件
            df = pd.read_csv(file_path, parse_dates=[CONFIG['date_col']], encoding='utf-8-sig')
            
            # 清洗列名
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            
            # 检查必要列是否存在
            required_cols = ['mapped_term', CONFIG['date_col'], CONFIG['rank_col']]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"   ⚠️ 文件 {file} 缺少列: {missing_cols}，跳过")
                skipped_files.append((file, f"缺少列: {missing_cols}"))
                continue
            
            # 排名转数字 (去除逗号)
            if df[CONFIG['rank_col']].dtype == object:
                df[CONFIG['rank_col']] = df[CONFIG['rank_col']].astype(str).str.replace(',', '')
            
            df[CONFIG['rank_col']] = pd.to_numeric(df[CONFIG['rank_col']], errors='coerce')
            
            # 去除空值
            df = df.dropna(subset=['mapped_term', CONFIG['date_col'], CONFIG['rank_col']])
            
            if len(df) == 0:
                print(f"   ⚠️ 文件 {file} 没有有效数据，跳过")
                skipped_files.append((file, "没有有效数据"))
                continue
            
            # 聚合与透视
            df_agg = df.groupby(['mapped_term', CONFIG['date_col']])[CONFIG['rank_col']].min().reset_index()
            df_pivot = df_agg.pivot(index='mapped_term', columns=CONFIG['date_col'], values=CONFIG['rank_col'])
            
            if not df_pivot.empty:
                all_dfs.append(df_pivot)
            else:
                skipped_files.append((file, "透视后为空"))
                
        except Exception as e:
            error_msg = str(e)[:100]  # 截取错误信息前100字符
            print(f"   ⚠️ 跳过文件 {file}: {error_msg}")
            skipped_files.append((file, error_msg))
    
    print(f"\n✅ 数据处理完成!")
    print(f"   📊 成功读取 {len(all_dfs)}/{len(files)} 个文件的数据块")
    
    if skipped_files:
        print(f"   ⚠️ 跳过了 {len(skipped_files)} 个文件")
        # 保存跳过的文件列表
        skipped_df = pd.DataFrame(skipped_files, columns=['文件名', '原因'])
        skipped_path = os.path.join(CONFIG['check_folder'], 'skipped_files.csv')
        skipped_df.to_csv(skipped_path, index=False, encoding='utf-8-sig')
        print(f"   📄 跳过的文件列表已保存: {skipped_path}")
    
    if not all_dfs:
        print("❌ 错误: 所有文件处理后均为空！无法生成矩阵。")
        return
    
    # 4. 合并矩阵
    print("🧩 正在拼接全量矩阵...")
    merge_start = time.time()
    
    # 分批合并，避免内存溢出
    batch_size = 10
    merged_dfs = []
    
    for i in range(0, len(all_dfs), batch_size):
        batch = all_dfs[i:i+batch_size]
        batch_df = pd.concat(batch, axis=1)
        
        # 处理重复列
        batch_df = batch_df.T.groupby(level=0).min().T
        merged_dfs.append(batch_df)
        
        # 清理内存
        del batch, batch_df
        gc.collect()
    
    # 最终合并
    if len(merged_dfs) > 1:
        final_df = pd.concat(merged_dfs, axis=1)
        final_df = final_df.T.groupby(level=0).min().T
    else:
        final_df = merged_dfs[0]
    
    print(f"   ✅ 矩阵合并完成，耗时: {time.time() - merge_start:.2f}秒")
    
    # 5. 按时间排序
    final_df = final_df.sort_index(axis=1)
    
    # 6. 最终清洗与保存
    print("🧹 最终清洗和保存...")
    
    # 填充空值
    final_df.fillna(2000000, inplace=True)
    
    # 过滤逻辑 (保留历史最高排名前 200万 的词)
    valid_mask = final_df.min(axis=1) < 2000000
    final_filtered = final_df.loc[valid_mask]
    
    print(f"   📊 过滤前: {final_df.shape[0]:,} 个词")
    print(f"   📊 过滤后: {final_filtered.shape[0]:,} 个词")
    print(f"   📊 过滤比例: {(1 - final_filtered.shape[0]/final_df.shape[0])*100:.1f}%")
    
    # 转换为log10矩阵
    final_matrix = np.log10(final_filtered.values + 1)
    kept_terms = final_filtered.index
    kept_dates = final_filtered.columns
    
    # 7. 保存文件
    print("💾 保存结果文件...")
    
    # 确保输出目录存在
    if not os.path.exists(CONFIG['npy_folder']):
        os.makedirs(CONFIG['npy_folder'])
    
    # 保存npy文件
    npy_path = os.path.join(CONFIG['npy_folder'], 'timesnet_input.npy')
    np.save(npy_path, final_matrix)
    print(f"   ✅ 矩阵文件: {npy_path}")
    print(f"      形状: {final_matrix.shape}")
    print(f"      大小: {final_matrix.nbytes / 1024 / 1024:.1f} MB")
    
    # 保存术语文件
    terms_path = os.path.join(CONFIG['npy_folder'], 'terms.csv')
    pd.Series(kept_terms, name='term').to_csv(terms_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 术语文件: {terms_path}")
    print(f"      术语数量: {len(kept_terms):,}")
    
    # 保存日期文件
    dates_path = os.path.join(CONFIG['npy_folder'], 'dates.csv')
    pd.Series(kept_dates, name='date').to_csv(dates_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 日期文件: {dates_path}")
    print(f"      日期数量: {len(kept_dates)}")
    print(f"      日期范围: {kept_dates.min()} 到 {kept_dates.max()}")
    
    # 保存矩阵统计信息
    stats = {
        'matrix_shape': final_matrix.shape,
        'min_value': float(final_matrix.min()),
        'max_value': float(final_matrix.max()),
        'mean_value': float(final_matrix.mean()),
        'std_value': float(final_matrix.std()),
        'n_terms': len(kept_terms),
        'n_dates': len(kept_dates),
        'date_range': [str(kept_dates.min()), str(kept_dates.max())],
        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'processing_time_seconds': time.time() - start_time
    }
    
    stats_path = os.path.join(CONFIG['npy_folder'], 'matrix_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 统计信息: {stats_path}")
    
    print(f"\n🎉 最终矩阵生成完成!")
    print(f"⏱️ 总耗时: {time.time() - start_time:.2f}秒")

# ================= 🎯 模块 5: 输出验证 =================
def verify_output():
    """验证输出文件的完整性"""
    print("\n" + "="*50)
    print("🔍 输出验证...")
    print("="*50)
    
    verification_passed = True
    
    # 检查npy文件
    npy_path = os.path.join(CONFIG['npy_folder'], 'timesnet_input.npy')
    if os.path.exists(npy_path):
        try:
            matrix = np.load(npy_path)
            print(f"✅ 矩阵文件: {matrix.shape}")
            print(f"   最小值: {matrix.min():.4f}")
            print(f"   最大值: {matrix.max():.4f}")
            print(f"   平均值: {matrix.mean():.4f}")
            print(f"   标准差: {matrix.std():.4f}")
            
            # 检查NaN值
            nan_count = np.isnan(matrix).sum()
            if nan_count > 0:
                print(f"⚠️ 警告: 矩阵中包含 {nan_count} 个NaN值")
                verification_passed = False
            else:
                print(f"✅ 矩阵无NaN值")
                
        except Exception as e:
            print(f"❌ 无法加载矩阵文件: {e}")
            verification_passed = False
    else:
        print("❌ 矩阵文件不存在")
        verification_passed = False
    
    # 检查术语文件
    terms_path = os.path.join(CONFIG['npy_folder'], 'terms.csv')
    if os.path.exists(terms_path):
        try:
            terms_df = pd.read_csv(terms_path, encoding='utf-8-sig')
            print(f"✅ 术语文件: {len(terms_df)} 个术语")
            
            # 检查重复项
            duplicates = terms_df['term'].duplicated().sum()
            if duplicates > 0:
                print(f"⚠️ 警告: 术语文件包含 {duplicates} 个重复项")
                verification_passed = False
            else:
                print(f"✅ 术语无重复")
                
        except Exception as e:
            print(f"❌ 无法加载术语文件: {e}")
            verification_passed = False
    else:
        print("❌ 术语文件不存在")
        verification_passed = False
    
    # 检查日期文件
    dates_path = os.path.join(CONFIG['npy_folder'], 'dates.csv')
    if os.path.exists(dates_path):
        try:
            dates_df = pd.read_csv(dates_path, encoding='utf-8-sig')
            print(f"✅ 日期文件: {len(dates_df)} 个日期")
            print(f"   日期范围: {dates_df['date'].min()} 到 {dates_df['date'].max()}")
        except Exception as e:
            print(f"❌ 无法加载日期文件: {e}")
            verification_passed = False
    else:
        print("❌ 日期文件不存在")
        verification_passed = False
    
    # 检查文件大小一致性
    if os.path.exists(npy_path) and os.path.exists(terms_path):
        matrix = np.load(npy_path)
        terms_df = pd.read_csv(terms_path, encoding='utf-8-sig')
        
        if matrix.shape[0] != len(terms_df):
            print(f"❌ 错误: 矩阵行数 ({matrix.shape[0]}) 与术语数量 ({len(terms_df)}) 不匹配")
            verification_passed = False
        else:
            print(f"✅ 矩阵与术语数量匹配: {matrix.shape[0]}")
    
    print("="*50)
    if verification_passed:
        print("🎉 所有验证通过!")
    else:
        print("⚠️ 验证失败，请检查输出文件")
    
    return verification_passed

# ================= 🚀 主程序入口 =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 语义聚类与矩阵生成系统 - GPU加速版")
    print("="*70)
    
    # 记录总开始时间
    total_start_time = time.time()
    
    try:
        # 步骤 0: 检查GPU状态
        device = check_gpu_status()
        
        # 步骤 1: 收集词表
        vocab = module_1_collect_vocab()
        
        if not vocab:
            print("❌ 错误: 没有收集到词汇，程序终止")
            sys.exit(1)
        
        # 步骤 2: 训练映射 (GPU加速)
        mapping, changes = module_2_build_mapping(vocab)
        
        # 清理词汇表内存
        del vocab
        gc.collect()
        
        # 步骤 3: 导出Excel对比文件 & 处理数据
        files = module_3_export_verification(mapping, changes)
        
        # 清理映射字典内存
        del mapping
        gc.collect()
        
        # 步骤 4: 生成最终矩阵
        module_4_generate_matrix()
        
        # 步骤 5: 验证输出
        verification_result = verify_output()
        
        # 总耗时统计
        total_time = time.time() - total_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*70)
        print("🎉 所有流程完成!")
        print("="*70)
        print(f"⏱️ 总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒")
        
        if verification_result:
            print("✅ 输出验证通过，可以用于TimesNet训练")
        else:
            print("⚠️ 输出验证失败，请检查输出文件")
            
        print(f"\n📁 输出文件位置:")
        print(f"   1. 处理后的数据: {CONFIG['processed_folder']}")
        print(f"   2. 检查文件: {CONFIG['check_folder']}")
        print(f"   3. 最终矩阵: {CONFIG['npy_folder']}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断，程序终止")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)