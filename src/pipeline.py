import pandas as pd
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
import faiss
import gc
import json
import time

# ================= ⚙️ 全局配置 (Config) =================
CONFIG = {
    'input_folder': './clean_data',          # 输入：清洗后的数据文件夹
    'processed_folder': './processed_data',  # 输出：加了映射列的数据 (for model)
    'check_folder': './mapping_check',       # 输出：人工检查用 Excel
    'npy_folder': './final_npy',             # 输出：最终矩阵
    
    'term_col': 'normalized_term',           # 搜索词列名
    'date_col': '报告日期',                   # 日期列名
    'rank_col': '搜索频率排名',               # 排名列名
    
    'similarity_threshold': 0.75,            # 相似度阈值
    'device': 'cpu'                          # AMD 780M 强制用 CPU
}

# 确保文件夹存在
for folder in [CONFIG['processed_folder'], CONFIG['check_folder'], CONFIG['npy_folder']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ================= 📦 模块 1: 全局词汇收集 =================
def module_1_collect_vocab():
    """
    遍历所有文件，提取所有唯一的搜索词。
    """
    print("\n📦 [模块 1] 启动: 全局词汇收集...")
    start_time = time.time()
    
    global_vocab = set()
    files = [f for f in os.listdir(CONFIG['input_folder']) if f.endswith('.csv')]
    
    for i, file in enumerate(files):
        path = os.path.join(CONFIG['input_folder'], file)
        try:
            # 只读一列，速度极快
            df = pd.read_csv(path, usecols=[CONFIG['term_col']], encoding='utf-8-sig')
            terms = df[CONFIG['term_col']].dropna().unique().tolist()
            global_vocab.update(terms)
        except Exception as e:
            print(f"   ⚠️ 跳过 {file}: {e}")
            
    vocab_list = sorted(list(global_vocab))
    print(f"   ✅ 收集完成! 全网唯一词数: {len(vocab_list)}")
    print(f"   ⏱️ 耗时: {time.time() - start_time:.2f}s")
    return vocab_list

# ================= 🧠 模块 2: 向量聚类与映射 =================
# ================= 🧠 模块 2（稳定版）: 向量化 + Top-K 相似词映射 =================
def module_2_build_mapping(vocab_list):
    """
    【推荐稳定实现】
    - 使用 Top-K 搜索替代 range_search
    - 避免 FAISS 在 Windows / CPU / 百万级下的 C++ abort
    - 语义效果与 range_search 基本一致
    """
    print("\n🧠 [模块 2 - 稳定版] 启动: 向量化 + Top-K 相似搜索 (CPU Safe Mode)")
    start_time = time.time()

    # ---------- 1️⃣ 加载模型 ----------
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',
        device=CONFIG['device']
    )

    # ---------- 2️⃣ 向量化 ----------
    print(f"   ⚡ 正在计算 {len(vocab_list)} 个词的向量...")
    embeddings = model.encode(
        vocab_list,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 单位化，内积 = cosine
    faiss.normalize_L2(embeddings)

    # ---------- 3️⃣ 构建 FAISS Index ----------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # ---------- 4️⃣ Top-K 搜索（核心差异点） ----------
    # ⚠️ 经验值：20～30 足够覆盖所有近义词
    TOP_K = 20
    SIM_THRESHOLD = CONFIG['similarity_threshold']

    print(f"   🔍 正在执行 Top-{TOP_K} 相似搜索 (threshold={SIM_THRESHOLD})...")
    D, I = index.search(embeddings, TOP_K)

    # ---------- 5️⃣ 构建映射 ----------
    mapping_dict = {}
    change_log = []

    for i, word in enumerate(vocab_list):
        # 当前词的 Top-K 相似词
        sim_scores = D[i]
        neighbor_indices = I[i]

        # 过滤：只保留 ≥ 阈值 的
        neighbors = [
            vocab_list[j]
            for j, score in zip(neighbor_indices, sim_scores)
            if score >= SIM_THRESHOLD
        ]

        # 理论上至少包含自身
        if not neighbors:
            mapping_dict[word] = word
            continue

        # Canonical 策略：最短字符串
        canonical = min(neighbors, key=len)
        mapping_dict[word] = canonical

        if word != canonical:
            change_log.append({
                '原始词 (Original)': word,
                '映射后 (Mapped)': canonical,
                '同组词数': len(neighbors)
            })

        # 可选进度提示（不影响性能）
        if (i + 1) % 100000 == 0:
            print(f"      已处理 {i + 1:,}/{len(vocab_list):,} 个词", end='\r')

    print(f"\n   ✅ 映射完成!")
    print(f"   🔁 发生映射的词数: {len(change_log):,}")
    print(f"   ⏱️ 总耗时: {time.time() - start_time:.2f}s")

    return mapping_dict, change_log

# ================= 📝 模块 3: 输出验证文件 (你需求的核心) =================
def module_3_export_verification(mapping_dict, change_log):
    """
    1. 生成一个汇总的 Change Log Excel 方便人工审查。
    2. 重新处理 52 个 CSV，增加 'mapped_term' 列并保存。
    """
    print("\n📝 [模块 3] 启动: 生成验证文件与处理数据...")
    
    # --- 任务 A: 导出“映射关系表” (只看变化的，方便你快速检查) ---
    if change_log:
        df_log = pd.DataFrame(change_log)
        log_path = os.path.join(CONFIG['check_folder'], '词义映射对照表_只看变化.xlsx')
        # 存为 Excel，方便你筛选
        df_log.to_excel(log_path, index=False)
        print(f"   👁️ [人工检查] 映射对照表已保存至: {log_path} (请务必打开看看!)")
    else:
        print("   ⚠️ 没有发现任何相似词合并，请检查阈值是否太高。")

    # --- 任务 B: 批量处理 52 个文件，增加新列 ---
    print("   🚀 正在批量处理原始文件 (增加映射列)...")
    files = [f for f in os.listdir(CONFIG['input_folder']) if f.endswith('.csv')]
    
    processed_file_paths = []
    
    for file in files:
        in_path = os.path.join(CONFIG['input_folder'], file)
        out_path = os.path.join(CONFIG['processed_folder'], f"mapped_{file}")
        
        try:
            # 读取
            df = pd.read_csv(in_path, encoding='utf-8-sig', parse_dates=[CONFIG['date_col']])
            
            # 核心操作：映射并新增一列
            # map不到的词保持原样 (fillna)
            df['mapped_term'] = df[CONFIG['term_col']].map(mapping_dict).fillna(df[CONFIG['term_col']])
            
            # 保存
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            processed_file_paths.append(out_path)
            
        except Exception as e:
            print(f"   ❌ 处理 {file} 失败: {e}")
            
    print(f"   ✅ 所有文件已处理并保存至: {CONFIG['processed_folder']}")
    return processed_file_paths

# ================= 📊 模块 4: 生成 TimesNet 矩阵 =================
def module_4_generate_matrix():
    """
    直接扫描 CONFIG['processed_folder'] 文件夹中的所有 CSV 文件。
    执行 Pivot 和 Concat，生成最终矩阵。
    """
    print("\n📊 [模块 4] 启动: 扫描文件夹并生成矩阵...")
    
    input_folder = CONFIG['processed_folder']
    
    # 1. 检查文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 错误: 输入文件夹不存在: {input_folder}")
        print("   请先运行模块 3 生成数据，或检查路径配置。")
        return

    # 2. 扫描所有 CSV 文件
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"   📂 发现 {len(files)} 个 CSV 文件。")
    
    if not files:
        print("❌ 错误: 文件夹是空的，没有 CSV 文件。")
        return

    all_dfs = []
    
    # 3. 循环处理
    print("   🚀 开始读取数据...")
    for i, file in enumerate(files):
        file_path = os.path.join(input_folder, file)
        
        try:
            # 读取 (utf-8-sig 兼容性最佳)
            df = pd.read_csv(file_path, parse_dates=[CONFIG['date_col']], encoding='utf-8-sig')
            
            # 清洗列名
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            
            # 排名转数字 (去除逗号)
            if df[CONFIG['rank_col']].dtype == object:
                df[CONFIG['rank_col']] = df[CONFIG['rank_col']].astype(str).str.replace(',', '')
            df[CONFIG['rank_col']] = pd.to_numeric(df[CONFIG['rank_col']], errors='coerce')
            
            # 聚合与透视
            df_agg = df.groupby(['mapped_term', CONFIG['date_col']])[CONFIG['rank_col']].min().reset_index()
            df_pivot = df_agg.pivot(index='mapped_term', columns=CONFIG['date_col'], values=CONFIG['rank_col'])
            
            if not df_pivot.empty:
                all_dfs.append(df_pivot)
            
            # 进度条
            if (i + 1) % 10 == 0:
                print(f"      已处理 {i + 1}/{len(files)} 个文件...", end='\r')
                
        except Exception as e:
            print(f"   ⚠️ 跳过文件 {file}: {e}")

    print(f"\n   ✅ 成功读取 {len(all_dfs)} 个有效文件的数据块。")

    if not all_dfs:
        print("❌ 错误: 所有文件处理后均为空！无法生成矩阵。")
        return

    # 4. 合并矩阵
    print("   🧩 正在拼接全量矩阵 (Concat)...")
    final_df = pd.concat(all_dfs, axis=1)
    
    # 5. 处理重复列 (Transpose Groupby 修复版)
    print("   🔄 处理重复日期列 (Transpose Groupby)...")
    final_df = final_df.T.groupby(level=0).min().T
    
    # 按时间排序
    final_df = final_df.sort_index(axis=1)
    
    # 6. 最终清洗与保存
    final_df.fillna(100000, inplace=True) # 填充空值
    
    # 过滤逻辑 (保留历史最高排名前 10万 的词)
    valid_mask = final_df.min(axis=1) < 100000
    
    final_matrix = np.log10(final_df.loc[valid_mask].values + 1)
    kept_terms = final_df.index[valid_mask]
    kept_dates = final_df.columns
    
    # 确保输出目录存在
    if not os.path.exists(CONFIG['npy_folder']):
        os.makedirs(CONFIG['npy_folder'])

    # 保存文件
    np.save(os.path.join(CONFIG['npy_folder'], 'timesnet_input.npy'), final_matrix)
    pd.Series(kept_terms, name='term').to_csv(os.path.join(CONFIG['npy_folder'], 'terms.csv'), index=False, encoding='utf-8-sig')
    pd.Series(kept_dates, name='date').to_csv(os.path.join(CONFIG['npy_folder'], 'dates.csv'), index=False, encoding='utf-8-sig')
    
    print(f"   🎉 最终矩阵形状: {final_matrix.shape}")
    print(f"   💾 结果已保存至: {CONFIG['npy_folder']}")

# ================= 🚀 主程序入口 =================
if __name__ == "__main__":
    # 1. 收集词表
    # vocab = module_1_collect_vocab()
    
    # 2. 训练映射 (最核心的一步)
    # mapping, changes = module_2_build_mapping(vocab)
    
    # 3. 导出Excel对比文件 & 处理数据
    # files = module_3_export_verification(mapping, changes)
    
    # 4. 生成最终矩阵
    module_4_generate_matrix()