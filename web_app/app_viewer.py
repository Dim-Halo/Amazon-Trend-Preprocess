import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os

# ================= 页面配置 =================
st.set_page_config(page_title="Amazon Semantic Trend Viewer", page_icon="🧠", layout="wide")

# ================= 侧边栏配置 =================
st.sidebar.title("📂 数据配置")
base_folder = st.sidebar.text_input("数据文件夹", "./final_npy")
npy_file = "timesnet_input.npy"
term_file = "terms.csv"
date_file = "dates.csv"
vector_file = "term_vectors.npy"  # 新增：向量文件

# ================= 数据加载 (带缓存) =================
@st.cache_data
def load_all_data(folder):
    try:
        # 1. 路径构建
        path_matrix = os.path.join(folder, npy_file)
        path_terms = os.path.join(folder, term_file)
        path_dates = os.path.join(folder, date_file)
        path_vectors = os.path.join(folder, vector_file)

        if not os.path.exists(path_matrix): return None, f"找不到 {npy_file}"
        
        # 2. 加载基础数据
        matrix = np.load(path_matrix)
        df_terms = pd.read_csv(path_terms, encoding='utf-8-sig')
        df_dates = pd.read_csv(path_dates, encoding='utf-8-sig')
        
        terms = df_terms.iloc[:, 0].astype(str).tolist()
        dates = df_dates.iloc[:, 0].astype(str).tolist()

        # 3. 加载向量 (如果有的话)
        vectors = None
        if os.path.exists(path_vectors):
            vectors = np.load(path_vectors)
        
        return {
            "matrix": matrix,
            "terms": terms,
            "dates": dates,
            "vectors": vectors
        }, None

    except Exception as e:
        return None, f"加载出错: {e}"

# ================= 核心：语义相似度计算 =================
@st.cache_data
def get_semantic_recommendations(main_term, all_terms, vectors, top_k=20):
    """
    计算余弦相似度，找出最相关的词
    """
    if vectors is None or main_term not in all_terms:
        return []
    
    # 1. 找到主词的向量
    idx = all_terms.index(main_term)
    target_vec = vectors[idx] # Shape: (384,)
    
    # 2. 矩阵运算：计算与所有词的相似度 (Dot Product)
    # 因为向量已经归一化，所以 点积 = 余弦相似度
    # scores Shape: (N_terms, )
    scores = np.dot(vectors, target_vec)
    
    # 3. 获取 Top K 的索引 (从大到小排序)
    # argsort 返回从小到大的索引，所以取最后 K 个并反转
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    recommendations = []
    for i in top_indices:
        term = all_terms[i]
        score = scores[i]
        if term == main_term: continue # 排除自己
        recommendations.append((term, score))
        
    return recommendations

# ================= 辅助：获取绘图数据 =================
def get_plot_df(term_list, all_terms, matrix, dates):
    plot_data = []
    for t in term_list:
        try:
            idx = all_terms.index(t)
            y_vals = matrix[idx, :]
            # 还原真实排名
            real_ranks = np.power(10, y_vals) - 1
            
            # 截断日期以匹配数据长度
            curr_dates = dates[:len(y_vals)]
            
            # 构建临时 DataFrame
            df_temp = pd.DataFrame({
                "Date": curr_dates,
                "Term": t,
                "Real_Rank": real_ranks
            })
            plot_data.append(df_temp)
        except:
            continue
            
    if plot_data:
        return pd.concat(plot_data, ignore_index=True)
    return pd.DataFrame()

# ================= 主程序 =================
st.title("🧠 Amazon Semantic Trend Viewer")

data, error = load_all_data(base_folder)

if error:
    st.error(error)
    st.stop()

# 解包数据
matrix = data['matrix']
terms = data['terms']
dates = data['dates']
vectors = data['vectors']

# 检查向量是否存在
if vectors is None:
    st.warning("⚠️ 未检测到 `term_vectors.npy`。目前仅支持文本匹配。请运行 `generate_vectors_for_ui.py` 生成向量以启用语义联想。")

# --- 搜索区 ---
col1, col2 = st.columns([3, 1])
with col1:
    selected_term = st.selectbox("🔍 搜索核心词:", options=terms, index=0)

with col2:
    st.write("")
    st.write("")
    st.markdown(f"**词库大小:** {len(terms):,} | **向量状态:** {'✅ 已加载' if vectors is not None else '❌ 未加载'}")

if selected_term:
    
    # --- 推荐算法区 ---
    st.divider()
    
    rec_col1, rec_col2 = st.columns(2)
    
    # 1. 语义联想 (Semantic)
    semantic_list = []
    with rec_col1:
        st.subheader("🧠 语义联想推荐")
        if vectors is not None:
            # 获取推荐
            recommendations = get_semantic_recommendations(selected_term, terms, vectors, top_k=30)
            
            # 提取词名用于 multiselect
            rec_options = [f"{r[0]} (相似度: {r[1]:.2f})" for r in recommendations]
            rec_map = {f"{r[0]} (相似度: {r[1]:.2f})": r[0] for r in recommendations}
            
            selected_semantic = st.multiselect(
                "基于含义相似 (即使拼写不同):",
                options=rec_options,
                placeholder="例如：选 running shoes 会推荐 sneakers"
            )
            # 还原回纯词名
            semantic_list = [rec_map[s] for s in selected_semantic]
        else:
            st.info("需要向量文件才能使用此功能。")

    # 2. 文本包含 (Token Matching)
    token_list = []
    with rec_col2:
        st.subheader("🔤 文本变体推荐")
        # 简单的包含匹配
        keywords = [w for w in selected_term.lower().split() if len(w) > 2]
        token_cands = [t for t in terms if any(k in t.lower() for k in keywords) and t != selected_term][:50]
        
        selected_token = st.multiselect(
            "基于拼写包含:",
            options=token_cands,
            placeholder="例如：选 gloves 会推荐 winter gloves"
        )
        token_list = selected_token

    # --- 绘图区 ---
    st.divider()
    
    # 合并用户选择的所有词
    final_compare_list = list(set([selected_term] + semantic_list + token_list))
    
    if len(final_compare_list) > 0:
        df_chart = get_plot_df(final_compare_list, terms, matrix, dates)
        
        if not df_chart.empty:
            title_txt = f"趋势对比: {selected_term} vs {len(final_compare_list)-1} 个关联词"
            
            fig = px.line(
                df_chart, 
                x="Date", 
                y="Real_Rank", 
                color="Term",
                title=title_txt,
                markers=True
            )
            fig.update_yaxes(autorange="reversed", title_text="排名 (越小越好)")
            fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 数据表
            with st.expander("查看详细数据"):
                st.dataframe(df_chart.pivot(index="Date", columns="Term", values="Real_Rank").sort_index(ascending=False))