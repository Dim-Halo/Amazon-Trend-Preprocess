import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from recommender import HybridRecommender

# ================= 配置 =================
st.set_page_config(page_title="Amazon 智能选品引擎", page_icon="🛍️", layout="wide")

# ================= 初始化引擎 (单例模式) =================
@st.cache_resource
def load_engine():
    return HybridRecommender()

engine = load_engine()

# ================= 侧边栏：策略控制 =================
st.sidebar.header("🎛️ 选品策略配置")

st.sidebar.subheader("权重分配")
w_semantic = st.sidebar.slider("语义权重 (找相似品)", 0.0, 1.0, 0.5, 0.1)
w_trend = st.sidebar.slider("趋势权重 (找同节奏品)", 0.0, 1.0, 0.5, 0.1)

st.sidebar.info(
    """
    **策略指南：**
    - **找竞品/替代品**：语义权重 > 0.8
    - **找互补品/搭售**：趋势权重 > 0.8 (语义低但趋势同步)
    - **季节性选品**：趋势权重 > 0.6
    """
)

# ================= 主界面 =================
st.title("🛍️ Amazon 双模态选品推荐系统")
st.markdown("基于 **Semantic Embedding (语义)** 与 **TimesNet Trend (趋势)** 的混合推荐引擎")

# 1. 种子词输入
col1, col2 = st.columns([2, 1])
with col1:
    seed_term = st.selectbox("🌱 选择一个‘种子商品’ (Seed Item):", options=engine.terms)

if seed_term:
    # 2. 执行推荐
    recommendations, seed_curve = engine.recommend(
        seed_term, 
        weight_semantic=w_semantic, 
        weight_trend=w_trend, 
        top_k=50
    )
    
    # 3. 结果展示区
    st.divider()
    
    # --- 左侧：推荐列表 ---
    col_list, col_chart = st.columns([1, 2])
    
    with col_list:
        st.subheader("📋 推荐候选清单")
        
        # 转换为 DataFrame 方便展示
        df_res = pd.DataFrame(recommendations)
        
        # 格式化数字
        df_display = df_res[["Term", "Final_Score", "Semantic_Score", "Trend_Corr"]].copy()
        
        # 交互式表格，允许用户勾选
        selected_rows = st.dataframe(
            df_display.style.background_gradient(subset=["Final_Score"], cmap="Greens"),
            use_container_width=True,
            on_select="rerun", # 允许点击行
            selection_mode="multi-row"
        )
        
        # 获取用户在表格中选中的词
        selected_indices = selected_rows.selection["rows"]
        selected_terms_from_table = df_display.iloc[selected_indices]["Term"].tolist()

        st.divider()
    
        # 创建三个选项卡
        tab_attr, tab_risk, tab_season = st.tabs(["☁️ 核心卖点", "📊 风险评估", "📅 季节性"])
        
        # ================= Tab 1: 高频属性词云 =================
        with tab_attr:
            # 1. 提取所有推荐词文本
            all_text = " ".join(df_display["Term"].tolist()).lower()
            
            # 2. 停用词过滤
            stop_words = set(['for', 'with', 'and', 'in', 'the', 'of', 'to', 'a', 'men', 'women', 'kids', 'pack', 'set'])
            seed_tokens = set(seed_term.lower().split())
            stop_words.update(seed_tokens) # 把种子词本身也去掉，只看修饰词
            
            # 3. 统计词频
            from collections import Counter
            tokens = [word for word in all_text.split() if word not in stop_words and len(word) > 2]
            
            if tokens:
                word_counts = Counter(tokens).most_common(10)
                df_words = pd.DataFrame(word_counts, columns=["Word", "Count"])
                
                # 4. 绘图
                fig_word = px.bar(
                    df_words, 
                    x="Count", 
                    y="Word", 
                    orientation='h',
                    # title="Top Attributes", # 标题省掉，节省空间
                    color="Count",
                    color_continuous_scale="Blues"
                )
                fig_word.update_layout(
                    yaxis={'categoryorder':'total ascending'}, 
                    height=300, 
                    margin=dict(l=0, r=0, t=10, b=0) # 极致压缩边距
                )
                st.plotly_chart(fig_word, use_container_width=True)
                st.caption("💡 这些是推荐列表中出现频率最高的修饰词/属性。")
            else:
                st.info("数据量不足以生成词频分析")

        # ================= Tab 2: 市场波动性分析 =================
        with tab_risk:
            if recommendations:
                vol_data = []
                for item in recommendations:
                    # 还原真实排名
                    ranks = np.power(10, item["Trend_Curve"]) - 1
                    vol_data.append({
                        "Term": item["Term"],
                        "Volatility": np.std(ranks), # 标准差作为波动率
                        "Avg_Rank": np.mean(ranks)
                    })
                
                df_vol = pd.DataFrame(vol_data)
                
                # 绘图
                fig_vol = px.scatter(
                    df_vol, 
                    x="Avg_Rank", 
                    y="Volatility",
                    size="Volatility", # 气泡大小
                    hover_name="Term",
                    color="Volatility",
                    color_continuous_scale="RdYlGn_r", # 红(高波动)->绿(低波动)
                    labels={"Avg_Rank": "平均排名", "Volatility": "波动率 (标准差)"}
                )
                fig_vol.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_vol, use_container_width=True)
                st.caption("🟢 低波动(下方)=稳健款 | 🔴 高波动(上方)=投机款")
            else:
                st.info("暂无推荐数据")

        # ================= Tab 3: 季节性热力图 =================
        with tab_season:
            if recommendations:
                # 转换日期
                dates_pd = pd.to_datetime(engine.dates)
                monthly_heat = {m: [] for m in range(1, 13)}
                
                for item in recommendations:
                    ranks = np.power(10, item["Trend_Curve"]) - 1
                    for date, rank in zip(dates_pd, ranks):
                        # 只统计排名前10万的数据，太差的数据不计入热度
                        if rank < 100000:
                            monthly_heat[date.month].append(rank)
                
                # 计算热度分
                viz_data = []
                for m in range(1, 13):
                    if monthly_heat[m]:
                        avg_r = np.mean(monthly_heat[m])
                        # 热度公式：分数越高越火
                        score = 100000 / (avg_r + 1) 
                        viz_data.append({"Month": m, "Heat": score})
                    else:
                        viz_data.append({"Month": m, "Heat": 0})
                
                df_season = pd.DataFrame(viz_data)
                
                fig_season = px.bar(
                    df_season,
                    x="Month",
                    y="Heat",
                    color="Heat",
                    color_continuous_scale="Magma",
                    labels={"Heat": "热度指数"}
                )
                fig_season.update_xaxes(tickmode='linear', tick0=1, dtick=1)
                fig_season.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_season, use_container_width=True)
                st.caption("🔥 颜色越亮，代表该月份此类商品整体排名越高。")
            else:
                st.info("暂无数据")

    # --- 右侧：多维分析图 ---
    with col_chart:
        st.subheader("📈 趋势共振分析")
        
        # 准备绘图数据
        # 默认显示 Top 5，如果有用户选中则显示选中的
        terms_to_plot = [seed_term]
        if selected_terms_from_table:
            terms_to_plot += selected_terms_from_table
        else:
            terms_to_plot += df_display["Term"].head(5).tolist()
            
        # 构建 Plot Data
        plot_data = []
        
        # 添加种子词数据
        seed_ranks = np.power(10, seed_curve) - 1
        for i, val in enumerate(seed_ranks):
            if i < len(engine.dates):
                plot_data.append({"Date": engine.dates[i], "Term": seed_term, "Rank": val, "Type": "Seed"})
                
        # 添加推荐词数据
        for item in recommendations:
            if item["Term"] in terms_to_plot and item["Term"] != seed_term:
                curve = item["Trend_Curve"]
                ranks = np.power(10, curve) - 1
                for i, val in enumerate(ranks):
                    if i < len(engine.dates):
                        plot_data.append({"Date": engine.dates[i], "Term": item["Term"], "Rank": val, "Type": "Recommendation"})
        
        df_plot = pd.DataFrame(plot_data)
        
        # 绘图
        fig = px.line(
            df_plot, x="Date", y="Rank", color="Term", 
            line_dash="Type", # 种子词实线，推荐词虚线
            title=f"趋势共振: {seed_term} vs 推荐选品"
        )
        fig.update_yaxes(autorange="reversed", title_text="Rank (Lower is Better)")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # 散点图：语义 vs 趋势
        st.subheader("💠 选品象限 (语义 vs 趋势)")
        fig_scatter = px.scatter(
            df_res, x="Semantic_Score", y="Trend_Corr", hover_name="Term", color="Final_Score",
            title="右上角: 完美替代品 | 左上角: 潜在互补品 | 右下角: 强语义弱趋势"
        )
        # 添加辅助线
        fig_scatter.add_hline(y=0.5, line_dash="dot", annotation_text="趋势强相关")
        fig_scatter.add_vline(x=0.7, line_dash="dot", annotation_text="语义强相关")
        st.plotly_chart(fig_scatter, use_container_width=True)