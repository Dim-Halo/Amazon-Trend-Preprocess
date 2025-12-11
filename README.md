# Amazon Trend Predictor & Recommendation System

基于 TimesNet 趋势预测与 Qwen 语义分析的亚马逊智能选品系统。

## 核心功能
1. **数据清洗 Pipeline**: 自动化处理搜索词数据，清洗、归一化、聚类。
2. **TimesNet 矩阵生成**: 构建时间序列矩阵，用于趋势预测。
3. **双模态选品引擎**: 结合 **语义相似度 (Embedding)** 与 **趋势共振 (Trend Correlation)** 进行推荐。
4. **可视化仪表盘**: 基于 Streamlit 的交互式选品决策界面。

## 目录结构
- `src/`: 核心数据处理与算法逻辑。
- `web_app/`: Streamlit 前端应用。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt