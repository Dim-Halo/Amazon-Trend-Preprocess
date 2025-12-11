import numpy as np
import pandas as pd
from numpy.linalg import norm

class HybridRecommender:
    def __init__(self, folder='./final_npy'):
        # 1. 加载数据
        print("⏳ 正在加载推荐引擎数据...")
        self.matrix = np.load(f'{folder}/timesnet_input.npy') # (N, T)
        self.vectors = np.load(f'{folder}/term_vectors.npy')  # (N, D)
        
        # 加载元数据
        df_terms = pd.read_csv(f'{folder}/terms.csv', encoding='utf-8-sig')
        self.terms = df_terms.iloc[:, 0].astype(str).tolist()
        
        df_dates = pd.read_csv(f'{folder}/dates.csv', encoding='utf-8-sig')
        self.dates = df_dates.iloc[:, 0].astype(str).tolist()
        
        print(f"✅ 数据加载完成: {len(self.terms)} 个商品, {len(self.dates)} 个时间点")

    def get_index(self, term):
        try:
            return self.terms.index(term)
        except ValueError:
            return -1

    def calculate_trend_correlation(self, target_idx):
        """
        计算目标词与所有其他词的趋势相关性 (Pearson Correlation)
        使用向量化计算，速度极快
        """
        # 目标曲线 (1, T)
        target_curve = self.matrix[target_idx]
        
        # 所有曲线 (N, T)
        all_curves = self.matrix
        
        # 1. 减去均值 (Center the data)
        target_centered = target_curve - np.mean(target_curve)
        all_centered = all_curves - np.mean(all_curves, axis=1, keepdims=True)
        
        # 2. 计算分子 (Covariance)
        numerator = np.dot(all_centered, target_centered)
        
        # 3. 计算分母 (Std Devs)
        target_norm = np.linalg.norm(target_centered)
        all_norms = np.linalg.norm(all_centered, axis=1)
        
        # 防止除零
        denominator = target_norm * all_norms
        denominator[denominator == 0] = 1e-9 
        
        # 4. 得到相关系数 (-1 到 1)
        correlations = numerator / denominator
        return correlations

    def calculate_semantic_similarity(self, target_idx):
        """
        计算语义相似度 (Cosine Similarity)
        前提：向量在生成时已经归一化了 (Norm=1)，否则需要除以 Norm
        """
        target_vec = self.vectors[target_idx]
        # Dot product
        similarities = np.dot(self.vectors, target_vec)
        return similarities

    def recommend(self, seed_term, weight_semantic=0.6, weight_trend=0.4, top_k=20):
        """
        核心推荐函数
        """
        idx = self.get_index(seed_term)
        if idx == -1:
            return []

        # 1. 计算两种分数
        sim_scores = self.calculate_semantic_similarity(idx) # 语义分
        corr_scores = self.calculate_trend_correlation(idx)  # 趋势分
        
        # 2. 混合加权
        # 归一化到 0-1 之间 (Correlation 原本是 -1 到 1)
        normalized_corr = (corr_scores + 1) / 2 
        
        final_scores = (weight_semantic * sim_scores) + (weight_trend * normalized_corr)
        
        # 3. 排序
        # argsort 从小到大，取最后 k 个并反转
        top_indices = np.argsort(final_scores)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            term = self.terms[i]
            if term == seed_term: continue # 排除自己
            
            results.append({
                "Term": term,
                "Final_Score": final_scores[i],
                "Semantic_Score": sim_scores[i],
                "Trend_Corr": corr_scores[i], # 保留原始相关系数，方便看正负相关
                "Trend_Curve": self.matrix[i] # 用于绘图
            })
            
        return results, self.matrix[idx] # 返回推荐列表 + 种子词的曲线