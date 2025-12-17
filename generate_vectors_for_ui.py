import pandas as pd
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer


# ================= 配置 =================
CONFIG = {
    'term_file': './final_npy/terms.csv',      # 你的 Terms 索引文件
    'output_vector': './final_npy/term_vectors.npy', # 输出的向量文件
    'model_name': 'all-MiniLM-L6-v2',          # 轻量级语义模型
    'device': 'cpu'                            # AMD 780M 用 CPU 即可
}

def generate_vectors():
    print(f"🚀 加载词表: {CONFIG['term_file']}")
    if not os.path.exists(CONFIG['term_file']):
        print("❌ 错误：找不到 terms.csv，请先运行 pipeline 生成最终矩阵。")
        return

    # 读取 terms
    df = pd.read_csv(CONFIG['term_file'], encoding='utf-8-sig')
    terms = df.iloc[:, 0].astype(str).tolist()
    print(f"   共 {len(terms)} 个词。")

    # 加载模型
    print(f"🧠 加载模型 {CONFIG['model_name']}...")
    model = SentenceTransformer(CONFIG['model_name'], device=CONFIG['device'])

    # 向量化
    print("⚡ 开始向量化计算 (可能需要几分钟)...")
    embeddings = model.encode(terms, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # 归一化 (这一步很关键，归一化后 点积(Dot Product) 等于 余弦相似度)
    # 这样前端计算速度会飞快
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # 保存
    print(f"💾 保存向量至: {CONFIG['output_vector']}")
    np.save(CONFIG['output_vector'], embeddings)
    print("✅ 完成！现在可以去运行 app.py 了。")

if __name__ == "__main__":
    generate_vectors()