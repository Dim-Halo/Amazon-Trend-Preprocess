import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils.logger import setup_logger
from sentence_transformers import SentenceTransformer

# 设置环境
Config.setup_environment()
Config.ensure_folders()
logger = setup_logger('generate_vectors', Config.LOG_FOLDER / 'generate_vectors.log')

def generate_vectors():
    """生成词向量文件"""
    logger.info("=" * 50)
    logger.info("🚀 开始生成词向量...")

    term_file = Config.NPY_FOLDER / 'terms.csv'
    output_file = Config.NPY_FOLDER / 'term_vectors.npy'

    # 检查输入文件
    if not term_file.exists():
        logger.error(f"找不到 terms.csv: {term_file}")
        logger.error("请先运行 pipeline 生成最终矩阵")
        return False

    # 读取词表
    logger.info(f"加载词表: {term_file}")
    df = pd.read_csv(term_file, encoding='utf-8-sig')
    terms = df.iloc[:, 0].astype(str).tolist()
    logger.info(f"共 {len(terms)} 个词")

    # 加载模型
    logger.info(f"加载模型 {Config.MODEL_NAME}...")
    model = SentenceTransformer(Config.MODEL_NAME, device=Config.DEVICE)

    # 向量化
    logger.info("开始向量化计算（可能需要几分钟）...")
    embeddings = model.encode(
        terms,
        batch_size=Config.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 归一化
    logger.info("归一化向量...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # 保存
    logger.info(f"保存向量至: {output_file}")
    np.save(output_file, embeddings)

    logger.info("✅ 向量生成完成！")
    logger.info(f"向量形状: {embeddings.shape}")
    logger.info("现在可以运行 Streamlit 应用了")

    return True

if __name__ == "__main__":
    try:
        success = generate_vectors()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"生成向量失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
