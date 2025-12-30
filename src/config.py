import os
from pathlib import Path

class Config:
    """统一配置管理"""

    # 基础路径
    BASE_DIR = Path(__file__).parent.parent
    INPUT_FOLDER = BASE_DIR / 'clean_data'
    PROCESSED_FOLDER = BASE_DIR / 'processed_data'
    CHECK_FOLDER = BASE_DIR / 'mapping_check'
    NPY_FOLDER = BASE_DIR / 'final_npy'
    CACHE_FOLDER = BASE_DIR / 'cache'
    LOG_FOLDER = BASE_DIR / 'logs'

    # 数据列名
    TERM_COL = 'normalized_term'
    DATE_COL = '报告日期'
    RANK_COL = '搜索频率排名'

    # 模型配置
    MODEL_NAME = 'all-MiniLM-L6-v2'
    SIMILARITY_THRESHOLD = float(os.getenv('SIM_THRESHOLD', '0.75'))
    TOP_K = 20

    # 设备配置
    DEVICE = os.getenv('DEVICE', 'cpu')

    # HuggingFace 配置
    HF_ENDPOINT = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')

    # 批处理配置
    BATCH_SIZE = 64
    VECTORIZATION_BATCH = 512

    # 数据处理配置
    FILL_VALUE = 100000
    RANK_THRESHOLD = 100000

    @classmethod
    def ensure_folders(cls):
        """确保所有必需的文件夹存在"""
        for folder in [cls.PROCESSED_FOLDER, cls.CHECK_FOLDER,
                      cls.NPY_FOLDER, cls.CACHE_FOLDER, cls.LOG_FOLDER]:
            folder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def setup_environment(cls):
        """设置环境变量"""
        os.environ['HF_ENDPOINT'] = cls.HF_ENDPOINT
