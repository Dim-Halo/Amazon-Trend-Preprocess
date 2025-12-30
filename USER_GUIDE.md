# Amazon Trend Pipeline 使用指南

## 📚 目录

1. [快速开始](#快速开始)
2. [安装配置](#安装配置)
3. [基础使用](#基础使用)
4. [高级功能](#高级功能)
5. [故障排除](#故障排除)
6. [最佳实践](#最佳实践)

## 快速开始

### 最简单的使用方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
# 将清洗后的 CSV 文件放入 ./clean_data 目录

# 3. 运行流水线
python -m src.cli

# 4. 生成向量
python scripts/generate_vectors.py

# 5. 启动 Web 应用
streamlit run web_app/app_selection.py
```

## 安装配置

### 系统要求

- Python 3.8+
- 8GB+ RAM（推荐 16GB）
- 磁盘空间：至少 5GB 可用空间

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd Amazon-Trend-Preprocess-main

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### GPU 支持（可选）

如果要使用 GPU 加速：

```bash
# 安装 PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 FAISS GPU 版本
pip uninstall faiss-cpu
pip install faiss-gpu
```

## 基础使用

### 1. 数据准备

将清洗后的 CSV 文件放入 `./clean_data` 目录。文件必须包含以下列：

- `normalized_term`: 标准化的搜索词
- `报告日期`: 日期列
- `搜索频率排名`: 排名数据

### 2. 运行流水线

#### 使用 CLI（推荐）

```bash
# 运行完整流水线
python -m src.cli

# 只运行特定模块
python -m src.cli --module 1 2 3

# 强制重新运行（忽略缓存）
python -m src.cli --force

# 查看流水线状态
python -m src.cli --status

# 重置流水线状态
python -m src.cli --reset
```

#### 使用 Pipeline V3（自动选择 CPU/GPU）

```bash
# 自动检测并使用最佳设备
python src/pipeline_v3.py

# 强制使用 CPU
python src/pipeline_v3.py --device cpu

# 强制使用 GPU
python src/pipeline_v3.py --device cuda

# 只运行特定模块
python src/pipeline_v3.py --module 1 2 3
```

#### 使用旧版本

```bash
# CPU 版本
python src/pipeline.py

# GPU 版本
python src/pipeline_GPU.py
```

### 3. 生成向量

```bash
# 必须在流水线完成后运行
python scripts/generate_vectors.py
```

### 4. 启动 Web 应用

```bash
# 产品选择界面
streamlit run web_app/app_selection.py

# 数据查看界面
streamlit run web_app/app_viewer.py
```

## 高级功能

### 环境变量配置

```bash
# 设置相似度阈值
export SIM_THRESHOLD=0.8

# 设置运行设备
export DEVICE=cuda

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 设置批处理大小
export BATCH_SIZE=128
```

### 模块说明

#### 模块 1: 全局词汇收集
- 功能：从所有 CSV 文件中提取唯一搜索词
- 输出：词汇表缓存 (`cache/vocab_list.json`)
- 耗时：通常几秒到几分钟

#### 模块 2: 向量聚类与映射
- 功能：计算词向量，使用 FAISS 进行相似度搜索，生成映射关系
- 输出：
  - 映射字典 (`cache/mapping_dict.json`)
  - 变更日志 (`cache/change_log.json`)
  - 向量缓存 (`cache/embeddings.npy`)
  - FAISS 索引 (`cache/faiss_index.bin`)
- 耗时：首次运行可能需要几分钟到几十分钟，后续运行几秒

#### 模块 3: 输出验证文件
- 功能：生成映射对照表，为所有文件添加 `mapped_term` 列
- 输出：
  - 对照表 (`mapping_check/词义映射对照表_只看变化.xlsx`)
  - 处理后的文件 (`processed_data/mapped_*.csv`)
- 耗时：取决于文件数量和大小

#### 模块 4: 生成 TimesNet 矩阵
- 功能：生成时间序列矩阵用于趋势预测
- 输出：
  - 矩阵文件 (`final_npy/timesnet_input.npy`)
  - 词表 (`final_npy/terms.csv`)
  - 日期表 (`final_npy/dates.csv`)
- 耗时：取决于数据规模

### 缓存管理

#### 查看缓存

```bash
# 查看流水线状态
python -m src.cli --status

# 查看缓存文件
ls -lh cache/
```

#### 清理缓存

```bash
# 删除所有缓存
rm -rf cache/

# 删除特定模块的缓存
rm cache/vocab_list.json  # 模块 1
rm cache/embeddings.npy cache/faiss_index.bin  # 模块 2 FAISS 缓存
rm cache/mapping_dict.json cache/change_log.json  # 模块 2 映射缓存
```

#### 重置状态

```bash
# 通过 CLI 重置
python -m src.cli --reset

# 或手动删除状态文件
rm cache/pipeline_state.json
```

### 内存优化

如果处理大规模数据集遇到内存问题：

```python
# 在代码中调整内存限制
from src.pipeline_v2 import module_4_generate_matrix
from src.utils.state import PipelineState

state = PipelineState()
module_4_generate_matrix(state, memory_limit_mb=1024)  # 设置为 1GB
```

或使用 Pipeline V3：

```python
from src.core.base_pipeline import CPUPipeline

pipeline = CPUPipeline()
pipeline.module_4_generate_matrix(memory_limit_mb=1024)
```

### 数据验证

独立使用数据验证器：

```python
from src.utils.validator import DataValidator
from src.config import Config

# 验证单个文件
result = DataValidator.validate_csv_structure(
    'path/to/file.csv',
    ['normalized_term', '报告日期', '搜索频率排名']
)

if result['is_valid']:
    print("✅ 文件验证通过")
else:
    print(f"❌ 验证失败: {result['error']}")

# 批量验证
results = DataValidator.validate_batch(
    Config.INPUT_FOLDER,
    [Config.TERM_COL, Config.DATE_COL, Config.RANK_COL]
)

# 检查结果
for filename, result in results.items():
    if not result['is_valid']:
        print(f"❌ {filename}: {result['error']}")
```

## 故障排除

### 常见问题

#### 1. 模块 2 运行很慢

**原因**：向量计算和 FAISS 索引构建耗时

**解决方案**：
- 首次运行是正常的，后续运行会使用缓存
- 考虑使用 GPU 加速
- 减少词汇量（提高数据质量）

#### 2. 内存不足 (OOM)

**原因**：数据集太大，超过可用内存

**解决方案**：
```bash
# 使用内存优化版本
python -m src.cli

# 或调整内存限制
# 在 pipeline_v2.py 中修改 memory_limit_mb 参数
```

#### 3. FAISS 索引构建失败

**原因**：Windows 系统上 FAISS 的兼容性问题

**解决方案**：
- 使用 Top-K 搜索（已在 pipeline_v2.py 中实现）
- 确保使用 `faiss-cpu` 而不是 `faiss`

#### 4. 找不到 terms.csv

**原因**：模块 4 未成功运行

**解决方案**：
```bash
# 检查流水线状态
python -m src.cli --status

# 重新运行模块 4
python -m src.cli --module 4 --force
```

#### 5. 编码错误

**原因**：CSV 文件编码问题

**解决方案**：
- 确保所有 CSV 文件使用 UTF-8 编码
- 代码已使用 `encoding='utf-8-sig'` 处理 BOM

#### 6. GPU 不可用

**原因**：PyTorch 或 CUDA 未正确安装

**解决方案**：
```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装 PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 日志查看

```bash
# 查看最新日志
tail -f logs/pipeline_*.log

# 查看所有日志
ls -lt logs/

# 搜索错误
grep -i error logs/pipeline_*.log
```

### 调试模式

```python
# 在代码中启用详细日志
import logging
from src.utils.logger import setup_logger

logger = setup_logger('debug', 'logs/debug.log', level=logging.DEBUG)
```

## 最佳实践

### 1. 数据准备

- ✅ 确保所有 CSV 文件格式一致
- ✅ 使用 UTF-8 编码
- ✅ 清理无效数据
- ✅ 验证必需列存在

### 2. 流水线运行

- ✅ 首次运行使用默认设置
- ✅ 利用缓存机制，避免重复计算
- ✅ 定期清理日志文件
- ✅ 监控内存使用

### 3. 参数调整

- ✅ 相似度阈值：0.7-0.8 之间通常效果较好
- ✅ Top-K：20-30 足够覆盖相似词
- ✅ 批处理大小：根据内存调整（64-512）

### 4. 性能优化

- ✅ 使用 GPU 加速（如果可用）
- ✅ 启用 FAISS 索引缓存
- ✅ 合理设置内存限制
- ✅ 使用 SSD 存储缓存文件

### 5. 版本选择

| 版本 | 适用场景 | 优势 |
|------|---------|------|
| CLI (`src.cli`) | 日常使用 | 最方便，功能完整 |
| Pipeline V3 | 需要 GPU 支持 | 自动选择设备 |
| Pipeline V2 | 需要自定义 | 灵活性高 |
| 原始版本 | 稳定性优先 | 经过充分测试 |

### 6. 工作流程建议

```bash
# 1. 首次运行完整流水线
python -m src.cli

# 2. 检查映射结果
# 打开 mapping_check/词义映射对照表_只看变化.xlsx

# 3. 如需调整阈值，重新运行模块 2-4
export SIM_THRESHOLD=0.8
python -m src.cli --module 2 3 4 --force

# 4. 生成向量
python scripts/generate_vectors.py

# 5. 启动应用
streamlit run web_app/app_selection.py
```

## 附录

### 文件结构

```
.
├── clean_data/          # 输入：清洗后的数据
├── processed_data/      # 输出：添加映射列的数据
├── mapping_check/       # 输出：映射对照表
├── final_npy/          # 输出：最终矩阵和向量
├── cache/              # 缓存文件
├── logs/               # 日志文件
├── src/
│   ├── config.py       # 配置管理
│   ├── cli.py          # CLI 接口
│   ├── pipeline_v2.py  # 改进的流水线
│   ├── pipeline_v3.py  # 统一的 CPU/GPU 流水线
│   ├── core/
│   │   └── base_pipeline.py  # 流水线基类
│   └── utils/
│       ├── logger.py   # 日志系统
│       ├── state.py    # 状态管理
│       └── validator.py # 数据验证
├── scripts/
│   └── generate_vectors.py  # 向量生成
└── web_app/
    ├── app_selection.py  # 选品界面
    └── app_viewer.py     # 数据查看界面
```

### 配置参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SIMILARITY_THRESHOLD | 0.75 | 相似度阈值 |
| TOP_K | 20 | Top-K 搜索参数 |
| BATCH_SIZE | 64 | 批处理大小 |
| DEVICE | 'cpu' | 运行设备 |
| FILL_VALUE | 100000 | 缺失值填充 |
| RANK_THRESHOLD | 100000 | 排名过滤阈值 |

### 性能基准

基于 10 万词汇量的测试结果：

| 模块 | 首次运行 | 缓存运行 | 内存使用 |
|------|---------|---------|---------|
| 模块 1 | ~30s | ~2s | ~500MB |
| 模块 2 | ~5min | ~5s | ~2GB |
| 模块 3 | ~2min | ~10s | ~1GB |
| 模块 4 | ~3min | ~3min | ~2-4GB |

*注：实际性能取决于硬件配置和数据规模*

### 支持与反馈

- 查看文档：`CLAUDE.md`, `ARCHITECTURE_OPTIMIZATION.md`, `OPTIMIZATION_PHASE2.md`
- 查看日志：`logs/` 目录
- 问题报告：通过 Git 提交 issue

---

**版本**: 3.0
**最后更新**: 2024-12
