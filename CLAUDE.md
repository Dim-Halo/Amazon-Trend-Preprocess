# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Amazon Trend Predictor & Recommendation System - An intelligent product selection system combining TimesNet trend prediction with semantic analysis using sentence transformers. The system processes Amazon search term data to provide dual-modal recommendations based on semantic similarity and trend correlation.

## Core Architecture

### Data Processing Pipeline (src/pipeline.py)

The pipeline consists of 4 sequential modules:

1. **Module 1 - Global Vocabulary Collection** (`module_1_collect_vocab`): Extracts all unique search terms from CSV files in `./clean_data`

2. **Module 2 - Vector Clustering & Mapping** (`module_2_build_mapping`):
   - Uses `all-MiniLM-L6-v2` sentence transformer model
   - Performs Top-K similarity search (K=20) with FAISS IndexFlatIP
   - Maps similar terms to canonical form (shortest string in cluster)
   - Threshold: 0.75 cosine similarity
   - **Note**: Uses Top-K instead of range_search to avoid FAISS C++ abort issues on Windows/CPU

3. **Module 3 - Verification Export** (`module_3_export_verification`):
   - Generates Excel mapping verification file in `./mapping_check`
   - Adds `mapped_term` column to all CSV files
   - Outputs to `./processed_data`

4. **Module 4 - TimesNet Matrix Generation** (`module_4_generate_matrix`):
   - Pivots data: rows=mapped_term, columns=date, values=rank
   - Applies log10 transformation: `log10(rank + 1)`
   - Fills missing values with 100000
   - Filters terms with historical best rank < 100000
   - Outputs: `timesnet_input.npy`, `terms.csv`, `dates.csv` to `./final_npy`

### Recommendation Engine (src/recommender.py)

`HybridRecommender` class combines two scoring mechanisms:

- **Semantic Similarity**: Cosine similarity via normalized dot product of term vectors
- **Trend Correlation**: Pearson correlation of time series curves
- **Hybrid Score**: `weight_semantic * sim_score + weight_trend * (corr + 1) / 2`

### GPU Acceleration (src/pipeline_GPU.py)

Alternative pipeline with CUDA support:
- Auto-detects GPU availability
- Uses FAISS GPU index when available
- Parallel file processing with multiprocessing
- Lower similarity threshold: 0.725

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Data Processing (推荐使用新版 CLI)
```bash
# 运行完整流水线（自动跳过已完成的模块）
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

### 旧版本命令（仍然可用）
```bash
# 原始流水线（需要手动编辑代码取消注释）
python src/pipeline.py

# GPU 加速版本
python src/pipeline_GPU.py
```

### Generate Vectors for UI
```bash
# Must run after pipeline generates terms.csv
python scripts/generate_vectors.py
```

### Launch Web Application
```bash
# Product selection interface
streamlit run web_app/app_selection.py

# Data viewer interface
streamlit run web_app/app_viewer.py
```

## Data Flow

```
./clean_data/*.csv
  → [Module 1] → global vocabulary
  → [Module 2] → mapping_dict
  → [Module 3] → ./processed_data/mapped_*.csv + ./mapping_check/*.xlsx
  → [Module 4] → ./final_npy/{timesnet_input.npy, terms.csv, dates.csv}
  → [generate_vectors] → ./final_npy/term_vectors.npy
  → [web_app] → Streamlit UI
```

## Key Configuration

All pipelines use CONFIG dict with these critical parameters:
- `term_col`: 'normalized_term' - the search term column
- `date_col`: '报告日期' - date column in Chinese
- `rank_col`: '搜索频率排名' - search frequency rank
- `similarity_threshold`: 0.75 (CPU) or 0.725 (GPU)
- `device`: 'cpu' or 'cuda:0'

## Architecture Improvements (2024)

### Phase 1: Core Infrastructure
- **Unified Configuration** (`src/config.py`): Centralized config management with environment variable support
- **Logging System** (`src/utils/logger.py`): Structured logging replacing print statements
- **State Management** (`src/utils/state.py`): Pipeline state tracking with caching and resume capability
- **CLI Interface** (`src/cli.py`): Command-line interface for flexible pipeline execution
- **Improved Pipeline** (`src/pipeline_v2.py`): Enhanced version with state management and caching

### Phase 2: Performance & Reliability
- **FAISS Index Caching**: Persistent FAISS index and embeddings cache (90%+ speedup on reruns)
- **Data Validation** (`src/utils/validator.py`): Automatic CSV structure and quality validation
- **Memory Optimization**: Module 4 with real-time memory monitoring and chunked processing
- **Refactored Vector Generation**: Updated `scripts/generate_vectors.py` with unified config and logging

### Phase 3: Code Refactoring & Documentation
- **Pipeline Base Class** (`src/core/base_pipeline.py`): Abstract base class eliminating CPU/GPU code duplication
- **Unified Pipeline** (`src/pipeline_v3.py`): Auto-detecting CPU/GPU pipeline with simplified interface
- **DuckDB Integration** (`src/duckdb_pipeline.py`): High-performance data processing with DuckDB (3-6x faster)
- **Complete User Guide** (`USER_GUIDE.md`): Comprehensive documentation with examples and troubleshooting
- **DuckDB Guide** (`DUCKDB_GUIDE.md`): Performance optimization guide using DuckDB
- **Updated Dependencies**: Added psutil and duckdb

### Key Benefits
- **Resume Capability**: Automatically skips completed modules, saves time on reruns
- **Better Observability**: Structured logs saved to `logs/` directory
- **Flexible Execution**: Run specific modules via CLI without code editing
- **Caching**: Module outputs and FAISS indices cached in `cache/` directory for faster reruns
- **Reliability**: Data validation, memory management, and comprehensive error handling

See `ARCHITECTURE_OPTIMIZATION.md` for Phase 1 details, `OPTIMIZATION_PHASE2.md` for Phase 2 details, `USER_GUIDE.md` for complete usage guide, and `FINAL_SUMMARY.md` for comprehensive summary.

## Important Notes

- **HuggingFace Mirror**: Code sets `HF_ENDPOINT='https://hf-mirror.com'` for China access
- **Encoding**: All CSV operations use `encoding='utf-8-sig'` for BOM compatibility
- **Memory Management**: Pipeline uses explicit `gc.collect()` for large datasets
- **FAISS Stability**: CPU version uses Top-K search instead of range_search to prevent crashes on Windows
- **Vector Normalization**: All embeddings are L2-normalized so dot product equals cosine similarity
