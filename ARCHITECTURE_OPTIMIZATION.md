# 架构优化说明

## 🎉 优化完成

已完成以下架构优化：

### 1. ✅ 统一配置管理 (`src/config.py`)
- 所有配置集中管理
- 支持环境变量覆盖
- 自动创建必需文件夹

### 2. ✅ 日志系统 (`src/utils/logger.py`)
- 替代 print 语句
- 支持文件和控制台输出
- 日志文件保存在 `logs/` 目录

### 3. ✅ 流水线状态管理 (`src/utils/state.py`)
- 自动追踪模块完成状态
- 支持缓存和断点续跑
- 避免重复执行已完成的模块

### 4. ✅ 代码去重
- 删除重复的 `generate_vectors_for_ui.py`
- 统一使用 `scripts/generate_vectors.py`

### 5. ✅ 改进的流水线 (`src/pipeline_v2.py`)
- 集成配置、日志、状态管理
- 支持模块级缓存
- 自动跳过已完成的模块
- 更好的错误处理

### 6. ✅ CLI 接口 (`src/cli.py`)
- 命令行运行特定模块
- 查看流水线状态
- 重置状态
- 强制重新运行

## 📖 使用指南

### 基础用法

```bash
# 运行完整流水线（自动跳过已完成的模块）
python -m src.cli

# 只运行特定模块
python -m src.cli --module 1        # 只运行模块 1
python -m src.cli --module 1 2 3    # 运行模块 1, 2, 3

# 强制重新运行（忽略缓存）
python -m src.cli --force

# 查看流水线状态
python -m src.cli --status

# 重置流水线状态
python -m src.cli --reset

# 指定运行设备
python -m src.cli --device cuda
```

### 模块说明

1. **模块 1** - 全局词汇收集
2. **模块 2** - 向量聚类与映射
3. **模块 3** - 输出验证文件
4. **模块 4** - 生成 TimesNet 矩阵

### 工作流程示例

```bash
# 第一次运行：执行所有模块
python -m src.cli

# 如果模块 3 失败，修复后只运行模块 3 和 4
python -m src.cli --module 3 4

# 调整相似度阈值后，重新运行模块 2-4
python -m src.cli --module 2 3 4 --force
```

## 🔧 配置说明

### 环境变量

可以通过环境变量覆盖配置：

```bash
# 设置相似度阈值
export SIM_THRESHOLD=0.8

# 设置运行设备
export DEVICE=cuda

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 运行流水线
python -m src.cli
```

### 配置文件

编辑 `src/config.py` 修改默认配置：

```python
class Config:
    SIMILARITY_THRESHOLD = 0.75  # 相似度阈值
    DEVICE = 'cpu'               # 运行设备
    BATCH_SIZE = 64              # 批处理大小
    # ... 更多配置
```

## 📁 新增文件结构

```
src/
├── config.py           # 统一配置管理
├── pipeline_v2.py      # 改进的流水线（推荐使用）
├── pipeline.py         # 原始流水线（保留）
├── cli.py              # 命令行接口
└── utils/
    ├── logger.py       # 日志系统
    └── state.py        # 状态管理

cache/                  # 缓存文件（自动创建）
├── pipeline_state.json # 流水线状态
├── vocab_list.json     # 词汇表缓存
├── mapping_dict.json   # 映射字典缓存
└── change_log.json     # 变更日志缓存

logs/                   # 日志文件（自动创建）
└── pipeline_*.log      # 运行日志
```

## 🚀 优势

### 1. 断点续跑
- 模块执行状态自动保存
- 失败后无需重新运行已完成的模块
- 节省大量时间（特别是模块 2 的向量计算）

### 2. 更好的可观测性
- 结构化日志输出
- 日志文件持久化
- 可追踪历史运行记录

### 3. 灵活的执行控制
- 通过 CLI 精确控制运行哪些模块
- 支持强制重新运行
- 查看流水线状态

### 4. 易于维护
- 配置集中管理
- 代码结构清晰
- 减少重复代码

## 🔄 迁移指南

### 从旧版本迁移

旧的使用方式：
```python
# 需要手动编辑 pipeline.py 取消注释
if __name__ == "__main__":
    # vocab = module_1_collect_vocab()
    # mapping, changes = module_2_build_mapping(vocab)
    # ...
```

新的使用方式：
```bash
# 直接通过 CLI 运行
python -m src.cli
```

### 保持兼容性

- 原始的 `pipeline.py` 仍然保留
- 新的 `pipeline_v2.py` 提供增强功能
- 可以根据需要选择使用

## 📝 注意事项

1. **首次运行**：会创建 `cache/` 和 `logs/` 目录
2. **缓存位置**：缓存文件保存在 `cache/` 目录
3. **日志位置**：日志文件保存在 `logs/` 目录
4. **状态重置**：如果需要完全重新运行，使用 `--reset` 或 `--force`

## 🐛 故障排除

### 问题：模块状态不正确

```bash
# 重置状态
python -m src.cli --reset

# 强制重新运行
python -m src.cli --force
```

### 问题：缓存文件损坏

```bash
# 删除缓存目录
rm -rf cache/

# 重新运行
python -m src.cli
```

### 问题：查看详细日志

```bash
# 日志文件位置
ls logs/

# 查看最新日志
tail -f logs/pipeline_*.log
```

## 🎯 下一步优化建议

以下是中低优先级的优化项，可以根据需要实施：

1. **GPU 流水线整合**：将 `pipeline_GPU.py` 整合到 `pipeline_v2.py`
2. **FAISS 索引缓存**：缓存 FAISS 索引到磁盘
3. **内存优化**：模块 4 的分块处理
4. **数据验证**：CSV 结构验证
5. **并行处理**：多进程处理文件

## 📚 相关文档

- 配置说明：`src/config.py`
- CLI 帮助：`python -m src.cli --help`
- 原始 README：`README.md`
