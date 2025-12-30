# Excel 文件大小限制问题解决方案

## 问题描述

```
ValueError: This sheet is too large!
Your sheet size is: 1115680, 3
Max sheet size is: 1048576, 16384
```

**原因**: Excel 最多支持 1,048,576 行，你的数据超过了这个限制。

## ✅ 已自动修复

最新版本的流水线已经自动处理这个问题：

```bash
# 直接运行，会自动选择最佳格式
python -m src.cli
```

### 自动处理逻辑

1. **数据 < 100万行**: 导出为单个 Excel 文件
2. **数据 100万-1000万行**: 导出为 CSV 文件（无限制）
3. **数据 > 1000万行**: 导出为 Parquet 文件（压缩率高）
4. **数据超过 Excel 限制**: 自动分割成多个 Excel 文件

## 🔧 手动解决方案

### 方案 1: 使用 CSV 格式（推荐）

CSV 没有行数限制，且兼容性好：

```python
from src.utils.large_data_exporter import LargeDataExporter
import pandas as pd

# 读取数据
df = pd.DataFrame(change_log)

# 导出为 CSV
LargeDataExporter.export_to_csv(df, './output/mapping_table.csv')
```

**优点**:
- 无行数限制
- 文件小
- Excel 可以打开
- 兼容性好

**缺点**:
- 不支持多个工作表
- 格式化选项少

### 方案 2: 分割成多个 Excel 文件

```python
from src.utils.large_data_exporter import LargeDataExporter
import pandas as pd

# 读取数据
df = pd.DataFrame(change_log)

# 自动分割
files = LargeDataExporter.export_to_multiple_excel(
    df,
    './output/mapping_table'
)

# 结果：
# mapping_table_part1.xlsx (1,048,575 行)
# mapping_table_part2.xlsx (67,105 行)
```

**优点**:
- 保持 Excel 格式
- 自动分割
- 每个文件都可以在 Excel 中打开

**缺点**:
- 多个文件，不方便查看全部数据

### 方案 3: 使用 Parquet 格式（最佳性能）

```python
from src.utils.large_data_exporter import LargeDataExporter
import pandas as pd

# 读取数据
df = pd.DataFrame(change_log)

# 导出为 Parquet
LargeDataExporter.export_to_parquet(df, './output/mapping_table.parquet')

# 读取 Parquet
df_read = pd.read_parquet('./output/mapping_table.parquet')
```

**优点**:
- 无行数限制
- 文件最小（压缩率高）
- 读取速度最快
- 保留数据类型

**缺点**:
- Excel 无法直接打开
- 需要 Python 或专门工具读取

### 方案 4: 智能自动选择

```python
from src.utils.large_data_exporter import LargeDataExporter
import pandas as pd

# 读取数据
df = pd.DataFrame(change_log)

# 自动选择最佳格式
files = LargeDataExporter.smart_export(
    df,
    './output/mapping_table',
    prefer_format='auto'  # 自动选择
)
```

## 📊 格式对比

| 格式 | 行数限制 | 文件大小 | 读取速度 | Excel 兼容 |
|------|---------|---------|---------|-----------|
| Excel | 1,048,576 | 大 | 慢 | ✅ 完美 |
| CSV | 无限制 | 中 | 中 | ✅ 可打开 |
| Parquet | 无限制 | 小 | 快 | ❌ 需转换 |
| 多个 Excel | 无限制 | 大 | 慢 | ✅ 分文件 |

## 🎯 推荐方案

### 场景 1: 需要在 Excel 中查看

```python
# 使用 CSV（Excel 可以打开）
LargeDataExporter.export_to_csv(df, './output/data.csv')
```

### 场景 2: 只需要程序处理

```python
# 使用 Parquet（最快最小）
LargeDataExporter.export_to_parquet(df, './output/data.parquet')
```

### 场景 3: 不确定

```python
# 使用智能导出（自动选择）
LargeDataExporter.smart_export(df, './output/data', prefer_format='auto')
```

## 🔄 在流水线中的应用

### 已自动集成

最新版本的 `pipeline_v2.py` 已经自动使用智能导出：

```python
# 模块 3 会自动处理大文件
python -m src.cli --module 3
```

### 手动指定格式

如果需要手动指定格式，可以修改代码：

```python
# 在 pipeline_v2.py 的 module_3_export_verification 中
output_files = LargeDataExporter.smart_export(
    df_log,
    log_path,
    prefer_format='csv'  # 强制使用 CSV
)
```

## 📝 查看分割后的文件

如果数据被分割成多个文件，会自动生成 `README.txt`：

```
mapping_check/
├── 词义映射对照表_只看变化_part1.xlsx
├── 词义映射对照表_只看变化_part2.xlsx
└── README.txt  # 说明文件
```

`README.txt` 内容：
```
映射对照表因数据量过大，已分割成多个文件：

1. 词义映射对照表_只看变化_part1.xlsx
2. 词义映射对照表_只看变化_part2.xlsx

总行数: 1,115,680
每个文件最多包含 1,048,575 行数据
```

## 🛠️ 故障排除

### 问题 1: 仍然报错

**解决**: 确保使用最新版本的代码

```bash
# 重新运行流水线
python -m src.cli --module 3 --force
```

### 问题 2: 想要单个文件

**解决**: 使用 CSV 或 Parquet

```python
# 修改 prefer_format
prefer_format='csv'  # 或 'parquet'
```

### 问题 3: 文件太大无法打开

**解决**: 使用 Python 读取

```python
import pandas as pd

# 读取 CSV
df = pd.read_csv('mapping_table.csv')

# 或读取 Parquet
df = pd.read_parquet('mapping_table.parquet')

# 查看前 100 行
print(df.head(100))

# 筛选特定数据
filtered = df[df['原始词 (Original)'].str.contains('关键词')]
```

## 📚 更多信息

- Excel 限制文档: [Microsoft 官方文档](https://support.microsoft.com/en-us/office/excel-specifications-and-limits-1672b34d-7043-467e-8e27-269d656771c3)
- Parquet 格式: [Apache Parquet](https://parquet.apache.org/)
- 大数据处理: 参考 `DUCKDB_GUIDE.md`

---

**总结**: 最新版本已自动处理此问题，无需手动干预。如果需要特定格式，可以使用 `LargeDataExporter` 工具类。
