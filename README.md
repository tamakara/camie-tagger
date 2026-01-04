# Camie Tagger CLI

这是一个基于 Camie Tagger ONNX 模型的图像标注命令行工具。

## 安装

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

使用图像路径运行脚本:

```bash
python run.py path/to/image.jpg
```

或者:

```bash
python -m src.cli path/to/image.jpg
```

### 选项

- `--threshold`: 设置全局置信度阈值 (默认: 0.35)。
- `--min-confidence`: 设置最小置信度 (默认: 0.1)。低于此值的标签将被忽略。

#### 类别特定阈值

您可以为特定类别设置不同的阈值。如果未设置，将使用全局 `--threshold`。

- `--threshold-artist`: Artist (画师) 类别阈值
- `--threshold-character`: Character (角色) 类别阈值
- `--threshold-copyright`: Copyright (作品) 类别阈值
- `--threshold-general`: General (常规) 类别阈值
- `--threshold-meta`: Meta (元数据) 类别阈值
- `--threshold-rating`: Rating (分级) 类别阈值
- `--threshold-year`: Year (年份) 类别阈值

示例:
```bash
# 设置全局阈值为 0.5
python run.py image.png --threshold 0.5

# 设置全局阈值为 0.3，但角色标签需要 0.6 的置信度
python run.py image.png --threshold 0.3 --threshold-character 0.6
```

## 输出

工具将按以下类别分组输出标签:
- 画师 (Artist)
- 角色 (Character)
- 作品 (Copyright)
- 常规 (General)
- 元数据 (Meta)
- 分级 (Rating)
- 年份 (Year)
