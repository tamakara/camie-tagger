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

- `--threshold`: 设置置信度阈值 (默认: 0.35)。

示例:
```bash
python run.py image.png --threshold 0.5
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

