import json
import os
import requests
from huggingface_hub import hf_hub_download

# 常量
MODEL_REPO = "Camais03/camie-tagger-v2"
ONNX_MODEL_FILE = "camie-tagger-v2.onnx"
METADATA_FILE = "camie-tagger-v2-metadata.json"
VALIDATION_FILE = "full_validation_results.json"


def get_model_files():
    """从 HF Hub 下载模型文件并返回路径"""
    try:
        # 使用当前目录下的 model_cache 目录
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # 1. 获取元数据
        metadata_path = None
        try:
            print("正在检查本地元数据...")
            metadata_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=METADATA_FILE,
                cache_dir=cache_dir,
                local_files_only=True
            )
            print("找到本地元数据。")
        except Exception:
            print("本地未找到元数据，正在下载...")
            metadata_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=METADATA_FILE,
                cache_dir=cache_dir
            )

        # 2. 获取 ONNX 模型
        onnx_path = None
        try:
            print("正在检查本地 ONNX 模型...")
            onnx_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=ONNX_MODEL_FILE,
                cache_dir=cache_dir,
                local_files_only=True
            )
            print("找到本地 ONNX 模型。")
        except Exception:
            print("本地未找到 ONNX 模型，正在下载 (这可能需要一些时间)...")
            # 尝试流式下载大型 ONNX 文件
            try:
                onnx_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=ONNX_MODEL_FILE,
                    cache_dir=cache_dir,
                    force_download=False
                )
            except Exception as e:
                print(f"ONNX 下载失败: {e}")
                # 回退方案：使用 requests 直接下载
                onnx_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{ONNX_MODEL_FILE}"
                onnx_path = os.path.join(cache_dir, ONNX_MODEL_FILE)

                print(f"尝试直接下载: {onnx_url}")
                response = requests.get(onnx_url, stream=True)
                response.raise_for_status()

                with open(onnx_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"直接下载成功: {onnx_path}")

        # 3. 获取验证结果文件 (可选)
        validation_path = None
        try:
            validation_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=VALIDATION_FILE,
                cache_dir=cache_dir,
                local_files_only=True
            )
        except Exception:
            try:
                validation_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=VALIDATION_FILE,
                    cache_dir=cache_dir
                )
            except Exception as e:
                print(f"验证结果不可用: {e}")
                validation_path = None

        return {
            'onnx_path': onnx_path,
            'metadata_path': metadata_path,
            'validation_path': validation_path
        }
    except Exception as e:
        print(f"下载模型文件失败: {e}")
        return None


def load_model_and_metadata():
    """从 HF Hub 加载模型和元数据"""

    # 下载模型文件
    model_files = get_model_files()
    if not model_files:
        return None, None, {}

    model_info = {
        'onnx_available': model_files['onnx_path'] is not None,
        'validation_results_available': model_files['validation_path'] is not None,
        'onnx_path': model_files['onnx_path'],
        'metadata_path': model_files['metadata_path'],
        'validation_path': model_files['validation_path']
    }

    # 加载元数据
    metadata = None
    if model_files['metadata_path']:
        try:
            with open(model_files['metadata_path'], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"加载元数据出错: {e}")

    return model_info, metadata

