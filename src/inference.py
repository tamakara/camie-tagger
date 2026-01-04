"""
基于 ONNX 的图像处理模块。
包含图像预处理和推理逻辑。
"""

import os
import time
import traceback
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms


def preprocess_image(image_path, image_size=512):
    """
    预处理图像以进行推理，包含 ImageNet 标准化。
    """
    if not os.path.exists(image_path):
        raise ValueError(f"未找到图像文件: {image_path}")

    # ImageNet 标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        with Image.open(image_path) as img:
            # 转换 RGBA 或 Palette 图像为 RGB
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # 获取原始尺寸
            width, height = img.size
            aspect_ratio = width / height

            # 计算新尺寸以保持纵横比
            if aspect_ratio > 1:
                new_width = image_size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = image_size
                new_width = int(new_height * aspect_ratio)

            # 使用 LANCZOS 滤镜调整大小
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建带有填充的新图像 (使用 ImageNet 均值填充)
            pad_color = (124, 116, 104)
            new_image = Image.new('RGB', (image_size, image_size), pad_color)
            paste_x = (image_size - new_width) // 2
            paste_y = (image_size - new_height) // 2
            new_image.paste(img, (paste_x, paste_y))

            # 应用变换 (包括 ImageNet 标准化)
            img_tensor = transform(new_image)
            return img_tensor.numpy()

    except Exception as e:
        raise Exception(f"处理图像 {image_path} 时出错: {str(e)}")


class ONNXImageTagger:
    """ONNX 图像标注器类"""

    def __init__(self, model_path, metadata):
        # 加载模型
        self.model_path = model_path

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        print(f"使用推理提供程序: {self.session.get_providers()}")

        # 存储元数据
        self.metadata = metadata

        # 从元数据中提取标签映射
        if 'dataset_info' in metadata:
            # 新元数据格式
            self.tag_mapping = metadata['dataset_info']['tag_mapping']
            self.idx_to_tag = self.tag_mapping['idx_to_tag']
            self.tag_to_category = self.tag_mapping['tag_to_category']
            self.total_tags = metadata['dataset_info']['total_tags']
        else:
            # 旧格式回退
            self.idx_to_tag = metadata.get('idx_to_tag', {})
            self.tag_to_category = metadata.get('tag_to_category', {})
            self.total_tags = metadata.get('total_tags', len(self.idx_to_tag))

        # 获取输入名称
        self.input_name = self.session.get_inputs()[0].name
        print(f"模型加载成功。输入名称: {self.input_name}")
        print(f"总标签数: {self.total_tags}, 分类数: {len(set(self.tag_to_category.values()))}")

    def predict_batch(self, image_arrays, threshold=0.5, category_thresholds=None, min_confidence=0.1):
        """对预处理后的图像数组进行批量推理"""
        # 堆叠数组
        batch_input = np.stack(image_arrays)

        # 运行推理
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: batch_input})

        # 处理输出
        if len(outputs) >= 2:
            # 多输出模型
            refined_logits = outputs[1]
            main_logits = refined_logits
        else:
            # 单输出模型
            main_logits = outputs[0]

        # 应用 sigmoid 获取概率
        main_probs = 1.0 / (1.0 + np.exp(-main_logits))

        # 处理结果
        batch_results = []

        for i in range(main_probs.shape[0]):
            probs = main_probs[i]

            # 提取并组织所有概率
            all_probs = {}
            for idx in range(probs.shape[0]):
                prob_value = float(probs[idx])
                if prob_value >= min_confidence:
                    idx_str = str(idx)
                    tag_name = self.idx_to_tag.get(idx_str, f"unknown-{idx}")
                    category = self.tag_to_category.get(tag_name, "general")

                    if category not in all_probs:
                        all_probs[category] = []

                    all_probs[category].append((tag_name, prob_value))

            # 在每个类别内按概率排序
            for category in all_probs:
                all_probs[category] = sorted(
                    all_probs[category],
                    key=lambda x: x[1],
                    reverse=True
                )

            # 根据阈值筛选标签
            tags = {}
            for category, cat_tags in all_probs.items():
                # 使用特定类别阈值或默认阈值
                if category_thresholds and category in category_thresholds:
                    cat_threshold = category_thresholds[category]
                else:
                    cat_threshold = threshold

                tags[category] = [(tag, prob) for tag, prob in cat_tags if prob >= cat_threshold]

            # 创建所有高于阈值标签的列表
            all_tags = []
            for category, cat_tags in tags.items():
                for tag, _ in cat_tags:
                    all_tags.append(tag)

            batch_results.append({
                'tags': tags,
                'all_probs': all_probs,
                'all_tags': all_tags,
                'success': True
            })

        return batch_results


def process_single_image(image_path, model_path, metadata, threshold=0.61, category_thresholds=None,
                         min_confidence=0.01):
    """
    处理单张图像
    """
    if category_thresholds is None:
        category_thresholds = {}
    try:
        # 创建或重用 tagger
        if hasattr(process_single_image, 'tagger'):
            tagger = process_single_image.tagger
        else:
            tagger = ONNXImageTagger(model_path, metadata)
            process_single_image.tagger = tagger

        # 预处理图像
        start_time = time.time()
        img_array = preprocess_image(image_path)

        # 运行推理
        results = tagger.predict_batch(
            [img_array],
            threshold=threshold,
            category_thresholds=category_thresholds,
            min_confidence=min_confidence
        )
        inference_time = time.time() - start_time

        if results:
            result = results[0]
            result['inference_time'] = inference_time
            result['success'] = True
            return result
        else:
            return {
                'success': False,
                'error': '处理图像失败',
                'all_tags': [],
                'tags': {}
            }

    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'all_tags': [],
            'tags': {}
        }
