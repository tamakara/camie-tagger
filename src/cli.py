import argparse
import os
import sys
from src.loader import load_model_and_metadata
from src.inference import process_single_image


def validate_probability(value):
    try:
        f_value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效的浮点数值: {value}")

    if not (0.0 <= f_value <= 1.0):
        raise argparse.ArgumentTypeError(f"数值 {value} 必须在 0.0 到 1.0 之间")
    return f_value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Camie 图像标注工具 CLI")
    parser.add_argument("image_path", help="图像文件路径")
    parser.add_argument("--threshold", type=validate_probability, default=0.61, help="全局置信度阈值 (默认: 0.61)")
    parser.add_argument("--min-confidence", type=validate_probability, default=0.01, help="最小置信度 (默认: 0.01)")

    # 类别特定阈值
    parser.add_argument("--threshold-artist", type=validate_probability, help="Artist 类别阈值")
    parser.add_argument("--threshold-character", type=validate_probability, help="Character 类别阈值")
    parser.add_argument("--threshold-copyright", type=validate_probability, help="Copyright 类别阈值")
    parser.add_argument("--threshold-general", type=validate_probability, help="General 类别阈值")
    parser.add_argument("--threshold-meta", type=validate_probability, help="Meta 类别阈值")
    parser.add_argument("--threshold-rating", type=validate_probability, help="Rating 类别阈值")
    parser.add_argument("--threshold-year", type=validate_probability, help="Year 类别阈值")

    return parser.parse_args()


def get_category_thresholds(args):
    active_category_thresholds = {}
    if args.threshold_artist is not None:
        active_category_thresholds['artist'] = args.threshold_artist
    if args.threshold_character is not None:
        active_category_thresholds['character'] = args.threshold_character
    if args.threshold_copyright is not None:
        active_category_thresholds['copyright'] = args.threshold_copyright
    if args.threshold_general is not None:
        active_category_thresholds['general'] = args.threshold_general
    if args.threshold_meta is not None:
        active_category_thresholds['meta'] = args.threshold_meta
    if args.threshold_rating is not None:
        active_category_thresholds['rating'] = args.threshold_rating
    if args.threshold_year is not None:
        active_category_thresholds['year'] = args.threshold_year
    return active_category_thresholds


def display_results(result):
    print("\n分析完成!")
    if 'inference_time' in result:
        print(f"推理耗时: {result['inference_time']:.4f} 秒")

    # 按类别显示标签
    target_categories = ["Artist", "Character", "Copyright", "General", "Meta", "Rating", "Year"]

    tags_by_category = result.get('tags', {})

    # 映射显示名称
    category_display_map = {
        "artist": "画师 (Artist)",
        "character": "角色 (Character)",
        "copyright": "作品 (Copyright)",
        "general": "常规 (General)",
        "meta": "元数据 (Meta)",
        "rating": "分级 (Rating)",
        "year": "年份 (Year)"
    }

    for display_cat in target_categories:
        found_key = None
        for key in tags_by_category.keys():
            if key.lower() == display_cat.lower():
                found_key = key
                break

        if found_key:
            cat_tags = tags_by_category[found_key]
            if cat_tags:
                display_name = category_display_map.get(found_key.lower(), display_cat)
                print(f"\n{display_name} ({len(cat_tags)}):")
                cat_tags.sort(key=lambda x: x[1], reverse=True)
                for tag, prob in cat_tags:
                    print(f"  {tag}: {prob:.2%}")


def main():
    args = parse_arguments()

    if not os.path.exists(args.image_path):
        print(f"错误: 未找到图像文件 {args.image_path}")
        sys.exit(1)

    print("正在加载模型...")
    model_path, metadata = load_model_and_metadata()

    if not model_path:
        print("错误: 模型不可用。")
        sys.exit(1)

    print(f"正在处理 {args.image_path}...")

    category_thresholds = get_category_thresholds(args)

    result = process_single_image(
        image_path=args.image_path,
        model_path=model_path,
        metadata=metadata,
        threshold=args.threshold,
        category_thresholds=category_thresholds,
        min_confidence=args.min_confidence
    )

    if not result['success']:
        print(f"处理图像时出错: {result.get('error')}")
        sys.exit(1)

    display_results(result)


if __name__ == "__main__":
    main()
