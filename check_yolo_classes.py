#!/usr/bin/env python3
"""
检查 YOLO 模型的类别信息
"""
from ultralytics import YOLO
import sys

def check_model_classes(model_path):
    """检查模型类别"""
    print(f"\n检查模型: {model_path}")
    print("=" * 50)

    try:
        model = YOLO(model_path)
        names = getattr(model, "names", None)

        print(f"模型类型: {type(model)}")
        print(f"类别数量: {len(names) if names else 0}")
        print(f"\n类别列表:")

        if isinstance(names, dict):
            for k, v in sorted(names.items(), key=lambda x: int(x[0])):
                marker = " <-- BIRD" if str(v).lower() == "bird" else ""
                print(f"  ID {k:2d}: {v}{marker}")
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                marker = " <-- BIRD" if str(v).lower() == "bird" else ""
                print(f"  ID {i:2d}: {v}{marker}")
        else:
            print(f"  未知格式: {names}")

        # 查找 bird
        bird_id = None
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == "bird":
                    bird_id = int(k)
                    break
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).lower() == "bird":
                    bird_id = i
                    break

        if bird_id is not None:
            print(f"\n✅ 找到 'bird' 类别，ID: {bird_id}")
        else:
            print(f"\n❌ 未找到 'bird' 类别")
            print("   尝试查找包含 'bird' 的类别名...")
            if isinstance(names, dict):
                for k, v in names.items():
                    if "bird" in str(v).lower():
                        print(f"     可能匹配: ID {k} = {v}")
            elif isinstance(names, (list, tuple)):
                for i, v in enumerate(names):
                    if "bird" in str(v).lower():
                        print(f"     可能匹配: ID {i} = {v}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查 YOLOv8 模型
    check_model_classes("models/yolov8x-seg.pt")

    # 检查 YOLO11 模型（如果存在）
    check_model_classes("models/yolo11x-seg.pt")
