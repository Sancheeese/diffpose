import os
import glob


def filter_images_by_names(txt_path, img_folder):
    # 从txt文件中读取要保留的人名列表
    with open(txt_path, 'r', encoding='utf-8') as f:
        keep_names = set(line.strip() for line in f)

    # 获取所有图片文件
    img_files = glob.glob(os.path.join(img_folder, "*.png"))
    deleted_count = 0
    kept_count = 0

    for img_path in img_files:
        # 提取文件名（不带路径）
        filename = os.path.basename(img_path)
        # 分割人名部分（第一个下划线前的部分）
        if '_' in filename:
            name_part = filename.split('_', 1)[0]

            # 检查人名是否在保留列表中
            if name_part in keep_names:
                kept_count += 1
            else:
                # 删除不在保留列表中的图片
                os.remove(img_path)
                deleted_count += 1
        else:
            # 处理不符合命名规则的文件
            print(f"警告：跳过不符合命名规则的文件 {filename}")
            os.remove(img_path)
            deleted_count += 1

    print(f"操作完成！共保留 {kept_count} 张图片，删除 {deleted_count} 张图片")


# 设置文件路径
txt_file = "names.txt"  # 包含人名的文本文件路径
image_folder = "/home/zsr/project/diffpose/ours/drrStyle_choose/trainA"  # 图片所在文件夹路径

# 执行清理操作
filter_images_by_names(txt_file, image_folder)