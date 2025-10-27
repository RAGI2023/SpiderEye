# extract folders from Beihang V2 dataset
import os
import shutil
import re
import unicodedata

# 原始数据根目录
src_root = os.path.expanduser("~/Downloads/V2")
# 目标数据集目录
dst_root = os.path.expanduser("~/Downloads/dataset")

os.makedirs(dst_root, exist_ok=True)

# 将中文/特殊字符转换为安全英文名
def safe_name(name):
    normalized = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', normalized)
    return safe or "unknown"

used_names = set()

# 遍历所有场景目录（停车场、教室、街道等）
for scene in os.listdir(src_root):
    scene_path = os.path.join(src_root, scene)
    if not os.path.isdir(scene_path):
        continue  # 跳过 .7z 等文件

    safe_scene = safe_name(scene)

    # 遍历场景下的第二级目录（A、B、I3等）
    for subdir in os.listdir(scene_path):
        subdir_path = os.path.join(scene_path, subdir)
        if not os.path.isdir(subdir_path):
            continue

        safe_subdir = safe_name(subdir)

        # 构造唯一目录名：scene_subdir
        dst_name = f"{safe_scene}_{safe_subdir}"
        # 确保不重复
        if dst_name in used_names:
            suffix = 1
            while f"{dst_name}_{suffix}" in used_names:
                suffix += 1
            dst_name = f"{dst_name}_{suffix}"
        used_names.add(dst_name)

        dst_path = os.path.join(dst_root, dst_name)

        if os.path.exists(dst_path):
            print(f"⚠️ 已存在，跳过：{dst_path}")
            continue

        print(f"复制目录：{subdir_path} -> {dst_path}")
        shutil.copytree(subdir_path, dst_path)

print("✅ 所有第二级文件夹已提取到 dataset/ 下，并确保目录名唯一。")
