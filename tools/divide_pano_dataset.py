import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm  

def split_dataset(data_dir, output_dir, test_ratio=0.2, seed=42):
    # 设置随机种子，保证可复现
    random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"

    # 创建输出目录
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    images = [f for f in data_dir.iterdir() if f.suffix.lower() in img_exts]

    if not images:
        print("❌ 未在指定路径下找到图片文件。")
        return

    # 打乱顺序
    random.shuffle(images)

    # 计算分割点
    test_size = int(len(images) * test_ratio)
    test_files = images[:test_size]
    train_files = images[test_size:]

    # 拷贝文件（带进度条）
    print(f"📦 正在复制训练集 ({len(train_files)} 张)...")
    for f in tqdm(train_files, desc="Train", ncols=80):
        shutil.copy(f, train_dir / f.name)

    print(f"🧪 正在复制验证集 ({len(test_files)} 张)...")
    for f in tqdm(test_files, desc="Test", ncols=80):
        shutil.copy(f, eval_dir / f.name)

    print("\n✅ 数据集划分完成！")
    print(f"训练集：{len(train_files)} 张 -> {train_dir}")
    print(f"验证集：{len(test_files)} 张 -> {eval_dir}")

def main():
    parser = argparse.ArgumentParser(description="将图片数据集随机划分为train和eval集（带进度条）")
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="原始图片目录路径")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="输出目录路径")
    parser.add_argument("-r", "--eval_ratio", type=float, default=0.2, help="验证集比例（默认0.2）")
    parser.add_argument("-s", "--seed", type=int, default=42, help="随机种子（默认42）")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    split_dataset(args.data_dir, args.output_dir, args.eval_ratio, args.seed)

if __name__ == "__main__":
    main()
