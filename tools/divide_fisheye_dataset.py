import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def split_dataset(data_dir, output_dir, test_ratio=0.2, seed=42):
    random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 获取所有子文件夹（每个文件夹代表一个样本）
    subfolders = [f for f in data_dir.iterdir() if f.is_dir()]

    if not subfolders:
        print("❌ 未找到任何子文件夹。请检查数据集结构。")
        return

    # 打乱顺序并划分
    random.shuffle(subfolders)
    test_size = int(len(subfolders) * test_ratio)
    eval_folders = subfolders[:test_size]
    train_folders = subfolders[test_size:]

    print(f"📦 训练集文件夹数量：{len(train_folders)}")
    print(f"🧪 验证集文件夹数量：{len(eval_folders)}")

    # ✅ 拷贝整个文件夹
    print("\n🚀 正在复制训练集文件夹...")
    for folder in tqdm(train_folders, desc="Train", ncols=80):
        shutil.copytree(folder, train_dir / folder.name, dirs_exist_ok=True)

    print("\n🚀 正在复制验证集文件夹...")
    for folder in tqdm(eval_folders, desc="Eval", ncols=80):
        shutil.copytree(folder, eval_dir / folder.name, dirs_exist_ok=True)

    print("\n✅ 数据集划分完成！")
    print(f"训练集路径：{train_dir}")
    print(f"验证集路径：{eval_dir}")

def main():
    parser = argparse.ArgumentParser(description="将数据集文件夹随机划分为 train / eval（每个子文件夹为一个样本）")
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="原始数据集路径")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="输出目录路径")
    parser.add_argument("-r", "--eval_ratio", type=float, default=0.2, help="验证集比例（默认0.2）")
    parser.add_argument("-s", "--seed", type=int, default=42, help="随机种子（默认42）")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    split_dataset(args.data_dir, args.output_dir, args.eval_ratio, args.seed)

if __name__ == "__main__":
    main()
