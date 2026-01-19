import os
import torch
import glob
import rasterio
import numpy as np
import argparse  # 导入argparse
import sys  # 用于退出程序
from torchvision import models
import torch.nn as nn
from colorama import init, Fore, Style

# 初始化colorama，自动重置样式
init(autoreset=True)
RED_BOLD = f"{Fore.RED}{Style.BRIGHT}"
RESET = Style.RESET_ALL


# --- 新增：命令行参数解析（完全手动控制）---
def parse_args():
    # 创建参数解析器（关闭原生help，手动控制）
    parser = argparse.ArgumentParser(
        description='提取TIF文件的CNN特征并保存为PT文件',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False  # 关闭原生help
    )

    # 关键：去掉required=True，改用手动校验
    parser.add_argument(
        '-d', '--data_dir',
        type=str,
        help='存放TIF文件的目录（必选），例如：\n./csv_tif \n或 /home/user/tif_files'
    )

    parser.add_argument(
        '-f', '--feature_dir',
        type=str,
        help='保存提取特征的目录（必选），例如：\n./extracted_features \n或 /home/user/features'
    )

    # 手动添加help参数
    parser.add_argument(
        '-h', '--help',
        action='store_true',
        help='显示此帮助信息并退出'
    )

    # 关键：使用parse_known_args()避免原生异常
    args, unknown = parser.parse_known_args()

    # 1. 处理帮助请求
    if args.help:
        print("=" * 80)
        print("📖 TIF特征提取工具 - 使用帮助")
        print("=" * 80)
        parser.print_help()
        sys.exit(0)

    # 2. 手动校验必选参数
    missing_args = []
    if not args.data_dir:
        missing_args.append('-d/--data_dir')
    if not args.feature_dir:
        missing_args.append('-f/--feature_dir')

    if missing_args:
        print(f"\n❌ 参数错误：缺少必选参数 → {', '.join(missing_args)}")
        print("\n📖 完整使用说明：")
        parser.print_help()
        sys.exit(1)

    # 3. 校验data_dir是否存在
    if not os.path.exists(args.data_dir):
        print(f"\n❌ 错误：指定的TIF目录不存在 → {args.data_dir}")
        print("\n📖 完整使用说明：")
        parser.print_help()
        sys.exit(1)

    # 4. 校验data_dir下是否有TIF文件
    tif_files = glob.glob(os.path.join(args.data_dir, "*.tif"))
    if len(tif_files) == 0:
        print(f"\n❌ 错误：{args.data_dir} 目录下未找到任何.tif文件")
        print("\n📖 完整使用说明：")
        parser.print_help()
        sys.exit(1)

    return args


# --- 主逻辑 ---
def main():
    # 解析命令行参数
    args = parse_args()

    # 配置（从命令行参数读取）
    data_dir = args.data_dir
    feature_dir = args.feature_dir
    os.makedirs(feature_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📌 使用配置：")
    print(f"   TIF文件目录: {data_dir}")
    print(f"   特征保存目录: {feature_dir}")
    print(f"   计算设备: {device}")

    # 1. 定义和加载CNN Backbone
    base = models.resnet18(pretrained=True).to(device)
    # 只要 ResNet 的前 8 层（去掉全连接和池化），输出 [512, H/32, W/32]
    cnn_backbone = nn.Sequential(*list(base.children())[:-2])
    # 修改第一层卷积以匹配20波段输入
    cnn_backbone[0] = nn.Conv2d(20, 64, kernel_size=7, stride=1, padding=3, bias=False).to(device)
    cnn_backbone.eval()

    def process_tif(img_path):
        with rasterio.open(img_path) as src:
            img_data = src.read().astype(np.float32)
            # 归一化逻辑 (与主程序保持一致)
            for i in range(img_data.shape[0]):
                ch_min, ch_max = img_data[i].min(), img_data[i].max()
                denom = ch_max - ch_min
                if denom > 1e-6:
                    img_data[i] = (img_data[i] - ch_min) / denom
                else:
                    img_data[i] = 0.0
            return torch.from_numpy(img_data).unsqueeze(0).to(device)

    # 2. 遍历并保存特征
    tif_files = glob.glob(os.path.join(data_dir, "*.tif"))
    print(f"\n🚀 开始提取 {len(tif_files)} 个文件的特征...")

    with torch.no_grad():
        for idx, t_path in enumerate(tif_files, 1):
            fname = os.path.basename(t_path).replace('.tif', '.pt')
            save_path = os.path.join(feature_dir, fname)

            # 跳过已存在的文件（可选优化）
            if os.path.exists(save_path):
                print(f"[{idx}/{len(tif_files)}] 已存在，跳过: {fname}")
                continue

            img_tensor = process_tif(t_path)
            feat_map = cnn_backbone(img_tensor)  # 得到特征图

            # 存入硬盘 (转到 CPU 节省显存)
            torch.save(feat_map.cpu(), save_path)
            print(f"[{idx}/{len(tif_files)}] 已保存: {fname}")

    # 修复：去掉args.args的笔误，改为args.feature_dir
    print(f"\n{RED_BOLD}保存提取特征的目录为：{feature_dir}{RESET}")
    print("\n✅ 特征提取完成！")


if __name__ == "__main__":
    main()