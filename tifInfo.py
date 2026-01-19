# -*- coding: gbk -*-
import os
import sys
import rasterio
import argparse  # 导入argparse库
from colorama import init, Fore, Style
init (autoreset=True) # 自动重置样式，避免后续文本变色
RED_BOLD = f"{Fore.RED}{Style.BRIGHT}"
RESET = Style.RESET_ALL

def open_large_tif_with_info(file_path, display_band=1):
    """
    打开TIF并可视化，标题显示总通道数、宽高、当前显示波段
    :param file_path: TIF文件路径（必填参数）
    :param display_band: 要显示的波段（默认第1波段，范围1-总通道数）
    :return: 归一化后的波段数据、总通道数、宽度、高度；失败则返回None
    """
    try:
        with rasterio.open(file_path) as src:
            # 提取核心信息
            total_bands = src.count  # 总通道数（波段数）
            width = src.width  # 宽度(W)
            height = src.height  # 高度(H)
            dtype = src.dtypes[0]  # 数据类型

            # 打印详细信息到控制台
            print("=" * 50)
            print(f"TIF文件核心信息：")
            print(f"文件路径: {file_path}")
            # 带红色加粗的打印语句
            print(f"\n{RED_BOLD}后续使用：总通道数C={total_bands}，宽度W={width}，高度H={height}{RESET}")
            print(f"数据类型: {dtype} | 坐标投影: {src.crs}")
            print("=" * 50)
    except ValueError as ve:
        print("参数错误：", str(ve))
        return None, None, None, None
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}，请检查路径是否正确")
        return None, None, None, None
    except Exception as e:
        print("打开失败，原因：", str(e))
        return None, None, None, None


def main():
    import os  # 新增：导入os模块

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='读取并可视化TIF文件，支持指定波段显示',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )

    # 添加必选参数：文件路径
    parser.add_argument(
        '-f', '--file_path',
        type=str,
        help='-f 或 --file_path（必选）\n输入单张TIF图像文件的完整路径，例如：\nD:\\lidongxing\\晏星老师\\pytorch-frame\\20171231_12_hourly_mean.tif \n或 /home/test.tif'
    )

    # 添加可选参数：显示波段
    parser.add_argument(
        '-b', '--band',
        type=int,
        default=1,
        help='-b 或 --band（可选）\n要显示的波段数（默认1，范围1-总通道数）'
    )

    # 手动添加help参数
    parser.add_argument(
        '-h', '--help',
        action='store_true',
        help='-h 或 --help（可选）\n显示此帮助信息并退出'
    )

    # ========== 关键修复：先解析参数，再处理异常 ==========
    args, unknown = parser.parse_known_args()

    # 处理帮助请求
    if args.help:
        print("=" * 80)
        print("? TIF文件查看工具 - 使用帮助")
        print("=" * 80)
        parser.print_help()
        sys.exit(0)

    # 检查必选参数是否存在
    if not hasattr(args, 'file_path') or args.file_path is None:
        print(f"\n? 参数解析错误：the following arguments are required: -f/--file_path")
        print("\n? 完整使用说明：")
        parser.print_help()
        sys.exit(1)

    # 校验文件路径是否存在
    if not os.path.exists(args.file_path):
        print(f"\n? 错误：文件路径不存在 → {args.file_path}")
        print("\n? 完整使用说明：")
        parser.print_help()
        sys.exit(1)

    # 调用核心函数
    open_large_tif_with_info(
        file_path=args.file_path,
        display_band=args.band
    )

if __name__ == "__main__":
    main()