# coding:utf-8
import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch_frame
from torch_frame import StatType
from torch_frame.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pickle

# 导入自定义模块
from models.pollution_model import CombinedPollutionModel
from utils.embedder import get_text_cfg
from utils.data_utils import load_feature_map, get_col_to_stype, load_single_tif


def get_args():
    parser = argparse.ArgumentParser(
        description="Air Pollution Multi-modal Training Pipeline (CNN + FT-Transformer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 自动在帮助信息中显示默认值
    )

    # --- 路径配置 ---
    group_path = parser.add_argument_group('Paths')
    group_path.add_argument('--data_dir', type=str, default='./csv_tif', help='Directory containing CSV and TIF files.')
    group_path.add_argument('--model_path', type=str, default='./my_bert_model',
                            help='Local path to the BERT/SentenceTransformer model.')
    group_path.add_argument('--save_path', type=str, default='best_model.pth',
                            help='Path to save the best model weights.')
    group_path.add_argument('--feat_dir', type=str, default='./extracted_features',
                            help='Directory where pre-extracted TIF features (.pt) are stored.')

    # --- 超参数配置 ---
    group_hparams = parser.add_argument_group('Hyperparameters')
    group_hparams.add_argument('--batch_size', type=int, default=1024, help='Number of samples per training batch.')
    group_hparams.add_argument('--epochs', type=int, default=50, help='Total number of training epochs.')
    group_hparams.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate for OneCycleLR scheduler.')
    group_hparams.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    group_hparams.add_argument('--img_h', type=int, default=850, help='Original height of the TIF images.')
    group_hparams.add_argument('--img_w', type=int, default=1110, help='Original width of the TIF images.')

    # --- 数据划分比例 ---
    group_split = parser.add_argument_group('Dataset Split Ratios')
    group_split.add_argument('--train_ratio', type=float, default=0.98, help='Proportion of data used for training.')
    group_split.add_argument('--val_ratio', type=float, default=0.01, help='Proportion of data used for validation.')
    group_split.add_argument('--test_ratio', type=float, default=0.01, help='Proportion of data used for testing.')

    return parser.parse_args()


def evaluate(model, loader, img_paths_dict, dataset, device, feat_dir):
    """
    模型评估函数
    Args:
        feat_dir: 显式传递特征目录，避免引用全局变量导致错误
    """
    model.eval()
    total_mae, num_samples = 0, 0
    num_cols = dataset.tensor_frame.col_names_dict[torch_frame.numerical]
    cat_cols = dataset.tensor_frame.col_names_dict[torch_frame.categorical]

    row_p, col_p = num_cols.index('row'), num_cols.index('col')
    fid_p = cat_cols.index('file_id')

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            rows = batch.feat_dict[torch_frame.numerical][:, row_p]
            cols = batch.feat_dict[torch_frame.numerical][:, col_p]
            fids = batch.feat_dict[torch_frame.categorical][:, fid_p]

            log_preds = model(batch, img_paths_dict, fids, rows, cols,
                              feature_loader_fn=load_feature_map,
                              feat_dir=feat_dir)
            y_pred = torch.expm1(log_preds).view(-1)
            y_true = batch.y.float().view(-1)

            total_mae += F.l1_loss(y_pred, y_true, reduction='sum').item()
            num_samples += y_true.size(0)

    return total_mae / num_samples if num_samples > 0 else 0


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 数据准备 ---
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    total_f = len(csv_files)
    train_end = int(args.train_ratio * total_f)
    val_end = int((args.train_ratio + args.val_ratio) * total_f)

    img_paths_dict = {}

    def prepare_df(file_list):
        dfs = []
        for c_path in file_list:
            c_name = os.path.basename(c_path)
            t_path = os.path.join(args.data_dir, f"{c_name.split('.')[0]}_12_hourly_mean.tif")
            if os.path.exists(t_path):
                fid = len(img_paths_dict)
                img_paths_dict[fid] = t_path
                df = pd.read_csv(c_path)
                df = df[(df['PM25'] > 0) & (df['PM25'] < 500)].copy()
                df['file_id'] = fid
                df['csv_filename'] = c_name
                df['orig_row_idx'] = df.index + 2
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    print(f"正在加载数据集 [Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}]...")
    train_df = prepare_df(csv_files[:train_end]).sample(frac=1, random_state=args.seed)
    val_df = prepare_df(csv_files[train_end:val_end])
    test_df = prepare_df(csv_files[val_end:])
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True).reset_index()
    full_df['file_id'] = full_df['file_id'].astype(str)

    # --- 2. 构造 Dataset ---
    col_to_stype = get_col_to_stype(torch_frame)
    text_cfg = get_text_cfg(args.model_path, device)

    dataset = Dataset(full_df, col_to_stype=col_to_stype, target_col='PM25',
                      col_to_text_embedder_cfg={'text': text_cfg})
    dataset.materialize()

    stats = dataset.col_stats

    # 1. 强制获取当前全量 file_id 列表
    all_fids = full_df['file_id'].unique().tolist()

    # 2. 找到 file_id 统计项中现有的那个枚举键（不管是 COUNT 还是别的）
    file_id_info = stats['file_id']
    existing_keys = list(file_id_info.keys())

    # 3. 寻找或伪造 CATEGORIES 键
    # 如果没有找到带 CATEGORIES 字样的键，我们就直接拿第一个键的类型来伪造一个
    cat_key = next((k for k in existing_keys if 'CATEGORIES' in str(k)), None)

    if cat_key is None and len(existing_keys) > 0:
        # 这一步很硬核：克隆 COUNT 键的类型，但改变其值为 CATEGORIES 的语义
        # 或者直接用字符串，因为 pickle 加载后通常能识别
        from torch_frame import StatType
        cat_key = StatType.CATEGORIES

    # 强行注入
    file_id_info[cat_key] = all_fids

    # 打印一下，确保在内存里已经有了
    print(f"DEBUG: 内存中 file_id 的键现在是: {list(file_id_info.keys())}")
    print(f"DEBUG: 注入的类别数量: {len(file_id_info[cat_key])}")

    # 4. 强制写入并刷新磁盘缓冲
    with open('col_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
        f.flush()
        os.fsync(f.fileno())  # 确保物理写入磁盘

    print("✅ col_stats.pkl 已物理更新到磁盘。")
    # --- 3. DataLoader ---
    # 计算 val 结束索引
    val_split_end = len(train_df) + len(val_df)

    train_loader = DataLoader(dataset[:len(train_df)], batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(dataset[len(train_df):val_split_end], batch_size=512)
    test_loader = DataLoader(dataset[val_split_end:], batch_size=512)

    # --- 4. 初始化模型与优化器 ---
    model = CombinedPollutionModel(
        dataset.col_stats,
        dataset.tensor_frame.col_names_dict,
        img_size=(args.img_h, args.img_w)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )
    scaler = GradScaler(enabled=True)

    # 索引位置预缓存
    num_names = dataset.tensor_frame.col_names_dict[torch_frame.numerical]
    cat_names = dataset.tensor_frame.col_names_dict[torch_frame.categorical]
    row_p, col_p, fid_p = num_names.index('row'), num_names.index('col'), cat_names.index('file_id')

    # --- 5. 训练循环 ---
    best_mae = float('inf')
    print(f"开始训练，设备: {device}, 总 Epochs: {args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            rows = batch.feat_dict[torch_frame.numerical][:, row_p]
            cols = batch.feat_dict[torch_frame.numerical][:, col_p]
            fids = batch.feat_dict[torch_frame.categorical][:, fid_p]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                preds = model(batch, img_paths_dict, fids, rows, cols,
                              feature_loader_fn=load_feature_map,
                              feat_dir=args.feat_dir)
                target = torch.log1p(batch.y.float().view(-1, 1))
                loss = F.mse_loss(preds, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        # 验证阶段
        val_mae = evaluate(model, val_loader, img_paths_dict, dataset, device, args.feat_dir)
        print(
            f"Epoch {epoch:02d} | Loss: {train_loss / len(train_loader):.4f} | Val MAE: {val_mae:.2f} | LR: {scheduler.get_last_lr()[0]:.6e}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), args.save_path)
            print(f"  [Save] New best MAE: {best_mae:.2f}, model saved to {args.save_path}")

    # --- 6. 最终测试汇报 ---
    print("\n" + "=" * 30 + " Final Test Evaluation " + "=" * 30)
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path))
    test_mae = evaluate(model, test_loader, img_paths_dict, dataset, device, args.feat_dir)
    print(f"Final Test MAE: {test_mae:.2f}")


if __name__ == '__main__':
    # 环境配置
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()