# coding:utf-8
import os
import glob
import argparse
import torch
import pandas as pd
import pickle
import torch_frame
import numpy as np
from torch_frame.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import copy
# 导入自定义模块
from models.pollution_model import CombinedPollutionModel
from utils.embedder import get_text_cfg
from utils.data_utils import load_feature_map, get_col_to_stype


def get_args():
    parser = argparse.ArgumentParser(
        description="Pollution Model Batch Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 路径配置 ---
    group_path = parser.add_argument_group('Path Configurations')
    group_path.add_argument('--inf_data_dir', type=str, required=True,
                            help='Directory containing raw CSV and TIF files for inference.')
    group_path.add_argument('--model_weight', type=str, default='./best_model.pth',
                            help='Path to the trained model weights (.pth).')
    group_path.add_argument('--bert_path', type=str, default='./my_bert_model',
                            help='Local path to the BERT model for text embedding.')
    group_path.add_argument('--feat_dir', type=str, default='./extracted_features',
                            help='Directory where pre-extracted TIF features (.pt) are stored.')

    # --- 推理与模型配置 ---
    group_model = parser.add_argument_group('Inference Settings')
    group_model.add_argument('--batch_size', type=int, default=512,
                             help='Batch size for inference.')
    group_model.add_argument('--save_suffix', type=str, default='_predicted',
                             help='Suffix added to the generated result CSV files.')
    group_model.add_argument('--img_h', type=int, default=850,
                             help='Original height of the TIF images.')
    group_model.add_argument('--img_w', type=int, default=1110,
                             help='Original width of the TIF images.')

    return parser.parse_args()


def run_batch_inference():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 搜集并对齐文件 (这一步必须在前面) ---
    all_csvs = sorted(glob.glob(os.path.join(args.inf_data_dir, "*.csv")))
    all_csvs = [f for f in all_csvs if not f.endswith(f"{args.save_suffix}.csv")]

    if not all_csvs:
        print(f"Error: No valid CSV files found in {args.inf_data_dir}")
        return

    img_paths_dict = {}
    dfs = []  # <--- 确保在这里定义了 dfs 列表

    print(f"Starting alignment for {len(all_csvs)} files...")
    for c_path in all_csvs:
        c_name = os.path.basename(c_path)
        t_path = os.path.join(args.inf_data_dir, f"{c_name.split('.')[0]}_12_hourly_mean.tif")

        if os.path.exists(t_path):
            fid = len(img_paths_dict)
            img_paths_dict[fid] = t_path

            df = pd.read_csv(c_path)
            df['file_id'] = fid
            df['csv_filename'] = c_name
            df['orig_row_idx'] = df.index + 2  # 用于最后回填结果

            if 'PM25' not in df.columns:
                df['PM25'] = 0.0
            dfs.append(df)
        else:
            print(f"Warning: Skipping {c_name}, TIF not found.")

    if not dfs:
        print("Error: No matching CSV-TIF pairs found.")
        return

    # --- 2. 生成原始数据的 DataFrame ---
    full_df = pd.concat(dfs, ignore_index=True)

    # 显式补齐推理数据缺失的 band 列（如果有的话）
    for i in range(1, 21):
        if f'band_{i}' not in full_df.columns:
            full_df[f'band_{i}'] = 0.0

    # 必须在这里创建 index 列
    full_df['index'] = range(len(full_df))
    full_df['sample_key'] = range(len(full_df))
    orig_len = len(full_df)

    # --- 3. 维度补齐逻辑 (Padding) ---
    stats_path = 'col_stats.pkl'
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing {stats_path}! Run training first.")

    with open(stats_path, 'rb') as f:
        train_col_stats = pickle.load(f)
        # 修正变量名并打印确认
        if 'file_id' in train_col_stats:
            print(f"成功加载统计信息，file_id 统计项包含: {list(train_col_stats['file_id'].keys())}")
        else:
            raise KeyError("col_stats.pkl 中缺少 file_id 信息，请检查训练脚本。")

    file_id_stats = train_col_stats['file_id']
    train_categories = None

    for k, v in file_id_stats.items():
        if 'CATEGORIES' in str(k).upper():
            train_categories = v
            break

    # --- 开始补齐 ---
    if train_categories is not None:
        train_cat_list = list(train_categories)
        current_fids = full_df['file_id'].unique()

        if len(current_fids) < len(train_cat_list):
            print(f"检测到维度不匹配，正在补齐...")

            # 创建 padding_rows 时，确保它包含 full_df 的所有列名
            # 这样 concat 时就不会产生缺失列
            padding_data = {col: 0 for col in full_df.columns}  # 先用 0 填充所有列

            # 覆盖关键列
            padding_data.update({
                'file_id': train_cat_list,
                'PM25': 0.0,
                'text': 'placeholder',
                'csv_filename': 'padding_placeholder',
                'sample_key': -1,
                'index': -1
            })

            # 使用列表包装 train_cat_list 来构造，防止长度不一报错
            # 但最稳妥的方法是直接用 DataFrame 构造循环
            padding_df = pd.DataFrame(columns=full_df.columns)
            for fid in train_cat_list:
                new_row = {col: 0 for col in full_df.columns}
                new_row.update({'file_id': fid, 'text': 'placeholder', 'sample_key': -1, 'index': -1})
                padding_df.loc[len(padding_df)] = new_row

            # 合并
            full_df = pd.concat([full_df, padding_df], ignore_index=True)
            # 强制转换类型，防止 torch_frame 误判
            full_df['index'] = full_df['index'].astype(int)
            full_df['sample_key'] = full_df['sample_key'].astype(int)

    # --- 4. 初始化 Dataset ---
    col_to_stype = get_col_to_stype(torch_frame)

    # 确保 sample_key 存在，但 index 移除
    if 'index' in col_to_stype: del col_to_stype['index']
    if 'orig_row_idx' in col_to_stype: del col_to_stype['orig_row_idx']

    # 关键：我们要它在数据里，但不能让它变成模型的输入特征
    col_to_stype['sample_key'] = torch_frame.numerical

    # 强制转换类别列为字符串
    cat_cols = ['year', 'month', 'day', 'hour', 'file_id']
    for col in cat_cols:
        full_df[col] = full_df[col].astype(str)

    text_cfg = get_text_cfg(args.bert_path, device)
    dataset = Dataset(full_df, col_to_stype=col_to_stype, target_col='PM25',
                          col_to_text_embedder_cfg={'text': text_cfg})
    dataset.materialize()

    # --- 5. 加载模型 ---
    # 深度拷贝一份列名字典，避免影响原始数据
    model_col_names = copy.deepcopy(dataset.tensor_frame.col_names_dict)

    # 【核心对齐】：从数值列名单中删掉 sample_key
    # 这样模型初始化时只会看到 24 列数值特征
    if 'sample_key' in model_col_names[torch_frame.numerical]:
        model_col_names[torch_frame.numerical].remove('sample_key')

    model = CombinedPollutionModel(
        train_col_stats,  # 46维对齐
        model_col_names,  # 24维对齐
        img_size=(args.img_h, args.img_w)
    ).to(device)

    model.load_state_dict(torch.load(args.model_weight, map_location=device))
    model.eval()
    print(f"✅ 模型完美对齐并加载成功")

    # --- 6. 执行推理 ---
    # 只取原始数据部分 [0:orig_len]
    test_dataset = dataset[:orig_len]
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    # 此时 dataset 包含 sample_key，所以我们可以找到它的位置
    all_num_cols = dataset.tensor_frame.col_names_dict[torch_frame.numerical]
    key_pos = all_num_cols.index('sample_key')

    # 模型需要的特征位置（此时 sample_key 已不在模型考虑范围内，但还在 tensor_frame 里）
    row_idx_pos = all_num_cols.index('row')
    col_idx_pos = all_num_cols.index('col')
    fid_pos = dataset.tensor_frame.col_names_dict[torch_frame.categorical].index('file_id')

    print(f"Inference starting...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # 【关键修改】：不再从 feat_dict 取 sample_key
            # 而是利用 batch.row_idx 从原始 full_df 中提取
            # current_batch_row_indices = batch.row_idx.cpu().numpy()
            indices = batch.feat_dict[torch_frame.numerical][:, key_pos].cpu().numpy().astype(int)

            # 提取模型计算需要的特征
            rows = batch.feat_dict[torch_frame.numerical][:, row_idx_pos]
            cols = batch.feat_dict[torch_frame.numerical][:, col_idx_pos]
            fids = batch.feat_dict[torch_frame.categorical][:, fid_pos]

            with autocast(enabled=True):
                log_preds = model(batch, img_paths_dict, fids, rows, cols,
                                  feature_loader_fn=load_feature_map,
                                  feat_dir=args.feat_dir)
                preds = torch.expm1(log_preds).cpu().numpy().flatten()

            for idx, p_val in zip(indices, preds):
                results.append((idx, p_val))

    # --- 7. 原地更新结果到原始 CSV ---
    pred_map = {idx: val for idx, val in results}
    # 截断回原始长度（去掉 Padding 部分）
    final_df = full_df.iloc[:orig_len].copy()
    final_df['Predicted_PM25'] = final_df['sample_key'].map(pred_map)

    print(f"Saving results to original files...")
    for fname in final_df['csv_filename'].unique():
        # 1. 提取当前文件对应的预测结果
        sub_df = final_df[final_df['csv_filename'] == fname]
        original_csv_path = os.path.join(args.inf_data_dir, fname)

        if os.path.exists(original_csv_path):
            # 2. 读取原始 CSV
            output_df = pd.read_csv(original_csv_path)

            # 3. 建立映射逻辑：通过 orig_row_idx 将预测值填回正确的行
            # 注意：这里减去 2 是因为之前在读取时加了 2 (df.index + 2)
            mapping = sub_df.set_index('orig_row_idx')['Predicted_PM25'].to_dict()

            # 4. 在原列基础上新增一列，或者覆盖已有的预测列
            output_df['Predicted_PM25'] = [mapping.get(i + 2, None) for i in range(len(output_df))]

            # 5. 直接覆盖保存原文件
            output_df.to_csv(original_csv_path, index=False)
            print(f"  [Updated] {original_csv_path}")
        else:
            print(f"  [Warning] Original file not found: {original_csv_path}")

    print("✅ All tasks completed successfully.")


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_batch_inference()