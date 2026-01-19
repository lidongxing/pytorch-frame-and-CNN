# coding:utf-8
import os
import torch
import numpy as np
import rasterio
from functools import lru_cache

@lru_cache(maxsize=512)
def load_feature_map(pt_path):
    """
    加载预计算好的 CNN 特征张量 (.pt 文件)
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"特征文件未找到: {pt_path}")
    # 返回维度应为 [512, h, w]
    return torch.load(pt_path, weights_only=True).squeeze(0)

@lru_cache(maxsize=128)
def load_single_tif(img_path):
    """
    读取原始 .tif 图像并进行最大最小归一化
    """
    with rasterio.open(img_path) as src:
        img_data = src.read().astype(np.float32)
        # 逐通道归一化
        for i in range(img_data.shape[0]):
            ch_min, ch_max = img_data[i].min(), img_data[i].max()
            denom = ch_max - ch_min
            if denom > 1e-6:
                img_data[i] = (img_data[i] - ch_min) / denom
            else:
                img_data[i] = 0.0
        return torch.from_numpy(img_data)

def get_col_to_stype(torch_frame):
    """
    统一管理列类型映射，确保 train 和 inference 严格一致
    """
    col_to_stype = {
        'index': torch_frame.numerical,
        'PM25': torch_frame.numerical,
        'lat': torch_frame.numerical,
        'lon': torch_frame.numerical,
        'year': torch_frame.categorical,
        'month': torch_frame.categorical,
        'day': torch_frame.categorical,
        'hour': torch_frame.categorical,
        'row': torch_frame.numerical,
        'col': torch_frame.numerical,
        'file_id': torch_frame.categorical,
        'text': torch_frame.text_embedded,
        'orig_row_idx': torch_frame.numerical
    }
    # 添加 20 个波段列
    for i in range(1, 21):
        col_to_stype[f'band_{i}'] = torch_frame.numerical
    return col_to_stype
