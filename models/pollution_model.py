# coding:utf-8
import os
import torch
import torch.nn as nn
from torchvision import models
import torch_frame
from torch_frame.data import TensorFrame
from torch_frame.nn.models import FTTransformer
from torch_frame.nn.encoder import LinearEncoder, EmbeddingEncoder, LinearEmbeddingEncoder


class CombinedPollutionModel(nn.Module):
    def __init__(self, col_stats, col_names_dict, embed_dim=32, img_size=(850, 1110)):
        """
        多模态污染预测模型：融合 CNN (遥感图像) 与 FT-Transformer (表格数据)

        Args:
            col_stats: torch_frame 的列统计信息，用于初始化表格编码器
            col_names_dict: torch_frame 的列名字典
            embed_dim: CNN 特征投影后的维度
            img_size: tif图像的H和W
        """
        super().__init__()
        self.H_orig, self.W_orig = img_size
        # --- 1. CNN 骨干网络 (处理 TIF 图像) ---
        # 使用 ResNet18 作为特征提取器，去掉最后两层 (全局池化和全连接)
        base = models.resnet18(weights=None)
        self.cnn_backbone = nn.Sequential(*list(base.children())[:-2])

        # 1x1 卷积将 512 通道降维，减少参数量并匹配后续融合层
        self.project = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.cnn_norm = nn.LayerNorm(embed_dim)

        # 运行模式控制：快速模式(使用预提取特征) vs 微调模式(端到端训练 CNN)
        self.fine_tune_mode = False

        # --- 2. 表格模型 (FT-Transformer) ---
        # 过滤掉不需要输入 Tabular 模型的辅助列（如 ID 和 索引）
        self.tabular_col_names_dict = {
            stype: [c for c in colnames if c not in ['file_id', 'index', 'orig_row_idx']]
            for stype, colnames in col_names_dict.items()
        }

        stype_encoder_dict = {
            torch_frame.numerical: LinearEncoder(),
            torch_frame.categorical: EmbeddingEncoder(),
        }

        # 处理嵌入类型列 (Text Embedding)
        if hasattr(torch_frame, 'embedding') and torch_frame.embedding in col_names_dict:
            stype_encoder_dict[torch_frame.embedding] = LinearEmbeddingEncoder()

        self.tabular_model = FTTransformer(
            channels=128,  # Transformer 隐层维度
            out_channels=1,  # 输出维度 (PM2.5 预测值)
            num_layers=3,  # Transformer 层数
            col_stats=col_stats,
            col_names_dict=self.tabular_col_names_dict,
            stype_encoder_dict=stype_encoder_dict
        )

        # --- 3. 融合层 ---
        # 将 CNN 空间特征 (embed_dim) + 归一化坐标 (2: row, col) 映射到 Tabular 的维度
        self.cnn_to_tab = nn.Linear(embed_dim + 2, 128)
        self.cnn_act = nn.GELU()
        self.final_norm = nn.LayerNorm(128)

    def forward(self, tf_batch, img_paths_dict, file_ids, row_indices, col_indices, feature_loader_fn=None, feat_dir='./extracted_features'):
        """
        Args:
            tf_batch: TensorFrame 格式的表格数据
            img_paths_dict: ID 到 TIF 路径的映射
            file_ids: 当前 Batch 对应的图像 ID
            row_indices, col_indices: 采样点在原始图像中的行列坐标
            feature_loader_fn: 外部传入的特征加载函数（用于快速模式下加载 .pt 文件）
            feat_dir: 外部传入的tif图像特征目录，包含形如20180115_12_hourly_mean.pt文件
        """
        unique_ids = torch.unique(file_ids)
        batch_size = file_ids.size(0)

        # 动态获取数据类型，确保计算一致性
        current_dtype = tf_batch.feat_dict[torch_frame.numerical].dtype
        cnn_tokens = torch.zeros((batch_size, 1, 128), device=file_ids.device, dtype=current_dtype)

        # 原始图像标准尺寸 (用于坐标映射)
        H_orig, W_orig = self.H_orig, self.W_orig

        for fid in unique_ids:
            mask = (file_ids == fid)
            img_path = img_paths_dict[int(fid.item())]

            # 图像特征获取逻辑
            if not self.fine_tune_mode and feature_loader_fn is not None:
                # 快速模式：从磁盘加载预提取的 [512, h, w] 特征
                c_name = os.path.basename(img_path)
                feat_path = os.path.join(feat_dir, c_name.replace('.tif', '.pt'))
                feat_map = feature_loader_fn(feat_path).to(file_ids.device).unsqueeze(0)
            else:
                # 微调模式：直接运行 CNN 骨干网络
                # 注意：此处假设 load_single_tif 在外部定义或已集成
                # img_tensor = load_single_tif(img_path).unsqueeze(0).to(file_ids.device)
                # feat_map = self.cnn_backbone(img_tensor)
                raise NotImplementedError("微调模式需配合图像加载函数使用")

            # 空间降维与变换
            feat_map = self.project(feat_map).squeeze(0)  # [embed_dim, h, w]
            h_feat, w_feat = feat_map.shape[-2], feat_map.shape[-1]
            feat_grid = feat_map.permute(1, 2, 0)  # [h, w, embed_dim]

            # 坐标双线性采样/近邻采样映射
            norm_rows = ((row_indices[mask] / (H_orig - 1)) * (h_feat - 1)).long().clamp(0, h_feat - 1)
            norm_cols = ((col_indices[mask] / (W_orig - 1)) * (w_feat - 1)).long().clamp(0, w_feat - 1)

            # 提取点位特征
            point_embeds = feat_grid[norm_rows, norm_cols, :]
            point_embeds = self.cnn_norm(point_embeds)

            # 拼接相对位置信息 [row, col]
            rel_row = (row_indices[mask].unsqueeze(1) / (H_orig - 1)).to(current_dtype)
            rel_col = (col_indices[mask].unsqueeze(1) / (W_orig - 1)).to(current_dtype)

            # 融合并生成 CNN Token
            enhanced_embeds = torch.cat([point_embeds, rel_row, rel_col], dim=1)
            token_out = self.cnn_to_tab(self.cnn_act(enhanced_embeds)).unsqueeze(1)
            cnn_tokens[mask] = token_out.to(cnn_tokens.dtype)

        # --- 表格数据处理 ---
        new_feat_dict = {}
        for stype, colnames in self.tabular_col_names_dict.items():
            orig_colnames = tf_batch.col_names_dict[stype]
            indices = torch.tensor([orig_colnames.index(c) for c in colnames], device=tf_batch.device)
            new_feat_dict[stype] = tf_batch.feat_dict[stype][:, indices]

        filtered_tf_batch = TensorFrame(
            feat_dict=new_feat_dict,
            col_names_dict=self.tabular_col_names_dict,
            y=tf_batch.y
        )

        # 将 Tabular Token 与 CNN Token 拼接
        x_tab, _ = self.tabular_model.encoder(filtered_tf_batch)
        x = torch.cat([x_tab, cnn_tokens.to(x_tab.dtype)], dim=1)

        # 经过 Transformer Backbone
        x, _ = self.tabular_model.backbone(x)

        # 池化与预测
        x = self.final_norm(x).mean(dim=1)
        return self.tabular_model.decoder(x)
