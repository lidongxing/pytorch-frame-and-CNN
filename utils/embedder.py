# coding:utf-8
import os
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer
from torch_frame.config.text_embedder import TextEmbedderConfig
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from sentence_transformers import SentenceTransformer
#
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model.save('./my_bert_model')
local_model_path = os.path.abspath('./my_bert_model')

class MyBERTTextEmbedder:
    def __init__(self, model_path, device):
        """
        本地 BERT 文本嵌入器
        Args:
            model_path: 本地 BERT 模型文件夹的绝对路径
            device: 运行设备 (cuda 或 cpu)
        """
        # 加载本地 SentenceTransformer 模型
        self.model = SentenceTransformer(model_path, device=device)

    def __call__(self, sentences: list[str]) -> Tensor:
        """
        将文本列表转换为 Tensor 向量
        torch-frame 期望返回的是 CPU 上的 Tensor（后续会统一物化到内存/显存）
        """
        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings.cpu()

def get_text_cfg(model_path, device, batch_size=32):
    """
    获取 torch_frame 所需的配置对象
    """
    my_embedder = MyBERTTextEmbedder(model_path=local_model_path, device=device)
    return TextEmbedderConfig(
        text_embedder=my_embedder,
        batch_size=batch_size
    )