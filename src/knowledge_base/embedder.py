"""CodeBERT嵌入封装，采用attention-mask均值池化。"""

import math

import numpy as np
import torch


class CodeBertEmbedder:
    """延迟加载本地或HuggingFace缓存中的CodeBERT模型。"""

    def __init__(self, config):
        self.config = config
        self.model_name = config["model_name"]
        self.revision = config.get("revision")
        self.device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu")
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
        ).to(self.device)
        self.model.eval()

    def encode(self, texts, normalize=True):
        """编码文本并确保向量有限且L2归一化。"""
        values = []
        with torch.no_grad():
            for offset in range(0, len(texts), int(self.config["batch_size"])):
                batch = texts[offset: offset + int(self.config["batch_size"])]
                tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=int(self.config["max_length"]), return_tensors="pt").to(self.device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                values.append(pooled.cpu().numpy().astype(np.float32))
        matrix = np.vstack(values) if values else np.empty((0, 0), dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("CodeBERT嵌入包含NaN或Inf")
        if not normalize:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("CodeBERT嵌入存在零范数向量")
        return matrix / norms
