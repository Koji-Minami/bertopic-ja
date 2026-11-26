import hashlib

import numpy as np
from bertopic.backend import BaseEmbedder


class HashEmbedder(BaseEmbedder):
    """テスト用の埋め込み。モデルの重みはダウンロードしない。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, documents, verbose=False):
        self.calls.append(list(documents))
        rows = []
        for document in documents:
            if "猫" in document:
                base = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            elif "車" in document or "ローン" in document:
                base = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
            else:
                digest = hashlib.sha256(document.encode()).digest()
                seed = int.from_bytes(digest[:4], "little")
                base = np.random.default_rng(seed).normal(size=4)
            rows.append(base)
        return np.vstack(rows)
