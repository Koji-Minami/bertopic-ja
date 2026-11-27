"""日本語文書の探索的トピック分析。

手法と実装の本体は Maarten Grootendorst の BERTopic です。
このパッケージは公式のサブプロジェクトではなく、Sudachi による分かち書きと
トピック語の調整メソッドを足した第三者の入口です。

https://github.com/MaartenGr/BERTopic
https://arxiv.org/abs/2203.05794
"""

from bertopic_ja._model import BERTopic
from bertopic_ja._tokenizer import DEFAULT_POS, DEFAULT_STOPWORDS

__all__ = ["BERTopic", "DEFAULT_POS", "DEFAULT_STOPWORDS"]
__version__ = "0.1.2"
