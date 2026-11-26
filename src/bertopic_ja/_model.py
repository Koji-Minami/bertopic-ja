"""BERTopic を日本語の初期値で包む。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from bertopic import BERTopic as _BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.backend._utils import select_backend
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from umap import UMAP

from bertopic_ja._config import Config
from bertopic_ja._tokenizer import build_vectorizer, explain_text, strip_fillers


class BERTopic(_BERTopic):
    """日本語文書から、どんなトピックが話されているかを探索する。

    呼び出し方は公式の BERTopic と同じです。埋め込みの初期値は
    ``language="multilingual"`` です。トピック語は Sudachi（モード C）の
    名詞で、フィラーはトピック語からだけ外します。

    これは Maarten Grootendorst の BERTopic の公式サブプロジェクトでは
    ありません。https://github.com/MaartenGr/BERTopic
    """

    def __init__(
        self,
        *,
        min_topic_size: int = 10,
        nr_topics: int | str | None = None,
        n_neighbors: int | None = None,
        reduce_outliers: bool = True,
        embedding_model: Any = None,
        exclude: list[str] | None = None,
        protect: list[str] | None = None,
        fillers: list[str] | None = None,
    ) -> None:
        self.ja_config = Config(
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            n_neighbors=n_neighbors,
            reduce_outliers=reduce_outliers,
            embedding_model=embedding_model,
            exclude=list(exclude or []),
            protect=list(protect or []),
            fillers=list(fillers or []),
        )
        self._documents: list[str] = []
        self._embeddings = None
        self._fitted = False
        umap_model = self._make_umap(n_docs=max(min_topic_size + 2, 32))
        init_kwargs: dict[str, Any] = {
            "language": "multilingual",
            "umap_model": umap_model,
            "hdbscan_model": self._make_hdbscan(self.ja_config.min_topic_size),
            "vectorizer_model": build_vectorizer(self.ja_config),
            "representation_model": KeyBERTInspired(),
            "min_topic_size": self.ja_config.min_topic_size,
            "nr_topics": self.ja_config.nr_topics,
            "calculate_probabilities": False,
            "verbose": False,
        }
        if embedding_model is not None:
            init_kwargs["embedding_model"] = embedding_model
        super().__init__(**init_kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> BERTopic:
        """YAML を読み、同じ項目の引数があれば引数を優先する。"""
        config = Config.from_yaml(path, overrides)
        return cls(
            min_topic_size=config.min_topic_size,
            nr_topics=config.nr_topics,
            n_neighbors=config.n_neighbors,
            reduce_outliers=config.reduce_outliers,
            embedding_model=config.embedding_model,
            exclude=config.exclude,
            protect=config.protect,
            fillers=config.fillers,
        )

    def to_yaml(self, path: str | Path) -> None:
        """今の設定を YAML に書き戻す。"""
        self.ja_config.to_yaml(path)

    def fit_transform(self, documents, embeddings=None, **kwargs):
        """文書をクラスタリングし、トピック番号の列を返す。"""
        docs = self._prepare_documents(documents)
        self._documents = docs
        self.umap_model = self._make_umap(len(docs))
        self.hdbscan_model = self._make_hdbscan(self.ja_config.min_topic_size)
        if embeddings is None:
            if not isinstance(self.embedding_model, BaseEmbedder):
                self.embedding_model = select_backend(self.embedding_model, language=self.language)
            embeddings = self.embedding_model.embed(docs)
        self._embeddings = embeddings
        topics, probs = super().fit_transform(docs, embeddings=embeddings, **kwargs)
        if self.ja_config.reduce_outliers:
            topics = self._apply_outlier_reduction(docs, topics)
        self._drop_blank_topic_words()
        self._fitted = True
        return topics, probs

    def exclude(self, *words: str | list[str]) -> pd.DataFrame | None:
        """トピック語から外す。埋め込みとクラスタは変えない。"""
        for word in _flatten(words):
            if word not in self.ja_config.exclude:
                self.ja_config.exclude.append(word)
        self._replace_vectorizer(update=self._fitted)
        if not self._fitted:
            return None
        return self.get_topic_info()

    def refit(self, min_topic_size: int | None = None, nr_topics: int | str | None = None):
        """クラスタの粗さだけを変えて学習し直す。埋め込みは再利用する。"""
        self._check_fitted()
        if min_topic_size is not None:
            self.ja_config.min_topic_size = int(min_topic_size)
            self.min_topic_size = self.ja_config.min_topic_size
            self.hdbscan_model = self._make_hdbscan(self.ja_config.min_topic_size)
        if nr_topics is not None:
            self.ja_config.nr_topics = nr_topics
            self.nr_topics = nr_topics
        self.umap_model = self._make_umap(len(self._documents))
        topics, probs = super().fit_transform(self._documents, embeddings=self._embeddings)
        if self.ja_config.reduce_outliers:
            topics = self._apply_outlier_reduction(self._documents, topics)
        self._drop_blank_topic_words()
        return topics, probs

    def suggest(self) -> pd.DataFrame:
        """多くのトピックに出る語を、文書頻度とともに返す。"""
        self._check_fitted()
        topic_words = self.get_topics()
        topic_ids = [topic_id for topic_id in topic_words if topic_id != -1]
        in_topics: dict[str, int] = {}
        for topic_id in topic_ids:
            seen: set[str] = set()
            for word, _score in topic_words[topic_id]:
                if word and word not in seen:
                    seen.add(word)
                    in_topics[word] = in_topics.get(word, 0) + 1
        analyzer = self.vectorizer_model.build_analyzer()
        doc_counts: dict[str, int] = {}
        for document in self._documents:
            for token in set(analyzer(document)):
                doc_counts[token] = doc_counts.get(token, 0) + 1
        n_topics = max(len(topic_ids), 1)
        n_docs = max(len(self._documents), 1)
        rows = []
        for token in set(in_topics) | set(doc_counts):
            topic_count = in_topics.get(token, 0)
            rows.append(
                {
                    "token": token,
                    "n_topics": topic_count,
                    "topic_fraction": round(topic_count / n_topics, 3),
                    "doc_frequency": round(doc_counts.get(token, 0) / n_docs, 3),
                }
            )
        frame = pd.DataFrame(rows, columns=["token", "n_topics", "topic_fraction", "doc_frequency"])
        if frame.empty:
            return frame
        return frame.sort_values(["topic_fraction", "doc_frequency"], ascending=False).reset_index(drop=True)

    def explain(self, text: str) -> dict[str, list]:
        """1文について、残す語と落とした理由を返す。"""
        return explain_text(text, self.ja_config)

    def preview_exclude(self, words: list[str]) -> list[dict[str, Any]]:
        """除外したあとのトピック語を、設定を変えずに見る。"""
        self._check_fitted()
        before = _topic_words(self)
        old_exclude = list(self.ja_config.exclude)
        old_vectorizer = self.vectorizer_model
        old_representation = self.representation_model
        try:
            self.ja_config.exclude = old_exclude + [word for word in words if word not in old_exclude]
            self.vectorizer_model = build_vectorizer(self.ja_config)
            self.update_topics(
                self._documents,
                vectorizer_model=self.vectorizer_model,
                representation_model=old_representation,
            )
            self._drop_blank_topic_words()
            after = _topic_words(self)
        finally:
            self.ja_config.exclude = old_exclude
            self.vectorizer_model = old_vectorizer
            self.update_topics(
                self._documents,
                vectorizer_model=old_vectorizer,
                representation_model=old_representation,
            )
            self._drop_blank_topic_words()
        topic_ids = sorted(set(before) | set(after))
        return [
            {"topic": topic_id, "before": before.get(topic_id, []), "after": after.get(topic_id, [])}
            for topic_id in topic_ids
        ]

    def _prepare_documents(self, documents) -> list[str]:
        docs = ["" if document is None else str(document) for document in documents]
        if len(docs) < 3:
            raise ValueError("文書は3件以上必要です。")
        if self.ja_config.min_topic_size >= len(docs):
            raise ValueError("min_topic_size は文書数より小さくしてください。")
        if not self.ja_config.fillers:
            return docs
        return [strip_fillers(document, self.ja_config.fillers) for document in docs]

    def _replace_vectorizer(self, update: bool) -> None:
        self.vectorizer_model = build_vectorizer(self.ja_config)
        if not update:
            return
        self.update_topics(
            self._documents,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model,
        )
        self._drop_blank_topic_words()

    def _apply_outlier_reduction(self, documents: list[str], topics) -> list[int]:
        if topics is None or -1 not in set(topics):
            return topics
        updated = self.reduce_outliers(documents, topics)
        self.update_topics(
            documents,
            topics=updated,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model,
        )
        self._drop_blank_topic_words()
        return list(self.topics_)

    def _make_umap(self, n_docs: int) -> UMAP:
        neighbors = self.ja_config.n_neighbors
        if neighbors is None:
            neighbors = 5 if n_docs < 500 else 15
        neighbors = max(2, min(int(neighbors), max(n_docs - 1, 2)))
        n_components = min(5, neighbors - 1)
        options: dict[str, Any] = {
            "n_neighbors": neighbors,
            "n_components": n_components,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
            "init": "random",
            "n_jobs": 1,
            "compatibility_layout": True,
        }
        try:
            return UMAP(**options)
        except TypeError:
            options.pop("compatibility_layout")
            return UMAP(**options)

    def _make_hdbscan(self, min_topic_size: int) -> HDBSCAN:
        return HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=max(1, min_topic_size // 2),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

    def _drop_blank_topic_words(self) -> None:
        """空文字だけのトピック語を、そのトピックの名詞の出現順に戻す。

        KeyBERTInspired は候補語が少ないと空文字を足す。類似度で空文字が
        上位になると、トピック名が空白になる。
        """
        representations = getattr(self, "topic_representations_", None)
        if not representations:
            return
        repaired: dict[int, list[tuple[str, float]]] = {}
        for topic_id, pairs in representations.items():
            words = [(str(word), float(score)) for word, score in pairs if word]
            repaired[topic_id] = words or self._words_from_documents(topic_id)
        self.topic_representations_ = repaired

    def _words_from_documents(self, topic_id: int) -> list[tuple[str, float]]:
        analyzer = self.vectorizer_model.build_analyzer()
        counts: dict[str, int] = {}
        for document, topic in zip(self._documents, self.topics_):
            if topic != topic_id:
                continue
            for token in analyzer(document):
                if token:
                    counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [(word, float(count)) for word, count in ranked[:10]]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit_transform を先に呼んでください。")


def _flatten(words: tuple[str | list[str], ...]) -> list[str]:
    flat: list[str] = []
    for word in words:
        if isinstance(word, (list, tuple, set)):
            flat.extend(str(item) for item in word)
        else:
            flat.append(str(word))
    return flat


def _topic_words(model: BERTopic) -> dict[int, list[str]]:
    return {
        topic_id: [word for word, _score in words]
        for topic_id, words in model.get_topics().items()
    }
