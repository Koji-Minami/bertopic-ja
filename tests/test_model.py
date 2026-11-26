import numpy as np

from bertopic_ja import BERTopic
from tests.fake_embedder import HashEmbedder


def _documents() -> list[str]:
    cats = [f"猫の病院とペットフードの話 {i}" for i in range(8)]
    cars = [f"車のローンと残価設定の話 {i}" for i in range(8)]
    return cats + cars


def _fitted() -> BERTopic:
    model = BERTopic(
        embedding_model=HashEmbedder(),
        min_topic_size=4,
        n_neighbors=5,
        reduce_outliers=False,
    )
    model.fit_transform(_documents())
    return model


def test_exclude_keeps_the_japanese_analyzer():
    model = _fitted()
    documents = _documents()
    model.exclude("ローン")
    tokens = model.vectorizer_model.build_analyzer()("このローンは病院である")
    assert "ローン" not in tokens
    assert "病院" in tokens
    assert "は" not in tokens
    assert model.embedding_model.calls[0] == documents
    assert documents not in model.embedding_model.calls[1:]


def test_refit_reuses_embeddings():
    model = _fitted()
    documents = _documents()
    model.refit(min_topic_size=4)
    assert model.embedding_model.calls[0] == documents
    assert documents not in model.embedding_model.calls[1:]
    assert model.topics_ is not None
    assert len(model.topics_) == 16


def test_from_yaml_override_reaches_the_model(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("min_topic_size: 6\nexclude:\n  - 残価\n", encoding="utf-8")
    model = BERTopic.from_yaml(path, min_topic_size=4, embedding_model=HashEmbedder(), n_neighbors=5, reduce_outliers=False)
    assert model.ja_config.min_topic_size == 4
    assert model.ja_config.exclude == ["残価"]
    model.fit_transform(_documents())
    tokens = model.vectorizer_model.build_analyzer()("残価の車")
    assert "残価" not in tokens


def test_explain_uses_current_exclude_list():
    model = BERTopic(embedding_model=HashEmbedder(), exclude=["車"])
    explained = model.explain("あの車は高い")
    assert "車" not in explained["kept"]
    reasons = {item["token"]: item["reason"] for item in explained["dropped"]}
    assert reasons["車"] == "除外"


def test_topic_words_are_content_nouns():
    model = _fitted()
    words = [
        word
        for topic_id in model.get_topics()
        if topic_id != -1
        for word, _score in model.get_topic(topic_id)
    ]
    assert words
    assert all(words)
    assert "病院" in words
    assert "ローン" in words


def test_embeddings_follow_the_documents():
    embedder = HashEmbedder()
    vectors = embedder.embed(["猫", "車"])
    assert np.argmax(vectors[0]) == 0
    assert np.argmax(vectors[1]) == 1
