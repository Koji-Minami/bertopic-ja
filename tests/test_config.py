from bertopic_ja._config import Config


def test_yaml_values_are_overridden_by_arguments(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "min_topic_size: 10\nexclude:\n  - 天気\nprotect:\n  - レシピ\n",
        encoding="utf-8",
    )
    config = Config.from_yaml(path, {"min_topic_size": 8})
    assert config.min_topic_size == 8
    assert config.exclude == ["天気"]
    assert config.protect == ["レシピ"]
    assert config.embedding_model is None
    assert config.fillers == []


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("umap_n_neighbors: 5\n", encoding="utf-8")
    try:
        Config.from_yaml(path)
    except ValueError as error:
        assert "umap_n_neighbors" in str(error)
    else:
        raise AssertionError("未知の項目を受け付けました")


def test_round_trip_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    config = Config(min_topic_size=8, exclude=["天気"], nr_topics="auto")
    config.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded.min_topic_size == 8
    assert loaded.exclude == ["天気"]
    assert loaded.nr_topics == "auto"
    assert loaded.pos == ["名詞", "形容詞", "動詞"]
    assert loaded.stopwords == []


def test_yaml_pos_and_stopwords_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "pos:\n  - 名詞\nstopwords:\n  - こと\n",
        encoding="utf-8",
    )
    config = Config.from_yaml(path, {"pos": ["名詞", "動詞"]})
    assert config.pos == ["名詞", "動詞"]
    assert config.stopwords == ["こと"]
