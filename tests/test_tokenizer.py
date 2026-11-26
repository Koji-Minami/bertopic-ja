from bertopic_ja._config import Config
from bertopic_ja._tokenizer import explain_text, strip_fillers


def test_topic_words_keep_nouns_and_drop_fillers():
    explained = explain_text("えー、あの車は病院です", Config())
    assert "病院" in explained["kept"]
    dropped = {item["token"]: item["reason"] for item in explained["dropped"]}
    assert "えー" in dropped
    assert dropped["えー"] in {"フィラー", "感動詞"}
    assert "は" in dropped
    assert dropped["車"] == "1文字"
    assert "病院" not in dropped


def test_fillers_do_not_eat_demonstratives():
    cleaned = strip_fillers("えー、あの車は高い", ["えー", "あの"])
    assert "えー" not in cleaned
    assert "車" in cleaned
    assert "あの" in cleaned
