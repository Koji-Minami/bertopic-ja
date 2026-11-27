import pickle

from bertopic_ja._config import Config
from bertopic_ja._tokenizer import build_vectorizer, explain_text, strip_fillers


def test_topic_words_keep_nouns_and_drop_fillers():
    explained = explain_text("えー、あの本は病院です", Config())
    assert "病院" in explained["kept"]
    dropped = {item["token"]: item["reason"] for item in explained["dropped"]}
    assert "えー" in dropped
    assert dropped["えー"] in {"フィラー", "感動詞"}
    assert "は" in dropped
    assert dropped["本"] == "1文字"
    assert "病院" not in dropped


def test_fillers_do_not_eat_demonstratives():
    cleaned = strip_fillers("えー、あの本は高い", ["えー", "あの"])
    assert "えー" not in cleaned
    assert "本" in cleaned
    assert "あの" in cleaned


def test_default_pos_emits_verb_and_adjective_lemmas():
    explained = explain_text("使いにくかった製品は高かった。新しい道具を使った。", Config())
    assert explained["kept"] == ["使いにくい", "製品", "高い", "新しい", "道具", "使う"]


def test_nouns_only_pos_drops_verbs_and_adjectives():
    explained = explain_text("高い病院で使った", Config(pos=["名詞"]))
    assert explained["kept"] == ["病院"]
    dropped = {item["token"]: item["reason"] for item in explained["dropped"]}
    assert dropped["高い"] == "形容詞"
    assert dropped["使っ"] == "動詞"


def test_default_keeps_verbs_and_formal_nouns():
    explained = explain_text("良い方法を見て行く。レシピを見ることができる。", Config())
    assert explained["kept"] == ["良い", "方法", "見る", "行く", "レシピ", "見る", "こと", "出来る"]
    assert Config().stopwords == []


def test_exclude_and_stopwords_drop_only_listed_words():
    excluded = explain_text("良い方法を見て行く", Config(exclude=["見る"]))
    assert excluded["kept"] == ["良い", "方法", "行く"]
    reasons = {item["token"]: item["reason"] for item in excluded["dropped"]}
    assert reasons["見"] == "除外"

    stopped = explain_text("作ったことがある", Config(stopwords=["こと"]))
    assert stopped["kept"] == ["作る", "有る"]
    stopped_reasons = {item["token"]: item["reason"] for item in stopped["dropped"]}
    assert stopped_reasons["こと"] == "ストップワード"


def test_protect_keeps_fillers_and_overrides_stopwords():
    explained = explain_text("えー、病院です", Config(protect=["えー"]))
    assert "えー" in explained["kept"]
    assert "病院" in explained["kept"]

    kept = explain_text("製品を作ったことがある", Config(protect=["こと"], exclude=["製品"], stopwords=["こと"]))
    assert "こと" in kept["kept"]
    assert "作る" in kept["kept"]
    reasons = {item["token"]: item["reason"] for item in kept["dropped"]}
    assert reasons["製品"] == "除外"
    assert "こと" not in reasons


def test_optional_function_words_use_normalized_forms():
    words = ["為る", "有る", "成る", "居る", "こと", "物", "為"]
    explained = explain_text("したことがあり、部屋にいた。見たもののためになった。", Config(exclude=words))
    assert explained["kept"] == ["部屋", "見る"]


def test_default_pos_keeps_na_adjectives():
    explained = explain_text("丁寧で簡単な静かな説明", Config())
    assert explained["kept"] == ["丁寧", "簡単", "静か", "説明"]


def test_analyzer_is_picklable():
    vectorizer = build_vectorizer(Config())
    restored = pickle.loads(pickle.dumps(vectorizer))
    text = "使いにくかった製品は高かった"
    assert restored.build_analyzer()(text) == vectorizer.build_analyzer()(text)
