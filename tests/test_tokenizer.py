import pickle

from bertopic_ja._config import Config
from bertopic_ja._tokenizer import DEFAULT_STOPWORDS, build_vectorizer, explain_text, strip_fillers


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


def test_formal_nouns_and_light_verbs_are_dropped():
    explained = explain_text("レシピを見ることができる。計画をすることになった。", Config())
    assert explained["kept"] == ["レシピ", "計画"]
    dropped = {item["token"]: item["reason"] for item in explained["dropped"]}
    assert dropped["こと"] == "ストップワード"
    assert dropped["見る"] == "非自立"
    assert dropped["できる"] == "非自立"
    assert dropped["する"] == "非自立"
    assert dropped["なっ"] == "非自立"

    forms = explain_text("確認のためのものを部屋にいることがある", Config())
    assert forms["kept"] == ["確認", "部屋"]
    reasons = {item["token"]: item["reason"] for item in forms["dropped"]}
    assert reasons["ため"] == "ストップワード"
    assert reasons["もの"] == "ストップワード"
    assert reasons["いる"] == "非自立"
    assert reasons["ある"] == "非自立"

    grammatical = explain_text("そのつもりでそのとおりに作った", Config())
    assert grammatical["kept"] == ["作る"]
    grammatical_reasons = {item["token"]: item["reason"] for item in grammatical["dropped"]}
    assert grammatical_reasons["つもり"] == "ストップワード"
    assert grammatical_reasons["とおり"] == "ストップワード"


def test_stopwords_can_be_replaced_or_extended():
    replaced = explain_text("作ったことがある", Config(stopwords=[]))
    assert "こと" in replaced["kept"]
    assert "作る" in replaced["kept"]
    assert "有る" not in replaced["kept"]

    extended = explain_text("病院のレシピ", Config(stopwords=[*DEFAULT_STOPWORDS, "病院"]))
    assert extended["kept"] == ["レシピ"]


def test_protect_and_exclude_still_apply_with_stopwords():
    explained = explain_text(
        "製品を作ったことがある",
        Config(protect=["こと"], exclude=["製品"]),
    )
    assert "こと" in explained["kept"]
    assert "作る" in explained["kept"]
    reasons = {item["token"]: item["reason"] for item in explained["dropped"]}
    assert reasons["製品"] == "除外"
    assert "こと" not in reasons


def test_analyzer_is_picklable():
    vectorizer = build_vectorizer(Config())
    restored = pickle.loads(pickle.dumps(vectorizer))
    text = "使いにくかった製品は高かった"
    assert restored.build_analyzer()(text) == vectorizer.build_analyzer()(text)
