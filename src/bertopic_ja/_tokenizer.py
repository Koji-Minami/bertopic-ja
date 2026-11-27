"""Sudachi によるトピック語の抽出。

初期値では名詞・形容詞・動詞を正規化形で残す。フィラーはトピック語から外す。
本文からの削除は fillers を渡したときだけで、そのときもトークン単位で行い、
名詞・動詞・形容詞・連体詞は残す。
"""

from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import CountVectorizer

from bertopic_ja._config import Config

# 感動詞として取りこぼしたときのための完全一致。指示語の「あの」は含めない。
FILLER_TOKENS = frozenset(
    {
        "えー",
        "えーっと",
        "えーと",
        "あのー",
        "あのぉ",
        "まあ",
        "まぁ",
        "まー",
    }
)

# 本文からフィラーを消すとき、この品詞は残す。「あの本」の「あの」は連体詞。
_KEEP_WHEN_STRIPPING = frozenset({"名詞", "動詞", "形容詞", "連体詞"})

# トピック語に使う品詞。Sudachi / UniDic の品詞大分類。
DEFAULT_POS: tuple[str, ...] = ("名詞", "形容詞", "動詞")

# 形式名詞と、軽い動詞・補助的な動詞。
# 表層・正規化形・辞書形のいずれかが一致したら落とす。
# 正規化で別表記になる語（する→為る、ある→有る、いる→居る）は両方入れておく。
DEFAULT_STOPWORDS: tuple[str, ...] = (
    "こと",
    "事",
    "もの",
    "もん",
    "ため",
    "よう",
    "ところ",
    "とこ",
    "とき",
    "の",
    "ほう",
    "方",
    "わけ",
    "はず",
    "つもり",
    "まま",
    "とおり",
    "ゆえ",
    "せい",
    "する",
    "為る",
    "ある",
    "有る",
    "在る",
    "なる",
    "成る",
    "いる",
    "おる",
    "居る",
    "できる",
    "出来る",
    "れる",
    "られる",
    "せる",
    "させる",
)

_NOUN_SUBCATEGORIES_TO_DROP = frozenset({"代名詞", "数詞"})


@lru_cache(maxsize=1)
def tokenizer():
    """Sudachi のタグガーと分割モード C を返す。"""
    try:
        from sudachipy import Dictionary, SplitMode
    except ImportError:  # sudachipy 0.6
        from sudachipy.dictionary import Dictionary
        from sudachipy.tokenizer import Tokenizer

        SplitMode = Tokenizer.SplitMode

    dictionary = Dictionary()
    tagger = dictionary.create() if hasattr(dictionary, "create") else dictionary.tokenizer()
    return tagger, SplitMode.C


def _readings(morpheme) -> tuple[str, str, str]:
    surface = morpheme.surface()
    try:
        normalized = morpheme.normalized_form()
    except Exception:
        normalized = surface
    if not normalized:
        normalized = surface
    try:
        dictionary = morpheme.dictionary_form()
    except Exception:
        dictionary = normalized
    if not dictionary:
        dictionary = normalized
    return surface, normalized, dictionary


def _forms(morpheme) -> tuple[str, str]:
    surface, normalized, _dictionary = _readings(morpheme)
    return surface, normalized


def iter_morphemes(text: str):
    if not text or not str(text).strip():
        return
    tagger, mode = tokenizer()
    for morpheme in tagger.tokenize(str(text), mode):
        yield morpheme


def strip_fillers(text: str, fillers: list[str]) -> str:
    """指定したフィラーを、トークン単位で本文から外す。"""
    targets = set(fillers)
    if not targets or not text:
        return "" if text is None else str(text)
    kept: list[str] = []
    for morpheme in iter_morphemes(text):
        surface, normalized = _forms(morpheme)
        pos = morpheme.part_of_speech()[0]
        listed = surface in targets or normalized in targets
        if listed and pos not in _KEEP_WHEN_STRIPPING:
            continue
        kept.append(surface)
    return "".join(kept)


def _pos(morpheme) -> tuple[str, ...]:
    pos = morpheme.part_of_speech()
    return tuple(pos) if pos else ()


def _is_adjectival_suffix(morpheme) -> bool:
    pos = _pos(morpheme)
    return len(pos) > 1 and pos[0] == "接尾辞" and pos[1] == "形容詞的"


def _hosts_adjectival_suffix(morpheme) -> bool:
    pos = _pos(morpheme)
    return bool(pos) and pos[0] in {"動詞", "名詞", "形容詞"}


def _compound_readings(host, suffix) -> tuple[str, str, str, tuple[str, ...]]:
    """「にくい」「やすい」などを、前の語とつないで一語の形容詞にする。

    Sudachi は「使いにくかった」を「使う」と「難い」に分ける。トピック語は
    活用を戻した「使いにくい」にする。動詞の語幹は連用形の表層、それ以外は
    正規化形を使う。
    """
    host_surface, host_normalized, _host_dictionary = _readings(host)
    suffix_surface, _suffix_normalized, suffix_dictionary = _readings(suffix)
    host_pos = _pos(host)
    stem = host_surface if host_pos and host_pos[0] == "動詞" else host_normalized
    lemma = f"{stem}{suffix_dictionary}"
    surface = f"{host_surface}{suffix_surface}"
    return surface, lemma, lemma, ("形容詞", "一般", "*", "*", "形容詞", "終止形-一般")


def _iter_tokens(text: str):
    """トピック語の候補を、(表層, 正規化形, 辞書形, 品詞) で返す。"""
    morphemes = list(iter_morphemes(text))
    index = 0
    while index < len(morphemes):
        current = morphemes[index]
        nxt = morphemes[index + 1] if index + 1 < len(morphemes) else None
        if nxt is not None and _hosts_adjectival_suffix(current) and _is_adjectival_suffix(nxt):
            yield _compound_readings(current, nxt)
            index += 2
            continue
        surface, normalized, dictionary = _readings(current)
        yield surface, normalized, dictionary, _pos(current)
        index += 1


def explain_text(text: str, config: Config) -> dict[str, list]:
    """1文について、残す語と落とした理由を返す。"""
    kept, dropped = _collect(
        text,
        frozenset(config.protect),
        frozenset(config.exclude),
        frozenset(config.pos),
        frozenset(config.stopwords),
    )
    return {"kept": kept, "dropped": dropped}


def _collect(
    text: str,
    protect: frozenset[str],
    exclude: frozenset[str],
    allowed_pos: frozenset[str],
    stopwords: frozenset[str],
) -> tuple[list[str], list[dict[str, str]]]:
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    for surface, token, dictionary, pos in _iter_tokens(text):
        reason = _drop_reason(
            token,
            surface,
            dictionary,
            pos,
            protect,
            exclude,
            allowed_pos,
            stopwords,
        )
        if reason is None:
            kept.append(token)
        else:
            dropped.append({"token": surface, "reason": reason})
    return kept, dropped


def _drop_reason(
    token: str,
    surface: str,
    dictionary: str,
    pos: tuple[str, ...],
    protect: frozenset[str],
    exclude: frozenset[str],
    allowed_pos: frozenset[str],
    stopwords: frozenset[str],
) -> str | None:
    if token in protect or surface in protect:
        return None
    if token in FILLER_TOKENS or surface in FILLER_TOKENS:
        return "フィラー"
    pos_main = pos[0] if pos else ""
    pos_sub = pos[1] if len(pos) > 1 else ""
    if pos_main == "感動詞":
        return "感動詞"
    if token in exclude or surface in exclude:
        return "除外"
    if pos_main not in allowed_pos:
        return pos_main or "品詞"
    if any("非自立" in part for part in pos):
        return "非自立"
    if stopwords.intersection((token, surface, dictionary)):
        return "ストップワード"
    if pos_main == "名詞" and pos_sub in _NOUN_SUBCATEGORIES_TO_DROP:
        return pos_sub
    if len(token) <= 1:
        return "1文字"
    if token.isdigit():
        return "数"
    return None


class _TopicWordAnalyzer:
    """CountVectorizer の analyzer。クロージャにしないので pickle できる。"""

    def __init__(
        self,
        protect: frozenset[str] | set[str],
        exclude: frozenset[str] | set[str],
        pos: frozenset[str] | set[str],
        stopwords: frozenset[str] | set[str],
    ) -> None:
        self.protect = frozenset(protect)
        self.exclude = frozenset(exclude)
        self.pos = frozenset(pos)
        self.stopwords = frozenset(stopwords)

    def __call__(self, text: str) -> list[str]:
        kept, _dropped = _collect(text, self.protect, self.exclude, self.pos, self.stopwords)
        return kept


def build_vectorizer(config: Config) -> CountVectorizer:
    """トピック語専用の CountVectorizer。クラスタ分けには使わない。"""
    analyzer = _TopicWordAnalyzer(config.protect, config.exclude, config.pos, config.stopwords)
    return CountVectorizer(analyzer=analyzer, token_pattern=None)
