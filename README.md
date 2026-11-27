# bertopic-ja

日本語の文書から、どんなトピックが話されているかを探索するなら [BERTopic](https://github.com/MaartenGr/BERTopic) です。埋め込み、UMAP、HDBSCAN、c-TF-IDF という今の探索的トピックモデルの手順を、日本語のトピック語が単語になる状態で始められます。

公式のトークナイザは空白で区切るため、分かち書きしない日本語では機能しません。トピック名が単語ではなく、文のかたまりになります。このパッケージはそこを Sudachi に置き換えた入口です。公式 BERTopic のサブプロジェクトではありません。手法と実装の本体は Maarten Grootendorst の BERTopic（論文: [arXiv:2203.05794](https://arxiv.org/abs/2203.05794)）です。

## インストール

```bash
uv add bertopic-ja
```

```bash
pip install bertopic-ja
```

## 使い方

```python
from bertopic_ja import BERTopic

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
topic_model.get_topic_info()
```

何も指定しないときの埋め込みは、BERTopic の `language="multilingual"`（`paraphrase-multilingual-MiniLM-L12-v2`）です。トピック語は Sudachi（分割モード C）の名詞・形容詞・動詞です。動詞と形容詞は正規化形で、活用を戻して出します。「使った」は「使う」、「高かった」は「高い」、「使いにくかった」は「使いにくい」です。可能動詞は Sudachi の正規化に合わせます。「書ける」は「書く」です。品詞は `pos` で指定します。`pos=["名詞"]` とすると名詞だけになります。「静か」のような形状詞は初期値に入っていません。`pos` に `"形状詞"` を足すと対象になります。

1文字の語、代名詞、数詞は出しません。「えー」「まあ」のようなフィラーはトピック語から外し、本文には残します。「あの本」の「あの」は消しません。`protect` に入れた語は、フィラーや1文字でも残します。「見る」「こと」「する」のような語は、初期値では残します。外すときは `exclude` か `stopwords` に、出てくる形を書きます。

トピックの粗さは `min_topic_size` です。UMAP と HDBSCAN は自分で組みません。

```python
topic_model = BERTopic(min_topic_size=8)
```

## 出てきた語を調整する

トピック語から外すだけなら、埋め込みはやり直しません。

```python
topic_model.exclude("天気", "気温")
topic_model.suggest()
topic_model.explain("このレシピって作ったことあります？")
topic_model.preview_exclude(["天気"])
```

クラスタの数や粗さを変えるときは、保存してある埋め込みで学習し直します。

```python
topic_model.refit(min_topic_size=8)
topic_model.refit(nr_topics=15)
```

本文からフィラーを消すとクラスタが変わるので、初期値では消しません。消すときだけ `fillers=` を渡します。

```python
topic_model = BERTopic(fillers=["えー", "あのー"])
```

品詞を変えるとき、最初から語を外すときは、次のように渡します。

```python
# 名詞だけにする
topic_model = BERTopic(pos=["名詞"])
# 出てくる形を指定して外す。する の正規化形は 為る
topic_model = BERTopic(stopwords=["こと", "為る"])
topic_model.exclude("見る")
```

別の埋め込みモデルにするときは、名前を一つ渡します。

```python
topic_model = BERTopic(embedding_model="cl-nagoya/ruri-v3-310m")
```

## 設定ファイル

引数と YAML は同じ項目です。両方あるときは引数が優先されます。

```yaml
min_topic_size: 10
nr_topics: auto
reduce_outliers: true
embedding_model: null
exclude:
  - 天気
protect:
  - レシピ
fillers: []
pos:
  - 名詞
  - 形容詞
  - 動詞
stopwords: []
```

`stopwords` を書かないときも空です。書いた語だけを外します。

```python
topic_model = BERTopic.from_yaml("config.yaml", min_topic_size=8)
topic_model.to_yaml("config.yaml")
```

## 公式との違い

公式の `BERTopic(language="multilingual")` は日本語の文をベクトルにできます。トピック語を作るトークナイザは英語向けの空白区切りのままで、日本語では単語に切れません。このパッケージはそのトークナイザを Sudachi に置き換え、品詞とストップワードを設定にし、語の除外とクラスタの再調整をメソッドにしています。

分類器でも、特定の業界向けの分析パイプラインでもありません。ラベルの付いていない文書から話題を探すためのものです。
