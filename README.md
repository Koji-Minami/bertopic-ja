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

何も指定しないときの埋め込みは、BERTopic の `language="multilingual"`（`paraphrase-multilingual-MiniLM-L12-v2`）です。トピック語は Sudachi（分割モード C）の名詞です。1文字の名詞、代名詞、数詞は出しません。「えー」「まあ」のようなフィラーはトピック語から外し、本文には残します。「あの車」の「あの」は消しません。

トピックの粗さは `min_topic_size` です。UMAP と HDBSCAN は自分で組みません。

```python
topic_model = BERTopic(min_topic_size=8)
```

## 出てきた語を調整する

トピック語から外すだけなら、埋め込みはやり直しません。

```python
topic_model.exclude("残価", "月々")
topic_model.suggest()
topic_model.explain("残価設定って聞いてることあります？")
topic_model.preview_exclude(["残価"])
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
  - 残価
protect:
  - 金利
fillers: []
```

```python
topic_model = BERTopic.from_yaml("config.yaml", min_topic_size=8)
topic_model.to_yaml("config.yaml")
```

## 公式との違い

公式の `BERTopic(language="multilingual")` は日本語の文をベクトルにできます。トピック語を作るトークナイザは英語向けの空白区切りのままで、日本語では単語に切れません。このパッケージはそのトークナイザを Sudachi に置き換え、語の除外とクラスタの再調整をメソッドにしています。

分類器でも、特定の業界向けの分析パイプラインでもありません。ラベルの付いていない文書から話題を探すためのものです。

## 公開

ローカルでのビルドは次です。

```bash
uv build
```

PyPI へ出すときは、PyPI の Trusted Publishing に GitHub リポジトリを登録してからタグを切ります。このリポジトリのワークフローはテストの実行までです。
