# python_att-seq2seq

## HOW TO USE
```
$ git clone $(This repository's URL)
$ cd seq2seq/
$ mv ~/Downloads/facebook-$(USER) ./raw/facebook
$ mv ~/Downloads/\[LINE\]\ Chat\ with\ *.txt ./raw/line/
```

### Input data
```
data = [["query data", "responce data"],
	["query data", "responce data"],
	[..., ...], ...]
```

### Important
parser.pyだけmainと切り離されている．

同一人物の連続した発話は除外

[日本語自然会話書き起こしコーパス（旧名大会話コーパス）](http://mmsrv.ninjal.ac.jp/nucc/)を使用
その後，parseには，[make-meidai-dialogue](https://github.com/knok/make-meidai-dialogue)を使用．
その`sequence.txt`ファイルだけ，`raw/corpus/`に移動させて使用する．

MeCabの辞書は，[mecab-ipadic-neologd](https://github.com/neologd/mecab-ipadic-neologd)を使用．

```
$ mv ./make-meidai-dialogue/sequence.txt ./raw/corpus/
```
