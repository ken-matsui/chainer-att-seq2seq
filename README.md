# chainer_att-seq2seq

## HOW TO USE
```
$ git clone $(This repository's URL)
$ cd seq2seq/

$ mv ~/Downloads/facebook-$(USER) ./raw/facebook
$ mkdir ./raw/line
$ mv ~/Downloads/\[LINE\]\ Chat\ with\ *.txt ./raw/line/
$ mkdir ./raw/corpus
$ mv ~/Downloads/make-meidai-dialogue/sequence.txt ./raw/corpus/

$ python parse.py
Parse facebook...
Parse line...
Parse corpus...
done.
$ python train.py
GPU: True
# Minibatch-size: 20
# embed_size: 100
# n_hidden: 100
# epoch: 200

Train
epoch: 1	tag: big
	loss: 108549.05
	accuracy: 3658.35
	time: 0:04:13.800777
.
.
.
$ python decode.py
Interactive decode from ./result/30.npz
> お元気ですか？
元気です
>
```

### Input data
```
data = [["query data", "responce data"],
	["query data", "responce data"],
	[..., ...], ...]
```

## Important
parser.pyだけmainと切り離されている．

同一人物の連続した発話は除外

[日本語自然会話書き起こしコーパス（旧名大会話コーパス）](http://mmsrv.ninjal.ac.jp/nucc/)を使用
その後，parseには，[make-meidai-dialogue](https://github.com/knok/make-meidai-dialogue)を使用．
その`sequence.txt`ファイルだけ，`raw/corpus/`に移動させて使用する．

MeCabの辞書は，[mecab-ipadic-neologd](https://github.com/neologd/mecab-ipadic-neologd)を使用．