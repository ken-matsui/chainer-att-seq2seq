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
epoch: 1	tag: bigdata
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
[日本語自然会話書き起こしコーパス（旧名大会話コーパス）](http://mmsrv.ninjal.ac.jp/nucc/)を使用
その後，parseには，[make-meidai-dialogue](https://github.com/knok/make-meidai-dialogue)を使用．
その`sequence.txt`ファイルだけ，`raw/corpus/`に移動させて使用する．

MeCabの辞書は，[mecab-ipadic-neologd](https://github.com/neologd/mecab-ipadic-neologd)を使用．

# CPP

mecabのインストールは，[mecab-ipadic-neologd](https://github.com/neologd/mecab-ipadic-neologd)を参考にインストールしてください．

:warning: -lboost_filesystemオプションは，昔の記事だと，-lboost-filesystemとなっている場合が多いですが，-lboost_filesystemが正しいです．  
  
**Ex. compile options**
```
$ g++ -std=c++1z -O3 -mtune=native -march=native -I/usr/local/Cellar/boost/1.65.1 -lboost_filesystem -lboost_system `mecab-config --cflags` `mecab-config --libs` -o parse parse.cpp
```

**速度検証**
```
$ instruments -s
...
Known Templates:
"Activity Monitor"
"Allocations"
"Blank"
"Cocoa Layout"
"Core Animation"
"Core Data"
"Counters"
"Energy Log"
"File Activity"
"Leaks"
"Metal System Trace"
"Network"
"SceneKit"
"System Trace"
"System Usage"
"Time Profiler"
"Zombies"
$ instruments -t "Time Profiler" -l 10000 ./parse
$ open ./instrumentscli0.trace
```
