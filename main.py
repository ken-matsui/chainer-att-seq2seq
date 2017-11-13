# coding: utf-8

import argparse

import numpy as np
from chainer import cuda, serializers

from att_seq2seq.model import AttSeq2Seq
from att_seq2seq.trainer import Trainer
from data_utils.converter import DataConverter

# TODO: 引数なににするの？？書かないと．
parser = argparse.ArgumentParser(description='This script is seq2seq with chainer. Show usable arguments below ...')
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', default=False, action='store_true', help='Train mode if this flag is set (default: False)')
group.add_argument('-d', '--decode', default=False, action='store_true', help='Decode mode if this flag is set (default: False)')
parser.add_argument('-i', '--interact', default=False, action='store_true', help='Interact mode if this flag is set (default: False)')
parser.add_argument('-g', '--gpu', default=False, action='store_true', help='GPU mode if this flag is set (default: False)')
# parser.add_argument() # --decodeを指定した時だけ必須にしたい．かつ，stringを受け取りたい．
FLAGS = parser.parse_args()

# TODO: 色々なところに散乱してるGPU系どうするの？？
# GPUのセット
# GPUを使用するかどうか
if FLAGS.gpu: # numpyかcuda.cupyか
	xp = cuda.cupy
	cuda.get_device(0).use()
else:
	xp = np

# 定数
EMBED_SIZE = 100
HIDDEN_SIZE = 100
BATCH_SIZE = 6 # ミニバッチ学習のバッチサイズ数
BATCH_COL_SIZE = 15
EPOCH_NUM = 70 # エポック数

def main():
	# TODO: いろいろなところに散乱してる，ディレクトリ必要系，try expect passで作らなくていいの？？
	# 教師データ
	DATA_PATH = './data/'
	with open(DATA_PATH+'input.txt', 'r') as fin, open(DATA_PATH+'output.txt', 'r') as fout:
		inp, out = fin.readlines(), fout.readlines()
	data = list(map(lambda l: list(l), list(zip(inp, out))))

	teacher_num = len(data) # 教師データの数

	# 教師データの読み込み
	data_converter = DataConverter(batch_col_size=BATCH_COL_SIZE) # データコンバーター
	data_converter.load(data) # 教師データ読み込み
	vocab_size = len(data_converter.vocab) # 単語数

	# モデルの宣言
	model = AttSeq2Seq(vocab_size=vocab_size,
					   embed_size=EMBED_SIZE,
					   hidden_size=HIDDEN_SIZE,
					   batch_col_size=BATCH_COL_SIZE)

	if FLAGS.train:
		print("Train")
		trainer = Trainer(model)
		trainer.fit(queries=data_converter.train_queries,
					responses=data_converter.train_responses,
					teacher_num=teacher_num,
					epoch_num=EPOCH_NUM,
					batch_size=BATCH_SIZE)
	elif FLAGS.decode:
		# ネットワークファイルの読み込み
		network = "./train/70.npz"
		serializers.load_npz(network, model)
		# Decode開始
		print("Predict")
		def predict(model, query):
			enc_query = data_converter.sentence2ids(query, train=False)
			dec_response = model(enc_words=enc_query, train=False)
			response = data_converter.ids2words(dec_response)
			if "<eos>" in response:
				for res in response[0:-1]: # 最後の<eos>を回避
					print(res, end="")
			else: # 含んでない時もある．(出力wordサイズが，15を超えた時？？？)
				for res in response:
					print(res, end="")
			print()

		while True:
			query = input("> ")
			predict(model, query)

if __name__ == '__main__':
	main()
