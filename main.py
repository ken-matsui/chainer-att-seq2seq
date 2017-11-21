# coding: utf-8

import os
import argparse
from glob import glob
from os.path import relpath, splitext

from att_seq2seq.model import AttSeq2Seq
from att_seq2seq.trainer import Trainer
from att_seq2seq.decoder import Decoder

parser = argparse.ArgumentParser(description='This script is seq2seq with chainer. Show usable arguments below ...')
parser.add_argument('-p', '--parse', default=False, action='store_true', help='Parse mode if this flag is set')
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', default=False, action='store_true', help='Train mode if this flag is set')
group.add_argument('-d', '--decode', default=False, action='store_true', help='Decode mode if this flag is set')
parser.add_argument('-r', '--resume', default=False, action='store_true', help='Resume mode if this flag is set')
parser.add_argument('-s', '--select', type=int, default=0, action='store', metavar='N', help="Select npz file's number")
parser.add_argument('-g', '--gpu', default=False, action='store_true', help='GPU mode if this flag is set')
FLAGS = parser.parse_args()

EMBED_SIZE = 500
HIDDEN_SIZE = 500
BATCH_SIZE = 20
# デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
BATCH_COL_SIZE = 15
EPOCH_NUM = 500
DATA_PATH = './data/'
TRAIN_PATH = './train/'

def main():
	if FLAGS.parse:
		import parser
		parser.main()

	# 単語辞書の読み込み
	vocab = load_vocab(DATA_PATH)
	model = AttSeq2Seq(vocab_size=len(vocab),
					   embed_size=EMBED_SIZE,
					   hidden_size=HIDDEN_SIZE)

	if FLAGS.train:
		# 学習用データを読み込む
		queries, responses, teacher_num = load_queres(DATA_PATH)
		if FLAGS.resume:
			if FLAGS.select == 0:
				# 最新のモデルデータを使用する．
				files = [splitext(relpath(s, TRAIN_PATH))[0] for s in glob(TRAIN_PATH + "*.npz")]
				num = max(list(map(int, files)))
			else:
				# 指定のモデルデータを使用する．
				num = FLAGS.select
			npz = TRAIN_PATH + str(num) + ".npz"
			print("Resume learning from", npz)
		else:
			try:
				os.mkdir(TRAIN_PATH)
			except:
				pass
			print("Train")
			npz = None
		trainer = Trainer(model, npz)
		trainer.fit(queries=queries,
					responses=responses,
					teacher_num=teacher_num,
					train_path=TRAIN_PATH,
					epoch_num=EPOCH_NUM,
					batch_size=BATCH_SIZE,
					flag_gpu=FLAGS.gpu)
	elif FLAGS.decode:
		if FLAGS.select == 0:
			# 最新のモデルデータを使用する．
			files = [splitext(relpath(s, TRAIN_PATH))[0] for s in glob(TRAIN_PATH + "*.npz")]
			num = max(list(map(int, files)))
		else:
			# 指定のモデルデータを使用する．
			num = FLAGS.select
		npz = TRAIN_PATH + str(num) + ".npz"
		print("Interactive decode from", npz)
		decoder = Decoder(model=model,
						  npz=npz,
						  vocab=vocab,
						  decode_max_size=BATCH_COL_SIZE,
						  flag_gpu=FLAGS.gpu)
		while True:
			query = input("> ")
			print(decoder(query))

def load_vocab(path):
	# 単語辞書データを取り出す
	with open(path + 'vocab.txt', 'r') as f:
		lines = f.readlines()
	return list(map(lambda s: s.replace("\n", ""), lines))

def load_queres(path):
	# 対話データ(ID版)を取り出す
	with open(path + 'query_id.txt', 'r') as fqid, open(path + 'response_id.txt', 'r') as frid:
		queid, resid = fqid.readlines(), frid.readlines()
	teacher_num = len(list(zip(queid, resid)))
	queries, responses = [], []
	for q, r in zip(queid, resid):
		# ミニバッチ対応のため，単語数サイズを調整してNumpy変換する
		query = list(map(int, q.replace('\n', '').split(',')))
		queries.append(batch_ids(query))
		respo = list(map(int, q.replace('\n', '').split(',')))
		responses.append(batch_ids(respo))
	return queries, responses, teacher_num

def batch_ids(ids):
	if len(ids) > BATCH_COL_SIZE: # ミニバッチ単語サイズになるように先頭から削る
		del ids[0:len(ids) - BATCH_COL_SIZE]
	else: # ミニバッチ単語サイズになるように先頭に付け足す
		ids = ([-1] * (BATCH_COL_SIZE - len(ids))) + ids
	return ids

if __name__ == '__main__':
	main()
