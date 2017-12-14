# coding: utf-8

import os
import argparse
from glob import glob
from os.path import relpath, splitext

from att_seq2seq.model import AttSeq2Seq
from att_seq2seq.trainer import Trainer

parser = argparse.ArgumentParser(description='This is a script to train seq2seq.')
parser.add_argument('-e', '--epoch', type=int, default=200,
					help="Number of epoch")
parser.add_argument('-r', '--resume', default=False, action='store_true',
					help="Resume mode if this flag is set")
parser.add_argument('-b', '--batchsize', type=int, default=20,
					help="Number of batch size")
parser.add_argument('-es', '--embed_size', type=int, default=100,
					help="Number of embed(vector) size")
parser.add_argument('-n', '--n_hidden', type=int, default=100,
					help="Number of hidden units")
parser.add_argument('-d', '--decode_max_size', type=int, default=15,
					help="Number of decode max size") # デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
parser.add_argument('-v', '--vocab_file', default='./data/vocab.txt',
					help="Directory to vocab file")
parser.add_argument('-i', '--infile', default='./data/dataid.txt',
					help="Directory to id file")
parser.add_argument('-o', '--out', default='./result/',
					help="Directory to output the result")
parser.add_argument('-s', '--select', type=int, default=0,
					help="Select npz file.")
parser.add_argument('-t', '--tag', default=None, help="TAG")
parser.add_argument('-g', '--gpu', default=False, action='store_true',
					help='GPU mode if this flag is set')
FLAGS = parser.parse_args()


def main():
	print('GPU: {}'.format(FLAGS.gpu))
	print('# Minibatch-size: {}'.format(FLAGS.batchsize))
	print('# embed_size: {}'.format(FLAGS.embed_size))
	print('# n_hidden: {}'.format(FLAGS.n_hidden))
	print('# epoch: {}'.format(FLAGS.epoch))
	print()

	# 単語辞書の読み込み
	vocab = load_vocab()
	model = AttSeq2Seq(vocab_size=len(vocab),
					   embed_size=FLAGS.embed_size,
					   hidden_size=FLAGS.n_hidden,
					   flag_gpu=FLAGS.gpu)
	# 学習用データを読み込む
	queries, responses = load_ids()
	if FLAGS.resume:
		if FLAGS.select == 0:
			# 最新のモデルデータを使用する．
			files = [splitext(relpath(s, FLAGS.out))[0] for s in glob(FLAGS.out + "*.npz")]
			num = max(list(map(int, files)))
		else:
			# 指定のモデルデータを使用する．
			num = FLAGS.select
		npz = FLAGS.out + str(num) + ".npz"
		print("Resume training from", npz)
	else:
		try:
			os.mkdir(FLAGS.out)
		except:
			pass
		print("Train")
		npz = None
	trainer = Trainer(model, npz, FLAGS.gpu)
	trainer.fit(queries=queries,
				responses=responses,
				train_path=FLAGS.out,
				epoch_num=FLAGS.epoch,
				batch_size=FLAGS.batchsize,
				tag=FLAGS.tag)

def load_vocab():
	# 単語辞書データを取り出す
	with open(FLAGS.vocab_file, 'r') as f:
		lines = f.readlines()
	return list(map(lambda s: s.replace("\n", ""), lines))

def load_ids():
	# 対話データ(ID版)を取り出す
	queries, responses = [], []
	with open(FLAGS.infile, 'r') as f:
		for l in f.read().split('\n')[:-1]:
			# queryとresponseで分割する
			d = l.split('\t')
			# ミニバッチ対応のため，単語数サイズを調整してNumpy変換する
			queries.append(batch_ids(list(map(int, d[0].split(',')[:-1])), "query"))
			responses.append(batch_ids(list(map(int, d[1].split(',')[:-1])), "response"))
	return queries, responses

def batch_ids(ids, sentence_type):
	if sentence_type == "query": # queryの場合は前方に-1を補填する
		if len(ids) > FLAGS.decode_max_size: # ミニバッチ単語サイズになるように先頭から削る
			del ids[0:len(ids) - FLAGS.decode_max_size]
		else: # ミニバッチ単語サイズになるように前方に付け足す
			ids = ([-1] * (FLAGS.decode_max_size - len(ids))) + ids
	elif sentence_type == "response": # responseの場合は後方に-1を補填する
		if len(ids) > FLAGS.decode_max_size: # ミニバッチ単語サイズになるように末尾から削る
			del ids[FLAGS.decode_max_size:]
		else: # ミニバッチ単語サイズになるように後方に付け足す
			ids = ids + ([-1] * (FLAGS.decode_max_size - len(ids)))
	return ids

if __name__ == '__main__':
	main()
