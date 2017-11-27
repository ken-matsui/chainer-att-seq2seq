# coding: utf-8

import os
import argparse
from glob import glob
from os.path import relpath, splitext

from att_seq2seq.model import AttSeq2Seq
from att_seq2seq.trainer import Trainer
from att_seq2seq.decoder import Decoder

parser = argparse.ArgumentParser(description='This script is seq2seq with chainer.')
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', default=False, action='store_true',
				   help='Train mode if this flag is set')
group.add_argument('-d', '--decode', default=False, action='store_true',
				   help='Decode mode if this flag is set')
parser.add_argument('-p', '--parse', default=False, action='store_true',
					help='Parse mode if this flag is set')
parser.add_argument('-r', '--resume', type=int, default=-1,
					help="Select npz file's number. You can specify the latest model file by specifying 0.")
parser.add_argument('-e', '--epoch', type=int, default=500,
					help="Number of epoch")
parser.add_argument('-es', '--embed_size', type=int, default=500,
					help="Number of embed(vector) size")
parser.add_argument('-n', '--n_hidden', type=int, default=500,
					help="Number of hidden units")
parser.add_argument('-b', '--batchsize', type=int, default=20,
					help="Number of batch size")
parser.add_argument('-dm', '--decode_max_size', type=int, default=15,
					help="Number of decode max size") # デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
parser.add_argument('-v', '--vocab_file', default='./data/vocab.txt',
					help="Directory to vocab file")
parser.add_argument('-qi', '--queid_file', default='./data/query_id.txt',
					help="Directory to query file")
parser.add_argument('-ri', '--resid_file', default='./data/response_id.txt',
					help="Directory to response file")
parser.add_argument('-o', '--out', default='./result/',
					help="Directory to output the result")
parser.add_argument('--tag', default='',
					help="TAG")
parser.add_argument('-g', '--gpu', default=False, action='store_true',
					help='GPU mode if this flag is set')
FLAGS = parser.parse_args()


def main():
	if FLAGS.parse:
		import parser
		parser.main()

	# 単語辞書の読み込み
	vocab = load_vocab()
	model = AttSeq2Seq(vocab_size=len(vocab),
					   embed_size=FLAGS.embed_size,
					   hidden_size=FLAGS.n_hidden,
					   flag_gpu=FLAGS.gpu)

	if FLAGS.train:
		print('GPU: {}'.format(FLAGS.gpu))
		print('# Minibatch-size: {}'.format(FLAGS.batchsize))
		print('# n_hidden: {}'.format(FLAGS.n_hidden))
		print('# epoch: {}'.format(FLAGS.epoch))
		print('')
		# 学習用データを読み込む
		queries, responses = load_queres()
		if FLAGS.resume != -1:
			if FLAGS.resume == 0:
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
	elif FLAGS.decode:
		if FLAGS.select == 0:
			# 最新のモデルデータを使用する．
			files = [splitext(relpath(s, FLAGS.out))[0] for s in glob(FLAGS.out + "*.npz")]
			num = max(list(map(int, files)))
		else:
			# 指定のモデルデータを使用する．
			num = FLAGS.select
		npz = FLAGS.out + str(num) + ".npz"
		print("Interactive decode from", npz)
		decoder = Decoder(model=model,
						  npz=npz,
						  vocab=vocab,
						  decode_max_size=FLAGS.decode_max_size,
						  flag_gpu=FLAGS.gpu)
		while True:
			query = input("> ")
			print(decoder(query))

def load_vocab():
	# 単語辞書データを取り出す
	with open(FLAGS.vocab_file, 'r') as f:
		lines = f.readlines()
	return list(map(lambda s: s.replace("\n", ""), lines))

def load_queres():
	# 対話データ(ID版)を取り出す
	queries, responses = [], []
	with open(FLAGS.queid_file, 'r') as fq, open(FLAGS.resid_file, 'r') as fr:
		for q, r in zip(fq.readlines(), fr.readlines()):
			# ミニバッチ対応のため，単語数サイズを調整してNumpy変換する
			query = list(map(int, q.replace('\n', '').split(',')))
			queries.append(batch_ids(query))
			respo = list(map(int, q.replace('\n', '').split(',')))
			responses.append(batch_ids(respo))
	return queries, responses

def batch_ids(ids):
	if len(ids) > FLAGS.decode_max_size: # ミニバッチ単語サイズになるように先頭から削る
		del ids[0:len(ids) - FLAGS.decode_max_size]
	else: # ミニバッチ単語サイズになるように先頭に付け足す
		ids = ([-1] * (FLAGS.decode_max_size - len(ids))) + ids
	return ids

if __name__ == '__main__':
	main()
