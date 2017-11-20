# coding: utf-8

import argparse
import glob

from att_seq2seq.model import AttSeq2Seq
from att_seq2seq.trainer import Trainer
from att_seq2seq.decoder import Decoder
from data_utils.converter import DataConverter

parser = argparse.ArgumentParser(description='This script is seq2seq with chainer. Show usable arguments below ...')
group = parser.add_mutually_exclusive_group()
group.add_argument('-t', '--train', default=False, action='store_true', help='Train mode if this flag is set')
group.add_argument('-d', '--decode', default=False, action='store_true', help='Decode mode if this flag is set')
parser.add_argument('-g', '--gpu', default=False, action='store_true', help='GPU mode if this flag is set')
parser.add_argument('-r', '--resume', default=False, action='store_true', help='Resume mode if this flag is set')
# parser.add_argument() # TODO: --decodeを指定した時だけ必須にしたい．かつ，stringを受け取りたい．
# TODO: 指定がなければ，最新のモデルファイルを使用したい．
FLAGS = parser.parse_args()

EMBED_SIZE = 100
HIDDEN_SIZE = 100
BATCH_SIZE = 20
# デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
BATCH_COL_SIZE = 15
EPOCH_NUM = 500
DATA_PATH = './data/'
TRAIN_PATH = './train/'

def main():
	data_converter = DataConverter(BATCH_COL_SIZE)
	data_converter.load(DATA_PATH)
	vocab_size = len(data_converter.vocab)

	model = AttSeq2Seq(vocab_size=vocab_size,
					   embed_size=EMBED_SIZE,
					   hidden_size=HIDDEN_SIZE)

	if FLAGS.train:
		if FLAGS.resume:
			# 最新のモデルデータを使用する．
			files = glob.glob(TRAIN_PATH + "*.npz")
			num = max(list(map(lambda s: int(s.replace(TRAIN_PATH, "").replace(".npz", "")), files)))
			npz = TRAIN_PATH + str(num) + ".npz"
			print("Resume learning from", npz)
		else:
			print("Train")
			npz = None
		trainer = Trainer(model, npz)
		trainer.fit(queries=data_converter.train_queries,
					responses=data_converter.train_responses,
					teacher_num=data_converter.teacher_num,
					epoch_num=EPOCH_NUM,
					batch_size=BATCH_SIZE,
					flag_gpu=FLAGS.gpu)
	elif FLAGS.decode:
		# 最新のモデルデータを使用する．
		files = glob.glob(TRAIN_PATH + "*.npz")
		num = max(list(map(lambda s: int(s.replace(TRAIN_PATH, "").replace(".npz", "")), files)))
		npz = TRAIN_PATH + str(num) + ".npz"
		print("Interactive decode from", npz)
		decoder = Decoder(model=model,
						  npz=npz,
						  decode_max_size=BATCH_COL_SIZE,
						  flag_gpu=FLAGS.gpu)
		while True:
			query = input("> ")
			print(decoder(query))

if __name__ == '__main__':
	main()
