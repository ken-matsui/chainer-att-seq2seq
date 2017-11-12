# coding: utf-8

import datetime
from chainer import optimizer, optimizers, serializers
import numpy as np

# OUTPUT_PATH = './train/model/seq2seq/EMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.npz'
OUTPUT_PATH = './train/EMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.npz'
FLAG_GPU = False

EMBED_SIZE = 300
HIDDEN_SIZE = 150

class Trainer(object):
	def __init__(self, model):
		self.model = model

	def fit(self, queries, responses, teacher_num, epoch_num=30, batch_size=40, plotting=False):
		# ネットワークファイルの読み込み
		#network = "./att_seq2seq_network/*******************network"
		#serializers.load_npz(network, self.model)

		opt = optimizers.Adam()
		opt.setup(self.model)
		opt.add_hook(optimizer.GradientClipping(5))
		# if FLAG_GPU:
		# 	self.model.to_gpu(0)
		self.model.reset()

		# 学習開始
		st = datetime.datetime.now()
		for epoch in range(epoch_num):
			# ミニバッチ学習
			perm = np.random.permutation(teacher_num) # ランダムな整数列リストを取得
			total_loss = 0
			for i in range(0, teacher_num, batch_size):
				enc_words = queries[perm[i:i+batch_size]]
				dec_words = responses[perm[i:i+batch_size]]
				self.model.reset()
				loss = self.model(enc_words=enc_words, dec_words=dec_words, train=True)
				loss.backward()
				loss.unchain_backward()
				total_loss += loss.data
				opt.update()
			output_path = "./train/{}_{}.network".format(epoch+1, total_loss)
			serializers.save_npz(output_path, self.model) # モデルの保存
			# if (epoch+1)%10 == 0: # 1epochがでかいので，毎epochで表示
			ed = datetime.datetime.now()
			print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
			st = datetime.datetime.now()