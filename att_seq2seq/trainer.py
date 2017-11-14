# coding: utf-8

import datetime
import re

from chainer import optimizer, optimizers, serializers
import numpy as np

FLAG_GPU = False

class Trainer(object):
	def __init__(self, model, npz=None):
		self.model = model
		if npz is not None:
			serializers.load_npz(npz, self.model)
			self.npz_num = int(re.search(r'[0-9]+', npz).group())
		else:
			self.npz_num = 0

	def fit(self, queries, responses, teacher_num, epoch_num=30, batch_size=40):
		opt = optimizers.Adam()
		opt.setup(self.model)
		opt.add_hook(optimizer.GradientClipping(5))
		if FLAG_GPU:
			self.model.to_gpu(0)
		self.model.reset()

		# 学習開始
		st = datetime.datetime.now()
		for epoch in range(self.npz_num, epoch_num):
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
			if (epoch+1) % 10 == 0:
				# モデルの保存
				serializers.save_npz("./train/" + str(epoch+1) + ".npz", self.model)
			# if (epoch+1)%10 == 0: # 1epochがでかいので，毎epochで表示
			ed = datetime.datetime.now()
			print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
			st = datetime.datetime.now()