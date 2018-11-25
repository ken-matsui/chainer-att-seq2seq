# coding: utf-8

import os
import re
import datetime

import chainer.functions as F
from chainer import optimizer, optimizers, serializers, cuda
import numpy as np


class Trainer(object):
	def __init__(self, model, npz=None, flag_gpu=False):
		self.model = model
		if npz is not None:
			serializers.load_npz(npz, self.model)
			self.npz_num = int(re.search(r'[0-9]+', npz).group())
		else:
			self.npz_num = 0
		if flag_gpu:
			import cupy as cp
			self.xp = cp
			cuda.get_device(0).use()
		else:
			self.xp = np
		self.flag_gpu = flag_gpu

	def fit(self, queries, responses, train_path, epoch_num=30, batch_size=40, tag=None):
		train_queries = self.xp.vstack(self.xp.array(queries))
		train_responses = self.xp.vstack(self.xp.array(responses))
		teacher_num = min(len(train_queries), len(train_responses))

		opt = optimizers.Adam()
		opt.setup(self.model)
		opt.add_hook(optimizer.GradientClipping(5))
		if self.flag_gpu:
			self.model.to_gpu(0)
		self.model.reset()

		# 学習開始
		st = datetime.datetime.now()
		for epoch in range(self.npz_num, epoch_num):
			# ミニバッチ学習
			perm = np.random.permutation(teacher_num) # ランダムでuniqueな整数列リストを取得
			total_loss = 0
			total_accuracy = 0
			for i in range(0, teacher_num, batch_size):
				# モデルの勾配などをリセット
				self.model.reset()
				# 整数列リストからそれぞれのwordを取得
				enc_words = train_queries[perm[i:i+batch_size]].T
				dec_words = train_responses[perm[i:i+batch_size]].T
				# エンコード時のバッチサイズ
				encode_batch_size = len(enc_words[0])
				# エンコードの計算
				self.model.encode(enc_words, encode_batch_size)
				# <eos>をデコーダーに読み込ませる
				t = self.xp.array([0] * encode_batch_size, dtype='int32')
				# 損失の初期化
				loss = self.xp.zeros((), dtype='float32')
				# 精度の初期化
				accuracy = self.xp.zeros((), dtype='float32')
				# １単語ずつデコードする
				for w in dec_words:
					y = self.model.decode(t)
					t = self.xp.array(w, dtype='int32') # 正解単語をarrayに変換
					loss += F.softmax_cross_entropy(y, t) # 正解単語と予測単語を照らし合わせて損失を計算
					accuracy += F.accuracy(y, t) # 精度の計算
				loss.backward()
				loss.unchain_backward()
				opt.update()
				total_loss += loss.data
				total_accuracy += accuracy.data
			if (epoch+1) % 10 == 0:
				# モデルの保存
				if self.flag_gpu: # modelをCPUでも使えるように
					self.model.to_cpu()
				serializers.save_npz(train_path + str(epoch+1) + ".npz", self.model)
				if self.flag_gpu:
					self.model.to_gpu(0)
			ed = datetime.datetime.now()
			epoch_data = "epoch: {}\ttag: {}\n".format(epoch + 1, str(tag))
			loss_data = "\tloss: {}\n".format(round(float(total_loss),2))
			accuracy_data = "\taccuracy: {}\n".format(round(float(total_accuracy),2))
			time_data = "\ttime: {}".format(ed-st)
			text = epoch_data + loss_data + accuracy_data + time_data
			print(text)
			st = datetime.datetime.now()