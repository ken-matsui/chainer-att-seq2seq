# coding: utf-8

import os
import re
import datetime
from os.path import join, dirname

import chainer.functions as F
from chainer import optimizer, optimizers, serializers, Variable
import numpy as np
from dotenv import load_dotenv
import slackweb

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)
slack = slackweb.Slack(os.environ.get("SLACK_WEBHOOK"))

class Trainer(object):
	def __init__(self, model, npz=None):
		self.model = model
		if npz is not None:
			serializers.load_npz(npz, self.model)
			self.npz_num = int(re.search(r'[0-9]+', npz).group())
		else:
			self.npz_num = 0

	def fit(self, queries, responses, teacher_num, epoch_num=30, batch_size=40, flag_gpu=False):
		if flag_gpu:
			import cupy as cp
			from chainer import cuda
			xp = cp
			cuda.get_device(0).use()
		else:
			xp = np

		queries = xp.vstack(xp.array(queries))
		responses = xp.vstack(xp.array(responses))

		opt = optimizers.Adam()
		opt.setup(self.model)
		opt.add_hook(optimizer.GradientClipping(5))
		if flag_gpu:
			self.model.to_gpu(0)
		self.model.reset()

		# 学習開始
		st = datetime.datetime.now()
		for epoch in range(self.npz_num, epoch_num):
			# ミニバッチ学習
			perm = np.random.permutation(teacher_num) # ランダムな整数列リストを取得
			total_loss = 0
			total_accuracy = 0
			for i in range(0, teacher_num, batch_size):
				# モデルの勾配などをリセット
				self.model.reset()
				# 整数列リストからそれぞれのwordを取得
				enc_words = queries[perm[i:i+batch_size]].T
				dec_words = responses[perm[i:i+batch_size]].T
				# エンコード時のバッチサイズ
				encode_batch_size = len(enc_words[0])
				# 発話リスト内の単語をVariable型に変更
				enc_words = [Variable(xp.array(row, dtype='int32')) for row in enc_words]
				# エンコードの計算
				self.model.encode(enc_words, encode_batch_size)
				# <eos>をデコーダーに読み込ませる
				t = Variable(xp.array([0] * encode_batch_size, dtype='int32'))
				# 損失の初期化
				loss = Variable(xp.zeros((), dtype='float32'))
				# 精度の初期化
				accuracy = Variable(xp.zeros((), dtype='float32'))
				# １単語ずつデコードする
				for w in dec_words:
					y = self.model.decode(t)
					t = Variable(xp.array(w, dtype='int32')) # 正解単語をVariable型に変換
					loss += F.softmax_cross_entropy(y, t) # 正解単語と予測単語を照らし合わせて損失を計算
					accuracy += F.accuracy(y, t) # 精度の計算
				loss.backward()
				loss.unchain_backward()
				opt.update()
				total_loss += loss.data
				total_accuracy += accuracy.data
			if (epoch+1) % 10 == 0:
				# モデルの保存
				if flag_gpu: # modelをCPUでも使えるように
					self.model.to_cpu()
				serializers.save_npz("./train/" + str(epoch+1) + ".npz", self.model)
				if flag_gpu:
					self.model.to_gpu(0)
			ed = datetime.datetime.now()
			data = "epoch: {}\n\tloss: {}\n\taccuracy: {}\n\ttime: {}".format(epoch+1, round(float(total_loss),2), round(float(total_accuracy),2), ed-st)
			slack.notify(text=data)
			print(data)
			st = datetime.datetime.now()