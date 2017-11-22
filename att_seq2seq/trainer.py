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

	def fit(self, queries, responses, train_path, epoch_num=30, batch_size=40, flag_gpu=False):
		if flag_gpu:
			import cupy as cp
			from chainer import cuda
			xp = cp
			cuda.get_device(0).use()
		else:
			xp = np

		# Train Data と Test Data に分割
		test_queries, train_queries = queries[:batch_size], queries[batch_size:]
		test_responses, train_responses = responses[:batch_size], responses[batch_size:]
		teacher_num = len(list(zip(train_queries, train_responses)))
		train_queries = xp.vstack(xp.array(train_queries))
		train_responses = xp.vstack(xp.array(train_responses))
		test_queries = xp.vstack(xp.array(test_queries))
		test_responses = xp.vstack(xp.array(test_responses))

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
				enc_words = train_queries[perm[i:i+batch_size]].T
				dec_words = train_responses[perm[i:i+batch_size]].T
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
			# 評価計算
			total_evaluation = 0
			for j in range(0, batch_size):
				self.model.reset()
				enc_words = test_queries[j].T
				dec_words = test_responses[j].T
				encode_batch_size = len(enc_words[0])
				enc_words = [Variable(xp.array(row, dtype='int32')) for row in enc_words]
				self.model.encode(enc_words, encode_batch_size)
				t = Variable(xp.array([0] * encode_batch_size, dtype='int32'))
				# 評価の初期化
				evaluation = Variable(xp.zeros((), dtype='float32'))
				for w in dec_words:
					y = self.model.decode(t)
					t = Variable(xp.array(w, dtype='int32'))
					evaluation += F.accuracy(y, t) # 評価の計算
				total_evaluation += evaluation
			if (epoch+1) % 10 == 0:
				# モデルの保存
				if flag_gpu: # modelをCPUでも使えるように
					self.model.to_cpu()
				serializers.save_npz(train_path + str(epoch+1) + ".npz", self.model)
				if flag_gpu:
					self.model.to_gpu(0)
			ed = datetime.datetime.now()
			epoch_data = "epoch: {}\n".format(epoch + 1)
			loss_data = "\tloss: {}\n".format(round(float(total_loss),2))
			accuracy_data = "\taccuracy: {}\n".format(round(float(total_accuracy),2))
			evaluation_data = "\tevaluation: {}\n".format(round(float(total_evaluation),2))
			time_data = "\ttime: {}".format(ed-st)
			data = epoch_data + loss_data + accuracy_data + evaluation_data + time_data
			slack.notify(text=data)
			print(data)
			st = datetime.datetime.now()