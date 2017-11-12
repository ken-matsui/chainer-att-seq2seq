# coding: utf-8

import chainer.functions as F
from chainer import optimizers
from chainer import optimizer
from chainer import serializers
from chainer import Variable
import random
import numpy as np

# OUTPUT_PATH = './train/model/seq2seq/EMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.npz'
OUTPUT_PATH = './train/EMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.npz'
FLAG_GPU = False

EMBED_SIZE = 300
HIDDEN_SIZE = 150

class Trainer(object):
	def __init__(self, model):
		self.model = model

	def __forward(self, enc_words, dec_words, batch_size, model, ARR):
		"""
		順伝播の計算を行う関数
		:param enc_words: 発話文の単語を記録したリスト
		:param dec_words: 応答文の単語を記録したリスト
		:param model: Seq2Seqのインスタンス
		:param ARR: cuda.cupyかnumpyか
		:return: 計算した損失の合計
		"""
		# バッチサイズを記録
		# batch_size = len(enc_words[0])
		# model内に保存されている勾配をリセット
		model.reset()
		# 発話リスト内の単語を、chainerの型であるVariable型に変更
		enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
		# エンコードの計算 ①
		model.encode(enc_words)
		# 損失の初期化
		loss = Variable(ARR.zeros((), dtype='float32'))
		# <eos>をデコーダーに読み込ませる ②
		t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
		# デコーダーの計算
		for w in dec_words:
			# 1単語ずつをデコードする ③
			y = model.decode(t)
			# 正解単語をVariable型に変換
			t = Variable(ARR.array(w, dtype='int32'))
			# 正解単語と予測単語を照らし合わせて損失を計算 ④
			loss += F.softmax_cross_entropy(y, t)
		return loss

	def __forward_test(self, enc_words, model, ARR):
		ret = []
		model.reset()
		enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
		model.encode(enc_words)
		t = Variable(ARR.array([0], dtype='int32'))
		counter = 0
		while counter < 50:
			y = model.decode(t)
			label = y.data.argmax()
			ret.append(label)
			t = Variable(ARR.array([label], dtype='int32'))
			counter += 1
			if label == 1:
				counter = 50
		return ret

	# trainの関数
	def __make_minibatch(self, minibatch):
		# enc_wordsの作成
		enc_words = [row[0] for row in minibatch]
		enc_max = np.max([len(row) for row in enc_words])
		enc_words = np.array([[-1]*(enc_max - len(row)) + row for row in enc_words], dtype='int32')
		enc_words = enc_words.T

		# dec_wordsの作成
		dec_words = [row[1] for row in minibatch]
		dec_max = np.max([len(row) for row in dec_words])
		dec_words = np.array([row + [-1]*(dec_max - len(row)) for row in dec_words], dtype='int32')
		dec_words = dec_words.T
		return enc_words, dec_words

	def fit(self, X1, X2, epoch_num=30, batch_size=40, plotting=False):
		# GPUのセット
		if FLAG_GPU:
			ARR = cuda.cupy
			cuda.get_device(0).use()
			self.model.to_gpu(0)
		else:
			ARR = np # TODO: xpとするのが一般的

		# 学習開始
		for epoch in range(epoch_num):
			# エポックごとにoptimizerの初期化
			opt = optimizers.Adam()
			opt.setup(self.model)
			opt.add_hook(optimizer.GradientClipping(5))
			# ミニバッチ学習
			perm = np.random.permutation(len(list(zip(X1, X2))))
			for num in range(len(list(zip(X1, X2)))//batch_size):
				# minibatch = X[num*batch_size: (num+1)*batch_size]
				# 読み込み用のデータ作成
				# enc_words, dec_words = self.__make_minibatch(minibatch)
				BATCH_SIZE = len(X1)
				X1 = np.array(X1)
				X2 = np.array(X2)
				enc_words = X1[perm[num:num+batch_size]][0]
				dec_words = X2[perm[num:num+batch_size]][1]
				# modelのリセット
				self.model.reset()
				# 順伝播
				total_loss = self.__forward(enc_words=enc_words,
											dec_words=dec_words,
											batch_size=BATCH_SIZE,
											model=self.model,
											ARR=ARR)
				# 学習
				total_loss.backward()
				opt.update()
				opt.zero_grads()
				# print (datetime.datetime.now())
			print ('Epoch %s 終了' % (epoch+1))
			outputpath = OUTPUT_PATH%(EMBED_SIZE, HIDDEN_SIZE, batch_size, epoch+1)
			serializers.save_npz(outputpath, self.model)