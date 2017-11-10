# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
import numpy as np

class LSTM_Encoder(chainer.Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		"""
		クラスの初期化
		:param vocab_size: 使われる単語の種類数（語彙数）
		:param embed_size: 単語をベクトル表現した際のサイズ
		:param hidden_size: 隠れ層のサイズ
		"""
		super(LSTM_Encoder, self).__init__(
			xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
			eh = links.Linear(embed_size, 4 * hidden_size),
			hh = links.Linear(hidden_size, 4 * hidden_size)
		)
	def __call__(self, x, c, h):
		"""
		:param x: one-hotな単語
		:param c: 内部メモリ
		:param h: 隠れ層
		:return: 次の内部メモリ、次の隠れ層
		"""
		e = functions.tanh(self.xe(x))
		return functions.lstm(c, self.eh(e) + self.hh(h))

class LSTM_Decoder(chainer.Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		"""
		クラスの初期化
		:param vocab_size: 使われる単語の種類数（語彙数）
		:param embed_size: 単語をベクトル表現した際のサイズ
		:param hidden_size: 隠れ層のサイズ
		"""
		super(LSTM_Decoder, self).__init__(
			ye = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
			eh = links.Linear(embed_size, 4 * hidden_size),
			hh = links.Linear(hidden_size, 4 * hidden_size),
			he = links.Linear(hidden_size, embed_size),
			ey = links.Linear(embed_size, vocab_size)
		)
	def __call__(self, y, c, h):
		"""
		:param y: one-hotな単語
		:param c: 内部メモリ
		:param h: 隠れ層
		:return: 予測単語、次の内部メモリ、次の隠れ層
		"""
		e = functions.tanh(self.ye(y))
		c, h = functions.lstm(c, self.eh(e) + self.hh(h))
		t = self.ey(functions.tanh(self.he(h)))
		return t, c, h

# optimizer.setup(model)で渡す理由
class Seq2Seq(chainer.Chain):
	"""docstring for Seq2Seq"""
	def __init__(self, vocab_size, embed_size, hidden_size, batch_size, flag_gpu=True):
		"""
		Seq2Seqの初期化
		:param vocab_size: 語彙サイズ
		:param embed_size: 単語ベクトルのサイズ
		:param hidden_size: 中間ベクトルのサイズ
		:param batch_size: ミニバッチのサイズ
		:param flag_gpu: GPUを使うかどうか
		"""
		# superはモデルとして定義する．（ネットワークの側）(chainerの仕様)
		super(Seq2Seq, self).__init__(
			# Encoderのインスタンス化
			encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
			# Decoderのインスタンス化
			decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
		)
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		# GPUで計算する場合はcupyをCPUで計算する場合はnumpyを使う
		if flag_gpu:
			self.ARR = cuda.cupy
		else:
			self.ARR = np

	def encode(self, words):
		"""
		Encoderを計算する部分
		:param words: 単語が記録されたリスト
		:return:
		"""
		# 内部メモリ、中間ベクトルの初期化
		c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
		h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

		# エンコーダーに単語を順番に読み込ませる
		for w in words:
			c, h = self.encoder(w, c, h)

		# 計算した中間ベクトルをデコーダーに引き継ぐためにインスタンス変数にする
		self.h = h
		# 内部メモリは引き継がないので、初期化
		self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

	def decode(self, w):
		"""
		デコーダーを計算する部分
		:param w: 単語
		:return: 単語数サイズのベクトルを出力する
		"""
		t, self.c, self.h = self.decoder(w, self.c, self.h)
		return t

	def reset(self):
		"""
		中間ベクトル、内部メモリ、勾配の初期化
		:return:
		"""
		self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
		self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

		self.zerograds()
