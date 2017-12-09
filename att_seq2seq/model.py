# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

__all__ = ['AttSeq2Seq']

# LSTMエンコーダークラス
class LSTMEncoder(chainer.Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		'''
		Encoderのインスタンス化
		:param vocab_size: 使われる単語の種類数
		:param embed_size: 単語をベクトル表現した際のサイズ
		:param hidden_size: 隠れ層のサイズ
		'''
		super(LSTMEncoder, self).__init__()
		with self.init_scope():
			self.xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1)
			self.eh = L.Linear(embed_size, 4 * hidden_size)
			self.hh = L.Linear(hidden_size, 4 * hidden_size)

	def __call__(self, x, c, h):
		'''
		Encoderの計算
		:param x: one-hotな単語
		:param c: 内部メモリ
		:param h: 隠れ層
		:return: 次の内部メモリ、次の隠れ層
		'''
		e = F.tanh(self.xe(x))
		return F.lstm(c, self.eh(e) + self.hh(h))

# Attention Model + LSTMデコーダークラス
class AttLSTMDecoder(chainer.Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		'''
		Attention ModelのためのDecoderのインスタンス化
		:param vocab_size: 語彙数
		:param embed_size: 単語ベクトルのサイズ
		:param hidden_size: 隠れ層のサイズ
		'''
		super(AttLSTMDecoder, self).__init__()
		with self.init_scope():
			self.ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1) # 単語を単語ベクトルに変換する層
			self.eh = L.Linear(embed_size, 4 * hidden_size) # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
			self.hh = L.Linear(hidden_size, 4 * hidden_size) # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
			self.fh = L.Linear(hidden_size, 4 * hidden_size) # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
			self.bh = L.Linear(hidden_size, 4 * hidden_size) # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
			self.he = L.Linear(hidden_size, embed_size) # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
			self.ey = L.Linear(embed_size, vocab_size) # 単語ベクトルを語彙数サイズのベクトルに変換する層

	def __call__(self, y, c, h, f, b):
		'''
		Decoderの計算
		:param y: Decoderに入力する単語
		:param c: 内部メモリ
		:param h: Decoderの中間ベクトル
		:param f: Attention Modelで計算された順向きEncoderの加重平均
		:param b: Attention Modelで計算された逆向きEncoderの加重平均
		:return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
		'''
		e = F.tanh(self.ye(y)) # 単語を単語ベクトルに変換
		c, h = F.lstm(c, self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b)) # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
		t = self.ey(F.tanh(self.he(h))) # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
		return t, c, h

# Attentionモデルクラス
class Attention(chainer.Chain):
	def __init__(self, hidden_size):
		'''
		Attentionのインスタンス化
		:param hidden_size: 隠れ層のサイズ
		'''
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
		with self.init_scope():
			self.fh = L.Linear(hidden_size, hidden_size) # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
			self.bh = L.Linear(hidden_size, hidden_size) # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
			self.hh = L.Linear(hidden_size, hidden_size) # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
			self.hw = L.Linear(hidden_size, 1) # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層

	def __call__(self, fs, bs, h):
		'''
		Attentionの計算
		:param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
		:param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
		:param h: Decoderで出力された中間ベクトル
		:return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
		'''
		# 引数の形から，numpyかcupyかを判断する．
		xp = cuda.get_array_module(h.data)
		batch_size = h.data.shape[0]
		# weight
		ws = []
		sum_w = xp.zeros((batch_size, 1), dtype='float32') # ウェイトの合計値を計算するための値を初期化
		# Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
		for f, b in zip(fs, bs):
			w = F.tanh(self.fh(f)+self.bh(b)+self.hh(h)) # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
			w = F.exp(self.hw(w)) # softmax関数を使って正規化する
			ws.append(w) # 計算したウェイトを記録
			sum_w += w
		# 出力する加重平均ベクトルの初期化
		att_f = xp.zeros((batch_size, self.hidden_size), dtype='float32')
		att_b = xp.zeros((batch_size, self.hidden_size), dtype='float32')
		for f, b, w in zip(fs, bs, ws):
			w /= sum_w # ウェイトの和が1になるように正規化
			# ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
			att_f += F.reshape(F.batch_matmul(f, w), (batch_size, self.hidden_size))
			att_b += F.reshape(F.batch_matmul(b, w), (batch_size, self.hidden_size))
		return att_f, att_b

# Attention Sequence to Sequence Modelクラス
class AttSeq2Seq(chainer.Chain):
	def __init__(self, vocab_size, embed_size, hidden_size, flag_gpu=False):
		'''
		Attention + Seq2Seqのインスタンス化
		:param vocab_size: 語彙数のサイズ
		:param embed_size: 単語ベクトルのサイズ
		:param hidden_size: 隠れ層のサイズ
		'''
		super(AttSeq2Seq, self).__init__()
		with self.init_scope():
			self.f_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size) # 順向きのEncoder
			self.b_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size) # 逆向きのEncoder
			self.attention = Attention(hidden_size) # Attention Model
			self.decoder = AttLSTMDecoder(vocab_size, embed_size, hidden_size) # Decoder
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		# 順向きのEncoderの中間ベクトルを保存する用
		self.fs = []
		# 逆向きのEncoderの中間ベクトルを保存する用
		self.bs = []

	def encode(self, words, batch_size):
		'''
		Encoderの計算
		:param words: 入力で使用する単語記録されたリスト
		:param batch_size: ミニバッチのサイズ
		:return:
		'''
		# 内部メモリ、中間ベクトルの初期化
		xp = cuda.get_array_module(words)
		# 発話リスト内の単語をrowで分割
		words = [xp.array(row, dtype='int32') for row in words]
		c = xp.zeros((batch_size, self.hidden_size), dtype='float32')
		h = xp.zeros((batch_size, self.hidden_size), dtype='float32')
		# 順向きのEncoderの計算
		for w in words:
			c, h = self.f_encoder(w, c, h)
			self.fs.append(h) # 計算された中間ベクトルを記録
		# 内部メモリ、中間ベクトルの初期化
		c = xp.zeros((batch_size, self.hidden_size), dtype='float32')
		h = xp.zeros((batch_size, self.hidden_size), dtype='float32')
		# 逆向きのEncoderの計算
		for w in reversed(words):
			c, h = self.b_encoder(w, c, h)
			self.bs.insert(0, h) # 計算された中間ベクトルを記録
		# 内部メモリ、中間ベクトルの初期化
		self.c = xp.zeros((batch_size, self.hidden_size), dtype='float32')
		self.h = xp.zeros((batch_size, self.hidden_size), dtype='float32')

	def decode(self, w):
		'''
		Decoderの計算
		:param w: Decoderで入力する単語
		:return: 予測単語
		'''
		att_f, att_b = self.attention(self.fs, self.bs, self.h)
		t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
		return t

	def reset(self):
		'''
		インスタンス変数を初期化する
		Encoderの中間ベクトルを記録するリストの初期化
		'''
		self.fs = []
		self.bs = []
		# 勾配の初期化
		self.zerograds()
